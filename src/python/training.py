import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import requests
import torch
from tqdm.rich import tqdm
from zipfile import ZipFile
from rich import print
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from build_data import set_data_types, creat_time_series, ts_to_secs
from train_utils import train_model, evaluate_model
from model import *
from cyclopts import App

app = App(
    name="MotionMetric Training"
)

# https://github.com/mmalekzadeh/motion-sense#labels
ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

@app.command
def get_ds(root: str, filename: str = "A_DeviceMotion_data.zip"):
    """Download and prepare the MotionSense dataset.

    Args:
        root (str): Root directory where the dataset will be stored.
        filename (str, optional): Name of the zip file to download. Please refer to https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data for available files.
    """
    # make the directories
    if not os.path.exists(f"{root}"):
        os.makedirs(f"{root}")

    # download the zip file
    url = f"https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/{filename}"
    filepath = f"{root}/{filename}" 

    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    response = requests.get(url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(desc="Downloading dataset...", total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    if total_size != 0 and progress_bar.n != total_size:
        exit("ERROR: Could not download file")

    # unzip
    zip_file = ZipFile(filepath)
    zip_file.extractall(f"{root}")
    zip_file.close()

    # remove the zip file
    os.remove(filepath)
    
    ## convert to time-series data
    trial_codes = [TRIAL_CODES[act] for act in ACT_LABELS]
    
    creat_time_series(set_data_types(["userAcceleration"]), ACT_LABELS, trial_codes, mode="raw", labeled=True, data_dir=filepath.replace(".zip", "")).to_csv(f"{root}/dataset.csv", index=False)
    

@app.default
def top(epochs: int,
          batch_size: int = 1,
          lr: float = 3e-4,
          w: int = 128,
          s: int = 4,
          validation_size: float = 0.2,
          num_columns: int = 3,
          lstm_units: int = 512,
          dense_units: int = 128,
          num_classes: int = len(ACT_LABELS),
          plots_folder='documents/plots',
          root: str = 'data',
          set_seed: int = 1,
          model_dir: str | None = None):
    """
    Main function to train the LSTM model.
    
    Args:
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        lstm_units (int): Number of units in the LSTM layer.
        dense_units (int): Number of units in the dense layer.
        lr (float): Learning rate for the optimizer.
        w (int): Size of the sliding window.
        s (int): Step size for building different snapshots from time-series data.
        validation_size (float): Proportion of the dataset to include in the validation split.
        num_columns (int): Number of columns in each window.
        num_classes (int): Number of classes in the dataset.
        plots_folder (str): Folder to save plots.
        root (str): Root directory for the data files.
        set_seed (int | None): Random seed for reproducibility. If -1, no seed is set.
        model_dir (str | None): Directory to save the trained model. If None, the model is not saved.
    """
    
    # print out the command entered
    print(f"Training with parameters: epochs={epochs}, batch_size={batch_size}, lstm_units={lstm_units}, dense_units={dense_units}, lr={lr}, w={w}, s={s}, validation_size={validation_size}, num_columns={num_columns}, num_classes={num_classes}:", file=open("latest-parameters.log", "a+"))
    print(f"Training with parameters: epochs={epochs}, batch_size={batch_size}, lstm_units={lstm_units}, dense_units={dense_units}, lr={lr}, w={w}, s={s}, validation_size={validation_size}, num_columns={num_columns}, num_classes={num_classes}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility# # Set the random seed for numpy
    if set_seed != -1:
        np.random.seed(set_seed)
        # Set the random seed for PyTorch
        torch.manual_seed(set_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(set_seed)
    else:
        print("-1 set_seed provided, not setting any seed for reproducibility.")


    ################################################################################################
    # SECTION I: DATA PREPARATION
    ################################################################################################
    
    # check if the csv files exist
    if not os.path.exists(f"{root}/dataset.csv"):
        get_ds(root)
        
    dataset = pd.read_csv(f"{root}/dataset.csv")
    
    test_trail = [11,12,13,14,15,16]  # split the dataset into train and test sets based on trial numbers (long trials for training, short trials for testing)

    test_ts = dataset.loc[(dataset['trial'].isin(test_trail))]
    train_ts = dataset.loc[~(dataset['trial'].isin(test_trail))]

    # save the train and test data as npy files
    train_ts.to_csv(f"{root}/train.csv", index=False)
    test_ts.to_csv(f"{root}/test.csv", index=False)

    train_data, act_train, id_train, train_mean, train_std = ts_to_secs(train_ts.copy(),
                                                                    w,
                                                                    s,
                                                                    standardize = True)
    test_data, act_test, id_test, test_mean, test_std = ts_to_secs(test_ts.copy(),
                                                                w,
                                                                s,
                                                                standardize = True,
                                                                mean = train_mean, 
                                                                std = train_std)
    
    # train_data, act_train, _, _, _ = ts_to_secs(train_ts.copy(), w, s)
    # test_data, act_test, _, _, _ = ts_to_secs(test_ts.copy(), w, s)
    
    # transpose the data to have the shape (num_samples, num_features, sequence_length)
    train_data = train_data.transpose(0, 2, 1)  # (num_samples, sequence_length, num_features)
    test_data = test_data.transpose(0, 2, 1)    # (num_samples, sequence_length, num_features)


    if set_seed != -1:
        random_state = set_seed
    else:
        random_state = None

    # Split the training data into training and validation sets
    train_data, val_data, act_train, act_val = train_test_split(train_data, act_train, test_size=validation_size, random_state=random_state, stratify=act_train)

    print(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}, Test data shape: {test_data.shape}")
    print(f"Train labels shape: {act_train.shape}, Validation labels shape: {act_val.shape}, Test labels shape: {act_test.shape}")

    # Create DataLoadera with consistent first dimensions.
    train_dataset = TensorDataset(torch.from_numpy(train_data).float().to(device), torch.from_numpy(act_train).long().to(device))
    val_dataset = TensorDataset(torch.from_numpy(val_data).float().to(device), torch.from_numpy(act_val).long().to(device))
    test_dataset = TensorDataset(torch.from_numpy(test_data).float().to(device), torch.from_numpy(act_test).long().to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ################################################################################################
    # SECTION II: MODEL DEFINITION
    ################################################################################################

    # Initialize the model
    model = LSTMModel(input_size=num_columns, 
                    lstm_units=lstm_units, 
                    dense_units=dense_units, 
                    num_classes=num_classes)
    
    print(f"Model: {model}")

    ################################################################################################
    # SECTION III: MODEL TRAINING
    ################################################################################################
    histroy = train_model(model,
                        train_loader,
                        val_loader,
                        "LSTMModel",
                        device,
                        epochs=epochs,
                        learning_rate=lr,
                        )

    ################################################################################################
    # SECTION IV: MODEL EVALUATION
    ################################################################################################

    loss, accuracy, predictions, labels = evaluate_model(model, test_loader)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\tTest Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%", file=open("latest-parameters.log", "a+"))
    
    # save the model if model_dir is provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"lstm_model-accuracy{accuracy:.4f}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    # print out accuracy for each class
    print("Test Accuracy per class:")
    for i, label in enumerate(ACT_LABELS):
        class_accuracy = np.mean(predictions[labels == i] == i)
        print(f"\t{label}: {class_accuracy * 100:.2f}%")
    
    # Plot the training and validation accuracy curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(histroy["accuracy"], label="LSTM Training Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(histroy["loss"], label="LSTM Training Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_folder, "training_curves.png"))
    print(f"Training curves saved to {os.path.join(plots_folder, 'training_curves.png')}")

if __name__ == "__main__":
    sys.exit(app())