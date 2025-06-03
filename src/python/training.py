import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import sklearn
import torch
try:
    import rich.traceback
    import rich
    from rich import print
except ImportError:
    print("Rich library not found. Skipping rich features.")
    rich = None
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from build_data import *
from train_utils import train_model, evaluate_model
from model import LSTMModel, LSTMSEQ


ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

# Set the random seed for numpy
np.random.seed(1)
# Set the random seed for PyTorch
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#########################################userparams##########################################

# The number of epochs
epochs = 32
# The batch size
batch_size = 128

# The number of units in the LSTM layer
lstm_units = 128
# The number of units in the dense layer
dense_units = 32
# The learning rate (lr)
learning_rate = 0.001

## This Variable Defines the Size of Sliding Window
## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor) 
w = 128 # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
s = 10

################################################################################################

# The test size
test_size = 0.2
# The validation size
validation_size = 0.2

# The number of columns in each window
num_columns = 3

# The number of classes
num_classes = len(ACT_LABELS)

################################################################################################
# SECTION I: DATA PREPARATION
################################################################################################

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
trial_codes = [TRIAL_CODES[act] for act in ACT_LABELS]
dataset = creat_time_series(set_data_types(["userAcceleration"]), ACT_LABELS, trial_codes, mode="raw", labeled=True, data_dir=glob("**/A_DeviceMotion_data", recursive=True)[0])
print("Shape of time-Series dataset:"+str(dataset.shape))    

test_trail = [11,12,13,14,15,16]  
print("Test Trials: "+str(test_trail))
test_ts = dataset.loc[(dataset['trial'].isin(test_trail))]
train_ts = dataset.loc[~(dataset['trial'].isin(test_trail))]


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


# TODO: determine if ID needed at all to identify unique subjects
# Split the training data into training and validation sets
train_data, val_data, act_train, act_val, id_train, id_val = train_test_split(train_data, 
                                                                              act_train, 
                                                                              id_train, 
                                                                              test_size=validation_size, 
                                                                              random_state=1, 
                                                                              stratify=act_train  # TODO: what is this and do we even need it?
)

print(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}, Test data shape: {test_data.shape}")
print(f"Train labels shape: {act_train.shape}, Validation labels shape: {act_val.shape}, Test labels shape: {act_test.shape}")

# convert the data to PyTorch tensors
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).permute(0, 2, 1).to(device)
val_data_tensor   = torch.tensor(val_data, dtype=torch.float32).permute(0, 2, 1).to(device)
test_data_tensor  = torch.tensor(test_data, dtype=torch.float32).permute(0, 2, 1).to(device)

train_labels_tensor = torch.tensor(act_train, dtype=torch.long).to(device)
val_labels_tensor = torch.tensor(act_val, dtype=torch.long).to(device)
test_labels_tensor = torch.tensor(act_test, dtype=torch.long).to(device)

# Create DataLoadera with consistent first dimensions.
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"dim train_data: {train_data_tensor.shape}, dim train_labels: {train_labels_tensor.shape}")
print(f"dim val_data: {val_data_tensor.shape}, dim val_labels: {val_labels_tensor.shape}")

################################################################################################
# SECTION II: MODEL DEFINITION
################################################################################################

# Initialize the model
model = LSTMModel(input_size=num_columns, 
                  lstm_units=lstm_units, 
                  dense_units=dense_units, 
                  num_classes=num_classes)

################################################################################################
# SECTION III: MODEL TRAINING
################################################################################################
histroy = train_model(model,
                      train_loader,
                      val_loader,
                      "LSTMModel",
                      device,
                      epochs=epochs
                    )

################################################################################################
# SECTION IV: MODEL EVALUATION
################################################################################################

loss, accuracy, predictions, labels = evaluate_model(model, test_loader, plots_folder='plots')
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

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
# plt.savefig(os.path.join(plots_folder, "training_curves.png"))
plt.show()

