import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm

# Define training function
def train_model(model, train_loader, validation_loader, model_name, device, epochs=64, learning_rate=0.001, models_folder='models'):
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit=" batch"):
            optimizer.zero_grad()
            outputs = model(data)
            # loss = criterion(outputs.squeeze(), labels)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)
            # train_correct += (predicted == labels).sum().item()
            predicted = outputs.argmax(dim=1)       # [B]
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(validation_loader, desc="Validating", unit=" batch"):
                outputs = model(data)
                # loss = criterion(outputs.squeeze(), labels)
                loss = criterion(outputs, labels.long())
                
                val_loss += loss.item()
                # predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                predicted = outputs.argmax(dim=1)       # [B]
                val_correct += (predicted == labels).sum().item()
                train_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        val_loss /= len(validation_loader)
        
        # Save history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save(model.state_dict(), os.path.join(models_folder, f"{model_name}_model.pth"))
        
        print(f"\tTrain Loss: {train_loss:.4f}, \tTrain Acc: {train_accuracy:.4f}, \tVal Loss: {val_loss:.4f}, \tVal Acc: {val_accuracy:.4f}")
    
    return history

# Evaluate models on test data
def evaluate_model(model, test_loader, plots_folder='plots'):
    os.makedirs(plots_folder, exist_ok=True)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing", unit=" batch"):
            outputs = model(data)
            # loss = criterion(outputs.squeeze(), labels)
            loss = criterion(outputs, labels.long())
            
            test_loss += loss.item()
            test_total += labels.size(0)
            # predicted = (outputs.squeeze() > 0.5).float()
            predicted = outputs.argmax(dim=1)       # [B]
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = test_correct / test_total
    test_loss /= len(test_loader)
    
    return test_loss, test_accuracy, np.array(all_predictions), np.array(all_labels)

# Generate confusion matrices
def plot_confusion_matrix(y_true, y_pred, model_name, plots_folder='plots'):
    os.makedirs(plots_folder, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non Falling", "Falling"])
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(plots_folder, f"{model_name.lower()}_cm.png"))
    plt.show()