import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units, num_classes):
        super(LSTMModel, self).__init__()
        # ADDED: batch norm 
        self.batch_norm = nn.BatchNorm1d(input_size) 
        
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True)
        
        # TODO: determin if we need activation functions for dense layers
        # Dense layer with ReLU activation
        self.dense = nn.Linear(lstm_units, dense_units)
        self.relu = nn.ReLU()
        
        # Output layer with Sigmoid (?) activation
        self.output = nn.Linear(dense_units, num_classes)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply batch normalization
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)
            x = self.batch_norm(x)
            x = x.reshape(batch_size, seq_len, features)
        else:
            x = self.batch_norm(x)
            
        # LSTM layer
        x, _ = self.lstm1(x)

        x = torch.max(x, dim=1)[0]  # Global max pooling
        
        # Dense layer with ReLU activation
        x = self.dense(x)
        x = self.relu(x)
        
        # Output layer with Sigmoid activation
        x = self.output(x)
        # x = self.sigmoid(x)
        
        return x
 
class LSTMRnnModel(nn.Module):
    """ Based from: https://www.mdpi.com/electronics/electronics-10-01715/article_deploy/html/images/electronics-10-01715-g007.png
    """
    def __init__(self, input_size, lstm_units, dense_units, num_classes):
        super(LSTMRnnModel, self).__init__()
        
        # Dense 1 layer
        self.dense1 = nn.Linear(input_size, dense_units)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(dense_units)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(dense_units, lstm_units, batch_first=True, num_layers=3)
        
        # Dense 2 (output layer)
        self.dense2 = nn.Linear(lstm_units, num_classes)
        
    def forward(self, x):
        # Dense 1
        x = self.dense1(x)
        
        # Batch Normalization
        # If input is 3D (batch, sequence, features), we need to reshape
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)
            x = self.batch_norm(x)
            x = x.reshape(batch_size, seq_len, features)
        else:
            x = self.batch_norm(x)
        
        # LSTM 1 + Dropout 1
        x, _ = self.lstm1(x)
        
        # pool
        x = torch.max(x, dim=1)[0]
        
        # Dense 2 (output)
        x = self.dense2(x)
        
        return x