import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.dense = nn.Linear(lstm_units, dense_units)
        self.relu = nn.ReLU()
        self.output = nn.Linear(dense_units, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output of the LSTM
        x = lstm_out[:, -1, :]
        x = self.relu(self.dense(x))
        x = self.sigmoid(self.output(x))
        return x
    
class LSTMSEQ(nn.Module):
    def __init__(self, input_size, lstm_units, dense_units, num_classes):
        super(LSTMSEQ, self).__init__()
        self.main = nn.Sequential(
            nn.LSTM(input_size, lstm_units, batch_first=True),
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.main[0](x)
        # Use the last output of the LSTM
        x = lstm_out[:, -1, :]
        for layer in self.main[1:]:
            x = layer(x)
        return x