import torch
import torch.nn as nn

class CNN_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=2)
        
        # RNN
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        # Permute to (batch_size, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # Apply convolution
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, features)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)

        # Extract the output at the last time step
        last_outputs = out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layer
        out = self.fc(last_outputs)
        return out



class CNN_LSTM(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Rearrange dimensions for Conv1D
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        
        # Pass through Conv1D
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, hidden_size, reduced_seq_len)
        
        # Rearrange dimensions for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, reduced_seq_len, hidden_size)

        # Pass through LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, reduced_seq_len, hidden_size)

        # Take the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        return out