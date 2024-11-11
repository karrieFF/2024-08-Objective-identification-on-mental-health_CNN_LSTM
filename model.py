import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F

# Define the CNN+LSTM model class
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=2) #, activation = 'relu' # Set conv input to match feature size (39)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, seq_lengths):
        # Pass through Conv1d layer
        x = x.permute(0, 2, 1)  # Rearrange for Conv1d: (batch_size, features, seq_len)
        x = self.conv(x)
        
        # Pack the sequence to handle variable lengths
        x = x.permute(0, 2, 1)  # Rearrange back for LSTM: (batch_size, seq_len, features)
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        #print(packed_input.shape)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #print(h0.shape)
        #print(c0.shape)
        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        
        # Unpack the sequence
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the output of the last valid time step
        out = out[torch.arange(out.size(0)), seq_lengths - 1]
        
        # Pass through the fully connected layer
        out = self.fc(out)

        # Apply softmax
        out = torch.sigmoid(out, dim=1)
        
        return out
        