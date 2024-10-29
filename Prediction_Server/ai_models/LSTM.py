# hyPro/Server/ai_models/LSTM.py
# Defines the default LSTM model for the hyPro Server.

# HS Analysis GmbH, 2024
# Author: Valentin Haas, Philipp Marquardt

# External Imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NUM_LAYERS = 3
HIDDEN_SIZE = 768
# 44 input size for combined


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)

        # First fully connected layer
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)

        # Second fully connected layer
        self.fc2 = nn.Linear(HIDDEN_SIZE // 2, output_size)

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM layer
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        last_outputs = []
        for i, length in enumerate(lengths):
            last_outputs.append(output[i, length - 1, :])
        out = torch.stack(last_outputs, dim=0)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        return out
