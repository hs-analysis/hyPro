# hyPro/Prediction_Server/ai_models/LSTM.py
# Defines the default LSTM model for the hyPro Server.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Python Imports

# External Imports
import torch
import torch.nn as nn


# Default Constants used during training
NUM_LAYERS = 2
HIDDEN_SIZE = 64


class LSTMModel(nn.Module):
    """
    Defines the default LSTM model for the hyPro Server.
    """

    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(0), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(0)
