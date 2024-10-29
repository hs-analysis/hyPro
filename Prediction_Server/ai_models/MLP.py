# hyPro/Server/ai_models/LSTM.py
# Defines the default LSTM model for the hyPro Server.

# HS Analysis GmbH, 2024
# Author: Philipp Marquardt

import torch
import torch.nn as nn
from typing import List


class TimeSeriesMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [256, 128, 64],
    ):
        """
        MLP model for time series prediction with 2 output values.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
