# hyPro/Prediction_Server/dataset/clean.py
# Description: This file contains the cleaning functions for the dataset module.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# External imports
import numpy as np
import pandas as pd
import torch

# Internal imports
from data_models.ai_model import AIModel

# Constants
VAC_SUCK_LABELS = {
    "Off": 0,
    "Suck": 1,
}


def preprocess_sequence(
    sequence_data: pd.DataFrame, presets: np.ndarray = None
) -> tuple:
    """
    Preprocess a single sequence by optionally concatenating preset data at each timestep.

    Args:
        sequence_data (pd.DataFrame): The input time series data.
        presets (np.ndarray, optional): An array of preset values to concatenate to each timestep.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (data tensor, length tensor)
    """
    # Convert sequence_data to a tensor and handle NaN values
    sequence_tensor = torch.tensor(sequence_data.values, dtype=torch.float32)
    sequence_tensor = torch.nan_to_num(sequence_tensor, nan=0.0)

    # Check if presets is provided and not None
    if presets is not None:
        # Expand presets to match the number of timesteps in sequence_data
        num_timesteps = sequence_tensor.size(0)
        preset_expanded = (
            torch.tensor(presets, dtype=torch.float32)
            .unsqueeze(0)
            .expand(num_timesteps, -1)
        )

        # Concatenate the expanded presets to the sequence data along the feature dimension
        sequence_tensor = torch.cat((sequence_tensor, preset_expanded), dim=1)

    # Add batch dimension
    sequence_tensor = sequence_tensor.unsqueeze(0)

    # Create length tensor
    length = torch.tensor([sequence_tensor.size(1)], dtype=torch.long)

    return sequence_tensor, length


def prepare_timeseries(
    input_df: pd.DataFrame, columns: list[str], presets: np.array = None
) -> pd.DataFrame:
    """
    Prepare the input data for time series analysis. This function will convert the input data to a time series format.

    Args:
        input_df (pd.DataFrame): The input data to prepare as a pandas DataFrame.
        ai_model (AIModel): The AI model to prepare the data for.

    Returns:
        pd.DataFrame: The prepared time series data as a pandas DataFrame.
    """
    # Sort the incoming columns in the same order as the model input columns
    input_df = input_df[columns]
    # Convert timestamp to seconds since start
    input_df["Timestamp"] = pd.to_datetime(input_df["Timestamp"]).astype("int64")
    start_time = input_df["Timestamp"][0]
    input_df["Timestamp"] = (input_df["Timestamp"] - start_time) / 1e9

    # Convert to numerical values
    input_df["VS"] = input_df["VS"].apply(lambda x: VAC_SUCK_LABELS[x])
    input_df.apply(pd.to_numeric)
    # prepare packed input
    input, length = preprocess_sequence(input_df, presets)
    return input, length
