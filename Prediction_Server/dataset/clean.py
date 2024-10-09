# hyPro/Prediction_Server/dataset/clean.py
# Description: This file contains the cleaning functions for the dataset module.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# External imports
import pandas as pd

# Internal imports
from data_models.ai_model import AIModel

# Constants
VAC_SUCK_LABELS = {
    "Off": 0,
    "Suck": 1,
}


def prepare_timeseries(input_df: pd.DataFrame, ai_model: AIModel) -> pd.DataFrame:
    """
    Prepare the input data for time series analysis. This function will convert the input data to a time series format.

    Args:
        input_df (pd.DataFrame): The input data to prepare as a pandas DataFrame.
        ai_model (AIModel): The AI model to prepare the data for.

    Returns:
        pd.DataFrame: The prepared time series data as a pandas DataFrame.
    """
    # Sort the incoming columns in the same order as the model input columns
    input_df = input_df[list(ai_model.input_cols.keys())]

    # Convert timestamp to seconds since start
    input_df["Timestamp"] = pd.to_datetime(input_df["Timestamp"]).astype("int64")
    start_time = input_df["Timestamp"][0]
    input_df["Timestamp"] = (input_df["Timestamp"] - start_time) / 1e9

    # Convert to numerical values
    input_df["VS"] = input_df["VS"].apply(lambda x: VAC_SUCK_LABELS[x])
    input_df.apply(pd.to_numeric)
    return input_df
