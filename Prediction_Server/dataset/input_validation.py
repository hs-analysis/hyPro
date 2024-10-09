# hyPro/Prediction_Server/dataset/input_validation.py
# Description: This file contains the input validation functions for the dataset module.

# HS Analsis GmbH, 2024
# Author: Valentin Haas


# python imports
import logging

# External Imports
import pandas as pd

# Internal Imports
from ai_models import list_models
from data_models.ai_model import AIModel


# Setup Logging
logger = logging.getLogger(__name__)


def check_for_model(model_id: str, model_version: str) -> AIModel:
    """
    Check if the requested model exists and return it.

    Args:
        model_id (str): The ID of the model to check for.
        model_version (str): The version of the model to check for.

    Raises:
        FileNotFoundError: If the requested model does not exist.
        FileNotFoundError: If the requested model version does not exist.
        ValueError: If multiple models with the same ID and version exist.

    Returns:
        AIModel: The requested AI model.
    """
    # Get the existing models
    models = list_models()

    # Check if the requested model exists
    models_with_id = [model for model in models if model.id == model_id]
    if not any(models_with_id):
        msg = f'Model with ID "{model_id}" not found.'
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Check if the requested model version exists
    models_with_version = [
        model for model in models_with_id if model.version == model_version
    ]
    if not any(models_with_version):
        msg = f'Model with ID "{model_id}" and version "{model_version}" not found.'
        logger.error(msg)
        raise FileNotFoundError(msg)

    elif len(models_with_version) > 1:
        msg = (
            f'Multiple models with ID "{model_id}" and version "{model_version}" found.'
        )
        logger.error(msg)
        raise ValueError(msg)

    return models_with_version[0]


def validate_columns(required_column_names: set, input_df: pd.DataFrame) -> None:
    """
    Validate the columns of the input data. Check for missing, extra, and duplicate columns between the input data and the required columns.

    Args:
        required_column_names (set): The required columns for the model input. Must be a set of strings.
        input_df (pd.DataFrame): The input data to validate as a pandas DataFrame.

    Raises:
        ValueError: If the input data contains duplicate columns.
        ValueError: If the input data is missing required columns.
        ValueError: If the input data contains extra columns.
    """
    # Drop id - excluded from training, must not appear in input data
    if "ID" in input_df.columns:
        input_df.drop("ID", axis=1, inplace=True)

    delivered_column_names = set(sorted(input_df.columns))

    # Check for duplicate columns
    if len(delivered_column_names) < len(input_df.columns):
        duplicate_cols = input_df.columns[input_df.columns.duplicated()]
        msg = "Input data contains duplicate columns. Duplicates: " + ", ".join(
            duplicate_cols
        )
        logger.error(msg)
        raise ValueError(msg)

    # Check for missing columns
    if not required_column_names.issubset(delivered_column_names):
        missing_cols = required_column_names - delivered_column_names
        msg = "Input data does not match model input, missing columns: " + ", ".join(
            missing_cols
        )
        logger.error(msg)
        raise ValueError(msg)

    # Check for extra columns
    if not delivered_column_names.issubset(required_column_names):
        extra_cols = delivered_column_names - required_column_names
        msg = "Input data does not match model input, extra columns: " + ", ".join(
            extra_cols
        )
        logger.error(msg)
        raise ValueError(msg)
