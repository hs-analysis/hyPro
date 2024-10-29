# hyPro/Prediction_Server/ai_models/helpers.py
# Description: This file contains helper functions for the AI models module.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Python Imports
from pathlib import Path
import logging

# Internal Imports
from data_models import AIModel

# Constants
CWD = Path(__file__).parent.resolve()
AI_MODEL_FOLDER = (CWD.parent / "trained_models").resolve()

# Setup Logging
logger = logging.getLogger(__name__)


def list_models_and_paths() -> list[tuple[AIModel, Path]]:
    """
    List all available AI models and the corresponding pth file.

    Returns:
        list[AIModel]: A list of AI models.
    """
    ai_model_files = Path(AI_MODEL_FOLDER).glob("*.pth")
    infos = []
    for file in ai_model_files:
        # Load the AI model information from the JSON file
        try:
            info = AIModel.load_info(file)
            infos.append((info, file))

        except Exception as e:
            # Skip the file if an exception occurs
            logger.warning(f"Error while loading model info from {file}: {e}")
    return infos


def list_models() -> list[AIModel]:
    """
    List all available AI models.

    Returns:
        list[AIModel]: A list of AI models.
    """

    # Get all .pt files in the AI model folder
    ai_model_files = Path(AI_MODEL_FOLDER).glob("*.pth")
    infos = []
    for file in ai_model_files:
        # Load the AI model information from the JSON file
        try:
            info = AIModel.load_info(file)
            infos.append(info)

        except Exception as e:
            # Skip the file if an exception occurs
            logger.warning(f"Error while loading model info from {file}: {e}")

    # Sort the models by name and version
    infos.sort(key=lambda x: (x.name, x.version))

    return infos
