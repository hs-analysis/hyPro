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


def list_models() -> list[AIModel]:
    """
    List all available AI models.

    Returns:
        list[AIModel]: A list of AI models.
    """

    # Get all .pt files in the AI model folder
    ai_model_files = Path(AI_MODEL_FOLDER).glob("*.pt")

    infos = []
    for file in ai_model_files:
        # Load the AI model information from the JSON file
        try:
            info = AIModel.load_info(file)
            infos.append(info)

        except Exception as e:
            # Skip the file if an exception occurs
            logger.warning(f"Error while loading model info from {file}: {e}")

    # region Dummy Data
    # Add some dummy data
    dummy_data = [
        {
            "id": "1",
            "name": "LSTM PTV and RMS Prediction",
            "version": "1.1",
            "description": "An LSTM model to predict the PTV and RMS values of a glass measurement.",
            "creation_date": "2024-09-16T17:02:26.473893",
            "input_cols": {
                "Timestamp": [0.0, 175.0],
                "TA1": [1372.0, 1372.0],
                "TA2": [437.0, 463.0],
                "TA3": [438.0, 515.0],
                "TA4": [421.0, 1372.0],
                "TA5": [444.0, 506.0],
                "TB1": [428.0, 529.0],
                "TB2": [445.0, 506.0],
                "TB3": [444.0, 511.0],
                "TB4": [405.0, 563.0],
                "TB5": [1372.0, 1372.0],
                "TC1": [1372.0, 1372.0],
                "TC2": [1372.0, 1372.0],
                "TC3": [1372.0, 1372.0],
                "TC4": [1372.0, 1372.0],
                "TC5": [1372.0, 1372.0],
                "TD1": [1372.0, 1372.0],
                "PZ": [-27.000117911576808, 141.0001529680744],
                "PX": [-0.0028763708495489, 2661.0017957372197],
                "FLLR1": [-0.9999992847442628, 0.9855342507362366],
                "FLHR1": [-15.176856994628906, -0.3143386840820312],
                "XV": [-104.11824035644533, 251.48486328125],
                "ZV": [-6.903645515441895, 41.00936126708984],
                "AV": [0.0, 90.0],
                "VS": [0.0, 1.0],
                "HT": [-1.0, 68.61799621582031],
                "ST": [-1.0, 26.420000076293945],
                "SP1": [-767.510009765625, 13.82489013671875],
                "PC": [7499.32666015625, 9842.64453125],
            },
            "output_cols": ["PTV", "RMS"],
        }
    ]

    infos.extend([AIModel(**data) for data in dummy_data])
    # endregion Dummy Data

    # region Dummy Data
    # Add some dummy data
    dummy_data = [
        {
            "id": "1",
            "name": "LSTM PTV and RMS Prediction",
            "version": "1.2",
            "description": "An LSTM model to predict the PTV and RMS values of a glass measurement.",
            "creation_date": "2024-09-16T17:02:26.473893",
            "input_cols": {
                "Timestamp": [0.0, 175.0],
                "TA1": [1372.0, 1372.0],
                "TA2": [437.0, 463.0],
                "TA3": [438.0, 515.0],
                "TA4": [421.0, 1372.0],
                "TA5": [444.0, 506.0],
                "TB1": [428.0, 529.0],
                "TB2": [445.0, 506.0],
                "TB3": [444.0, 511.0],
                "TB4": [405.0, 563.0],
                "TB5": [1372.0, 1372.0],
                "TC1": [1372.0, 1372.0],
                "TC2": [1372.0, 1372.0],
                "TC3": [1372.0, 1372.0],
                "TC4": [1372.0, 1372.0],
                "TC5": [1372.0, 1372.0],
                "TD1": [1372.0, 1372.0],
                "PZ": [-27.000117911576808, 141.0001529680744],
                "PX": [-0.0028763708495489, 2661.0017957372197],
                "FLLR1": [-0.9999992847442628, 0.9855342507362366],
                "FLHR1": [-15.176856994628906, -0.3143386840820312],
                "XV": [-104.11824035644533, 251.48486328125],
                "ZV": [-6.903645515441895, 41.00936126708984],
                "AV": [0.0, 90.0],
                "VS": [0.0, 1.0],
                "HT": [-1.0, 68.61799621582031],
                "ST": [-1.0, 26.420000076293945],
                "SP1": [-767.510009765625, 13.82489013671875],
                "PC": [7499.32666015625, 9842.64453125],
            },
            "output_cols": ["PTV", "RMS"],
        }
    ]

    infos.extend([AIModel(**data) for data in dummy_data])
    # endregion Dummy Data

    # Sort the models by name and version
    infos.sort(key=lambda x: (x.name, x.version))

    return infos
