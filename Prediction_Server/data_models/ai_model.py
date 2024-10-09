# hyPro/Prediction_Server/models/ai_model.py
# Defines the data model for an AI model.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Pyhton Imports
from pathlib import Path
import json

# External Imports
from pydantic import BaseModel
import torch
import torch.nn as nn

# Local Imports
from ai_models.LSTM import LSTMModel


class AIModel(BaseModel):
    """
    Defines the data model for an AI model.
    """

    id: str
    """The ID of the AI model."""

    name: str
    """The name of the AI model."""

    version: str
    """The version of the AI model."""

    description: str
    """A description of the AI model."""

    creation_date: str
    """The timestamp when the AI model was created in the ISO 8601 format."""

    input_cols: dict
    """The columns used for the AI model, with the min and max values for global normalization."""

    output_cols: list
    """The columns used for the AI model output."""

    def save(self, file_path: Path, model: nn.Module) -> None:
        """
        Saves the AI model to a JSON file.

        Args:
            file_path (Path): The path to the file to save the AI model to.
            model (nn.Module): The PyTorch model to save.
        """

        custom_data = self.model_dump()
        custom_data["model_state_dict"] = model.state_dict()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(custom_data, file_path.with_suffix(".pt"))

        # Make sure the cols are json serializable
        self.input_cols = {
            col: [float(min_val), float(max_val)]
            for col, (min_val, max_val) in self.input_cols.items()
        }

        # Save the model data to a JSON file
        with open(file_path.with_suffix(".json"), "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_info(cls, file_path: Path):
        """
        Loads an AI model from a JSON file.

        Args:
            file_path (Path): The path to the file to load the AI model from.

        Returns:
            AIModel: The loaded AI model.
        """
        with open(file_path.with_suffix(".json"), "r") as f:
            # Load the data from the JSON file using uf8 encoding
            data = json.load(f)

        return AIModel(**data)

    @classmethod
    def load_model(cls, file_path: Path):
        """
        Loads a PyTorch model from a file.

        Args:
            file_path (Path): The path to the file to load the PyTorch model from.

        Returns:
            nn.Module: The loaded PyTorch model.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        model_info = AIModel(**checkpoint)
        model = LSTMModel(
            input_size=len(model_info.input_cols),
            output_size=len(model_info.output_cols),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model_info, model
