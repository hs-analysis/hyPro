# hyPro/Prediction_Server/models/prediction_response.py
# Defines the response model for a calculated prediction.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Internal Imports
from typing import List

# External Imports
from pydantic import BaseModel

# Local Imports
from .prediction_segment import PredictionSegment


class PredictionResponse(BaseModel):
    """
    Defines the response model for a calculated prediction.
    """

    machine_id: str
    """The ID of the machine for which the prediction was requested."""

    ai_model_id: str
    """The ID of the model that was used for the prediction."""

    ai_model_version: str
    """The version of the model that was used for the prediction."""

    request_timestamp: str
    """Timestamp when the prediction was requested in the ISO 8601 format."""

    response_time_s: float
    """Time it took to calculate the prediction in seconds."""

    segments: List[PredictionSegment]
    """List of segments that make up the prediction."""
