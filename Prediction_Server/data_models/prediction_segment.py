# hyPro/Prediction_Server/models/prediction_segment.py
# Defines the dataformat for a single segment in a prediction response.

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Internal Imports
from typing import Tuple

# External Imports
from pydantic import BaseModel


class PredictionSegment(BaseModel):
    """
    Defines the dataformat for a single segment in a prediction response.
    """

    id: int
    """ID of the segment."""

    pred_p2v: float
    """Predicted P2V value for the segment."""

    pred_rms: float
    """Predicted RMS value for the segment."""

    x_range_mm: Tuple[float, float]
    """X range of the segment in mm."""

    y_range_mm: Tuple[float, float]
    """Y range of the segment in mm."""

    xy_location: Tuple[int, int]
    """Location of the segment in the XY grid."""
