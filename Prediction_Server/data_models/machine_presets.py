# hyPro/Prediction_Server/data_models/machine_presets.py
# Description: This file contains the required properties for machine presets

# HS Analysis GmbH, 2024
# Author: Valentin Haas


# External Imports
from pydantic import BaseModel
import numpy as np
import torch


class MachinePreset(BaseModel):
    """The required properties for machine presets."""

    ang_open_percent: float
    """Opening angle in percent."""

    T_furnace_C: float
    """Furnace temperature in degrees Celsius."""

    T_load_C: float
    """Load temperature in degrees Celsius."""

    T_start_C: float
    """Start temperature in degrees Celsius."""

    t_heat_s: float
    """Heating time in seconds."""

    t_heat_total_s: float
    """Total heating time in seconds."""

    t_process_s: float
    """Process time in seconds."""

    t_vacuum_s: float
    """Vacuum time in seconds."""

    t_z_cool_s: float
    """Cooling time in seconds."""

    v_cool_mm_s: float
    """Cooling speed in mm/s."""

    v_x_back_mm_s: float
    """X speed back in mm/s."""

    v_x_towards_mm_s: float
    """X speed towards in mm/s."""

    v_z_up_mm_s: float
    """Z speed up in mm/s."""

    z_heat_mm: float
    """Heating Z position in mm."""

    z_home_mm: float
    """Home Z position in mm."""

    def to_array(self) -> np.ndarray:
        """
        Convert the MachinePreset instance to a NumPy array.

        Returns:
            np.ndarray: The preset values as a NumPy array.
        """
        return np.array(
            [
                self.ang_open_percent,
                self.T_start_C,
                self.T_load_C,
                self.T_furnace_C,
                self.t_z_cool_s,
                self.t_process_s,
                self.t_heat_total_s,
                self.t_heat_s,
                self.t_vacuum_s,
                self.v_x_towards_mm_s,
                self.v_z_up_mm_s,
                self.v_x_back_mm_s,
                self.v_cool_mm_s,
                self.z_home_mm,
                self.z_heat_mm,
            ],
            dtype=float,
        )

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the MachinePreset instance to a PyTorch tensor with batch size 1.

        Returns:
            torch.Tensor: The preset values as a PyTorch tensor with batch size 1.
        """
        # List of values in specified order
        values = [
            self.ang_open_percent,
            self.T_start_C,
            self.T_load_C,
            self.T_furnace_C,
            self.t_z_cool_s,
            self.t_process_s,
            self.t_heat_total_s,
            self.t_heat_s,
            self.t_vacuum_s,
            self.v_x_towards_mm_s,
            self.v_z_up_mm_s,
            self.v_x_back_mm_s,
            self.v_cool_mm_s,
            self.z_home_mm,
            self.z_heat_mm,
        ]

        # Convert list to a PyTorch tensor and add batch dimension
        return torch.tensor([values], dtype=torch.float32)  # Shape: (1, 15)
