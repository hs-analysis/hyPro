# hyPro/Prediction_Server/data_models/machine_presets.py
# Description: This file contains the required properties for machine presets

# HS Analysis GmbH, 2024
# Author: Valentin Haas


# External Imports
from pydantic import BaseModel


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
