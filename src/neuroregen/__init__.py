"""
NeuroRegen — 3-axis pulsed coil simulation.
"""

from .constants import (
    SIM_TIME,
    DT,
    PULSE_FREQ,
    PULSE_WIDTH,
    TEMP_LIMIT_F,
    Z_MAX_M,
    Z_POINTS,
    B_THRESHOLD_T,
)
from .coil import Axis, coil_geom, resistance, B_loop, effective_depth_cm, c_to_f, f_to_c
from .thermal import thermal_gate_update
from .simulation import run_simulation, default_axes
from .plotting import plot_and_save
from .config_loader import load_config
from .field_mapping import (
    B_field_3d_loop,
    B_magnitude_3d,
    create_spatial_grid,
    calculate_field_map,
    plot_field_contours_2d,
    plot_field_interactive_slice,
    plot_targeting_volume,
)

__all__ = [
    "load_config",
    "run_simulation",
    "default_axes",
    "plot_and_save",
    "B_field_3d_loop",
    "B_magnitude_3d",
    "create_spatial_grid",
    "calculate_field_map",
    "plot_field_contours_2d",
    "plot_field_interactive_slice",
    "plot_targeting_volume",
    "Axis",
    "coil_geom",
    "resistance",
    "B_loop",
    "effective_depth_cm",
    "c_to_f",
    "f_to_c",
    "thermal_gate_update",
    "SIM_TIME",
    "DT",
    "PULSE_FREQ",
    "PULSE_WIDTH",
    "TEMP_LIMIT_F",
    "Z_MAX_M",
    "Z_POINTS",
    "B_THRESHOLD_T",
]
