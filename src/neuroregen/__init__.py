"""
NeuroRegen — 3-axis pulsed coil simulation & multicoil deep-brain targeting.
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
from .simulation import run_simulation, run_simulation_stepwise, default_axes
from .plotting import plot_and_save, live_plot_init
from .state_machine import ControllerState
from .controller import Controller
from .csv_logger import open_log
from .config_loader import (
    load_config,
    load_multicoil_config,
    build_multicoil_objects,
    load_pulsed_config,
)
from .ansys_field_map import AnsysFieldMap, load_ansys_field_maps
from .ansys_connection import AnsysLiveConnection, AnsysLiveFieldMap
from .field_mapping import (
    B_field_3d_loop,
    B_magnitude_3d,
    create_spatial_grid,
    calculate_field_map,
    plot_field_contours_2d,
    plot_field_interactive_slice,
    plot_targeting_volume,
)
from .pulsed import (
    coil_inductance,
    discharge_params,
    max_rep_rate,
    B_peak_on_axis,
    E_induced_peak,
    heat_per_pulse,
    superposed_B_peak_at,
    run_pulsed_thermal_sim,
    PulsedSimResult,
)
from .multicoil import (
    Coil,
    Target,
    DepthGateResult,
    MulticoilArray,
    B_field_at_point,
    compute_coil_distances,
    compute_cosine_factors,
    compute_distance_weights,
    compute_weighted_powers,
    E_field_at_surface,
    check_depth_gate,
    superposed_B_at_target,
    superposed_B_on_grid,
    surface_to_deep_ratio,
    run_multicoil_simulation,
    run_multicoil_simulation_stepwise,
    E_FIELD_CORTICAL_MAX_VM,
    SCALP_TO_CORTEX_M,
)

__all__ = [
    # --- pulsed TMS ---
    "load_pulsed_config",
    "coil_inductance",
    "discharge_params",
    "max_rep_rate",
    "B_peak_on_axis",
    "E_induced_peak",
    "heat_per_pulse",
    "superposed_B_peak_at",
    "run_pulsed_thermal_sim",
    "PulsedSimResult",
    # --- single-coil ---
    "load_config",
    "run_simulation",
    "run_simulation_stepwise",
    "default_axes",
    "plot_and_save",
    "live_plot_init",
    "ControllerState",
    "Controller",
    "open_log",
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
    # --- ANSYS integration ---
    "AnsysFieldMap",
    "load_ansys_field_maps",
    "AnsysLiveConnection",
    "AnsysLiveFieldMap",
    # --- multicoil ---
    "load_multicoil_config",
    "build_multicoil_objects",
    "Coil",
    "Target",
    "DepthGateResult",
    "MulticoilArray",
    "B_field_at_point",
    "compute_coil_distances",
    "compute_cosine_factors",
    "compute_distance_weights",
    "compute_weighted_powers",
    "E_field_at_surface",
    "check_depth_gate",
    "superposed_B_at_target",
    "superposed_B_on_grid",
    "surface_to_deep_ratio",
    "run_multicoil_simulation",
    "run_multicoil_simulation_stepwise",
    "E_FIELD_CORTICAL_MAX_VM",
    "SCALP_TO_CORTEX_M",
]
