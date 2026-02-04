"""
Simulation constants and parameters.
Physical constants (copper, magnetic), thermal limits, and run settings.
"""

import numpy as np

# --- Physical (copper) ---
MU0 = 4e-7 * np.pi
RHO_CU = 1.68e-8
ALPHA_CU = 0.0039
DENSITY_CU = 8960
CP_CU = 385

# --- Thermal environment ---
H_CONV = 10.0
T_AMB_C = 22.0

# --- Thermal gating ---
TEMP_LIMIT_F = 75.0
HYST_F = 0.7

# --- Depth / B-field ---
Z_MAX_M = 0.03
Z_POINTS = 300
B_THRESHOLD_T = 1e-4

# --- Simulation ---
SIM_TIME = 1800.0  # 30 minutes (s)
DT = 0.1  # time step (s)
PULSE_FREQ = 5.0  # Hz
PULSE_WIDTH = 0.02  # s ON per pulse
