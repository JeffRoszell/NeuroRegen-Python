"""
Coil geometry, resistance, B-field, and effective depth.
"""

import numpy as np
from dataclasses import dataclass

from .constants import (
    MU0,
    RHO_CU,
    ALPHA_CU,
    DENSITY_CU,
    Z_MAX_M,
    Z_POINTS,
    B_THRESHOLD_T,
)


@dataclass
class Axis:
    name: str
    wire_mm: float
    loop_mm: float
    turns: int
    pulse_power_w: float


def c_to_f(tc: float) -> float:
    return tc * 9 / 5 + 32


def f_to_c(tf: float) -> float:
    return (tf - 32) * 5 / 9


def coil_geom(axis: Axis):
    """Return (R, L, A, S, m): radius m, length m, cross-section m², surface m², mass kg."""
    R = axis.loop_mm / 2000
    L = axis.turns * 2 * np.pi * R
    A = np.pi * (axis.wire_mm / 2000) ** 2
    S = L * np.pi * (axis.wire_mm / 1000)
    m = DENSITY_CU * L * A
    return R, L, A, S, m


def resistance(L: float, A: float, T: float) -> float:
    """Copper resistance (Ω) at temperature T (°C)."""
    R0 = RHO_CU * L / A
    return R0 * (1 + ALPHA_CU * (T - 20))


def B_loop(I: float, R: float, z: np.ndarray, N: int) -> np.ndarray:
    """B-field (T) on axis at distances z (m) for loop current I (A), radius R (m), N turns."""
    return MU0 * I * R**2 / (2 * (R**2 + z**2) ** 1.5) * N


def effective_depth_cm(Pin: float, L: float, A: float, R: float, N: int, T_c: float) -> float:
    """
    Effective penetration depth (cm) for given power Pin (W), geometry, and coil temp T_c (°C).
    Depth is max z where B >= B_THRESHOLD_T, capped at Z_MAX_M (3 cm).
    Returns 0 if no power or B never exceeds threshold.
    """
    if Pin <= 0:
        return 0.0
    z = np.linspace(0, Z_MAX_M, Z_POINTS)
    R_ohm = resistance(L, A, T_c)
    I = np.sqrt(Pin / R_ohm)
    Bz = B_loop(I, R, z, N)
    above = Bz >= B_THRESHOLD_T
    if not np.any(above):
        return 0.0
    return float(z[above][-1] * 100)
