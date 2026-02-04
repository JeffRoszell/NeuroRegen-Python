"""
Thermal gating: skip pulses when at limit, resume when below hysteresis.
"""

from .constants import H_CONV, T_AMB_C, TEMP_LIMIT_F, HYST_F
from .coil import f_to_c

# Celsius equivalents for gating
LIMIT_C = f_to_c(TEMP_LIMIT_F)
HYST_C = f_to_c(TEMP_LIMIT_F - HYST_F)


def thermal_gate_update(
    T_prev: float, gated_off: bool, limit_c: float | None = None, hyst_c: float | None = None
) -> bool:
    """
    Update gated_off for next step.
    - If T_prev >= limit_c → return True (gated off, skip pulse).
    - If T_prev < hyst_c → return False (resume).
    - Else → keep current state (hysteresis band).
    limit_c/hyst_c None → use constants (TEMP_LIMIT_F, HYST_F).
    """
    if limit_c is None:
        limit_c = LIMIT_C
    if hyst_c is None:
        hyst_c = HYST_C
    if T_prev >= limit_c:
        return True
    if T_prev < hyst_c:
        return False
    return gated_off


def temp_step(T_prev: float, Pin: float, Pcool: float, Cth: float, dt: float) -> float:
    """Next temperature (°C) given power in, cooling, thermal mass, and dt."""
    return T_prev + (Pin - Pcool) / Cth * dt


def cooling_power(
    T: float, S: float, h_conv: float | None = None, t_amb_c: float | None = None
) -> float:
    """Convective cooling power (W) at temperature T (°C), surface S (m²)."""
    h = H_CONV if h_conv is None else h_conv
    t_amb = T_AMB_C if t_amb_c is None else t_amb_c
    return h * S * (T - t_amb)
