"""
Main simulation loop: time array, axes, thermal gating, power, temperature, depth.
"""

import numpy as np

from .constants import (
    SIM_TIME,
    DT,
    PULSE_FREQ,
    PULSE_WIDTH,
    T_AMB_C,
    Z_MAX_M,
    Z_POINTS,
    B_THRESHOLD_T,
    CP_CU,
)
from .coil import Axis, coil_geom, resistance, B_loop, f_to_c
from .thermal import thermal_gate_update, cooling_power, temp_step


def default_axes():
    return [
        Axis("X", 1.2, 80, 20, 45),
        Axis("Y", 1.2, 80, 20, 45),
        Axis("Z", 1.2, 80, 20, 45),
    ]


def run_simulation(
    sim_time: float | None = None,
    dt: float | None = None,
    pulse_freq: float | None = None,
    pulse_width: float | None = None,
    axes: list | None = None,
    config: dict | None = None,
):
    """
    Run the 3-axis pulsed thermal simulation.
    If config is provided, it overrides other kwargs and supplies axes.
    Otherwise uses constants and default_axes() for missing values.
    Returns (t, T, P, depth): time (s), temperature (°C) [n_time, 3], power (W) [n_time, 3], depth (cm) [n_time, 3].
    """
    if config is not None:
        sim_time = config["sim_time"]
        dt = config["dt"]
        pulse_freq = config["pulse_freq"]
        pulse_width = config["pulse_width"]
        axes = config["axes"]
        t_amb_c = config["t_amb_c"]
        h_conv = config["h_conv"]
        z_max_m = config["z_max_m"]
        z_points = config["z_points"]
        b_threshold_t = config["b_threshold_t"]
        limit_c = f_to_c(config["temp_limit_f"])
        hyst_c = f_to_c(config["temp_limit_f"] - config["hyst_f"])
    else:
        sim_time = sim_time if sim_time is not None else SIM_TIME
        dt = dt if dt is not None else DT
        pulse_freq = pulse_freq if pulse_freq is not None else PULSE_FREQ
        pulse_width = pulse_width if pulse_width is not None else PULSE_WIDTH
        axes = default_axes() if axes is None else axes
        t_amb_c = T_AMB_C
        h_conv = None  # use constants in cooling_power
        z_max_m = Z_MAX_M
        z_points = Z_POINTS
        b_threshold_t = B_THRESHOLD_T
        limit_c = None  # thermal_gate_update uses its defaults
        hyst_c = None

    t = np.arange(0, sim_time, dt)
    z = np.linspace(0, z_max_m, z_points)
    n = len(t)
    T = np.full((n, 3), t_amb_c)
    P = np.zeros((n, 3))
    depth = np.zeros((n, 3))

    geom = [coil_geom(a) for a in axes]
    Cth = [g[4] * CP_CU for g in geom]
    gated_off = [False] * len(axes)

    for k in range(1, n):
        pulse_on = (t[k] % (1 / pulse_freq)) < pulse_width

        for i, a in enumerate(axes):
            R, L, A, S, m = geom[i]
            gated_off[i] = thermal_gate_update(
                T[k - 1, i], gated_off[i], limit_c=limit_c, hyst_c=hyst_c
            )
            on = pulse_on and not gated_off[i]

            Pin = a.pulse_power_w if on else 0
            P[k, i] = Pin

            Pcool = cooling_power(T[k - 1, i], S, h_conv=h_conv, t_amb_c=t_amb_c)
            T[k, i] = temp_step(T[k - 1, i], Pin, Pcool, Cth[i], dt)

            if Pin > 0:
                I = np.sqrt(Pin / resistance(L, A, T[k - 1, i]))
                Bz = B_loop(I, R, z, a.turns)
                depth[k, i] = (
                    z[Bz >= b_threshold_t][-1] * 100 if np.any(Bz >= b_threshold_t) else 0
                )
            else:
                depth[k, i] = 0.0

    return t, T, P, depth
