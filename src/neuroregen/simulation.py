"""
Main simulation loop: time array, axes, thermal gating, power, temperature, depth.
Stepwise generator for interactive/live use with axis enable and stop check.
"""

import numpy as np
from typing import Callable, Generator

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


def _sim_params_from_config(config: dict | None):
    """Build simulation parameters from config or defaults. Returns dict with keys needed for stepwise run."""
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
    if config is not None:
        return {
            "sim_time": config["sim_time"],
            "dt": config["dt"],
            "pulse_freq": config["pulse_freq"],
            "pulse_width": config["pulse_width"],
            "axes": config["axes"],
            "t_amb_c": config["t_amb_c"],
            "h_conv": config["h_conv"],
            "z_max_m": config["z_max_m"],
            "z_points": config["z_points"],
            "b_threshold_t": config["b_threshold_t"],
            "limit_c": f_to_c(config["temp_limit_f"]),
            "hyst_c": f_to_c(config["temp_limit_f"] - config["hyst_f"]),
        }
    axes = default_axes()
    return {
        "sim_time": SIM_TIME,
        "dt": DT,
        "pulse_freq": PULSE_FREQ,
        "pulse_width": PULSE_WIDTH,
        "axes": axes,
        "t_amb_c": T_AMB_C,
        "h_conv": None,
        "z_max_m": Z_MAX_M,
        "z_points": Z_POINTS,
        "b_threshold_t": B_THRESHOLD_T,
        "limit_c": None,
        "hyst_c": None,
    }


def run_simulation_stepwise(
    config: dict | None = None,
    axis_enabled: list[bool] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> Generator:
    """
    Run simulation one step at a time for interactive/live use.
    Yields ("step", k, t_k, T_k, P_k, depth_k) each step, then ("final", t, T, P, depth).
    axis_enabled: list of 3 bools; if False, that axis gets no pulses. Default all True.
    stop_check: callable(); if True, stop and yield final arrays.
    """
    params = _sim_params_from_config(config)
    sim_time = params["sim_time"]
    dt = params["dt"]
    pulse_freq = params["pulse_freq"]
    pulse_width = params["pulse_width"]
    axes = params["axes"]
    t_amb_c = params["t_amb_c"]
    h_conv = params["h_conv"]
    z_max_m = params["z_max_m"]
    z_points = params["z_points"]
    b_threshold_t = params["b_threshold_t"]
    limit_c = params["limit_c"]
    hyst_c = params["hyst_c"]

    if axis_enabled is None:
        axis_enabled = [True] * len(axes)
    n_axes = len(axes)
    axis_enabled = list(axis_enabled)[:n_axes]
    if len(axis_enabled) < n_axes:
        axis_enabled.extend([True] * (n_axes - len(axis_enabled)))

    t = np.arange(0, sim_time, dt)
    z = np.linspace(0, z_max_m, z_points)
    n = len(t)
    T = np.full((n, 3), t_amb_c)
    P = np.zeros((n, 3))
    depth = np.zeros((n, 3))

    geom = [coil_geom(a) for a in axes]
    Cth = [g[4] * CP_CU for g in geom]
    gated_off = [False] * len(axes)

    last_k = 0
    for k in range(1, n):
        if stop_check and stop_check():
            last_k = k
            break
        pulse_on = (t[k] % (1 / pulse_freq)) < pulse_width

        for i, a in enumerate(axes):
            R, L, A, S, m = geom[i]
            if not axis_enabled[i]:
                P[k, i] = 0
                Pcool = cooling_power(T[k - 1, i], S, h_conv=h_conv, t_amb_c=t_amb_c)
                T[k, i] = temp_step(T[k - 1, i], 0, Pcool, Cth[i], dt)
                depth[k, i] = 0.0
                continue
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
        last_k = k
        yield ("step", k, t[k], T[k, :].copy(), P[k, :].copy(), depth[k, :].copy())

    K = last_k + 1
    yield ("final", t[:K], T[:K], P[:K], depth[:K])
