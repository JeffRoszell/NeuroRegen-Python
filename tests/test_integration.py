"""
Integration tests: run simulation for a short time and check outputs are in range.
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.neuroregen import run_simulation
from src.neuroregen.coil import f_to_c
from src.neuroregen.constants import T_AMB_C, TEMP_LIMIT_F, Z_MAX_M


def test_run_simulation_shapes():
    """Short run: t, T, P, depth have correct shapes and dtypes."""
    sim_time = 20.0  # seconds
    dt = 0.1
    t, T, P, depth = run_simulation(sim_time=sim_time, dt=dt)
    n = len(t)
    assert n >= 2
    assert t.shape == (n,)
    assert T.shape == (n, 3)
    assert P.shape == (n, 3)
    assert depth.shape == (n, 3)
    assert np.issubdtype(t.dtype, np.floating)
    assert np.issubdtype(T.dtype, np.floating)
    assert t[0] == 0 and t[-1] < sim_time + dt


def test_run_simulation_temperature_bounds():
    """Temperatures stay between ambient and thermal limit (with small tolerance)."""
    sim_time = 30.0
    dt = 0.1
    limit_c = f_to_c(TEMP_LIMIT_F)
    t, T, P, depth = run_simulation(sim_time=sim_time, dt=dt)
    assert T.min() >= T_AMB_C - 1.0  # allow small numerical drift
    # Gating should keep T at or below limit; allow 0.5°C tolerance
    assert T.max() <= limit_c + 0.5


def test_run_simulation_depth_bounds():
    """Depth stays in [0, z_max] cm."""
    sim_time = 20.0
    z_max_cm = Z_MAX_M * 100
    t, T, P, depth = run_simulation(sim_time=sim_time)
    assert depth.min() >= 0
    assert depth.max() <= z_max_cm + 0.01  # small tolerance


def test_run_simulation_power_values():
    """Power is between 0 and axis pulse power (45 W for default axes)."""
    sim_time = 20.0
    t, T, P, depth = run_simulation(sim_time=sim_time)
    assert P.min() >= 0
    assert P.max() <= 45.0 + 0.1


def test_run_simulation_with_config():
    """Short run with config: same shape checks, respects config bounds."""
    from src.neuroregen.config_loader import load_config

    config_path = os.path.join(ROOT, "config", "default.yaml")
    if not os.path.isfile(config_path):
        return  # skip if no config
    config = load_config(config_path)
    # Override for fast test
    config = dict(config)
    config["sim_time"] = 15.0
    config["dt"] = 0.1
    t, T, P, depth = run_simulation(config=config)
    n = len(t)
    assert T.shape == (n, 3)
    assert P.shape == (n, 3)
    assert depth.shape == (n, 3)
    limit_c = f_to_c(config["temp_limit_f"])
    assert T.max() <= limit_c + 0.5
    assert depth.max() <= config["z_max_m"] * 100 + 0.01


def test_heat_limit_never_exceeded():
    """Heat limit: temperatures must never exceed configured limit (gating enforces this)."""
    sim_time = 90.0  # long enough for heating
    dt = 0.1
    limit_f = 75.0
    limit_c = f_to_c(limit_f)
    t, T, P, depth = run_simulation(sim_time=sim_time, dt=dt)
    # Allow at most one step overshoot (we gate after T >= limit)
    tolerance_c = 0.5
    assert T.max() <= limit_c + tolerance_c, (
        f"Heat limit {limit_f}°F ({limit_c}°C) exceeded: T.max()={T.max():.3f}°C"
    )
    for i in range(3):
        assert T[:, i].max() <= limit_c + tolerance_c, f"Axis {i} exceeded heat limit"


def test_depth_limit_never_exceeded():
    """Depth limit: penetration depth must never exceed configured z_max (from constants)."""
    t, T, P, depth = run_simulation(sim_time=30.0)
    z_max_cm = Z_MAX_M * 100
    tolerance_cm = 0.01
    assert depth.max() <= z_max_cm + tolerance_cm, (
        f"Depth limit {z_max_cm} cm exceeded: depth.max()={depth.max():.3f} cm"
    )


def test_depth_limit_respected_with_custom_config():
    """Depth limit from config is respected when using custom z_max_m."""
    from src.neuroregen.simulation import default_axes

    z_max_m = 0.02  # 2 cm
    z_max_cm = 2.0
    config = {
        "sim_time": 25.0,
        "dt": 0.1,
        "pulse_freq": 5.0,
        "pulse_width": 0.02,
        "axes": default_axes(),
        "t_amb_c": 22.0,
        "h_conv": 10.0,
        "temp_limit_f": 75.0,
        "hyst_f": 0.7,
        "z_max_m": z_max_m,
        "z_points": 300,
        "b_threshold_t": 1e-4,
    }
    t, T, P, depth = run_simulation(config=config)
    assert depth.max() <= z_max_cm + 0.01
