"""
Tests that the controller enforces heat limit (FAULT on over-temp) and run respects limits.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.neuroregen.controller import Controller
from src.neuroregen.state_machine import ControllerState


def test_controller_fault_on_over_temperature():
    """Controller must transition to FAULT when any axis exceeds heat limit."""
    from src.neuroregen.coil import Axis

    # Temp limit just above ambient; high power and no cooling so we fault in a few steps
    temp_limit_f = 23.2  # ~73.8°F, just above 22°C ambient
    hot_axes = [
        Axis("X", 1.2, 80, 20, 200),
        Axis("Y", 1.2, 80, 20, 200),
        Axis("Z", 1.2, 80, 20, 200),
    ]
    config = {
        "sim_time": 120.0,
        "dt": 0.1,
        "pulse_freq": 5.0,
        "pulse_width": 0.02,
        "axes": hot_axes,
        "t_amb_c": 22.0,
        "h_conv": 0.0,  # no cooling so we heat fast
        "temp_limit_f": temp_limit_f,
        "hyst_f": 0.5,
        "z_max_m": 0.03,
        "z_points": 100,
        "b_threshold_t": 1e-4,
    }
    ctrl = Controller(
        config=config, csv_dir=os.path.join(ROOT, "outputs", "test_logs"), log_run=False
    )
    ctrl.arm()
    ctrl.start()
    t, T, P, depth = ctrl.run_firing_loop()
    assert ctrl.state == ControllerState.FAULT, f"Expected FAULT, got {ctrl.state}"
    assert ctrl.fault_reason == "over_temperature", (
        f"Expected over_temperature, got {ctrl.fault_reason}"
    )
    assert len(t) < 1200, "Run should have stopped early due to fault"
