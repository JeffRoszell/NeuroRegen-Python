"""
Unit tests for thermal gating (hysteresis).
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.neuroregen.thermal import thermal_gate_update
from src.neuroregen.coil import f_to_c


def test_gate_off_at_limit():
    # At or above limit → gated off
    assert thermal_gate_update(f_to_c(75.0), False) is True
    assert thermal_gate_update(f_to_c(76.0), False) is True


def test_resume_below_hyst():
    # Below hysteresis → resume (gated_off = False)
    assert thermal_gate_update(f_to_c(74.0), True) is False
    assert thermal_gate_update(f_to_c(73.0), True) is False


def test_hysteresis_band():
    # In band (74.3–75 °F): keep current state
    # 74.3 °F = 75 - 0.7, so hyst is 74.3 °F
    t_hyst = f_to_c(74.3)
    t_limit = f_to_c(75.0)
    # Just below limit, was gated off → stay gated off
    assert thermal_gate_update(t_limit - 0.01, True, limit_c=t_limit, hyst_c=t_hyst) is True
    # Just below limit, was not gated → stay not gated
    assert thermal_gate_update(t_limit - 0.01, False, limit_c=t_limit, hyst_c=t_hyst) is False


def test_heat_limit_gating_prevents_rise():
    """Once at or above limit, gating is on so no power is applied; T cannot keep rising from heating."""
    from src.neuroregen.thermal import temp_step, cooling_power
    from src.neuroregen.coil import coil_geom
    from src.neuroregen.constants import CP_CU
    from src.neuroregen.simulation import default_axes

    limit_c = f_to_c(75.0)
    # At limit, gated_off=True → no Pin → only cooling. So next T should be <= T_prev (cooling).
    axes = default_axes()
    R, L, A, S, m = coil_geom(axes[0])
    Cth = m * CP_CU
    T_at_limit = limit_c
    Pcool = cooling_power(T_at_limit, S, t_amb_c=22.0)
    T_next = temp_step(T_at_limit, Pin=0, Pcool=Pcool, Cth=Cth, dt=0.1)
    assert T_next <= T_at_limit
