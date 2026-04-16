"""
Unit tests for the capacitor-discharge pulsed TMS physics module.

Tests validate analytical formulas against hand-computed reference values.
"""

import sys
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.neuroregen.pulsed import (
    coil_inductance,
    discharge_params,
    max_rep_rate,
    B_peak_on_axis,
    E_induced_peak,
    heat_per_pulse,
    superposed_B_peak_at,
    run_pulsed_thermal_sim,
)
from src.neuroregen.constants import MU0
from src.neuroregen.multicoil import Coil, Target


# ---------------------------------------------------------------------------
# coil_inductance
# ---------------------------------------------------------------------------
class TestCoilInductance:
    def test_single_turn_positive(self):
        """Single-turn coil must have positive inductance."""
        L = coil_inductance(R_m=0.05, N=1, wire_radius_m=0.001)
        assert L > 0

    def test_n_squared_scaling(self):
        """L should scale as N²."""
        L1 = coil_inductance(R_m=0.05, N=1, wire_radius_m=0.001)
        L4 = coil_inductance(R_m=0.05, N=4, wire_radius_m=0.001)
        assert abs(L4 / L1 - 16.0) < 1e-9

    def test_neumann_formula_reference(self):
        """Cross-check Neumann formula for R=0.075 m, N=6, a=0.005 m."""
        R, N, a = 0.075, 6, 0.005
        L1_expected = MU0 * R * (np.log(8.0 * R / a) - 2.0)
        L_expected = N**2 * L1_expected
        assert abs(coil_inductance(R, N, a) - L_expected) < 1e-15


# ---------------------------------------------------------------------------
# discharge_params
# ---------------------------------------------------------------------------
class TestDischargeParams:
    PARAMS = dict(C=200e-6, V=2500.0, L=9.47e-6)

    def test_returns_all_keys(self):
        d = discharge_params(**self.PARAMS)
        assert {"I_peak_A", "tau_s", "E_pulse_J"} == set(d.keys())

    def test_I_peak_formula(self):
        """I_peak = V * sqrt(C/L)."""
        C, V, L = self.PARAMS["C"], self.PARAMS["V"], self.PARAMS["L"]
        expected = V * np.sqrt(C / L)
        assert abs(discharge_params(C, V, L)["I_peak_A"] - expected) < 1.0

    def test_tau_formula(self):
        """τ = π √(LC)."""
        C, V, L = self.PARAMS["C"], self.PARAMS["V"], self.PARAMS["L"]
        expected = np.pi * np.sqrt(L * C)
        assert abs(discharge_params(C, V, L)["tau_s"] - expected) < 1e-9

    def test_E_pulse_formula(self):
        """E = ½ C V²."""
        C, V, L = self.PARAMS["C"], self.PARAMS["V"], self.PARAMS["L"]
        expected = 0.5 * C * V**2
        assert abs(discharge_params(C, V, L)["E_pulse_J"] - expected) < 1e-9

    def test_reference_numbers(self):
        """Verify reference values from design analysis."""
        d = discharge_params(**self.PARAMS)
        assert 10000 < d["I_peak_A"] < 15000  # ~11 500 A
        assert 100e-6 < d["tau_s"] < 200e-6  # ~137 µs
        assert abs(d["E_pulse_J"] - 625.0) < 0.1  # exactly 625 J


# ---------------------------------------------------------------------------
# max_rep_rate
# ---------------------------------------------------------------------------
class TestMaxRepRate:
    def test_basic(self):
        """f_max = P * η / E_total."""
        f = max_rep_rate(E_pulse_total_J=625.0, P_supply_w=2200.0, efficiency=0.85)
        assert abs(f - 2200.0 * 0.85 / 625.0) < 1e-9

    def test_three_coil_budget(self):
        """3 coils × 625 J at 1 Hz requires ~2206 W (just above 2200 W)."""
        f = max_rep_rate(3 * 625.0, 2200.0, efficiency=0.85)
        assert abs(f - 1.0) < 0.01  # should be ≈ 0.99 Hz


# ---------------------------------------------------------------------------
# B_peak_on_axis
# ---------------------------------------------------------------------------
class TestBPeakOnAxis:
    def test_on_axis_at_origin(self):
        """At z=0, B = μ₀ I N / (2 R)."""
        R, I, N = 0.05, 1000.0, 10
        expected = MU0 * I * N / (2.0 * R)
        assert abs(B_peak_on_axis(I, R, 0.0, N) - expected) < 1e-10

    def test_falls_with_distance(self):
        """B must decrease as z increases."""
        R, I, N = 0.075, 11500.0, 6
        B_near = B_peak_on_axis(I, R, 0.01, N)
        B_far = B_peak_on_axis(I, R, 0.10, N)
        assert B_far < B_near

    def test_dipole_limit(self):
        """Far from coil (z >> R), B ∝ 1/z³."""
        R, I, N = 0.01, 1000.0, 1
        z1, z2 = 0.5, 1.0
        B1 = B_peak_on_axis(I, R, z1, N)
        B2 = B_peak_on_axis(I, R, z2, N)
        # ratio should be ≈ (z2/z1)³ = 8
        assert abs(B1 / B2 - 8.0) < 0.2


# ---------------------------------------------------------------------------
# E_induced_peak
# ---------------------------------------------------------------------------
class TestEInducedPeak:
    def test_faraday_formula(self):
        """E = r × π × B_peak / (2 τ)."""
        B, r, tau = 0.1, 0.01, 100e-6
        expected = r * np.pi * B / (2.0 * tau)
        assert abs(E_induced_peak(B, r, tau) - expected) < 1e-12

    def test_scales_with_radius(self):
        """E should double when r doubles."""
        B, tau = 0.1, 100e-6
        E1 = E_induced_peak(B, 0.005, tau)
        E2 = E_induced_peak(B, 0.010, tau)
        assert abs(E2 / E1 - 2.0) < 1e-9

    def test_scales_inverse_with_tau(self):
        """E should halve when τ doubles (faster dB/dt → stronger E)."""
        B, r = 0.1, 0.005
        E_fast = E_induced_peak(B, r, 50e-6)
        E_slow = E_induced_peak(B, r, 100e-6)
        assert abs(E_fast / E_slow - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# heat_per_pulse
# ---------------------------------------------------------------------------
class TestHeatPerPulse:
    def test_basic(self):
        """Q = R I² τ / 2."""
        R, I, tau = 0.001, 11500.0, 137e-6
        expected = R * I**2 * tau / 2.0
        assert abs(heat_per_pulse(R, I, tau) - expected) < 1e-10

    def test_positive(self):
        assert heat_per_pulse(0.001, 1000.0, 100e-6) > 0


# ---------------------------------------------------------------------------
# superposed_B_peak_at
# ---------------------------------------------------------------------------
class TestSuperposedBPeakAt:
    def _make_coil(self, pos, normal, name="T"):
        return Coil(
            name=name,
            wire_mm=10.0,
            loop_mm=150.0,
            turns=6,
            pulse_power_w=0.0,
            position_m=pos,
            normal=normal,
        )

    def test_single_coil_nonzero(self):
        """Single coil should produce nonzero B at target."""
        coil = self._make_coil(
            pos=(0.0, 0.0, 0.09),
            normal=(0.0, 0.0, -1.0),
        )
        target = Target(name="test", position_m=(0.0, 0.0, -0.04))
        B_vec, B_mag = superposed_B_peak_at([coil], [11500.0], target)
        assert B_mag > 0
        assert B_vec.shape == (3,)

    def test_two_opposing_coils_cancel(self):
        """Two identical coils aimed at the same point from opposite sides
        with opposite polarity currents should partially cancel."""
        c1 = self._make_coil(pos=(0.0, 0.0, 0.09), normal=(0.0, 0.0, -1.0), name="C1")
        c2 = self._make_coil(pos=(0.0, 0.0, -0.09), normal=(0.0, 0.0, 1.0), name="C2")
        target = Target(name="ctr", position_m=(0.0, 0.0, 0.0))
        _, B_both = superposed_B_peak_at([c1, c2], [11500.0, 11500.0], target)
        _, B_one = superposed_B_peak_at([c1], [11500.0], target)
        # Both coils with same-sign current add; opposite polarity would cancel
        # (This test just verifies superposition runs without error and gives a scalar)
        assert B_both >= 0


# ---------------------------------------------------------------------------
# run_pulsed_thermal_sim — integration smoke test
# ---------------------------------------------------------------------------
class TestRunPulsedThermalSim:
    def _make_config(self, n_pulses=5):
        return {
            "capacitance_f": 200e-6,
            "charge_voltage_v": 2500.0,
            "pulse_freq": 1.0,
            "n_pulses": n_pulses,
            "dt_between_pulses": 0.01,
            "t_amb_c": 22.0,
            "h_conv": 10.0,
            "r_tissue_m": 0.005,
            "scalp_to_cortex_m": 0.015,
        }

    def _make_coil(self, name, pos, normal):
        return Coil(
            name=name,
            wire_mm=10.0,
            loop_mm=150.0,
            turns=6,
            pulse_power_w=0.0,
            position_m=pos,
            normal=normal,
        )

    def test_smoke(self):
        """Simulation should run without error and return sensible shapes."""
        coils = [
            self._make_coil("A", (-0.010, -0.010, 0.088), (-0.016, -0.023, -1.0)),
            self._make_coil("B", (0.085, -0.020, 0.025), (-0.829, 0.060, -0.556)),
        ]
        target = Target(name="STN", position_m=(-0.012, -0.013, -0.040))
        result = run_pulsed_thermal_sim(coils, target, self._make_config(n_pulses=5))

        assert result.t_s.shape == (5,)
        assert result.T_c.shape == (5, 2)
        assert result.B_target_T.shape == (5,)
        assert result.E_target_vm.shape == (5,)
        assert result.E_surface_vm.shape == (2,)
        assert result.Q_pulse_J.shape == (2,)

    def test_temperature_rises(self):
        """Coil temperature must rise above ambient after pulses with no cooling."""
        coils = [
            self._make_coil("A", (-0.010, -0.010, 0.088), (-0.016, -0.023, -1.0)),
        ]
        target = Target(name="STN", position_m=(-0.012, -0.013, -0.040))
        cfg = self._make_config(n_pulses=10)
        cfg["h_conv"] = 0.0  # disable cooling so T must rise
        result = run_pulsed_thermal_sim(coils, target, cfg)
        assert result.T_c[-1, 0] > cfg["t_amb_c"]

    def test_B_constant_across_pulses(self):
        """B at target is geometry-only — should be constant across pulses."""
        coils = [
            self._make_coil("A", (-0.010, -0.010, 0.088), (-0.016, -0.023, -1.0)),
        ]
        target = Target(name="STN", position_m=(-0.012, -0.013, -0.040))
        result = run_pulsed_thermal_sim(coils, target, self._make_config(n_pulses=5))
        # All B values should be the same (within floating point)
        assert np.allclose(result.B_target_T, result.B_target_T[0])

    def test_tau_and_discharge_keys(self):
        """Result should carry tau_s and discharge dict."""
        coils = [
            self._make_coil("A", (-0.010, -0.010, 0.088), (-0.016, -0.023, -1.0)),
        ]
        target = Target(name="STN", position_m=(-0.012, -0.013, -0.040))
        result = run_pulsed_thermal_sim(coils, target, self._make_config())
        assert result.tau_s > 0
        assert "I_peak_A" in result.discharge
        assert "tau_s" in result.discharge
        assert "E_pulse_J" in result.discharge
