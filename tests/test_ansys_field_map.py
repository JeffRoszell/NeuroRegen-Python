"""
Tests for AnsysFieldMap — ANSYS Maxwell field-map integration.

All tests use in-memory CSV content (via tmp_path) so no real ANSYS licence
or pre-generated files are required.
"""

import math
import sys
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

scipy = pytest.importorskip("scipy", reason="scipy required for AnsysFieldMap")

from src.neuroregen.ansys_field_map import AnsysFieldMap, load_ansys_field_maps


# ---------------------------------------------------------------------------
# Helpers — build minimal CSV content
# ---------------------------------------------------------------------------


def _write_regular_csv(
    tmp_path: Path,
    filename: str = "coil_A.csv",
    ref_current: float = 1.0,
    with_efield: bool = False,
    x_vals=(-0.05, 0.0, 0.05),
    y_vals=(-0.05, 0.0, 0.05),
    z_vals=(-0.05, 0.0, 0.05),
) -> Path:
    """Write a complete regular-grid CSV and return the path."""
    lines = [
        "# NeuroRegen test field map",
        f"# reference_current_A: {ref_current}",
        "#",
    ]
    if with_efield:
        lines.append("x_m,y_m,z_m,Bx_T,By_T,Bz_T,Ex_Vm,Ey_Vm,Ez_Vm")
    else:
        lines.append("x_m,y_m,z_m,Bx_T,By_T,Bz_T")

    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                # Simple synthetic field: Bz = 1e-4 T everywhere, others 0
                Bx, By, Bz = 0.0, 0.0, 1e-4
                if with_efield:
                    Ex, Ey, Ez = 5.0, 0.0, 0.0
                    lines.append(
                        f"{x:.6f},{y:.6f},{z:.6f},"
                        f"{Bx:.8e},{By:.8e},{Bz:.8e},"
                        f"{Ex:.8e},{Ey:.8e},{Ez:.8e}"
                    )
                else:
                    lines.append(f"{x:.6f},{y:.6f},{z:.6f},{Bx:.8e},{By:.8e},{Bz:.8e}")

    csv_path = tmp_path / filename
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def _write_mm_units_csv(tmp_path: Path) -> Path:
    """CSV using millimetre coordinates and millitesla B values."""
    lines = [
        "# reference_current_A: 2.0",
        "x_mm,y_mm,z_mm,Bx_mT,By_mT,Bz_mT",
    ]
    # 3×3×3 grid
    for x in (-50, 0, 50):
        for y in (-50, 0, 50):
            for z in (-50, 0, 50):
                # Bz = 0.1 mT = 1e-4 T
                lines.append(f"{x},{y},{z},0.0,0.0,0.1")
    csv_path = tmp_path / "coil_mm.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def _write_scattered_csv(tmp_path: Path) -> Path:
    """Four non-grid points → forces LinearNDInterpolator path."""
    lines = [
        "# reference_current_A: 1.0",
        "x_m,y_m,z_m,Bx_T,By_T,Bz_T",
        "0.0,0.0,0.0,0.0,0.0,1.0e-4",
        "0.1,0.0,0.0,0.0,0.0,0.5e-4",
        "0.0,0.1,0.0,0.0,0.0,0.5e-4",
        "0.0,0.0,0.1,0.0,0.0,0.5e-4",
        "0.1,0.1,0.1,0.0,0.0,0.2e-4",
    ]
    csv_path = tmp_path / "scattered.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------


class TestAnsysFieldMapLoad:
    def test_loads_regular_grid(self, tmp_path):
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path, coil_name="A")
        assert fm.is_regular_grid
        assert fm.n_points == 27  # 3×3×3
        assert fm.coil_name == "A"
        assert math.isclose(fm.reference_current_a, 1.0)

    def test_loads_mm_units(self, tmp_path):
        path = _write_mm_units_csv(tmp_path)
        fm = AnsysFieldMap(path)
        assert fm.is_regular_grid
        lo, hi = fm.bounds
        # Coordinates should have been converted to metres: ±50 mm → ±0.05 m
        assert math.isclose(lo[0], -0.05, abs_tol=1e-9)
        assert math.isclose(hi[0], 0.05, abs_tol=1e-9)
        assert math.isclose(fm.reference_current_a, 2.0)

    def test_loads_with_efield(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=True)
        fm = AnsysFieldMap(path)
        assert fm.has_efield

    def test_no_efield_flag(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=False)
        fm = AnsysFieldMap(path)
        assert not fm.has_efield

    def test_scattered_grid_detected(self, tmp_path):
        path = _write_scattered_csv(tmp_path)
        fm = AnsysFieldMap(path)
        assert not fm.is_regular_grid
        assert fm.n_points == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnsysFieldMap(tmp_path / "nonexistent.csv")

    def test_default_ref_current_when_comment_absent(self, tmp_path):
        lines = ["x_m,y_m,z_m,Bx_T,By_T,Bz_T", "0.0,0.0,0.0,0.0,0.0,1e-4"]
        p = tmp_path / "no_comment.csv"
        p.write_text("\n".join(lines))
        fm = AnsysFieldMap(p)
        assert math.isclose(fm.reference_current_a, 1.0)

    def test_repr(self, tmp_path):
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path, coil_name="TestCoil")
        r = repr(fm)
        assert "TestCoil" in r
        assert "regular" in r


# ---------------------------------------------------------------------------
# Interpolation tests — B_at_point
# ---------------------------------------------------------------------------


class TestBAtPoint:
    def test_exact_grid_node(self, tmp_path):
        """Query at an exact grid node should return the stored value."""
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        B = fm.B_at_point((0.0, 0.0, 0.0), current_A=1.0)
        assert B.shape == (3,)
        assert math.isclose(B[2], 1e-4, rel_tol=1e-6)
        assert math.isclose(B[0], 0.0, abs_tol=1e-12)

    def test_current_scaling(self, tmp_path):
        """B should scale linearly with current_A / reference_current_a."""
        path = _write_regular_csv(tmp_path, ref_current=1.0)
        fm = AnsysFieldMap(path)
        B1 = fm.B_at_point((0.0, 0.0, 0.0), current_A=1.0)
        B2 = fm.B_at_point((0.0, 0.0, 0.0), current_A=3.0)
        np.testing.assert_allclose(B2, B1 * 3.0, rtol=1e-9)

    def test_ref_current_scaling(self, tmp_path):
        """With ref_current=2 A, querying at 2 A should give the stored value."""
        path = _write_mm_units_csv(tmp_path)  # ref_current = 2.0 A, Bz = 1e-4 T
        fm = AnsysFieldMap(path)
        B = fm.B_at_point((0.0, 0.0, 0.0), current_A=2.0)
        assert math.isclose(B[2], 1e-4, rel_tol=1e-5)

    def test_out_of_bounds_returns_nan(self, tmp_path):
        """Point outside the grid should return NaN vector."""
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        B = fm.B_at_point((1.0, 1.0, 1.0))  # far outside ±5 cm grid
        assert np.all(np.isnan(B))

    def test_interpolation_between_nodes(self, tmp_path):
        """Midpoint between nodes should interpolate to the node value (uniform field)."""
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        # All nodes have Bz = 1e-4; midpoint should too
        B_mid = fm.B_at_point((0.025, 0.025, 0.025), current_A=1.0)
        assert not np.any(np.isnan(B_mid))
        assert math.isclose(B_mid[2], 1e-4, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Vectorised B_at_points
# ---------------------------------------------------------------------------


class TestBAtPoints:
    def test_shape(self, tmp_path):
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        B = fm.B_at_points(pts, current_A=1.0)
        assert B.shape == (3, 3)

    def test_vectorised_matches_pointwise(self, tmp_path):
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        pts = np.array(
            [
                [-0.05, -0.05, -0.05],
                [0.0, 0.0, 0.0],
                [0.05, 0.05, 0.05],
            ]
        )
        B_vec = fm.B_at_points(pts, current_A=1.5)
        for j, pt in enumerate(pts):
            B_pt = fm.B_at_point(tuple(pt), current_A=1.5)
            np.testing.assert_allclose(B_vec[j], B_pt, atol=1e-15)

    def test_nan_for_out_of_bounds_rows(self, tmp_path):
        path = _write_regular_csv(tmp_path)
        fm = AnsysFieldMap(path)
        pts = np.array([[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]])
        B = fm.B_at_points(pts)
        assert not np.any(np.isnan(B[0]))
        assert np.all(np.isnan(B[1]))


# ---------------------------------------------------------------------------
# E-field tests
# ---------------------------------------------------------------------------


class TestEField:
    def test_e_magnitude_at_grid_node(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=True)
        fm = AnsysFieldMap(path)
        E = fm.E_magnitude_at_point((0.0, 0.0, 0.0), current_A=1.0)
        assert E is not None
        assert math.isclose(E, 5.0, rel_tol=1e-5)

    def test_e_field_none_when_not_loaded(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=False)
        fm = AnsysFieldMap(path)
        E = fm.E_magnitude_at_point((0.0, 0.0, 0.0))
        assert E is None

    def test_e_scales_with_current(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=True)
        fm = AnsysFieldMap(path)
        E1 = fm.E_magnitude_at_point((0.0, 0.0, 0.0), current_A=1.0)
        E2 = fm.E_magnitude_at_point((0.0, 0.0, 0.0), current_A=2.0)
        assert math.isclose(E2, E1 * 2.0, rel_tol=1e-9)

    def test_e_out_of_bounds_returns_nan(self, tmp_path):
        path = _write_regular_csv(tmp_path, with_efield=True)
        fm = AnsysFieldMap(path)
        E = fm.E_magnitude_at_point((5.0, 5.0, 5.0))
        assert math.isnan(E)


# ---------------------------------------------------------------------------
# Integration: B_field_at_point fallback behaviour
# ---------------------------------------------------------------------------


class TestMulticoilFallback:
    """
    Verify that multicoil.B_field_at_point uses the ANSYS map when provided
    and gracefully falls back to analytical for out-of-bounds points.
    """

    def test_uses_map_when_in_bounds(self, tmp_path):
        from src.neuroregen.multicoil import Coil, B_field_at_point

        coil = Coil(
            name="A",
            wire_mm=1.2,
            loop_mm=80,
            turns=20,
            pulse_power_w=60,
            position_m=(0.0, 0.0, 0.09),
            normal=(0.0, 0.0, -1.0),
        )
        path = _write_regular_csv(tmp_path)  # uniform Bz = 1e-4 T at 1A
        fm = AnsysFieldMap(path)

        B = B_field_at_point(coil, 1.0, (0.0, 0.0, 0.0), field_map=fm)
        # Map value must be used (Bz = 1e-4)
        assert math.isclose(B[2], 1e-4, rel_tol=1e-5)

    def test_falls_back_to_analytical_for_out_of_bounds(self, tmp_path):
        from src.neuroregen.multicoil import Coil, B_field_at_point

        coil = Coil(
            name="A",
            wire_mm=1.2,
            loop_mm=80,
            turns=20,
            pulse_power_w=60,
            position_m=(0.0, 0.0, 0.09),
            normal=(0.0, 0.0, -1.0),
        )
        path = _write_regular_csv(tmp_path)  # grid only covers ±5 cm
        fm = AnsysFieldMap(path)

        # Query 20 cm away — outside the ±5 cm grid → must fall back to analytical
        B_with_map = B_field_at_point(coil, 1.0, (0.0, 0.0, -0.20), field_map=fm)
        B_no_map = B_field_at_point(coil, 1.0, (0.0, 0.0, -0.20))

        # Should not be NaN
        assert not np.any(np.isnan(B_with_map))
        np.testing.assert_allclose(B_with_map, B_no_map, rtol=1e-9)

    def test_no_map_returns_analytical(self, tmp_path):
        from src.neuroregen.multicoil import Coil, B_field_at_point

        coil = Coil(
            name="A",
            wire_mm=1.2,
            loop_mm=80,
            turns=20,
            pulse_power_w=60,
            position_m=(0.0, 0.0, 0.09),
            normal=(0.0, 0.0, -1.0),
        )
        B_a = B_field_at_point(coil, 1.0, (0.0, 0.0, 0.04), field_map=None)
        B_b = B_field_at_point(coil, 1.0, (0.0, 0.0, 0.04))
        np.testing.assert_array_equal(B_a, B_b)


# ---------------------------------------------------------------------------
# load_ansys_field_maps config helper
# ---------------------------------------------------------------------------


class TestLoadAnsysFieldMaps:
    def test_returns_none_when_section_absent(self):
        config = {"coils": [{"name": "A"}, {"name": "B"}]}
        result = load_ansys_field_maps(config)
        assert result is None

    def test_loads_matching_coils(self, tmp_path):
        path_a = _write_regular_csv(tmp_path, "coil_A.csv")
        path_b = _write_regular_csv(tmp_path, "coil_B.csv")
        config = {
            "coils": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "ansys_field_maps": {
                "coil_maps": {
                    "A": str(path_a),
                    "B": str(path_b),
                }
            },
        }
        maps = load_ansys_field_maps(config)
        assert maps is not None
        assert len(maps) == 3
        assert isinstance(maps[0], AnsysFieldMap)
        assert isinstance(maps[1], AnsysFieldMap)
        assert maps[2] is None  # coil C has no map

    def test_warns_when_no_coil_names_match(self, tmp_path):
        path_a = _write_regular_csv(tmp_path, "coil_A.csv")
        config = {
            "coils": [{"name": "X"}, {"name": "Y"}],
            "ansys_field_maps": {
                "coil_maps": {"A": str(path_a)},
            },
        }
        with pytest.warns(UserWarning, match="no coil CSV paths matched"):
            result = load_ansys_field_maps(config)
        assert result is None
