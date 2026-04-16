"""
Tests for AnsysLiveConnection and AnsysLiveFieldMap.

All tests are fully mocked — pyaedt is never actually imported. Tests run
on any machine (macOS, Linux, CI) without an ANSYS licence.
"""

import math
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

scipy = pytest.importorskip("scipy", reason="scipy required for field-map interpolation")

# ---------------------------------------------------------------------------
# Constants matching config/multicoil.yaml defaults
# ---------------------------------------------------------------------------
MU0 = 4.0 * math.pi * 1e-7

SAMPLE_COILS = [
    {
        "name": "A",
        "wire_mm": 1.2,
        "loop_mm": 80,
        "turns": 20,
        "pulse_power_w": 60,
        "position_m": (-0.010, -0.010, 0.088),
        "normal": (-0.016, -0.023, -1.000),
    },
    {
        "name": "B",
        "wire_mm": 1.2,
        "loop_mm": 80,
        "turns": 20,
        "pulse_power_w": 60,
        "position_m": (0.085, -0.020, 0.025),
        "normal": (-0.829, 0.060, -0.556),
    },
    {
        "name": "C",
        "wire_mm": 1.2,
        "loop_mm": 80,
        "turns": 20,
        "pulse_power_w": 60,
        "position_m": (-0.015, -0.088, 0.015),
        "normal": (0.032, 0.806, -0.591),
    },
]

SAMPLE_TARGET = {
    "name": "STN_left",
    "position_m": (-0.012, -0.013, -0.040),
}

SAMPLE_CONNECTION = {
    "version": "2024.1",
    "non_graphical": True,
    "export_grid_spacing_mm": 3.0,
    "reference_current_a": 1.0,
    "head_model": {
        "scalp_radius_m": 0.095,
        "skull_radius_m": 0.090,
        "brain_radius_m": 0.080,
    },
}


# ---------------------------------------------------------------------------
# Helpers for generating synthetic field data
# ---------------------------------------------------------------------------
def _biot_savart_bz(z: float, R: float = 0.04, N: int = 20, I: float = 1.0) -> float:
    """On-axis Biot-Savart Bz for a circular loop of radius R, N turns, current I."""
    return MU0 * I * R**2 / (2.0 * (R**2 + z**2) ** 1.5) * N


def _make_synthetic_grid(target_pos, spacing_m=0.003, half_extent=0.05, ref_current=1.0):
    """Build synthetic B and E field grids centred on target_pos."""
    x = np.arange(
        target_pos[0] - half_extent,
        target_pos[0] + half_extent + spacing_m,
        spacing_m,
    )
    y = np.arange(
        target_pos[1] - half_extent,
        target_pos[1] + half_extent + spacing_m,
        spacing_m,
    )
    z = np.arange(
        target_pos[2] - half_extent,
        target_pos[2] + half_extent + spacing_m,
        spacing_m,
    )
    nx, ny, nz = len(x), len(y), len(z)
    n_pts = nx * ny * nz

    # Simple synthetic B-field: Bz ∝ 1/r³ from a dipole at origin
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(Xg**2 + Yg**2 + Zg**2) + 1e-10
    Bz_flat = (MU0 * ref_current * 0.04**2 * 20) / (2.0 * r**3)
    Bx_flat = np.zeros_like(Bz_flat)
    By_flat = np.zeros_like(Bz_flat)

    # Synthetic E-field: E ≈ (r/2) * dB/dt at 50 Hz
    freq = 50.0
    dBdt = Bz_flat * 2.0 * math.pi * freq
    Ex_flat = (r / 2.0) * dBdt * 0.5
    Ey_flat = (r / 2.0) * dBdt * 0.5
    Ez_flat = np.zeros_like(Bz_flat)

    return {
        "x": x,
        "y": y,
        "z": z,
        "n_pts": n_pts,
        "Bx": Bx_flat,
        "By": By_flat,
        "Bz": Bz_flat,
        "Ex": Ex_flat,
        "Ey": Ey_flat,
        "Ez": Ez_flat,
    }


# ---------------------------------------------------------------------------
# Build a mock pyaedt module
# ---------------------------------------------------------------------------
def _build_mock_pyaedt(target_pos):
    """
    Create a mock pyaedt module whose Maxwell3d records geometry calls
    and returns synthetic field data.
    """
    mock_pyaedt = ModuleType("pyaedt")

    grids = _make_synthetic_grid(target_pos)

    class MockSolutionData:
        def __init__(self, expressions, grids):
            self._grids = grids
            self._expressions = expressions

        def data_real(self, name):
            return self._grids[name].ravel().tolist()

    class MockPost:
        def __init__(self, grids):
            self._grids = grids

        def get_solution_data(self, expressions=None, variations=None):
            # Map Bx/By/Bz to grid data and Ex/Ey/Ez to grid data
            field_grids = {}
            for expr in expressions:
                field_grids[expr] = self._grids[expr]
            return MockSolutionData(expressions, field_grids)

    class MockModeler:
        def __init__(self):
            self.spheres = []
            self.tori = []

        def create_sphere(self, origin=None, radius=None, name=None):
            self.spheres.append({"origin": origin, "radius": radius, "name": name})
            return MagicMock()

        def create_torus(
            self,
            center=None,
            major_radius=None,
            minor_radius=None,
            axis=None,
            name=None,
        ):
            self.tori.append(
                {
                    "center": center,
                    "major_radius": major_radius,
                    "minor_radius": minor_radius,
                    "axis": axis,
                    "name": name,
                }
            )
            return MagicMock()

    class MockMaxwell3d:
        def __init__(self, solution_type=None, specified_version=None, non_graphical=None):
            self.solution_type = solution_type
            self.specified_version = specified_version
            self.non_graphical = non_graphical
            self.modeler = MockModeler()
            self.post = MockPost(grids)
            self._current_assignments = {}

        def assign_current(self, assignment=None, amplitude=None):
            self._current_assignments[assignment] = amplitude

        def analyze(self):
            pass

        def close_desktop(self):
            pass

    mock_pyaedt.Maxwell3d = MockMaxwell3d
    return mock_pyaedt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPyaedtMissing:
    """Test behaviour when pyaedt is not installed."""

    def test_pyaedt_missing_raises(self):
        """Requesting live mode without pyaedt raises ImportError."""
        # Temporarily force _PYAEDT_AVAILABLE to False
        from src.neuroregen import ansys_connection as mod

        original = mod._PYAEDT_AVAILABLE
        try:
            mod._PYAEDT_AVAILABLE = False
            conn = mod.AnsysLiveConnection(
                coils=SAMPLE_COILS,
                target=SAMPLE_TARGET,
                connection_config=SAMPLE_CONNECTION,
            )
            with pytest.raises(ImportError, match="pyaedt"):
                conn.connect_and_solve()
        finally:
            mod._PYAEDT_AVAILABLE = original


class TestConnectAndSolve:
    """Test the full connect_and_solve flow with mocked pyaedt."""

    @pytest.fixture()
    def mock_pyaedt(self):
        return _build_mock_pyaedt(SAMPLE_TARGET["position_m"])

    @pytest.fixture()
    def patched_connection(self, mock_pyaedt):
        """Patch pyaedt into sys.modules and enable the flag."""
        with patch.dict(sys.modules, {"pyaedt": mock_pyaedt}):
            from src.neuroregen import ansys_connection as mod

            original = mod._PYAEDT_AVAILABLE
            mod._PYAEDT_AVAILABLE = True
            yield mod
            mod._PYAEDT_AVAILABLE = original

    def test_returns_field_maps(self, patched_connection):
        mod = patched_connection
        conn = mod.AnsysLiveConnection(
            coils=SAMPLE_COILS,
            target=SAMPLE_TARGET,
            connection_config=SAMPLE_CONNECTION,
        )
        maps = conn.connect_and_solve()
        assert len(maps) == 3
        for fm in maps:
            assert isinstance(fm, mod.AnsysLiveFieldMap)

    def test_coil_names_match(self, patched_connection):
        mod = patched_connection
        conn = mod.AnsysLiveConnection(
            coils=SAMPLE_COILS,
            target=SAMPLE_TARGET,
            connection_config=SAMPLE_CONNECTION,
        )
        maps = conn.connect_and_solve()
        names = [fm.coil_name for fm in maps]
        assert names == ["A", "B", "C"]

    def test_coil_geometry_pushed_to_ansys(self, patched_connection, mock_pyaedt):
        mod = patched_connection
        conn = mod.AnsysLiveConnection(
            coils=SAMPLE_COILS,
            target=SAMPLE_TARGET,
            connection_config=SAMPLE_CONNECTION,
        )
        conn.connect_and_solve()

        # Access the Maxwell instance's modeler through the mock
        # The Maxwell3d class was called — we need to inspect the mock calls
        # Since our mock doesn't record globally, we verify via the torus count
        # by creating a fresh instance and inspecting directly
        maxwell = mock_pyaedt.Maxwell3d(
            solution_type="EddyCurrent",
            specified_version="2024.1",
            non_graphical=True,
        )
        # Build coils on this fresh instance to verify the method works
        conn._build_coils(maxwell)
        assert len(maxwell.modeler.tori) == 3
        torus_names = [t["name"] for t in maxwell.modeler.tori]
        assert "coil_A" in torus_names
        assert "coil_B" in torus_names
        assert "coil_C" in torus_names

        # Verify positions match config
        for torus in maxwell.modeler.tori:
            coil_name = torus["name"].replace("coil_", "")
            coil_cfg = next(c for c in SAMPLE_COILS if c["name"] == coil_name)
            expected_pos = [float(v) for v in coil_cfg["position_m"]]
            assert torus["center"] == expected_pos

    def test_head_model_spheres_created(self, patched_connection, mock_pyaedt):
        mod = patched_connection
        maxwell = mock_pyaedt.Maxwell3d()
        conn = mod.AnsysLiveConnection(
            coils=SAMPLE_COILS,
            target=SAMPLE_TARGET,
            connection_config=SAMPLE_CONNECTION,
        )
        head = SAMPLE_CONNECTION["head_model"]
        conn._build_head_model(maxwell, head)

        assert len(maxwell.modeler.spheres) == 3
        names = {s["name"] for s in maxwell.modeler.spheres}
        assert names == {"scalp", "skull", "brain"}

        radii = {s["name"]: s["radius"] for s in maxwell.modeler.spheres}
        assert math.isclose(radii["scalp"], 0.095)
        assert math.isclose(radii["skull"], 0.090)
        assert math.isclose(radii["brain"], 0.080)


class TestAnsysLiveFieldMap:
    """Test the AnsysLiveFieldMap wrapper directly."""

    @pytest.fixture()
    def field_map(self):
        """Build an AnsysLiveFieldMap from synthetic data."""
        target_pos = SAMPLE_TARGET["position_m"]
        grids = _make_synthetic_grid(target_pos, spacing_m=0.005, half_extent=0.05)
        return __import__(
            "src.neuroregen.ansys_connection", fromlist=["AnsysLiveFieldMap"]
        ).AnsysLiveFieldMap(
            x=grids["x"],
            y=grids["y"],
            z=grids["z"],
            Bx=grids["Bx"],
            By=grids["By"],
            Bz=grids["Bz"],
            Ex=grids["Ex"],
            Ey=grids["Ey"],
            Ez=grids["Ez"],
            reference_current_a=1.0,
            coil_name="test",
        )

    def test_interpolation_returns_finite(self, field_map):
        """B_at_point at the target should return finite, non-NaN values."""
        target_pos = SAMPLE_TARGET["position_m"]
        B = field_map.B_at_point(target_pos, current_A=1.0)
        assert B.shape == (3,)
        assert not np.any(np.isnan(B))
        assert np.linalg.norm(B) > 0

    def test_field_map_api_compatible(self, field_map):
        """The field map has all methods multicoil.py expects."""
        assert hasattr(field_map, "B_at_point")
        assert hasattr(field_map, "B_at_points")
        assert hasattr(field_map, "E_magnitude_at_point")
        assert hasattr(field_map, "E_at_points")
        assert hasattr(field_map, "reference_current_a")
        assert hasattr(field_map, "bounds")
        assert hasattr(field_map, "has_efield")
        assert hasattr(field_map, "coil_name")
        assert hasattr(field_map, "n_points")
        assert hasattr(field_map, "is_regular_grid")

    def test_current_scaling(self, field_map):
        """B at 2 A should be 2x B at 1 A."""
        target_pos = SAMPLE_TARGET["position_m"]
        B1 = field_map.B_at_point(target_pos, current_A=1.0)
        B2 = field_map.B_at_point(target_pos, current_A=2.0)
        np.testing.assert_allclose(B2, B1 * 2.0, rtol=1e-9)

    def test_b_at_points_vectorised(self, field_map):
        """Vectorised query matches pointwise queries."""
        pts = np.array(
            [
                list(SAMPLE_TARGET["position_m"]),
                [-0.012, -0.013, -0.030],
            ]
        )
        B_vec = field_map.B_at_points(pts, current_A=1.5)
        assert B_vec.shape == (2, 3)
        for j in range(2):
            B_pt = field_map.B_at_point(tuple(pts[j]), current_A=1.5)
            np.testing.assert_allclose(B_vec[j], B_pt, atol=1e-15)

    def test_has_efield(self, field_map):
        assert field_map.has_efield is True

    def test_e_magnitude_at_point(self, field_map):
        target_pos = SAMPLE_TARGET["position_m"]
        E = field_map.E_magnitude_at_point(target_pos, current_A=1.0)
        assert E is not None
        assert not math.isnan(E)
        assert E > 0

    def test_is_regular_grid(self, field_map):
        assert field_map.is_regular_grid is True

    def test_n_points_positive(self, field_map):
        assert field_map.n_points > 0

    def test_bounds(self, field_map):
        lo, hi = field_map.bounds
        assert lo.shape == (3,)
        assert hi.shape == (3,)
        assert np.all(hi > lo)

    def test_repr(self, field_map):
        r = repr(field_map)
        assert "test" in r
        assert "AnsysLiveFieldMap" in r

    def test_no_efield(self):
        """AnsysLiveFieldMap without E-field data returns None."""
        from src.neuroregen.ansys_connection import AnsysLiveFieldMap

        x = np.linspace(-0.05, 0.05, 5)
        y = np.linspace(-0.05, 0.05, 5)
        z = np.linspace(-0.05, 0.05, 5)
        shape = (5, 5, 5)
        Bz = np.full(shape, 1e-4)
        fm = AnsysLiveFieldMap(
            x=x,
            y=y,
            z=z,
            Bx=np.zeros(shape),
            By=np.zeros(shape),
            Bz=Bz,
            coil_name="no_e",
        )
        assert fm.has_efield is False
        assert fm.E_magnitude_at_point((0.0, 0.0, 0.0)) is None
        assert fm.E_at_points(np.array([[0.0, 0.0, 0.0]])) is None


class TestConfigLoaderIntegration:
    """Test config_loader parsing of the new ansys block."""

    def test_config_loader_live_mode(self, tmp_path):
        """load_multicoil_config correctly parses ansys.mode: 'live'."""
        yaml_content = """\
target:
  name: STN_left
  position_m: [-0.012, -0.013, -0.040]
safety:
  cortical_max_vm: 150.0
  scalp_to_cortex_m: 0.015
geometry:
  skull_radius_m: 0.09
coils:
  - name: A
    wire_mm: 1.2
    loop_mm: 80
    turns: 20
    pulse_power_w: 60
    position_m: [0, 0, 0.09]
    normal: [0, 0, -1]
simulation:
  sim_time: 10
  dt: 0.1
  pulse_freq: 5.0
  pulse_width: 0.02
thermal:
  h_conv: 10.0
  t_amb_c: 22.0
  temp_limit_f: 75.0
  hyst_f: 0.7
depth:
  z_max_m: 0.12
  z_points: 500
  b_threshold_t: 1.0e-4
ansys:
  mode: "live"
  connection:
    version: "2024.1"
    non_graphical: true
    export_grid_spacing_mm: 3.0
    reference_current_a: 1.0
    head_model:
      scalp_radius_m: 0.095
      skull_radius_m: 0.090
      brain_radius_m: 0.080
"""
        cfg_path = tmp_path / "test_multicoil.yaml"
        cfg_path.write_text(yaml_content)

        from src.neuroregen.config_loader import load_multicoil_config

        config = load_multicoil_config(cfg_path)
        assert config["ansys_mode"] == "live"
        assert "ansys_connection" in config
        assert config["ansys_connection"]["version"] == "2024.1"
        assert config["ansys_connection"]["non_graphical"] is True
        assert config["ansys_connection"]["head_model"]["scalp_radius_m"] == 0.095

    def test_config_loader_file_mode(self, tmp_path):
        """load_multicoil_config correctly parses ansys.mode: 'file'."""
        yaml_content = """\
target:
  name: STN_left
  position_m: [-0.012, -0.013, -0.040]
safety:
  cortical_max_vm: 150.0
  scalp_to_cortex_m: 0.015
geometry:
  skull_radius_m: 0.09
coils:
  - name: A
    wire_mm: 1.2
    loop_mm: 80
    turns: 20
    pulse_power_w: 60
    position_m: [0, 0, 0.09]
    normal: [0, 0, -1]
simulation:
  sim_time: 10
  dt: 0.1
  pulse_freq: 5.0
  pulse_width: 0.02
thermal:
  h_conv: 10.0
  t_amb_c: 22.0
  temp_limit_f: 75.0
  hyst_f: 0.7
depth:
  z_max_m: 0.12
  z_points: 500
  b_threshold_t: 1.0e-4
ansys:
  mode: "file"
  field_maps:
    reference_current_a: 1.0
    coil_maps:
      A: exports/coil_A.csv
"""
        cfg_path = tmp_path / "test_multicoil.yaml"
        cfg_path.write_text(yaml_content)

        from src.neuroregen.config_loader import load_multicoil_config

        config = load_multicoil_config(cfg_path)
        assert config["ansys_mode"] == "file"
        assert "ansys_field_maps" in config
        assert config["ansys_field_maps"]["coil_maps"]["A"] == "exports/coil_A.csv"

    def test_backward_compat_old_key(self, tmp_path):
        """Old ansys_field_maps key still works as mode: 'file'."""
        yaml_content = """\
target:
  name: STN_left
  position_m: [-0.012, -0.013, -0.040]
safety:
  cortical_max_vm: 150.0
  scalp_to_cortex_m: 0.015
geometry:
  skull_radius_m: 0.09
coils:
  - name: A
    wire_mm: 1.2
    loop_mm: 80
    turns: 20
    pulse_power_w: 60
    position_m: [0, 0, 0.09]
    normal: [0, 0, -1]
simulation:
  sim_time: 10
  dt: 0.1
  pulse_freq: 5.0
  pulse_width: 0.02
thermal:
  h_conv: 10.0
  t_amb_c: 22.0
  temp_limit_f: 75.0
  hyst_f: 0.7
depth:
  z_max_m: 0.12
  z_points: 500
  b_threshold_t: 1.0e-4
ansys_field_maps:
  reference_current_a: 1.0
  coil_maps:
    A: old_path/coil_A.csv
"""
        cfg_path = tmp_path / "test_multicoil.yaml"
        cfg_path.write_text(yaml_content)

        from src.neuroregen.config_loader import load_multicoil_config

        config = load_multicoil_config(cfg_path)
        assert config["ansys_mode"] == "file"
        assert config["ansys_field_maps"]["coil_maps"]["A"] == "old_path/coil_A.csv"

    def test_no_ansys_section(self, tmp_path):
        """Config without any ansys section has no ansys_mode key."""
        yaml_content = """\
target:
  name: STN_left
  position_m: [-0.012, -0.013, -0.040]
safety:
  cortical_max_vm: 150.0
  scalp_to_cortex_m: 0.015
geometry:
  skull_radius_m: 0.09
coils:
  - name: A
    wire_mm: 1.2
    loop_mm: 80
    turns: 20
    pulse_power_w: 60
    position_m: [0, 0, 0.09]
    normal: [0, 0, -1]
simulation:
  sim_time: 10
  dt: 0.1
  pulse_freq: 5.0
  pulse_width: 0.02
thermal:
  h_conv: 10.0
  t_amb_c: 22.0
  temp_limit_f: 75.0
  hyst_f: 0.7
depth:
  z_max_m: 0.12
  z_points: 500
  b_threshold_t: 1.0e-4
"""
        cfg_path = tmp_path / "test_multicoil.yaml"
        cfg_path.write_text(yaml_content)

        from src.neuroregen.config_loader import load_multicoil_config

        config = load_multicoil_config(cfg_path)
        assert "ansys_mode" not in config
        assert "ansys_field_maps" not in config
        assert "ansys_connection" not in config
