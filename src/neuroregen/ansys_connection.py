"""
ANSYS Maxwell live connection for NeuroRegen multicoil simulation.

Uses ``pyaedt`` to programmatically drive ANSYS Maxwell 3D, push coil
geometry, run an Eddy Current solve, and extract B/E fields in memory.

The resulting :class:`AnsysLiveFieldMap` objects expose the same API as
:class:`~neuroregen.ansys_field_map.AnsysFieldMap` so that
``multicoil.py`` works unchanged with either mode.

``pyaedt`` is imported lazily — the module can be imported on any machine
and will raise a clear error only when a live connection is actually
requested.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import pyaedt  # noqa: F401

    _PYAEDT_AVAILABLE = True
except ImportError:
    _PYAEDT_AVAILABLE = False

from .ansys_field_map import _FieldMapQueryMixin


# ---------------------------------------------------------------------------
# AnsysLiveFieldMap — same API as AnsysFieldMap but built from numpy arrays
# ---------------------------------------------------------------------------
class AnsysLiveFieldMap(_FieldMapQueryMixin):
    """
    Field-map wrapper built from in-memory numpy arrays extracted via pyaedt.

    Provides the same query interface as
    :class:`~neuroregen.ansys_field_map.AnsysFieldMap` so that
    ``multicoil.py`` can use either transparently.

    Parameters
    ----------
    x, y, z : 1-D arrays
        Sorted unique grid coordinates (metres).
    Bx, By, Bz : 3-D arrays
        Magnetic field components (Tesla) on the ``(x, y, z)`` grid,
        shape ``(len(x), len(y), len(z))``.
    Ex, Ey, Ez : 3-D arrays or None
        Electric field components (V/m), same shape.  ``None`` if
        E-field data is unavailable.
    reference_current_a : float
        Current (A) at which the field was computed.
    coil_name : str
        Human-readable label.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        Bx: np.ndarray,
        By: np.ndarray,
        Bz: np.ndarray,
        Ex: Optional[np.ndarray] = None,
        Ey: Optional[np.ndarray] = None,
        Ez: Optional[np.ndarray] = None,
        reference_current_a: float = 1.0,
        coil_name: str = "",
    ) -> None:
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for field-map interpolation. Install with:  pip install scipy"
            )

        self._ref_current = float(reference_current_a)
        self._coil_name = coil_name

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        self._bounds_min = np.array([x.min(), y.min(), z.min()])
        self._bounds_max = np.array([x.max(), y.max(), z.max()])
        self._n_points = len(x) * len(y) * len(z)

        def _make_interp(data: np.ndarray) -> RegularGridInterpolator:
            return RegularGridInterpolator(
                (x, y, z),
                data,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

        self._bx_interp = _make_interp(Bx)
        self._by_interp = _make_interp(By)
        self._bz_interp = _make_interp(Bz)

        self._has_efield = Ex is not None and Ey is not None and Ez is not None
        if self._has_efield:
            self._ex_interp = _make_interp(Ex)
            self._ey_interp = _make_interp(Ey)
            self._ez_interp = _make_interp(Ez)
        else:
            self._ex_interp = self._ey_interp = self._ez_interp = None

    # ---- properties ---------------------------------------------------------

    @property
    def reference_current_a(self) -> float:
        return self._ref_current

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._bounds_min.copy(), self._bounds_max.copy()

    @property
    def has_efield(self) -> bool:
        return self._has_efield

    @property
    def coil_name(self) -> str:
        return self._coil_name

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def is_regular_grid(self) -> bool:
        return True  # always regular — built from grid arrays

    def __repr__(self) -> str:
        return (
            f"AnsysLiveFieldMap(coil={self._coil_name!r}, "
            f"n_pts={self._n_points:,}, "
            f"I_ref={self._ref_current:.2f} A, "
            f"has_efield={self._has_efield})"
        )


# ---------------------------------------------------------------------------
# AnsysLiveConnection — pyaedt driver
# ---------------------------------------------------------------------------
class AnsysLiveConnection:
    """
    Drives ANSYS Maxwell 3D via ``pyaedt`` to compute per-coil field maps.

    Parameters
    ----------
    coils : list of dict
        Coil definitions from the multicoil config (name, position_m,
        normal, loop_mm, turns, wire_mm, pulse_power_w).
    target : dict
        Target definition (name, position_m).
    connection_config : dict
        Keys: ``version``, ``non_graphical``, ``export_grid_spacing_mm``,
        ``reference_current_a``, ``head_model`` (with ``scalp_radius_m``,
        ``skull_radius_m``, ``brain_radius_m``).
    """

    def __init__(
        self,
        coils: list[dict],
        target: dict,
        connection_config: dict,
    ) -> None:
        self.coils = coils
        self.target = target
        self.config = connection_config

    def connect_and_solve(self) -> list[AnsysLiveFieldMap]:
        """
        Launch ANSYS Maxwell, build geometry, solve, and extract field maps.

        Returns one :class:`AnsysLiveFieldMap` per coil (isolated solves at
        ``reference_current_a`` with all other coils at 0 A).

        Raises
        ------
        ImportError
            If ``pyaedt`` is not installed.
        """
        if not _PYAEDT_AVAILABLE:
            raise ImportError(
                "pyaedt is required for ANSYS live connection mode. "
                "Install with:  pip install pyaedt\n"
                "An ANSYS Electronics Desktop licence is also required."
            )

        import pyaedt

        version = self.config.get("version", "2024.1")
        non_graphical = self.config.get("non_graphical", True)
        spacing_mm = self.config.get("export_grid_spacing_mm", 3.0)
        ref_current = self.config.get("reference_current_a", 1.0)
        head = self.config.get("head_model", {})

        maxwell = pyaedt.Maxwell3d(
            solution_type="EddyCurrent",
            specified_version=version,
            non_graphical=non_graphical,
        )

        try:
            # --- Build head tissue spheres ---
            self._build_head_model(maxwell, head)

            # --- Build coil geometries ---
            self._build_coils(maxwell)

            # --- Per-coil isolated solves ---
            field_maps: list[AnsysLiveFieldMap] = []
            for coil in self.coils:
                fm = self._solve_single_coil(maxwell, coil, ref_current, spacing_mm)
                field_maps.append(fm)

        finally:
            maxwell.close_desktop()

        return field_maps

    # ---- internal helpers ---------------------------------------------------

    def _build_head_model(self, maxwell, head: dict) -> None:
        """Create concentric spheres for scalp, skull, and brain."""
        origin = [0.0, 0.0, 0.0]

        for label, key in [
            ("scalp", "scalp_radius_m"),
            ("skull", "skull_radius_m"),
            ("brain", "brain_radius_m"),
        ]:
            radius = head.get(key)
            if radius is not None:
                maxwell.modeler.create_sphere(
                    origin=origin,
                    radius=float(radius),
                    name=label,
                )

    def _build_coils(self, maxwell) -> None:
        """Create a torus for each coil at its configured position and orientation."""
        for coil in self.coils:
            pos = [float(v) for v in coil["position_m"]]
            normal = [float(v) for v in coil["normal"]]
            major_radius = coil["loop_mm"] / 2000.0  # mm → m
            minor_radius = coil["wire_mm"] / 2000.0  # mm → m

            maxwell.modeler.create_torus(
                center=pos,
                major_radius=major_radius,
                minor_radius=minor_radius,
                axis=normal,
                name=f"coil_{coil['name']}",
            )

    def _solve_single_coil(
        self, maxwell, active_coil: dict, ref_current: float, spacing_mm: float
    ) -> AnsysLiveFieldMap:
        """Run an isolated solve for one coil and extract the field grid."""
        target_pos = self.target["position_m"]

        # Set up excitations: active coil at ref_current, others at 0
        for coil in self.coils:
            current = ref_current if coil["name"] == active_coil["name"] else 0.0
            maxwell.assign_current(
                assignment=f"coil_{coil['name']}",
                amplitude=current,
            )

        # Solve
        maxwell.analyze()

        # Build extraction grid centred on target
        spacing_m = spacing_mm / 1000.0
        half_extent = 0.05  # 5 cm around target
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

        # Build meshgrid for field extraction
        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
        points = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

        # Extract fields at grid points
        b_data = maxwell.post.get_solution_data(
            expressions=["Bx", "By", "Bz"],
            variations={"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]},
        )
        e_data = maxwell.post.get_solution_data(
            expressions=["Ex", "Ey", "Ez"],
            variations={"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]},
        )

        nx, ny, nz = len(x), len(y), len(z)
        Bx = np.array(b_data.data_real("Bx")).reshape(nx, ny, nz)
        By = np.array(b_data.data_real("By")).reshape(nx, ny, nz)
        Bz = np.array(b_data.data_real("Bz")).reshape(nx, ny, nz)

        Ex = np.array(e_data.data_real("Ex")).reshape(nx, ny, nz)
        Ey = np.array(e_data.data_real("Ey")).reshape(nx, ny, nz)
        Ez = np.array(e_data.data_real("Ez")).reshape(nx, ny, nz)

        return AnsysLiveFieldMap(
            x=x,
            y=y,
            z=z,
            Bx=Bx,
            By=By,
            Bz=Bz,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            reference_current_a=ref_current,
            coil_name=active_coil["name"],
        )
