"""
ANSYS Maxwell field-map integration for NeuroRegen multicoil simulation.

Loads B-field (and optionally E-field) data exported from ANSYS Maxwell 3D
and provides linear interpolation at arbitrary points, replacing the
analytical Biot–Savart approximation in ``multicoil.py``.

CSV Format
----------
The loader accepts a CSV file with an optional metadata header (comment lines
beginning with ``#``) followed by a row of column names and then numerical
data.  Supported column names are case-insensitive; units are detected from
the suffix:

Coordinate columns (exactly one group required)::

    x_m,  y_m,  z_m      – metres (preferred)
    x_mm, y_mm, z_mm     – millimetres (converted automatically)

Magnetic field columns (required)::

    Bx_T,  By_T,  Bz_T      – Tesla (preferred)
    Bx_mT, By_mT, Bz_mT     – millitesla (converted automatically)
    Bx_G,  By_G,  Bz_G      – Gauss     (converted automatically)

Optional induced E-field columns (used for the cortical safety gate)::

    Ex_Vm, Ey_Vm, Ez_Vm     – V/m

Optional metadata comments (before the column-header row)::

    # reference_current_A: 1.0

When ``reference_current_A`` is declared, the stored field is assumed to
correspond to that excitation current and is linearly scaled to the actual
current at query time.  Defaults to ``1.0 A`` if the comment is absent.

Grid types
----------
The loader automatically detects whether the data lies on a **regular
Cartesian grid** (``len(xu) × len(yu) × len(zu) == len(rows)``).

* **Regular grid** → ``scipy.interpolate.RegularGridInterpolator``.
  Vectorised, memory-efficient, supports grids with millions of nodes.
* **Scattered / unstructured** → ``scipy.interpolate.LinearNDInterpolator``.
  Works with tetrahedral FEM meshes exported directly from ANSYS.

Out-of-bounds behaviour
-----------------------
Both interpolators return ``NaN`` for points outside the exported domain.
When ``B_field_at_point`` in ``multicoil.py`` receives a ``NaN`` result it
transparently falls back to the analytical Biot–Savart formula, so a
partial-domain export is safe.

Typical ANSYS Maxwell export workflow
--------------------------------------
1. Solve the magnetostatic problem at a reference current (e.g. 1 A per coil).
2. Right-click the *Field Overlays* → *Export to File* → choose *Rectangular
   Grid* or *All Mesh Nodes*, select ``Bx``, ``By``, ``Bz`` components.
3. Set units to **metres** and **Tesla**, name the file ``coil_A.csv``, etc.
4. Add ``# reference_current_A: 1.0`` as the first line of each file.
5. Point ``multicoil.yaml`` to the files under ``ansys_field_maps:``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unit-conversion lookup tables  (normalised column name → scale factor)
# ---------------------------------------------------------------------------

_COORD_COLS = {
    "x_m": ("x", 1.0),
    "y_m": ("y", 1.0),
    "z_m": ("z", 1.0),
    "x_mm": ("x", 1e-3),
    "y_mm": ("y", 1e-3),
    "z_mm": ("z", 1e-3),
}

_B_COLS = {
    "bx_t": ("bx", 1.0),
    "by_t": ("by", 1.0),
    "bz_t": ("bz", 1.0),
    "bx_mt": ("bx", 1e-3),
    "by_mt": ("by", 1e-3),
    "bz_mt": ("bz", 1e-3),
    "bx_millitesla": ("bx", 1e-3),
    "by_millitesla": ("by", 1e-3),
    "bz_millitesla": ("bz", 1e-3),
    "bx_g": ("bx", 1e-4),
    "by_g": ("by", 1e-4),
    "bz_g": ("bz", 1e-4),
    "bx_gauss": ("bx", 1e-4),
    "by_gauss": ("by", 1e-4),
    "bz_gauss": ("bz", 1e-4),
}

_E_COLS = {
    "ex_vm": ("ex", 1.0),
    "ey_vm": ("ey", 1.0),
    "ez_vm": ("ez", 1.0),
    "ex_v/m": ("ex", 1.0),
    "ey_v/m": ("ey", 1.0),
    "ez_v/m": ("ez", 1.0),
    "ex_mv/m": ("ex", 1e-3),
    "ey_mv/m": ("ey", 1e-3),
    "ez_mv/m": ("ez", 1e-3),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_csv(path: Path) -> tuple[dict[str, np.ndarray], float]:
    """
    Read an ANSYS-format CSV, strip metadata comments, normalise units.

    Returns
    -------
    arrays : dict mapping logical name → ndarray
        Keys include ``"x"``, ``"y"``, ``"z"`` (metres), ``"bx"``, ``"by"``,
        ``"bz"`` (Tesla), and optionally ``"ex"``, ``"ey"``, ``"ez"`` (V/m).
    ref_current : float
        Reference current (A) parsed from ``# reference_current_A:`` line,
        or ``1.0`` if not found.
    """
    text_lines: list[str] = path.read_text().splitlines()

    ref_current = 1.0
    data_lines: list[str] = []
    header_line: Optional[str] = None

    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            low = stripped.lstrip("#").strip().lower()
            if low.startswith("reference_current_a:"):
                try:
                    ref_current = float(low.split(":", 1)[1].strip())
                except ValueError:
                    pass
            continue
        if header_line is None:
            header_line = stripped
        else:
            data_lines.append(stripped)

    if header_line is None:
        raise ValueError(f"No header row found in {path}")
    if not data_lines:
        raise ValueError(f"No data rows found in {path}")

    # Parse column names (strip quotes and whitespace)
    col_names = [c.strip().strip('"').strip("'") for c in header_line.split(",")]
    col_lower = [c.lower() for c in col_names]

    # Parse numerical data
    raw = np.array(
        [[float(v) for v in row.split(",")] for row in data_lines],
        dtype=float,
    )
    if raw.shape[1] != len(col_names):
        raise ValueError(f"{path}: expected {len(col_names)} columns, got {raw.shape[1]}")

    # Build column index for quick lookup
    col_idx: dict[str, int] = {name: i for i, name in enumerate(col_lower)}

    arrays: dict[str, np.ndarray] = {}

    for lookup in (_COORD_COLS, _B_COLS, _E_COLS):
        for key, (logical, scale) in lookup.items():
            if key in col_idx:
                arrays[logical] = raw[:, col_idx[key]] * scale

    # Validate required fields
    for req in ("x", "y", "z", "bx", "by", "bz"):
        if req not in arrays:
            raise ValueError(f"{path}: missing required column for '{req}'. Available: {col_names}")

    return arrays, ref_current


def _is_regular_grid(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Check whether (x, y, z) form a complete Cartesian product.

    Returns (is_regular, xu, yu, zu).
    """
    xu = np.unique(x)
    yu = np.unique(y)
    zu = np.unique(z)
    regular = len(xu) * len(yu) * len(zu) == len(x)
    return regular, xu, yu, zu


def _build_regular_interps(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xu: np.ndarray,
    yu: np.ndarray,
    zu: np.ndarray,
    components: list[np.ndarray],
) -> list:
    """
    Build one ``RegularGridInterpolator`` per field component.

    The raw data must already cover all (xu × yu × zu) combinations; this
    function sorts and reshapes it to match the grid axes.
    """
    # Sort by (x, y, z) so that reshape produces a consistent grid
    order = np.lexsort((z, y, x))
    nx, ny, nz = len(xu), len(yu), len(zu)

    interps = []
    for comp in components:
        grid = comp[order].reshape(nx, ny, nz)
        interps.append(
            RegularGridInterpolator(
                (xu, yu, zu),
                grid,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        )
    return interps


def _build_scattered_interps(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    components: list[np.ndarray],
) -> list:
    """
    Build one ``LinearNDInterpolator`` per field component for scattered data.
    """
    points = np.stack([x, y, z], axis=1)
    return [LinearNDInterpolator(points, comp, fill_value=np.nan) for comp in components]


# ---------------------------------------------------------------------------
# Shared query mixin
# ---------------------------------------------------------------------------


class _FieldMapQueryMixin:
    """Shared B/E-field query methods for file-based and live field maps."""

    _bx_interp: object
    _by_interp: object
    _bz_interp: object
    _ex_interp: object
    _ey_interp: object
    _ez_interp: object
    _ref_current: float
    has_efield: bool

    def B_at_point(
        self,
        point_m: tuple[float, float, float],
        current_A: float = 1.0,
    ) -> np.ndarray:
        """Interpolated B-field vector (T) at a single 3-D point."""
        pt = np.asarray(point_m, dtype=float).reshape(1, 3)
        scale = current_A / self._ref_current
        Bx = float(self._bx_interp(pt)[0])
        By = float(self._by_interp(pt)[0])
        Bz = float(self._bz_interp(pt)[0])
        return np.array([Bx, By, Bz]) * scale

    def B_at_points(
        self,
        points_m: np.ndarray,
        current_A: float = 1.0,
    ) -> np.ndarray:
        """Vectorised B-field query for an array of points."""
        pts = np.asarray(points_m, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        scale = current_A / self._ref_current
        Bx = self._bx_interp(pts)
        By = self._by_interp(pts)
        Bz = self._bz_interp(pts)
        return np.stack([Bx, By, Bz], axis=1) * scale

    def E_magnitude_at_point(
        self,
        point_m: tuple[float, float, float],
        current_A: float = 1.0,
    ) -> Optional[float]:
        """Interpolated E-field magnitude (V/m) at a single point."""
        if not self.has_efield:
            return None
        pt = np.asarray(point_m, dtype=float).reshape(1, 3)
        scale = current_A / self._ref_current
        Ex = float(self._ex_interp(pt)[0])
        Ey = float(self._ey_interp(pt)[0])
        Ez = float(self._ez_interp(pt)[0])
        if any(np.isnan(v) for v in (Ex, Ey, Ez)):
            return float("nan")
        return float(np.sqrt(Ex**2 + Ey**2 + Ez**2)) * scale

    def E_at_points(
        self,
        points_m: np.ndarray,
        current_A: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Vectorised E-field magnitude query."""
        if not self.has_efield:
            return None
        pts = np.asarray(points_m, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        scale = current_A / self._ref_current
        Ex = self._ex_interp(pts)
        Ey = self._ey_interp(pts)
        Ez = self._ez_interp(pts)
        return np.sqrt(Ex**2 + Ey**2 + Ez**2) * scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AnsysFieldMap(_FieldMapQueryMixin):
    """
    ANSYS Maxwell 3D field export loaded as an interpolation table.

    Parameters
    ----------
    path : str or Path
        CSV file exported from ANSYS Maxwell.  See module docstring for the
        expected format.
    coil_name : str, optional
        Human-readable label used in warning messages.

    Attributes
    ----------
    coil_name : str
    reference_current_a : float
        Current (A) at which the ANSYS field was computed.
    is_regular_grid : bool
        ``True`` if a ``RegularGridInterpolator`` was used (structured export);
        ``False`` for scattered / FEM mesh exports.
    has_efield : bool
        ``True`` if Ex, Ey, Ez columns were present in the file.
    bounds : tuple[ndarray, ndarray]
        ``(min_xyz, max_xyz)`` bounding box of the exported domain (metres).
    n_points : int
        Number of data rows loaded.
    """

    def __init__(self, path: str | Path, coil_name: str = "") -> None:
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for ANSYS field-map interpolation. "
                "Install with:  pip install scipy"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ANSYS field map not found: {path}")

        arrays, ref_current = _parse_csv(path)
        self.coil_name = coil_name or path.stem
        self._ref_current: float = float(ref_current)
        self.n_points: int = len(arrays["x"])

        x, y, z = arrays["x"], arrays["y"], arrays["z"]
        bx, by, bz = arrays["bx"], arrays["by"], arrays["bz"]

        self._bounds_min = np.array([x.min(), y.min(), z.min()])
        self._bounds_max = np.array([x.max(), y.max(), z.max()])

        # Detect grid type
        is_reg, xu, yu, zu = _is_regular_grid(x, y, z)
        self.is_regular_grid: bool = is_reg

        if is_reg:
            b_interps = _build_regular_interps(x, y, z, xu, yu, zu, [bx, by, bz])
        else:
            b_interps = _build_scattered_interps(x, y, z, [bx, by, bz])
        self._bx_interp, self._by_interp, self._bz_interp = b_interps

        # Optional E-field
        self.has_efield: bool = all(k in arrays for k in ("ex", "ey", "ez"))
        if self.has_efield:
            ex, ey, ez = arrays["ex"], arrays["ey"], arrays["ez"]
            if is_reg:
                e_interps = _build_regular_interps(x, y, z, xu, yu, zu, [ex, ey, ez])
            else:
                e_interps = _build_scattered_interps(x, y, z, [ex, ey, ez])
            self._ex_interp, self._ey_interp, self._ez_interp = e_interps
        else:
            self._ex_interp = self._ey_interp = self._ez_interp = None

        grid_type = "regular" if is_reg else "scattered"
        e_note = " + E-field" if self.has_efield else ""
        print(
            f"  [AnsysFieldMap] Loaded '{self.coil_name}' — {self.n_points:,} pts "
            f"({grid_type}{e_note}), I_ref={self._ref_current:.2f} A, "
            f"bounds x=[{self._bounds_min[0] * 100:.1f}, {self._bounds_max[0] * 100:.1f}] cm "
            f"y=[{self._bounds_min[1] * 100:.1f}, {self._bounds_max[1] * 100:.1f}] cm "
            f"z=[{self._bounds_min[2] * 100:.1f}, {self._bounds_max[2] * 100:.1f}] cm"
        )

    # ---- properties ---------------------------------------------------------

    @property
    def reference_current_a(self) -> float:
        """Current (A) at which the ANSYS data was computed."""
        return self._ref_current

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min_xyz, max_xyz)`` bounding box of the exported domain (metres)."""
        return self._bounds_min.copy(), self._bounds_max.copy()

    def __repr__(self) -> str:
        grid = "regular" if self.is_regular_grid else "scattered"
        return (
            f"AnsysFieldMap(coil={self.coil_name!r}, "
            f"n_pts={self.n_points:,}, grid={grid}, "
            f"I_ref={self._ref_current:.2f} A, "
            f"has_efield={self.has_efield})"
        )


# ---------------------------------------------------------------------------
# Config-driven loader
# ---------------------------------------------------------------------------


def load_ansys_field_maps(
    config: dict,
    base_dir: Optional[str | Path] = None,
) -> Optional[list[Optional["AnsysFieldMap"]]]:
    """
    Load ANSYS field maps from a multicoil config dict.

    Reads the optional ``ansys_field_maps`` section of the config produced by
    :func:`~neuroregen.config_loader.load_multicoil_config`.  Returns a list
    with one entry per coil (``None`` for coils without a map), or ``None``
    if the section is absent.

    Parameters
    ----------
    config : dict
        Flat config dict from ``load_multicoil_config``.
    base_dir : path, optional
        Directory used to resolve relative CSV paths.  Defaults to the
        current working directory.

    Returns
    -------
    list[AnsysFieldMap | None] | None
    """
    section = config.get("ansys_field_maps")
    if not section:
        return None

    coil_specs = config.get("coils", [])
    coil_names = [c["name"] for c in coil_specs]
    map_paths: dict[str, str] = section.get("coil_maps", {})

    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    result: list[Optional[AnsysFieldMap]] = []
    for name in coil_names:
        if name in map_paths:
            csv_path = Path(map_paths[name])
            if not csv_path.is_absolute():
                csv_path = base_dir / csv_path
            result.append(AnsysFieldMap(csv_path, coil_name=name))
        else:
            result.append(None)

    loaded = sum(1 for m in result if m is not None)
    if loaded == 0:
        warnings.warn(
            "ansys_field_maps section found in config but no coil CSV paths matched "
            f"coil names {coil_names}. Check 'coil_maps' keys.",
            stacklevel=2,
        )
        return None

    return result
