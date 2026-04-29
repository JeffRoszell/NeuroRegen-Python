"""
NeuroRegen — External B-field simulator (ANSYS Maxwell replacement).

Pipeline
--------
    YAML / CLI overrides
        ↓
    Build N coils — stacked Circle loops (default) or helix Polyline
    (magpylib idiomatic; Circle uses the Ortner-2022 closed-form formula)
        ↓
    Load head mesh                         (PyVista — .stl / .ply / .vtk / .msh)
        ↓
    Coil ↔ scalp standoff check            (signed distance vs SimNIBS-style
                                            skin offset, default 4 mm)
        ↓
    Build 3-D observation grid             (numpy meshgrid → flat (N,3))
        ↓
    Vectorised B = Collection.getB(observers)
        ↓
    Wrap as pyvista.ImageData, render:
        - semi-transparent head
        - 3-D coil geometry
        - magnetic field streamlines
        - orthogonal |B| slice heatmap
        - target marker

All units are SI (m, A, T) end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyvista as pv
import yaml
from scipy.spatial.transform import Rotation as R_scipy

import magpylib as magpy

from constraints import LimitsSpec, evaluate_constraints, report_constraints

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CoilSpec:
    name: str
    loop_diameter_m: float
    wire_diameter_m: float
    turns: int
    helix_height_m: float
    segments_per_turn: int
    current_a: float
    position_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    geometry: str = "stacked"            # "stacked" | "helix"
    peak_dIdt_a_per_us: Optional[float] = None


@dataclass
class GridSpec:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    spacing_m: float

    def dimensions(self) -> tuple[int, int, int]:
        nx = int(round((self.xmax - self.xmin) / self.spacing_m)) + 1
        ny = int(round((self.ymax - self.ymin) / self.spacing_m)) + 1
        nz = int(round((self.zmax - self.zmin) / self.spacing_m)) + 1
        return nx, ny, nz

    def origin(self) -> tuple[float, float, float]:
        return (self.xmin, self.ymin, self.zmin)


@dataclass
class VizSpec:
    head_opacity: float = 0.25
    coil_color: str = "orange"
    log_color_scale: bool = True
    streamline_density: int = 200
    streamline_max_steps: int = 4000
    slice_planes: tuple[str, ...] = ("x", "y", "z")
    show_target_marker: bool = True
    background: str = "white"


@dataclass
class SafetySpec:
    skin_offset_m: float = 0.004        # SimNIBS default (~hair thickness)
    tolerance_m: float = 0.003
    intersection_sample_points: int = 256


@dataclass
class SimConfig:
    mesh_path: Path
    mesh_scale_to_m: float
    mesh_translate_m: tuple[float, float, float]
    target_name: str
    target_position_m: tuple[float, float, float]
    coils: list[CoilSpec]
    grid: GridSpec
    viz: VizSpec
    safety: SafetySpec
    limits: LimitsSpec = field(default_factory=LimitsSpec)
    config_dir: Path = field(default_factory=Path.cwd)


def load_config(path: Path) -> SimConfig:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    mesh_cfg = cfg["mesh"]
    mesh_path = Path(mesh_cfg["path"])
    if not mesh_path.is_absolute():
        mesh_path = (path.parent / mesh_path).resolve()

    coils: list[CoilSpec] = []

    # 1. Explicit per-coil block (back-compat with original layout)
    for c in cfg.get("coils", []) or []:
        coils.append(
            CoilSpec(
                name=c["name"],
                loop_diameter_m=float(c["loop_diameter_m"]),
                wire_diameter_m=float(c["wire_diameter_m"]),
                turns=int(c["turns"]),
                helix_height_m=float(c["helix_height_m"]),
                segments_per_turn=int(c["segments_per_turn"]),
                current_a=float(c["current_a"]),
                position_m=tuple(c["position_m"]),
                normal=tuple(c["normal"]),
                geometry=str(c.get("geometry", "stacked")).lower(),
                peak_dIdt_a_per_us=(
                    float(c["peak_dIdt_a_per_us"]) if "peak_dIdt_a_per_us" in c else None
                ),
            )
        )

    # 2. Pod block — each pod expands to 3 orthogonal coils sharing one
    #    position. Pod's `normal` becomes the local +z (radial inward); two
    #    tangent vectors span the plane perpendicular to it. Resulting coils
    #    are named "<pod>_z" (radial), "<pod>_x" (first tangent),
    #    "<pod>_y" (second tangent).
    for pod in cfg.get("pods", []) or []:
        n = np.asarray(pod["normal"], dtype=float)
        n = n / np.linalg.norm(n)
        e1, e2 = orthonormal_tangents(n)

        # Optional scalp standoff: shift the pod centre OUTWARD (away from
        # head centre) by `scalp_offset_m`. The pod normal points inward
        # toward the target, so outward = −n.
        offset = float(pod.get("scalp_offset_m", 0.0))
        scalp_pos = np.asarray(pod["position_m"], dtype=float)
        pos = tuple(scalp_pos + offset * (-n))

        common = dict(
            loop_diameter_m=float(pod["loop_diameter_m"]),
            wire_diameter_m=float(pod["wire_diameter_m"]),
            turns=int(pod["turns"]),
            helix_height_m=float(pod["helix_height_m"]),
            segments_per_turn=int(pod["segments_per_turn"]),
            current_a=float(pod["current_a"]),
            position_m=pos,
            geometry=str(pod.get("geometry", "stacked")).lower(),
            peak_dIdt_a_per_us=(
                float(pod["peak_dIdt_a_per_us"]) if "peak_dIdt_a_per_us" in pod else None
            ),
        )
        pod_name = pod["name"]
        for axis_label, axis_vec in (("z", n), ("x", e1), ("y", e2)):
            coils.append(
                CoilSpec(
                    name=f"{pod_name}_{axis_label}",
                    normal=tuple(float(v) for v in axis_vec),
                    **common,
                )
            )

    g = cfg["grid"]["bounds_m"]
    grid = GridSpec(
        xmin=float(g["xmin"]),
        xmax=float(g["xmax"]),
        ymin=float(g["ymin"]),
        ymax=float(g["ymax"]),
        zmin=float(g["zmin"]),
        zmax=float(g["zmax"]),
        spacing_m=float(cfg["grid"]["spacing_m"]),
    )

    vcfg = cfg.get("viz", {})
    viz = VizSpec(
        head_opacity=float(vcfg.get("head_opacity", 0.25)),
        coil_color=vcfg.get("coil_color", "orange"),
        log_color_scale=bool(vcfg.get("log_color_scale", True)),
        streamline_density=int(vcfg.get("streamline_density", 200)),
        streamline_max_steps=int(vcfg.get("streamline_max_steps", 4000)),
        slice_planes=tuple(vcfg.get("slice_planes", ["x", "y", "z"])),
        show_target_marker=bool(vcfg.get("show_target_marker", True)),
        background=vcfg.get("background", "white"),
    )

    scfg = cfg.get("safety", {})
    safety = SafetySpec(
        skin_offset_m=float(scfg.get("skin_offset_m", 0.004)),
        tolerance_m=float(scfg.get("tolerance_m", 0.003)),
        intersection_sample_points=int(scfg.get("intersection_sample_points", 256)),
    )

    limits = LimitsSpec.from_yaml(cfg.get("limits"))

    return SimConfig(
        mesh_path=mesh_path,
        mesh_scale_to_m=float(mesh_cfg.get("scale_to_m", 1.0)),
        mesh_translate_m=tuple(mesh_cfg.get("translate_m", (0.0, 0.0, 0.0))),
        target_name=cfg["target"]["name"],
        target_position_m=tuple(cfg["target"]["position_m"]),
        coils=coils,
        grid=grid,
        viz=viz,
        safety=safety,
        limits=limits,
        config_dir=path.parent.resolve(),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def orthonormal_tangents(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to `normal`.

    Used to expand a pod (one position + one radial normal) into three
    mutually orthogonal coil axes.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _rotation_align_z_to(target: np.ndarray) -> np.ndarray:
    """3x3 rotation that maps the local +z axis onto the unit-vector `target`.

    Uses Rodrigues' formula. Handles parallel and antiparallel cases.
    """
    z = np.array([0.0, 0.0, 1.0])
    t = np.asarray(target, dtype=float)
    t = t / np.linalg.norm(t)
    v = np.cross(z, t)
    s = np.linalg.norm(v)
    c = float(np.dot(z, t))

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    K = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _circle_vertices(center: np.ndarray, rot: np.ndarray, R: float, n_seg: int) -> np.ndarray:
    """Closed-loop polyline (n_seg+1 points) for a planar circle.

    The circle lies in the plane normal to (rot @ ẑ), centred at `center`.
    First and last vertex coincide so the line tube renders without a gap.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_seg + 1)
    local = np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)])
    return local @ rot.T + np.asarray(center)


def _helix_vertices(spec: CoilSpec) -> np.ndarray:
    """(M, 3) true-helix vertices in world coordinates for `geometry: helix`."""
    R = spec.loop_diameter_m / 2.0
    n_pts = max(2, spec.turns * spec.segments_per_turn + 1)
    theta = np.linspace(0.0, 2.0 * np.pi * spec.turns, n_pts)
    z = np.linspace(-spec.helix_height_m / 2.0, spec.helix_height_m / 2.0, n_pts)
    local = np.column_stack([R * np.cos(theta), R * np.sin(theta), z])

    rot = _rotation_align_z_to(np.asarray(spec.normal))
    return local @ rot.T + np.asarray(spec.position_m)


def _stacked_circle_sources(spec: CoilSpec) -> tuple[list[magpy.current.Circle], list[np.ndarray]]:
    """Build N planar Circle sources distributed along the coil axis.

    This is magpylib's idiomatic coil pattern — uses the closed-form
    Ortner-2022 current-loop field, no near-wire singularities, and is
    significantly faster than a polyline helix for the same physical
    geometry.
    """
    rot = _rotation_align_z_to(np.asarray(spec.normal))
    R_loop = spec.loop_diameter_m / 2.0

    if spec.turns <= 1 or spec.helix_height_m <= 0:
        z_offsets = np.array([0.0])
    else:
        z_offsets = np.linspace(
            -spec.helix_height_m / 2.0, spec.helix_height_m / 2.0, spec.turns
        )

    orientation = R_scipy.from_matrix(rot)
    base = np.asarray(spec.position_m)

    sources: list[magpy.current.Circle] = []
    polylines: list[np.ndarray] = []
    for z in z_offsets:
        center = rot @ np.array([0.0, 0.0, float(z)]) + base
        sources.append(
            magpy.current.Circle(
                current=spec.current_a,
                diameter=spec.loop_diameter_m,
                position=center,
                orientation=orientation,
            )
        )
        polylines.append(_circle_vertices(center, rot, R_loop, spec.segments_per_turn))

    return sources, polylines


def build_coil_sources(spec: CoilSpec) -> tuple[list, list[np.ndarray]]:
    """Dispatch on `spec.geometry` → magpylib sources + plotting polylines."""
    if spec.geometry == "stacked":
        return _stacked_circle_sources(spec)
    if spec.geometry == "helix":
        verts = _helix_vertices(spec)
        return (
            [magpy.current.Polyline(current=spec.current_a, vertices=verts)],
            [verts],
        )
    raise ValueError(
        f"coil '{spec.name}': unknown geometry '{spec.geometry}' "
        f"(expected 'stacked' or 'helix')"
    )


def build_collection(
    specs: Sequence[CoilSpec],
) -> tuple[magpy.Collection, list[list[np.ndarray]]]:
    """Build the full magpylib Collection plus per-coil plotting polylines.

    Returns
    -------
    collection : magpy.Collection
        All current sources from every coil; ``collection.getB(obs)`` returns
        the vector sum at observers.
    polylines_per_coil : list[list[ndarray]]
        Outer index: coil. Inner index: turn (1 entry for `helix`,
        ``turns`` entries for `stacked`).
    """
    all_sources: list = []
    polylines_per_coil: list[list[np.ndarray]] = []
    for s in specs:
        sources, polylines = build_coil_sources(s)
        all_sources.extend(sources)
        polylines_per_coil.append(polylines)
    return magpy.Collection(*all_sources), polylines_per_coil


# ---------------------------------------------------------------------------
# Mesh loading + standoff check
# ---------------------------------------------------------------------------
def load_head_mesh(cfg: SimConfig) -> Optional[pv.PolyData]:
    """Load and condition the SimNIBS head mesh.

    Returns ``None`` if the mesh file is missing — the rest of the pipeline
    still runs (useful when iterating on coil geometry only).
    """
    if not cfg.mesh_path.exists():
        print(f"[mesh] WARN: '{cfg.mesh_path}' not found — skipping head mesh.")
        return None

    print(f"[mesh] loading {cfg.mesh_path}")
    mesh = pv.read(str(cfg.mesh_path))

    if cfg.mesh_scale_to_m != 1.0:
        mesh.points *= cfg.mesh_scale_to_m
    if any(t != 0 for t in cfg.mesh_translate_m):
        mesh.points += np.asarray(cfg.mesh_translate_m)

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    return mesh


def _standoff_status(min_signed_m: float, offset_m: float, tol_m: float) -> str:
    if min_signed_m < 0:
        return "INSIDE"
    if min_signed_m < offset_m - tol_m:
        return "TOO_CLOSE"
    if min_signed_m > offset_m + tol_m:
        return "TOO_FAR"
    return "OK"


def check_coil_standoff(
    polylines_per_coil: list[list[np.ndarray]],
    coil_specs: Sequence[CoilSpec],
    head: Optional[pv.PolyData],
    safety: SafetySpec,
) -> list[dict]:
    """Per-coil standoff vs the SimNIBS-style skin offset.

    Min signed distance d (negative = inside scalp):
        d < 0                    → INSIDE      (winding pierces scalp)
        0 ≤ d < offset − tol     → TOO_CLOSE   (closer than intended air gap)
        offset − tol ≤ d ≤ offset+tol → OK
        d > offset + tol         → TOO_FAR     (wasting drive — bad coupling)
    """
    if head is None:
        return [
            dict(
                name=s.name,
                status="NO_MESH",
                n_inside=0,
                min_signed_distance_m=float("nan"),
                target_offset_m=safety.skin_offset_m,
            )
            for s in coil_specs
        ]

    results: list[dict] = []
    for poly_list, spec in zip(polylines_per_coil, coil_specs):
        all_verts = np.vstack(poly_list)
        n = min(safety.intersection_sample_points, len(all_verts))
        idx = np.linspace(0, len(all_verts) - 1, n).astype(int)
        sample = pv.PolyData(all_verts[idx])

        enc = sample.select_enclosed_points(head, tolerance=1e-6, check_surface=False)
        inside_mask = np.asarray(enc["SelectedPoints"], dtype=bool)
        n_inside = int(inside_mask.sum())

        unsigned = np.abs(sample.compute_implicit_distance(head)["implicit_distance"])
        signed = np.where(inside_mask, -unsigned, unsigned)
        min_signed = float(signed.min())

        results.append(
            dict(
                name=spec.name,
                status=_standoff_status(min_signed, safety.skin_offset_m, safety.tolerance_m),
                n_inside=n_inside,
                min_signed_distance_m=min_signed,
                target_offset_m=safety.skin_offset_m,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Observation grid + vectorised B-field
# ---------------------------------------------------------------------------
def build_observation_grid(grid: GridSpec) -> tuple[pv.ImageData, np.ndarray]:
    """Return (ImageData container, flat (N, 3) observer point array)."""
    nx, ny, nz = grid.dimensions()
    img = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(grid.spacing_m, grid.spacing_m, grid.spacing_m),
        origin=grid.origin(),
    )
    observers = np.asarray(img.points, dtype=float)
    print(f"[grid] {nx} × {ny} × {nz} = {observers.shape[0]:,} points")
    return img, observers


def compute_B_on_grid(collection: magpy.Collection, observers: np.ndarray) -> np.ndarray:
    """Vectorised B-field for the whole grid in one magpylib call (Tesla)."""
    print(f"[field] computing B at {observers.shape[0]:,} points (vectorised)")
    return np.asarray(collection.getB(observers), dtype=float)


def attach_field_to_grid(img: pv.ImageData, B: np.ndarray) -> pv.ImageData:
    img["B"] = B
    img["B_mag"] = np.linalg.norm(B, axis=1)
    return img


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _coil_polydata_multi(polylines: list[np.ndarray]) -> pv.PolyData:
    """Pack one coil's turns into a single PolyData with N polyline cells."""
    points: list[np.ndarray] = []
    cells: list[int] = []
    offset = 0
    for verts in polylines:
        n = len(verts)
        points.append(verts)
        cells.append(n)
        cells.extend(range(offset, offset + n))
        offset += n
    poly = pv.PolyData()
    poly.points = np.vstack(points)
    poly.lines = np.asarray(cells, dtype=np.int64)
    return poly


def _streamline_seed_disks(
    coil_specs: Sequence[CoilSpec],
    n_per_coil: int,
) -> pv.PolyData:
    """Disk of seed points in each coil's plane — streamlines thread the windings."""
    seeds = []
    for s in coil_specs:
        R = 0.4 * s.loop_diameter_m / 2.0
        rot = _rotation_align_z_to(np.asarray(s.normal))
        rng = np.random.default_rng(seed=hash(s.name) & 0xFFFFFFFF)
        u = rng.uniform(0, 1, n_per_coil)
        v = rng.uniform(0, 1, n_per_coil)
        r = R * np.sqrt(u)
        th = 2 * np.pi * v
        local = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n_per_coil)])
        world = local @ rot.T + np.asarray(s.position_m)
        seeds.append(world)
    return pv.PolyData(np.vstack(seeds))


def render_scene(
    cfg: SimConfig,
    head: Optional[pv.PolyData],
    polylines_per_coil: list[list[np.ndarray]],
    field_grid: pv.ImageData,
    *,
    save_html: Optional[Path] = None,
    show: bool = True,
) -> None:
    pv.global_theme.background = cfg.viz.background

    plotter = pv.Plotter(window_size=(1280, 900))
    plotter.add_axes()

    if head is not None:
        plotter.add_mesh(
            head,
            color="lightpink",
            opacity=cfg.viz.head_opacity,
            smooth_shading=True,
            name="scalp",
        )

    for poly_list, spec in zip(polylines_per_coil, cfg.coils):
        plotter.add_mesh(
            _coil_polydata_multi(poly_list),
            color=cfg.viz.coil_color,
            line_width=4,
            render_lines_as_tubes=True,
            name=f"coil_{spec.name}",
        )
        plotter.add_point_labels(
            np.asarray(spec.position_m).reshape(1, 3),
            [f"Coil {spec.name}"],
            point_size=10,
            font_size=14,
            text_color="black",
            shape_opacity=0.25,
        )

    n_per = max(1, cfg.viz.streamline_density // max(1, len(cfg.coils)))
    seeds = _streamline_seed_disks(cfg.coils, n_per)
    try:
        streamlines = field_grid.streamlines_from_source(
            seeds,
            vectors="B",
            integration_direction="both",
            max_steps=cfg.viz.streamline_max_steps,
            initial_step_length=cfg.grid.spacing_m * 0.5,
            terminal_speed=1e-9,
        )
        if streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines.tube(radius=cfg.grid.spacing_m * 0.15),
                scalars="B_mag",
                cmap="plasma",
                log_scale=cfg.viz.log_color_scale,
                scalar_bar_args=dict(title="|B| (T)", n_labels=5),
                name="streamlines",
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] streamline generation failed: {exc}")

    target = np.asarray(cfg.target_position_m)
    slices = field_grid.slice_orthogonal(x=target[0], y=target[1], z=target[2])
    plotter.add_mesh(
        slices,
        scalars="B_mag",
        cmap="inferno",
        log_scale=cfg.viz.log_color_scale,
        opacity=0.85,
        show_scalar_bar=False,
        name="bmag_slices",
    )

    if cfg.viz.show_target_marker:
        marker = pv.Sphere(radius=0.004, center=target)
        plotter.add_mesh(marker, color="red", name="target")
        plotter.add_point_labels(
            target.reshape(1, 3),
            [cfg.target_name],
            font_size=14,
            text_color="red",
            shape_opacity=0.0,
        )

    geom_mix = "/".join(sorted({c.geometry for c in cfg.coils}))
    plotter.add_text(
        f"NeuroRegen — {len(cfg.coils)}-coil B-field [{geom_mix}]   "
        f"target: {cfg.target_name}",
        font_size=11,
    )

    if save_html is not None:
        save_html.parent.mkdir(parents=True, exist_ok=True)
        plotter.export_html(str(save_html))
        print(f"[viz] saved interactive scene → {save_html}")

    if show:
        plotter.show()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_field_at_target(collection: magpy.Collection, cfg: SimConfig) -> None:
    target = np.asarray(cfg.target_position_m, dtype=float)
    B = np.asarray(collection.getB(target)).flatten()
    mag = float(np.linalg.norm(B))
    print(
        f"[target] B at {cfg.target_name} = "
        f"({B[0] * 1e3:+.3f}, {B[1] * 1e3:+.3f}, {B[2] * 1e3:+.3f}) mT  "
        f"|B|={mag * 1e3:.3f} mT"
    )


def report_dIdt(coils: Sequence[CoilSpec]) -> None:
    """Echo the peak dI/dt declared in the config (for downstream E reports).

    Per the validated-coil-database convention (Brain Stim 2022), simulations
    are normalised to peak dI/dt rather than RMS. We don't *use* this value
    here (B is linear in I, full stop), but printing it makes the link to the
    SimNIBS / E-field stage explicit.
    """
    has_any = any(c.peak_dIdt_a_per_us is not None for c in coils)
    if not has_any:
        return
    print("[dI/dt] declared peak (for downstream E-field scaling):")
    for c in coils:
        if c.peak_dIdt_a_per_us is not None:
            print(f"  {c.name}: {c.peak_dIdt_a_per_us:.2f} A/μs   (I_peak = {c.current_a:.0f} A)")


def report_standoff(results: list[dict]) -> int:
    """Print the standoff table; return number of non-OK coils."""
    n_bad = 0
    print("\n[standoff check]")
    print(f"  {'coil':<6} {'status':>10} {'min |d|':>11} {'target':>9} {'verts in':>9}")
    for r in results:
        status = r["status"]
        if status not in ("OK", "NO_MESH"):
            n_bad += 1
        d = r["min_signed_distance_m"]
        if np.isnan(d):
            d_str = "    n/a"
        else:
            sign = "-" if d < 0 else " "
            d_str = f"{sign}{abs(d) * 1000:7.2f} mm"
        target_str = f"{r['target_offset_m'] * 1000:5.1f} mm"
        print(f"  {r['name']:<6} {status:>10} {d_str:>11} {target_str:>9} {r['n_inside']:>9d}")
    if n_bad:
        print(f"  WARN: {n_bad} coil(s) outside the standoff envelope.")
    return n_bad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="simulate_bfield",
        description="External B-field calculator + 3-D viewer for the NeuroRegen 3-coil array.",
    )
    here = Path(__file__).resolve().parent
    p.add_argument(
        "-c", "--config", type=Path, default=here / "config.yaml",
        help="YAML config file (default: ./config.yaml)",
    )
    p.add_argument(
        "--mesh", type=Path, default=None,
        help="Override the head mesh path from the config.",
    )
    p.add_argument(
        "--spacing", type=float, default=None,
        help="Override observation-grid spacing in metres (e.g. 0.003).",
    )
    p.add_argument(
        "--geometry", choices=("stacked", "helix"), default=None,
        help="Override coil geometry mode for ALL coils.",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Skip interactive PyVista window (still computes / can save HTML).",
    )
    p.add_argument(
        "--save-html", type=Path, default=None,
        help="Export the 3-D scene to a self-contained HTML file.",
    )
    p.add_argument(
        "--save-vti", type=Path, default=None,
        help="Save the field grid (B + |B|) as a .vti for ParaView.",
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="Run geometry + standoff check, skip field calculation and rendering.",
    )
    p.add_argument(
        "--check-constraints", action="store_true",
        help="Evaluate E-field, energy, wall-power, thermal vs `limits:` block "
             "and exit. Skips B-field grid and rendering.",
    )
    p.add_argument(
        "--optimize", action="store_true",
        help="Run constrained optimizer over coil currents + pod positions to "
             "maximise |B| at the target. Writes config_optimized.yaml.",
    )
    p.add_argument(
        "--opt-out", type=Path, default=None,
        help="Output YAML path for --optimize (default: <config>_optimized.yaml).",
    )
    p.add_argument(
        "--opt-maxiter", type=int, default=60,
        help="SLSQP outer-iteration cap (default 60).",
    )
    p.add_argument(
        "--opt-imax", type=float, default=5000.0,
        help="Per-coil current upper bound during optimization (A, default 5000).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.mesh is not None:
        cfg.mesh_path = args.mesh
    if args.spacing is not None:
        cfg.grid.spacing_m = args.spacing
    if args.geometry is not None:
        for c in cfg.coils:
            c.geometry = args.geometry

    print(f"[config] {args.config}")
    print(f"[target] {cfg.target_name} at {cfg.target_position_m} m")
    print(
        f"[coils ] {len(cfg.coils)}: "
        + ", ".join(f"{c.name}({c.geometry})" for c in cfg.coils)
    )
    report_dIdt(cfg.coils)

    collection, polylines_per_coil = build_collection(cfg.coils)

    head = load_head_mesh(cfg)

    standoff = check_coil_standoff(polylines_per_coil, cfg.coils, head, cfg.safety)
    n_bad = report_standoff(standoff)
    if args.check_only:
        return 1 if n_bad else 0

    if args.check_constraints:
        cr = evaluate_constraints(cfg.coils, head, cfg.limits)
        n_viol = report_constraints(cr)
        return (1 if n_bad else 0) | (2 if n_viol else 0)

    if args.optimize:
        from optimize import optimize_array, write_optimized_yaml, PodLayout

        if head is None:
            print("[opt] ERROR: optimizer requires a head mesh; aborting.")
            return 4

        target_xyz = np.asarray(cfg.target_position_m, dtype=float).reshape(1, 3)
        print(f"\n[opt] target {cfg.target_name} at {tuple(target_xyz.flatten())}")
        print(f"[opt] caps: E_scalp≤{cfg.limits.e_max_scalp_v_per_m} V/m, "
              f"E_pulse≤{cfg.limits.max_pulse_energy_j} J, "
              f"P_wall≤{cfg.limits.max_avg_power_w} W @ {cfg.limits.prf_hz} Hz")

        # Initial constraint snapshot
        cr0 = evaluate_constraints(cfg.coils, head, cfg.limits)
        print("\n[opt] initial state:")
        report_constraints(cr0)

        best_specs, info = optimize_array(
            cfg.coils,
            head,
            target_xyz,
            cfg.limits,
            i_max=args.opt_imax,
            maxiter=args.opt_maxiter,
            verbose=True,
        )

        print(f"\n[opt] {info['message']}")
        print(f"[opt] |B(target)|: {info['b_target_mT_initial']:.3f} mT  →  "
              f"{info['b_target_mT_final']:.3f} mT  "
              f"({(info['b_target_mT_final'] / max(info['b_target_mT_initial'], 1e-12) - 1) * 100:+.1f} %)")

        # Final constraint snapshot
        cr1 = evaluate_constraints(best_specs, head, cfg.limits)
        print("\n[opt] final state:")
        report_constraints(cr1)

        out_path = args.opt_out or args.config.with_name(
            args.config.stem + "_optimized.yaml"
        )
        layout = PodLayout(cfg.coils, head)
        write_optimized_yaml(args.config, best_specs, layout, out_path)
        print(f"\n[opt] wrote optimized config → {out_path}")
        return 0

    img, observers = build_observation_grid(cfg.grid)
    B = compute_B_on_grid(collection, observers)
    field_grid = attach_field_to_grid(img, B)

    report_field_at_target(collection, cfg)

    if args.save_vti is not None:
        args.save_vti.parent.mkdir(parents=True, exist_ok=True)
        field_grid.save(str(args.save_vti))
        print(f"[output] wrote {args.save_vti}")

    render_scene(
        cfg,
        head,
        polylines_per_coil,
        field_grid,
        save_html=args.save_html,
        show=not args.no_show,
    )

    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
