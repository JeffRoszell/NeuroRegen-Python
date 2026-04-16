#!/usr/bin/env python3
"""
Generate synthetic ANSYS-format field-map CSV files for all three coils.

This script uses the existing analytical Biot–Savart model (``B_field_at_point``
in ``multicoil.py``) to compute B-field values on a uniform Cartesian grid
centred on the STN target and writes them out in the CSV format expected by
:class:`~neuroregen.ansys_field_map.AnsysFieldMap`.

Purpose
-------
1. **Validate the integration** — running the simulation after generating
   these files and pointing ``multicoil.yaml`` at them should produce
   numerically identical results to the pure-analytical run (within
   interpolation error).
2. **Document the expected format** — researchers exporting real ANSYS data
   can compare their files against the generated ones to ensure column names
   and units are compatible.
3. **CI smoke test** — the generated files are used by ``test_ansys_field_map``
   to verify the loader and interpolator without requiring a real ANSYS licence.

Output
------
Creates ``ansys_exports/`` in the project root containing:

    coil_A.csv  coil_B.csv  coil_C.csv

Each file has a metadata comment block, then columns::

    x_m, y_m, z_m, Bx_T, By_T, Bz_T, Ex_Vm, Ey_Vm, Ez_Vm

The E-field columns are a simple Faraday estimate (``E ≈ R/2 × dB/dt``) at
each grid point — good enough for format validation, but not a true
volumetric E-field solve.

Usage
-----
    python scripts/generate_ansys_sample.py
    python scripts/generate_ansys_sample.py --grid-cm 8 --resolution 20
    python scripts/generate_ansys_sample.py --no-efield
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.neuroregen.config_loader import load_multicoil_config, build_multicoil_objects
from src.neuroregen.multicoil import B_field_at_point
from src.neuroregen.coil import coil_geom, resistance


# ---------------------------------------------------------------------------
# E-field estimate (Faraday, simple circular-path approximation)
# ---------------------------------------------------------------------------


def _E_field_estimate(coil, I: float, point_m: tuple, pulse_width: float = 0.02) -> np.ndarray:
    """
    Rough induced E-field vector (V/m) at a point.

    Uses the on-axis B magnitude and a Faraday approximation to give a
    non-zero E-field in the CSV for format-validation purposes.  This is
    NOT a rigorous volumetric E-field solve.
    """
    B_vec = B_field_at_point(coil, I, point_m)
    B_mag = float(np.linalg.norm(B_vec))
    # dB/dt ≈ π B_peak / pulse_width (half-sine pulse)
    dBdt = B_mag * np.pi / pulse_width
    R_loop = coil.loop_mm / 2000.0
    # E ≈ (R/2) × dB/dt, directed along the local B direction
    E_mag = (R_loop / 2.0) * dBdt
    B_hat = B_vec / (B_mag + 1e-30)
    return E_mag * B_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ANSYS-format B/E field CSV files."
    )
    parser.add_argument(
        "--grid-cm",
        type=float,
        default=6.0,
        help="Half-width of the cubic grid centred on the STN target (cm). "
        "Default: 6 cm → 12 cm cube.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=15,
        help="Number of grid points along each axis.  Default: 15 "
        "(15×15×15 = 3375 points per file).  Use 25+ for smoother "
        "interpolation.",
    )
    parser.add_argument(
        "--ref-current",
        type=float,
        default=1.0,
        help="Reference current (A) to use in the ANSYS comment header. Default: 1.0 A.",
    )
    parser.add_argument(
        "--no-efield",
        action="store_true",
        help="Omit Ex/Ey/Ez columns (test B-only format).",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to multicoil YAML config (default: config/multicoil.yaml).",
    )
    args = parser.parse_args()

    # ---- load coils from config -------------------------------------------
    config_path = args.config or os.path.join(ROOT, "config", "multicoil.yaml")
    config = load_multicoil_config(config_path)
    array, config = build_multicoil_objects(config, config_path=config_path)

    target_pos = array.target.position_m
    half = args.grid_cm / 100.0  # convert cm → m

    # ---- build grid centred on the STN target -----------------------------
    n = args.resolution
    xs = np.linspace(target_pos[0] - half, target_pos[0] + half, n)
    ys = np.linspace(target_pos[1] - half, target_pos[1] + half, n)
    zs = np.linspace(target_pos[2] - half, target_pos[2] + half, n)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
    n_pts = pts.shape[0]

    ref_I = args.ref_current

    # ---- output directory -------------------------------------------------
    out_dir = Path(ROOT) / "ansys_exports"
    out_dir.mkdir(exist_ok=True)

    print(f"Grid: {n}×{n}×{n} = {n_pts:,} points")
    print(f"Bounds: ±{args.grid_cm:.1f} cm around target {array.target.name}")
    print(f"Reference current: {ref_I} A")
    print(f"Output → {out_dir}/\n")

    # ---- generate one CSV per coil ----------------------------------------
    for coil in array.coils:
        ax = coil.to_axis()
        _, L, A, _, _ = coil_geom(ax)
        R_ohm = resistance(L, A, 22.0)  # ambient temperature
        I_actual = ref_I  # field computed at ref_I

        print(f"  Computing coil {coil.name} ({n_pts:,} points)...")
        B_arr = np.zeros((n_pts, 3))
        E_arr = np.zeros((n_pts, 3)) if not args.no_efield else None

        for j, pt in enumerate(pts):
            B_arr[j] = B_field_at_point(coil, I_actual, tuple(pt))
            if E_arr is not None:
                E_arr[j] = _E_field_estimate(coil, I_actual, tuple(pt))

        # ---- write CSV ----------------------------------------------------
        csv_path = out_dir / f"coil_{coil.name}.csv"
        with open(csv_path, "w") as f:
            # Metadata header
            f.write("# NeuroRegen ANSYS-format synthetic field map\n")
            f.write(f"# Coil: {coil.name}\n")
            f.write("# Generated by: scripts/generate_ansys_sample.py\n")
            f.write("# source: analytical Biot-Savart (N-turn circular loop)\n")
            f.write(f"# reference_current_A: {ref_I:.6f}\n")
            f.write(f"# coil_position_m: {coil.position_m}\n")
            f.write(f"# coil_normal: {coil.normal}\n")
            f.write(f"# coil_loop_mm: {coil.loop_mm}, turns: {coil.turns}\n")
            f.write(f"# grid: {n}x{n}x{n}, centre {target_pos}\n")
            f.write("#\n")

            # Column header
            if E_arr is not None:
                f.write("x_m,y_m,z_m,Bx_T,By_T,Bz_T,Ex_Vm,Ey_Vm,Ez_Vm\n")
            else:
                f.write("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n")

            # Data rows
            for j, pt in enumerate(pts):
                row = f"{pt[0]:.8e},{pt[1]:.8e},{pt[2]:.8e},"
                row += f"{B_arr[j, 0]:.8e},{B_arr[j, 1]:.8e},{B_arr[j, 2]:.8e}"
                if E_arr is not None:
                    row += f",{E_arr[j, 0]:.8e},{E_arr[j, 1]:.8e},{E_arr[j, 2]:.8e}"
                f.write(row + "\n")

        B_peak = np.linalg.norm(B_arr, axis=1).max()
        print(f"    Saved {csv_path.name}  (peak |B| = {B_peak * 1e6:.1f} µT)")

    print(
        "\nDone.  To activate the maps, uncomment and fill in "
        "'ansys_field_maps' in config/multicoil.yaml:\n"
        "\n"
        "  ansys_field_maps:\n"
        "    coil_maps:\n"
        "      A: ansys_exports/coil_A.csv\n"
        "      B: ansys_exports/coil_B.csv\n"
        "      C: ansys_exports/coil_C.csv\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
