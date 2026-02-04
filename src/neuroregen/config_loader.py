"""
Load simulation config from a YAML file.
Returns a flat dict with run parameters and a list of Axis.
"""

import os
from pathlib import Path

from .coil import Axis

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str | os.PathLike | None = None) -> dict:
    """
    Load config from a YAML file. If path is None, use config/default.yaml
    next to the project root (parent of src/).
    Returns a dict with keys: sim_time, dt, pulse_freq, pulse_width,
    t_amb_c, h_conv, temp_limit_f, hyst_f, z_max_m, z_points, b_threshold_t, axes.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for config. Install with: pip install pyyaml")

    if path is None:
        # Default: config/default.yaml relative to repo root (parent of src/)
        pkg_dir = Path(__file__).resolve().parent  # .../src/neuroregen
        root = pkg_dir.parent.parent  # .../Python Code
        path = root / "config" / "default.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    sim = raw.get("simulation", {})
    therm = raw.get("thermal", {})
    depth = raw.get("depth", {})
    axes_spec = raw.get("axes", [])

    axes = [
        Axis(
            name=a["name"],
            wire_mm=float(a["wire_mm"]),
            loop_mm=float(a["loop_mm"]),
            turns=int(a["turns"]),
            pulse_power_w=float(a["pulse_power_w"]),
        )
        for a in axes_spec
    ]

    return {
        "sim_time": float(sim.get("sim_time", 1800.0)),
        "dt": float(sim.get("dt", 0.1)),
        "pulse_freq": float(sim.get("pulse_freq", 5.0)),
        "pulse_width": float(sim.get("pulse_width", 0.02)),
        "t_amb_c": float(therm.get("t_amb_c", 22.0)),
        "h_conv": float(therm.get("h_conv", 10.0)),
        "temp_limit_f": float(therm.get("temp_limit_f", 75.0)),
        "hyst_f": float(therm.get("hyst_f", 0.7)),
        "z_max_m": float(depth.get("z_max_m", 0.03)),
        "z_points": int(depth.get("z_points", 300)),
        "b_threshold_t": float(depth.get("b_threshold_t", 1e-4)),
        "axes": axes,
    }
