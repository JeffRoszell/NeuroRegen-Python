"""
Load simulation config from a YAML file.

Supports two config formats:
    - **Single-coil** (``load_config``) — 3-axis pulsed coil setup.
    - **Multicoil** (``load_multicoil_config``) — N-coil deep-brain
      focusing array with 3-D positions, orientations, and safety limits.
"""

import os
from pathlib import Path

from .coil import Axis

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """Return the repo root (parent of ``src/``)."""
    pkg_dir = Path(__file__).resolve().parent  # .../src/neuroregen
    return pkg_dir.parent.parent               # .../Python Code


# ---------------------------------------------------------------------------
# Single-coil config loader (unchanged API)
# ---------------------------------------------------------------------------
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
        path = _project_root() / "config" / "default.yaml"

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


# ---------------------------------------------------------------------------
# Multicoil config loader
# ---------------------------------------------------------------------------
def load_multicoil_config(path: str | os.PathLike | None = None) -> dict:
    """
    Load a multicoil YAML config.

    If *path* is ``None``, falls back to ``config/multicoil.yaml``.

    Returns
    -------
    dict with keys:
        target      – dict(name, position_m)
        coils       – list of dict(name, wire_mm, loop_mm, turns,
                       pulse_power_w, position_m, normal)
        cortical_max_vm, scalp_to_cortex_m, skull_radius_m
        sim_time, dt, pulse_freq, pulse_width
        t_amb_c, h_conv, temp_limit_f, hyst_f
        z_max_m, z_points, b_threshold_t
    """
    if yaml is None:
        raise ImportError("PyYAML is required for config. Install with: pip install pyyaml")

    if path is None:
        path = _project_root() / "config" / "multicoil.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Multicoil config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # --- target ---
    tgt = raw.get("target", {})
    target = dict(
        name=tgt.get("name", "unknown"),
        position_m=tuple(float(v) for v in tgt.get("position_m", [0, 0, 0])),
    )

    # --- safety ---
    safety = raw.get("safety", {})
    cortical_max_vm = float(safety.get("cortical_max_vm", 150.0))
    scalp_to_cortex_m = float(safety.get("scalp_to_cortex_m", 0.015))

    # --- geometry ---
    geom = raw.get("geometry", {})
    skull_radius_m = float(geom.get("skull_radius_m", 0.09))

    # --- coils ---
    coils_spec = raw.get("coils", [])
    coils = []
    for c in coils_spec:
        coils.append(dict(
            name=c["name"],
            wire_mm=float(c["wire_mm"]),
            loop_mm=float(c["loop_mm"]),
            turns=int(c["turns"]),
            pulse_power_w=float(c["pulse_power_w"]),
            position_m=tuple(float(v) for v in c["position_m"]),
            normal=tuple(float(v) for v in c["normal"]),
        ))

    # --- simulation ---
    sim = raw.get("simulation", {})
    therm = raw.get("thermal", {})
    depth = raw.get("depth", {})

    return {
        "target": target,
        "coils": coils,
        "cortical_max_vm": cortical_max_vm,
        "scalp_to_cortex_m": scalp_to_cortex_m,
        "skull_radius_m": skull_radius_m,
        "sim_time": float(sim.get("sim_time", 1800.0)),
        "dt": float(sim.get("dt", 0.1)),
        "pulse_freq": float(sim.get("pulse_freq", 5.0)),
        "pulse_width": float(sim.get("pulse_width", 0.02)),
        "t_amb_c": float(therm.get("t_amb_c", 22.0)),
        "h_conv": float(therm.get("h_conv", 10.0)),
        "temp_limit_f": float(therm.get("temp_limit_f", 75.0)),
        "hyst_f": float(therm.get("hyst_f", 0.7)),
        "z_max_m": float(depth.get("z_max_m", 0.12)),
        "z_points": int(depth.get("z_points", 500)),
        "b_threshold_t": float(depth.get("b_threshold_t", 1e-4)),
    }


def load_pulsed_config(path: str | os.PathLike | None = None) -> dict:
    """
    Load a pulsed-TMS capacitor-discharge config (e.g. ``config/pulsed_tms.yaml``).

    Extends the multicoil format with a ``capacitor`` section and pulsed
    simulation parameters.

    If *path* is ``None``, falls back to ``config/pulsed_tms.yaml``.

    Returns
    -------
    dict with all keys from :func:`load_multicoil_config` plus:
        capacitance_f        – capacitance per coil (F)
        charge_voltage_v     – charge voltage (V)
        n_pulses             – number of pulse events to simulate
        dt_between_pulses    – cooling time step between pulses (s)
        r_tissue_m           – effective tissue loop radius for E-field (m)
    """
    if yaml is None:
        raise ImportError("PyYAML is required for config. Install with: pip install pyyaml")

    if path is None:
        path = _project_root() / "config" / "pulsed_tms.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pulsed TMS config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Reuse multicoil base sections
    tgt = raw.get("target", {})
    target = dict(
        name=tgt.get("name", "unknown"),
        position_m=tuple(float(v) for v in tgt.get("position_m", [0, 0, 0])),
    )

    safety = raw.get("safety", {})
    geom = raw.get("geometry", {})
    coils_spec = raw.get("coils", [])
    coils = []
    for c in coils_spec:
        coils.append(dict(
            name=c["name"],
            wire_mm=float(c["wire_mm"]),
            loop_mm=float(c["loop_mm"]),
            turns=int(c["turns"]),
            pulse_power_w=float(c.get("pulse_power_w", 0.0)),
            position_m=tuple(float(v) for v in c["position_m"]),
            normal=tuple(float(v) for v in c["normal"]),
        ))

    cap = raw.get("capacitor", {})
    sim = raw.get("simulation", {})
    therm = raw.get("thermal", {})

    return {
        "target": target,
        "coils": coils,
        "cortical_max_vm": float(safety.get("cortical_max_vm", 150.0)),
        "scalp_to_cortex_m": float(safety.get("scalp_to_cortex_m", 0.015)),
        "skull_radius_m": float(geom.get("skull_radius_m", 0.09)),
        # capacitor
        "capacitance_f": float(cap.get("capacitance_f", 200e-6)),
        "charge_voltage_v": float(cap.get("charge_voltage_v", 2500.0)),
        # pulsed simulation
        "n_pulses": int(sim.get("n_pulses", 60)),
        "pulse_freq": float(sim.get("pulse_freq", 1.0)),
        "dt_between_pulses": float(sim.get("dt_between_pulses", 0.001)),
        # thermal
        "t_amb_c": float(therm.get("t_amb_c", 22.0)),
        "h_conv": float(therm.get("h_conv", 10.0)),
        "temp_limit_f": float(therm.get("temp_limit_f", 75.0)),
        "hyst_f": float(therm.get("hyst_f", 0.7)),
        "r_tissue_m": float(therm.get("r_tissue_m", 0.005)),
    }


def build_multicoil_objects(config: dict):
    """
    Convenience: turn the flat dict from :func:`load_multicoil_config` into
    ready-to-use ``(MulticoilArray, flat_config)`` for the simulation.

    The returned *flat_config* is the same dict, suitable for passing to
    :func:`run_multicoil_simulation`.
    """
    from .multicoil import Coil, Target, MulticoilArray

    target = Target(
        name=config["target"]["name"],
        position_m=config["target"]["position_m"],
    )
    coils = [
        Coil(
            name=c["name"],
            wire_mm=c["wire_mm"],
            loop_mm=c["loop_mm"],
            turns=c["turns"],
            pulse_power_w=c["pulse_power_w"],
            position_m=c["position_m"],
            normal=c["normal"],
        )
        for c in config["coils"]
    ]
    array = MulticoilArray(
        coils=coils,
        target=target,
        cortical_max_vm=config["cortical_max_vm"],
        surface_distance_m=config["scalp_to_cortex_m"],
    )
    return array, config
