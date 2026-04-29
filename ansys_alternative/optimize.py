"""
NeuroRegen — Constrained optimizer for the multipod TMS array.

Maximises |B| at the deep target (STN_left by default) subject to the
`limits:` block in the YAML, varying:

    9 currents      (one per coil)
    3 pod positions (each as (θ, φ) on the scalp; ray-cast onto the
                     skin mesh, then offset outward by scalp_offset_m)

The pod's three orthogonal coil axes are recomputed from the new normal
at every iteration via `orthonormal_tangents`, so per-axis currents
remain meaningful as the pod moves.

Hardware coupling: each coil keeps its **initial** angular frequency
ω_i = (dI/dt)_init / I_init constant (a fixed LC tank); when the
optimizer changes I_i, the implied (dI/dt)_i scales linearly. So
E-field is linear in I, energy and thermal are quadratic in I.

Algorithm: SLSQP with finite-difference Jacobians via scipy. ~15
unknowns → typical convergence in 30–60 outer iterations, ~5 min.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Optional

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

import magpylib as magpy

import constraints as cn


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
def _direction_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def _angles_from_direction(d: np.ndarray) -> tuple[float, float]:
    d = d / np.linalg.norm(d)
    theta = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
    phi = float(np.arctan2(d[1], d[0]))
    return theta, phi


def _ray_cast_scalp(
    direction: np.ndarray,
    head: pv.PolyData,
    origin: np.ndarray = np.zeros(3),
    max_radius_m: float = 0.30,
) -> Optional[np.ndarray]:
    """First intersection of a ray (origin → direction) with the scalp mesh."""
    end = origin + direction * max_radius_m
    pts, _ = head.ray_trace(origin, end, first_point=True)
    if pts is None or (hasattr(pts, "size") and pts.size == 0):
        return None
    return np.asarray(pts).reshape(3)


def _pod_pos_normal(
    theta: float,
    phi: float,
    head: pv.PolyData,
    scalp_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pod centre and inward-pointing normal for given spherical angles."""
    d = _direction_from_angles(theta, phi)
    scalp_pt = _ray_cast_scalp(d, head)
    if scalp_pt is None:
        scalp_pt = d * 0.10                  # fallback: nominal 10 cm head radius
    pod_centre = scalp_pt + scalp_offset_m * d
    return pod_centre, -d                    # normal points inward


def _orthonormal_tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


# ---------------------------------------------------------------------------
# Pod-aware parameter vector (de)packing
# ---------------------------------------------------------------------------
class PodLayout:
    """
    Holds everything *fixed* about the array (geometry per coil, hardware ω,
    pod assignments) and lets us cheaply rebuild a coil list given a parameter
    vector x = [I_1, …, I_9, θ_A, φ_A, θ_B, φ_B, θ_C, φ_C].
    """

    def __init__(self, base_specs, head: pv.PolyData):
        # Group coils by pod prefix (everything before the last "_axis" suffix).
        # Pod block in the YAML names them <pod>_z / <pod>_x / <pod>_y.
        groups: dict[str, list] = {}
        for s in base_specs:
            if "_" not in s.name:
                raise ValueError(
                    f"coil '{s.name}' is not a pod-style name (expected <pod>_z/x/y)"
                )
            pod_name, axis = s.name.rsplit("_", 1)
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    f"coil '{s.name}' suffix must be _x, _y, or _z; got '{axis}'"
                )
            groups.setdefault(pod_name, []).append((axis, s))

        self.pod_names: list[str] = sorted(groups.keys())
        self.head = head
        self.coil_order: list[tuple[str, str]] = []   # (pod_name, axis)
        self.specs_by_key: dict[tuple[str, str], object] = {}
        self.scalp_offset_by_pod: dict[str, float] = {}
        self.omega_by_key: dict[tuple[str, str], float] = {}

        for pod_name in self.pod_names:
            axes = {ax: s for ax, s in groups[pod_name]}
            for ax in ("x", "y", "z"):              # canonical order
                if ax not in axes:
                    raise ValueError(f"pod {pod_name} missing _{ax} coil")
                self.coil_order.append((pod_name, ax))
                self.specs_by_key[(pod_name, ax)] = axes[ax]
                self.omega_by_key[(pod_name, ax)] = cn.fundamental_omega(axes[ax])

            # The pod's "scalp_offset" isn't stored on CoilSpec — recover it
            # from how far the pod centre sits from a head-radius estimate.
            # In practice all three coils share one position, so we can pull
            # the value from the loader's already-applied offset by reading
            # the spec back-of-envelope. Simplest: assume same as YAML value,
            # default 0.025 m if nothing else.
            self.scalp_offset_by_pod[pod_name] = 0.025

        # Build the *initial* (θ, φ) for each pod from its current normal.
        self.theta0: dict[str, float] = {}
        self.phi0: dict[str, float] = {}
        for pod_name in self.pod_names:
            spec_z = self.specs_by_key[(pod_name, "z")]
            n = -np.asarray(spec_z.normal, dtype=float)   # spec.normal is inward; outward = -n
            theta, phi = _angles_from_direction(n)
            self.theta0[pod_name] = theta
            self.phi0[pod_name] = phi

        self.n_pods = len(self.pod_names)
        self.n_coils = len(self.coil_order)
        self.n_vars = self.n_coils + 2 * self.n_pods

    # ------------------------- (de)packers --------------------------------
    # Currents are rescaled to kA (×1e-3) so the optimizer's finite-difference
    # step works on similar magnitudes for currents and angles.
    I_SCALE = 1.0e-3

    def x0(self) -> np.ndarray:
        x = np.zeros(self.n_vars)
        for i, (pod, ax) in enumerate(self.coil_order):
            x[i] = self.specs_by_key[(pod, ax)].current_a * self.I_SCALE
        for j, pod in enumerate(self.pod_names):
            x[self.n_coils + 2 * j + 0] = self.theta0[pod]
            x[self.n_coils + 2 * j + 1] = self.phi0[pod]
        return x

    def bounds(
        self,
        i_min: float = 0.0,
        i_max: float = 5000.0,
        theta_min: float = 0.0,
        theta_max: float = np.pi / 2,        # upper hemisphere only
    ) -> list[tuple[float, float]]:
        b = [(i_min * self.I_SCALE, i_max * self.I_SCALE)] * self.n_coils
        for _ in self.pod_names:
            b.append((theta_min, theta_max))
            b.append((-np.pi, np.pi))
        return b

    def specs_from_x(self, x: np.ndarray) -> list:
        """Build a fresh list[CoilSpec] from a parameter vector."""
        currents = x[: self.n_coils] / self.I_SCALE     # back to A
        pod_angles = x[self.n_coils :].reshape(self.n_pods, 2)

        # Pod-specific position + normal
        pod_centre: dict[str, np.ndarray] = {}
        pod_normal: dict[str, np.ndarray] = {}
        pod_e1: dict[str, np.ndarray] = {}
        pod_e2: dict[str, np.ndarray] = {}
        for j, pod_name in enumerate(self.pod_names):
            theta, phi = pod_angles[j]
            centre, normal = _pod_pos_normal(
                theta, phi, self.head, self.scalp_offset_by_pod[pod_name]
            )
            pod_centre[pod_name] = centre
            pod_normal[pod_name] = normal
            e1, e2 = _orthonormal_tangents(normal)
            pod_e1[pod_name] = e1
            pod_e2[pod_name] = e2

        new_specs: list = []
        for i, (pod, ax) in enumerate(self.coil_order):
            base = self.specs_by_key[(pod, ax)]
            I_new = float(currents[i])
            omega = self.omega_by_key[(pod, ax)]
            dIdt_new = (I_new * omega) / 1e6 if omega > 0 else base.peak_dIdt_a_per_us

            normal = {
                "z": pod_normal[pod],
                "x": pod_e1[pod],
                "y": pod_e2[pod],
            }[ax]

            new_specs.append(
                replace(
                    base,
                    current_a=I_new,
                    position_m=tuple(pod_centre[pod].tolist()),
                    normal=tuple(float(v) for v in normal),
                    peak_dIdt_a_per_us=dIdt_new,
                )
            )
        return new_specs


# ---------------------------------------------------------------------------
# Objective + constraints
# ---------------------------------------------------------------------------
def _b_at_target(specs, target_xyz: np.ndarray) -> float:
    """|B| in Tesla at the target. Uses magpylib (closed-form Circle)."""
    from simulate_bfield import build_collection
    coll, _ = build_collection(specs)
    B = np.asarray(coll.getB(target_xyz)).flatten()
    return float(np.linalg.norm(B))


def _scalp_subsample(head: pv.PolyData, n_max: int = 2000) -> pv.PolyData:
    """Decimate the scalp mesh for fast E-field evaluation."""
    if head.n_points <= n_max:
        return head
    frac = 1.0 - (n_max / head.n_points)
    # decimate_pro keeps topology + supports preserve_topology; the plain
    # `decimate` filter dropped that kwarg in pyvista 0.47.
    return head.decimate_pro(frac, preserve_topology=True)


def _scalp_e_peak(specs, scalp_sub: pv.PolyData) -> float:
    _, stats = cn.induced_e_field_on_surface(specs, scalp_sub, in_phase=True)
    return stats.e_peak_v_per_m


def _cortex_e_peak(specs, cortex_sub: pv.PolyData) -> float:
    _, stats = cn.induced_e_field_on_surface(specs, cortex_sub, in_phase=True)
    return stats.e_peak_v_per_m


def _energy_j(specs) -> float:
    return float(0.5 * sum(cn.self_inductance(s) * s.current_a ** 2 for s in specs))


def _wall_power_w(specs, limits: cn.LimitsSpec) -> float:
    return _energy_j(specs) * limits.prf_hz / max(1e-6, limits.charger_efficiency)


def _max_thermal_w(specs, limits: cn.LimitsSpec) -> float:
    if not specs:
        return 0.0
    return max(cn.coil_thermal_w(s, limits) for s in specs)


# ---------------------------------------------------------------------------
# Top-level optimizer
# ---------------------------------------------------------------------------
def optimize_array(
    base_specs,
    head: pv.PolyData,
    target_xyz: np.ndarray,
    limits: cn.LimitsSpec,
    *,
    i_max: float = 5000.0,
    n_scalp_sub: int = 4000,
    maxiter: int = 60,
    verbose: bool = True,
) -> tuple[list, dict]:
    """Run the constrained optimization. Returns (best_specs, info_dict)."""
    if head is None:
        raise ValueError("optimizer needs a head mesh for E-field + ray-cast")

    layout = PodLayout(base_specs, head)
    scalp_sub = _scalp_subsample(head, n_max=n_scalp_sub)
    cortex_sub = cn.cortex_proxy_mesh(scalp_sub, limits.cortex_offset_m)

    target = np.asarray(target_xyz, dtype=float).reshape(1, 3)

    # ---- Feasibility pre-pass --------------------------------------------
    # SLSQP can terminate at an infeasible KKT point if the start is
    # infeasible. Pull the start onto the feasible side first by scaling
    # currents down uniformly until E_scalp ≤ cap and E_pulse ≤ cap.
    x_init = layout.x0()
    specs_init = layout.specs_from_x(x_init)
    e_s = _scalp_e_peak(specs_init, scalp_sub)
    e_p = _energy_j(specs_init)
    k_e = limits.e_max_scalp_v_per_m / max(e_s, 1e-12)
    k_pe = (limits.max_pulse_energy_j / max(e_p, 1e-12)) ** 0.5
    k = min(1.0, k_e, k_pe) * 0.99           # 1% slack for FD noise
    if k < 1.0:
        if verbose:
            print(f"[opt] feasibility pre-pass: scaling currents by ×{k:.3f} "
                  f"(E_scalp {e_s:.1f}→{e_s * k:.1f} V/m, "
                  f"E_pulse {e_p:.1f}→{e_p * k * k:.1f} J)")
        x_init[: layout.n_coils] *= k

    # State for callback / final report
    history: list[dict] = []

    def objective(x):
        specs = layout.specs_from_x(x)
        bmag = _b_at_target(specs, target)
        return -bmag                 # minimize -|B|

    def cons_e_scalp(x):
        specs = layout.specs_from_x(x)
        return limits.e_max_scalp_v_per_m - _scalp_e_peak(specs, scalp_sub)

    def cons_e_cortex(x):
        specs = layout.specs_from_x(x)
        return limits.e_max_cortex_v_per_m - _cortex_e_peak(specs, cortex_sub)

    def cons_energy(x):
        specs = layout.specs_from_x(x)
        return limits.max_pulse_energy_j - _energy_j(specs)

    def cons_wall(x):
        specs = layout.specs_from_x(x)
        return limits.max_avg_power_w - _wall_power_w(specs, limits)

    def cons_thermal(x):
        specs = layout.specs_from_x(x)
        return limits.max_coil_dissipation_w - _max_thermal_w(specs, limits)

    def cons_pod_separation_factory(i: int, j: int):
        """||p_i − p_j|| − d_min ≥ 0 for pods i,j."""
        def f(x):
            specs = layout.specs_from_x(x)
            # pull pod centres from the _z coil of each pod
            p_i = np.asarray(
                next(s.position_m for s in specs
                     if s.name == f"{layout.pod_names[i]}_z")
            )
            p_j = np.asarray(
                next(s.position_m for s in specs
                     if s.name == f"{layout.pod_names[j]}_z")
            )
            return float(np.linalg.norm(p_i - p_j) - limits.pod_min_separation_m)
        return f

    constraints = [
        {"type": "ineq", "fun": cons_e_scalp},
        {"type": "ineq", "fun": cons_e_cortex},
        {"type": "ineq", "fun": cons_energy},
        {"type": "ineq", "fun": cons_wall},
        {"type": "ineq", "fun": cons_thermal},
    ]
    for i in range(layout.n_pods):
        for j in range(i + 1, layout.n_pods):
            constraints.append(
                {"type": "ineq", "fun": cons_pod_separation_factory(i, j)}
            )

    bounds = layout.bounds(i_min=0.0, i_max=i_max)

    if verbose:
        b0 = -objective(x_init)
        print(f"[opt] start: |B(target)| = {b0 * 1e3:.3f} mT, "
              f"{layout.n_coils} coils, {layout.n_pods} pods, {layout.n_vars} vars")

    n_eval = [0]

    def cb(xk):
        n_eval[0] += 1
        if verbose and (n_eval[0] % 3 == 0):
            specs = layout.specs_from_x(xk)
            bm = _b_at_target(specs, target)
            print(f"[opt]   iter {n_eval[0]}: |B|={bm * 1e3:.3f} mT, "
                  f"E_pulse={_energy_j(specs):.1f} J, "
                  f"E_scalp={_scalp_e_peak(specs, scalp_sub):.1f} V/m")

    res = minimize(
        objective,
        x_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=cb,
        options=dict(
            maxiter=maxiter,
            ftol=1e-9,                     # objective ~ 1e-2 T, want sub-µT moves
            eps=5e-3,                      # FD step: 5 A in current, 0.005 rad
            disp=verbose,
        ),
    )

    best_specs = layout.specs_from_x(res.x)

    # Post-correction: SLSQP enforces constraints on the *subsampled* scalp
    # (and a sub-cortex). Validate against the full mesh — peaks can hide
    # between subsampled vertices, especially after pods migrate. If the
    # full-mesh E_scalp exceeds cap, scale all currents down uniformly
    # (E ∝ I, energy ∝ I² — uniform down-scale is monotone-feasible).
    full_cortex = cn.cortex_proxy_mesh(head, limits.cortex_offset_m)
    final_e_scalp_full = _scalp_e_peak(best_specs, head)
    final_e_cortex_full = (
        _cortex_e_peak(best_specs, full_cortex) if full_cortex is not None else 0.0
    )
    k_scalp = limits.e_max_scalp_v_per_m / max(final_e_scalp_full, 1e-12)
    k_cortex = limits.e_max_cortex_v_per_m / max(final_e_cortex_full, 1e-12)
    k = min(1.0, k_scalp, k_cortex) * 0.999
    if k < 0.999:
        if verbose:
            print(
                f"[opt] post-correction (full mesh): E_scalp={final_e_scalp_full:.1f}, "
                f"E_cortex={final_e_cortex_full:.1f} → scaling currents ×{k:.3f}"
            )
        x_corr = res.x.copy()
        x_corr[: layout.n_coils] *= k
        best_specs = layout.specs_from_x(x_corr)
        res_x = x_corr
    else:
        res_x = res.x

    info = dict(
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit) if hasattr(res, "nit") else len(history),
        b_target_mT_initial=float(_b_at_target(layout.specs_from_x(layout.x0()), target) * 1e3),
        b_target_mT_final=float(_b_at_target(best_specs, target) * 1e3),
        x0=layout.x0().tolist(),
        x_opt=res_x.tolist(),
    )
    return best_specs, info


# ---------------------------------------------------------------------------
# Write back optimized config
# ---------------------------------------------------------------------------
def write_optimized_yaml(
    src_yaml_path,
    optimized_specs,
    layout: PodLayout,
    out_yaml_path,
) -> None:
    """Update the source YAML with optimized currents + pod positions/normals.

    Preserves comments / formatting where possible by editing in place via a
    string-rewrite for the `pods:` block. (Round-trip yaml is messier than
    it needs to be for our 60-line config.)
    """
    import yaml

    with open(src_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Rebuild pod blocks
    pods_out = []
    for pod_name in layout.pod_names:
        specs = [s for s in optimized_specs if s.name.startswith(f"{pod_name}_")]
        if not specs:
            continue
        # All three axes share position; recover from any one
        pos = list(specs[0].position_m)
        # pod normal (inward) was stored on the _z spec
        zspec = next(s for s in specs if s.name.endswith("_z"))
        # Currents per axis go into a new per-coil override map
        currents = {s.name.split("_")[-1]: float(s.current_a) for s in specs}

        # Find the original pod block in the source config to keep comments
        original = next(
            (p for p in (cfg.get("pods") or []) if p.get("name") == pod_name), {}
        )
        new_block = dict(original)
        new_block["position_m"] = [float(x) for x in pos]
        new_block["normal"] = [float(x) for x in zspec.normal]
        # Per-axis override of current_a (loader doesn't know this yet —
        # we just write the average/max here as documentation; per-axis
        # currents are emitted as separate `coils:` entries below).
        new_block["current_a"] = float(np.mean(list(currents.values())))
        # Optimized peak dI/dt scaled with current
        new_block["peak_dIdt_a_per_us"] = float(
            np.mean([s.peak_dIdt_a_per_us or 0.0 for s in specs])
        )
        pods_out.append(new_block)

    # Also emit explicit per-coil overrides as a `coils:` block so per-axis
    # currents are honored exactly. The loader already supports this.
    coils_out = []
    for s in optimized_specs:
        coils_out.append(
            dict(
                name=s.name,
                loop_diameter_m=float(s.loop_diameter_m),
                wire_diameter_m=float(s.wire_diameter_m),
                turns=int(s.turns),
                helix_height_m=float(s.helix_height_m),
                segments_per_turn=int(s.segments_per_turn),
                current_a=float(s.current_a),
                position_m=[float(x) for x in s.position_m],
                normal=[float(x) for x in s.normal],
                geometry=str(s.geometry),
                peak_dIdt_a_per_us=(
                    float(s.peak_dIdt_a_per_us)
                    if s.peak_dIdt_a_per_us is not None
                    else None
                ),
            )
        )

    # Write the optimized YAML — use explicit `coils:` (one entry per coil)
    # and drop `pods:` so the loader doesn't double-build.
    cfg_out = deepcopy(cfg)
    cfg_out.pop("pods", None)
    cfg_out["coils"] = coils_out

    with open(out_yaml_path, "w") as f:
        f.write(
            "# Auto-generated by optimize.py from "
            f"{src_yaml_path}\n"
            "# All pods expanded into explicit per-coil entries with the\n"
            "# optimized currents, pod positions, and induced-E-aware caps.\n\n"
        )
        yaml.safe_dump(cfg_out, f, sort_keys=False)
