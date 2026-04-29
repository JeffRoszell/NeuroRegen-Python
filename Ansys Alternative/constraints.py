"""
NeuroRegen — Physical constraint modeling for the multipod TMS array.

Companion to simulate_bfield.py. Adds:

    Induced E-field on the scalp surface (and a cortex proxy 12 mm inward),
    per-pulse magnetic energy, wall-plug average power, I²R thermal
    dissipation per coil — all evaluated against a `limits:` block from
    the YAML config.

Key formulas
------------
Vector potential of a circular loop, axis along local ẑ, in the loop frame
(Smythe §7.10 / Jackson 5.37):

    A_φ(ρ, z) = (μ₀ I) / (π √m) · √(a/ρ) · [(1 − m/2) K(m) − E(m)]
                m = 4 a ρ / [(a + ρ)² + z²]

K(m), E(m) are the complete elliptic integrals (scipy parameter convention,
m = k²). On axis (ρ → 0) the expression has a 0/0 limit equal to zero.

Induced E-field in the quasi-static limit (worst case: all coils peak in
phase, no tissue conductivity correction):

    E(r) = −Σ_coils  (dI/dt)_peak · A_per_unit_current(r)

Self-inductance (close-wound short coil, Maxwell single-loop scaled by N²):

    L_self ≈ N² · μ₀ R [ln(8R / r_wire) − 2]

Mutual inductance between coils is dropped (orthogonal coils in the same
pod have M = 0 by symmetry; pod-to-pod mutuals are O(10 nH) at 10 cm
separation versus L_self ~ 10 µH — three orders of magnitude smaller, so
the diagonal approximation costs <0.1 % on the energy budget).

AC resistance (engineering skin-effect proxy):

    R_ac = R_dc · litz_factor · max(1, a / 2δ)        δ = √(2ρ / ωμ₀)

RMS current for a repeated half-sine pulse of width τ at rate PRF:

    I_rms = I_peak · √(τ · PRF / 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyvista as pv
from scipy.special import ellipe, ellipk

MU_0 = 4.0 * np.pi * 1e-7        # T·m/A
RHO_CU = 1.68e-8                 # Ω·m  — annealed copper @ 20 °C


# ---------------------------------------------------------------------------
# Limits dataclass + YAML loader
# ---------------------------------------------------------------------------
@dataclass
class LimitsSpec:
    e_max_scalp_v_per_m: float = 100.0          # IEC 60601-2-33 / Reilly PNS
    e_max_cortex_v_per_m: float = 150.0         # TMS rMT band, off-target cap
    cortex_offset_m: float = 0.012              # scalp → cortex proxy inward
    max_pulse_energy_j: float = 200.0           # Magstim-class bench
    max_avg_power_w: float = 1800.0             # 15 A US wall, 80 % derate
    charger_efficiency: float = 0.85            # cap-bank charger eff
    prf_hz: float = 1.0                         # pulse repetition rate
    max_coil_dissipation_w: float = 50.0        # passive convection per coil
    wire_resistivity_ohm_m: float = RHO_CU      # copper default
    litz_skin_factor: float = 1.3               # bundled-Litz penalty
    in_phase_firing: bool = True                # worst-case E-field assumption
    pod_min_separation_m: float = 0.06          # centre-to-centre, prevents collapse

    @classmethod
    def from_yaml(cls, block: Optional[dict]) -> "LimitsSpec":
        if not block:
            return cls()
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in block:
                kwargs[f] = type(getattr(cls(), f))(block[f])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Vector potential — closed form for one circular loop
# ---------------------------------------------------------------------------
def _rotation_align_z_to(target: np.ndarray) -> np.ndarray:
    """3×3 rotation taking local +ẑ onto the unit vector `target`. Rodrigues."""
    z = np.array([0.0, 0.0, 1.0])
    t = np.asarray(target, dtype=float)
    t = t / np.linalg.norm(t)
    v = np.cross(z, t)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, t))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def vector_potential_circle(
    center: np.ndarray,
    normal: np.ndarray,
    radius_m: float,
    current_a: float,
    observers: np.ndarray,
) -> np.ndarray:
    """Vector potential A (Tesla·m) at `observers` from one circular loop."""
    rot = _rotation_align_z_to(np.asarray(normal, dtype=float))
    obs_local = (np.asarray(observers, dtype=float) - np.asarray(center, dtype=float)) @ rot

    x_loc, y_loc, z_loc = obs_local[:, 0], obs_local[:, 1], obs_local[:, 2]
    rho = np.hypot(x_loc, y_loc)
    eps = 1e-12
    rho_safe = np.maximum(rho, eps)
    a = float(radius_m)

    m = 4.0 * a * rho_safe / ((a + rho_safe) ** 2 + z_loc ** 2)
    m = np.clip(m, 0.0, 1.0 - 1e-12)        # K(m) singular at m → 1

    K = ellipk(m)
    E = ellipe(m)

    # √m in denominator cancels analytically with √(aρ)/(...) — keep it
    # explicit; the 0/0 on axis is handled by the rho < eps mask below.
    A_phi = (
        (MU_0 * current_a / np.pi)
        * np.sqrt(a / rho_safe)
        * ((1.0 - 0.5 * m) * K - E)
        / np.sqrt(m + 1e-30)
    )
    A_phi = np.where(rho < eps, 0.0, A_phi)

    cos_phi = x_loc / rho_safe
    sin_phi = y_loc / rho_safe

    A_local = np.column_stack(
        [-A_phi * sin_phi, A_phi * cos_phi, np.zeros_like(A_phi)]
    )
    return A_local @ rot.T


def vector_potential_collection(
    coil_specs,
    observers: np.ndarray,
) -> np.ndarray:
    """Sum A from every turn of every coil. Uses the same Circle stack the
    main simulator builds (one Circle per turn, evenly spaced along the
    coil axis). `coil_specs` is a Sequence[CoilSpec] from simulate_bfield.
    """
    A_total = np.zeros_like(observers, dtype=float)
    for spec in coil_specs:
        rot = _rotation_align_z_to(np.asarray(spec.normal, dtype=float))
        R_loop = spec.loop_diameter_m / 2.0
        if spec.turns <= 1 or spec.helix_height_m <= 0:
            z_offsets = np.array([0.0])
        else:
            z_offsets = np.linspace(
                -spec.helix_height_m / 2.0, spec.helix_height_m / 2.0, spec.turns
            )
        base = np.asarray(spec.position_m, dtype=float)
        for z in z_offsets:
            center = rot @ np.array([0.0, 0.0, float(z)]) + base
            A_total += vector_potential_circle(
                center, spec.normal, R_loop, spec.current_a, observers
            )
    return A_total


# ---------------------------------------------------------------------------
# Cortex proxy mesh (inward offset of scalp)
# ---------------------------------------------------------------------------
def cortex_proxy_mesh(
    head: Optional[pv.PolyData], offset_m: float
) -> Optional[pv.PolyData]:
    if head is None:
        return None
    with_normals = head.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        non_manifold_traversal=False,
    )
    proxy = with_normals.copy()
    proxy.points = (
        np.asarray(with_normals.points)
        - np.asarray(with_normals.point_normals) * float(offset_m)
    )
    return proxy


# ---------------------------------------------------------------------------
# Induced E-field on a mesh
# ---------------------------------------------------------------------------
@dataclass
class EFieldStats:
    surface_name: str
    n_points: int
    e_peak_v_per_m: float                        # max |E| over the surface
    e_p99_v_per_m: float                         # 99th percentile |E|
    argmax_xyz_m: tuple[float, float, float]     # location of E_peak


def induced_e_field_on_surface(
    coil_specs,
    surface: pv.PolyData,
    *,
    in_phase: bool = True,
) -> tuple[np.ndarray, EFieldStats]:
    """E (V/m) at every point of `surface`, plus summary stats.

    E = −Σ_i (dI/dt)_i · A_i(per unit I, evaluated at surface)
    """
    pts = np.asarray(surface.points, dtype=float)

    # A is linear in current; we want A_per_unit_I per coil. Easiest: build
    # a temporary copy of each spec at I=1 A, sum -dIdt_i · A_i.
    E = np.zeros_like(pts)
    for spec in coil_specs:
        if spec.peak_dIdt_a_per_us is None:
            continue
        dIdt = float(spec.peak_dIdt_a_per_us) * 1e6  # A/s
        if not in_phase:
            # Random sign per coil — gives a (loose) upper bound on the
            # incoherent-sum case. Not used by default.
            dIdt *= np.random.choice([-1.0, 1.0])
        # Reuse the collection routine on a one-element list with I=1.
        unit_spec = _spec_with_unit_current(spec)
        A_unit = vector_potential_collection([unit_spec], pts)
        E -= dIdt * A_unit

    Emag = np.linalg.norm(E, axis=1)
    arg = int(np.argmax(Emag))
    stats = EFieldStats(
        surface_name=surface.field_data.get("name", ["surface"])[0]
        if "name" in surface.field_data
        else "surface",
        n_points=Emag.size,
        e_peak_v_per_m=float(Emag.max()),
        e_p99_v_per_m=float(np.quantile(Emag, 0.99)),
        argmax_xyz_m=tuple(pts[arg].tolist()),
    )
    return E, stats


def _spec_with_unit_current(spec):
    """Lightweight clone of a CoilSpec with current_a = 1.0."""
    from dataclasses import replace
    return replace(spec, current_a=1.0)


# ---------------------------------------------------------------------------
# Self-inductance, AC resistance, energy, power
# ---------------------------------------------------------------------------
def self_inductance(spec) -> float:
    """Maxwell single-loop L scaled by N² for a tightly close-wound coil."""
    R = spec.loop_diameter_m / 2.0
    r_w = spec.wire_diameter_m / 2.0
    if r_w <= 0 or R <= 0 or 8.0 * R / r_w <= np.e ** 2:
        return 0.0
    L_single = MU_0 * R * (np.log(8.0 * R / r_w) - 2.0)
    return float(spec.turns ** 2 * L_single)


def wire_length(spec) -> float:
    return float(spec.turns * np.pi * spec.loop_diameter_m)


def dc_resistance(spec, rho_ohm_m: float) -> float:
    A_wire = np.pi * (spec.wire_diameter_m / 2.0) ** 2
    if A_wire <= 0:
        return float("inf")
    return float(rho_ohm_m * wire_length(spec) / A_wire)


def fundamental_omega(spec) -> float:
    """Effective angular frequency from the half-sine peak dI/dt → I_peak."""
    if spec.peak_dIdt_a_per_us is None or spec.current_a <= 0:
        return 0.0
    return float(spec.peak_dIdt_a_per_us * 1e6 / spec.current_a)


def ac_resistance(spec, rho_ohm_m: float, litz_factor: float) -> float:
    R_dc = dc_resistance(spec, rho_ohm_m)
    omega = fundamental_omega(spec)
    if omega <= 0:
        return R_dc * litz_factor
    skin_depth = np.sqrt(2.0 * rho_ohm_m / (omega * MU_0))
    a = spec.wire_diameter_m / 2.0
    skin_factor = max(1.0, a / (2.0 * skin_depth))
    return float(R_dc * litz_factor * skin_factor)


def pulse_width_s(spec) -> float:
    """Half-sine pulse width: τ = π / ω_peak."""
    omega = fundamental_omega(spec)
    return float(np.pi / omega) if omega > 0 else 0.0


def rms_current(spec, prf_hz: float) -> float:
    """I_rms over one period for a repeated half-sine: I_pk·√(τ·PRF/2)."""
    tau = pulse_width_s(spec)
    if tau <= 0 or prf_hz <= 0:
        return 0.0
    duty = min(1.0, tau * prf_hz)
    return float(spec.current_a * np.sqrt(duty / 2.0))


def coil_thermal_w(spec, limits: LimitsSpec) -> float:
    R_ac = ac_resistance(spec, limits.wire_resistivity_ohm_m, limits.litz_skin_factor)
    I_rms = rms_current(spec, limits.prf_hz)
    return float(I_rms ** 2 * R_ac)


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------
@dataclass
class ConstraintReport:
    e_scalp: Optional[EFieldStats]
    e_cortex: Optional[EFieldStats]
    pulse_energy_j: float
    wall_power_w: float
    thermal_w_per_coil: dict
    limits: LimitsSpec
    violations: list

    def passed(self) -> bool:
        return not self.violations


def evaluate_constraints(
    coil_specs,
    head_mesh: Optional[pv.PolyData],
    limits: LimitsSpec,
) -> ConstraintReport:
    # 1. E-field on scalp + cortex proxy
    e_scalp_stats: Optional[EFieldStats] = None
    e_cortex_stats: Optional[EFieldStats] = None
    if head_mesh is not None:
        _, e_scalp_stats = induced_e_field_on_surface(
            coil_specs, head_mesh, in_phase=limits.in_phase_firing
        )
        cortex = cortex_proxy_mesh(head_mesh, limits.cortex_offset_m)
        if cortex is not None:
            _, e_cortex_stats = induced_e_field_on_surface(
                coil_specs, cortex, in_phase=limits.in_phase_firing
            )

    # 2. Per-pulse energy (diagonal: ½ Σ Lᵢ Iᵢ²)
    Ls = [self_inductance(s) for s in coil_specs]
    pulse_energy = float(0.5 * sum(L * s.current_a ** 2 for L, s in zip(Ls, coil_specs)))

    # 3. Wall-plug average power
    eta = max(1e-6, limits.charger_efficiency)
    wall_power = float(pulse_energy * limits.prf_hz / eta)

    # 4. Thermal per coil
    thermal = {s.name: coil_thermal_w(s, limits) for s in coil_specs}

    # 5. Violations (1% slack — at-cap is OK, treat as "meets")
    SLACK = 1.01
    violations: list[str] = []
    if e_scalp_stats and e_scalp_stats.e_peak_v_per_m > limits.e_max_scalp_v_per_m * SLACK:
        violations.append(
            f"E_scalp_peak={e_scalp_stats.e_peak_v_per_m:.1f} V/m "
            f"> cap {limits.e_max_scalp_v_per_m:.1f}"
        )
    if e_cortex_stats and e_cortex_stats.e_peak_v_per_m > limits.e_max_cortex_v_per_m * SLACK:
        violations.append(
            f"E_cortex_peak={e_cortex_stats.e_peak_v_per_m:.1f} V/m "
            f"> cap {limits.e_max_cortex_v_per_m:.1f}"
        )
    if pulse_energy > limits.max_pulse_energy_j * SLACK:
        violations.append(
            f"E_pulse={pulse_energy:.1f} J > cap {limits.max_pulse_energy_j:.1f}"
        )
    if wall_power > limits.max_avg_power_w * SLACK:
        violations.append(
            f"P_wall={wall_power:.0f} W > cap {limits.max_avg_power_w:.0f}"
        )
    for name, p in thermal.items():
        if p > limits.max_coil_dissipation_w * SLACK:
            violations.append(
                f"P_thermal[{name}]={p:.1f} W > cap {limits.max_coil_dissipation_w:.1f}"
            )

    return ConstraintReport(
        e_scalp=e_scalp_stats,
        e_cortex=e_cortex_stats,
        pulse_energy_j=pulse_energy,
        wall_power_w=wall_power,
        thermal_w_per_coil=thermal,
        limits=limits,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------
def report_constraints(report: ConstraintReport) -> int:
    L = report.limits
    print("\n[constraints]")
    print(f"  {'metric':<28} {'value':>14}   {'cap':>10}   {'status'}")

    def row(label: str, value: float, cap: float, fmt: str = "{:.2f}") -> None:
        ok = value <= cap * 1.01
        print(
            f"  {label:<28} {fmt.format(value):>14}   {fmt.format(cap):>10}   "
            f"{'OK' if ok else 'OVER'}"
        )

    if report.e_scalp is not None:
        row(
            "E_scalp peak (V/m)",
            report.e_scalp.e_peak_v_per_m,
            L.e_max_scalp_v_per_m,
            "{:.1f}",
        )
        print(
            f"  {'  ↳ scalp argmax (mm)':<28} "
            f"({report.e_scalp.argmax_xyz_m[0] * 1e3:+.1f}, "
            f"{report.e_scalp.argmax_xyz_m[1] * 1e3:+.1f}, "
            f"{report.e_scalp.argmax_xyz_m[2] * 1e3:+.1f})"
        )
    else:
        print(f"  {'E_scalp peak':<28} {'(no head mesh)':>14}")

    if report.e_cortex is not None:
        row(
            "E_cortex peak (V/m)",
            report.e_cortex.e_peak_v_per_m,
            L.e_max_cortex_v_per_m,
            "{:.1f}",
        )
    else:
        print(f"  {'E_cortex peak':<28} {'(no head mesh)':>14}")

    row(
        "Pulse energy (J)",
        report.pulse_energy_j,
        L.max_pulse_energy_j,
        "{:.1f}",
    )
    row(
        f"Wall power @ {L.prf_hz:g} Hz (W)",
        report.wall_power_w,
        L.max_avg_power_w,
        "{:.0f}",
    )
    p_max = max(report.thermal_w_per_coil.values()) if report.thermal_w_per_coil else 0.0
    hottest = (
        max(report.thermal_w_per_coil.items(), key=lambda kv: kv[1])[0]
        if report.thermal_w_per_coil
        else "-"
    )
    row(
        f"Thermal max ({hottest}) (W)",
        p_max,
        L.max_coil_dissipation_w,
        "{:.1f}",
    )

    if report.violations:
        print(f"\n  VIOLATIONS ({len(report.violations)}):")
        for v in report.violations:
            print(f"    - {v}")
    else:
        print("\n  ALL CONSTRAINTS OK.")
    return len(report.violations)
