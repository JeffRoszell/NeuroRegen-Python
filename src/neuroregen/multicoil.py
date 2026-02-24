"""
Multicoil focusing system for deep brain stimulation.

Implements the physics and safety logic required to drive an array of coils
positioned around the skull so that their fields superpose constructively at
a deep target (e.g. the Subthalamic Nucleus) while keeping cortical exposure
within safe limits.

Key features
------------
1. **Sixth-power distance compensation** – each coil's drive amplitude is
   automatically scaled so that every coil delivers equal field strength at
   the focal point, regardless of its distance.  Because B ∝ √P / d³ in
   the far field, equalising B requires power weights ∝ d⁶.
2. **Depth Gate (surface-to-target ratio)** – the induced E-field at the
   cortical surface directly under each coil is estimated.  If any coil
   exceeds the safety threshold (default 150 V/m) the entire pulse is
   blocked ("Depth Gate Failure").
3. **Tilt and orientation (cosine effect)** – a rotation matrix maps each
   coil's local B-field into the global frame based on its 3D orientation
   (normal vector).  The effective field contribution is reduced by the
   cosine of the misalignment angle.
4. **Weighting array** – coil powers are automatically "gained up" for far
   coils and "gained down" for close ones so the combined field peaks at
   the target rather than drifting toward the nearest coil.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Generator

from .constants import MU0, CP_CU, B_THRESHOLD_T
from .coil import Axis, coil_geom, resistance, f_to_c
from .thermal import thermal_gate_update, cooling_power, temp_step


# ---------------------------------------------------------------------------
# Safety defaults
# ---------------------------------------------------------------------------
E_FIELD_CORTICAL_MAX_VM: float = 150.0   # V/m — max induced E at cortex
SCALP_TO_CORTEX_M: float = 0.015         # 1.5 cm (scalp + skull)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Coil:
    """
    A physical coil with 3-D position and orientation.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "A", "B", "C").
    wire_mm : float
        Wire diameter (mm).
    loop_mm : float
        Loop diameter (mm).
    turns : int
        Number of turns.
    pulse_power_w : float
        *Base* power per pulse (W) **before** distance compensation.
    position_m : tuple[float, float, float]
        (x, y, z) of the coil centre, metres, relative to head origin.
    normal : tuple[float, float, float]
        Unit-normal of the coil plane — points toward the brain by
        convention (direction of the principal B-field axis).
    """

    name: str
    wire_mm: float
    loop_mm: float
    turns: int
    pulse_power_w: float
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float)
        mag = float(np.linalg.norm(n))
        if mag > 0:
            n = n / mag
        self.normal = (float(n[0]), float(n[1]), float(n[2]))

    def to_axis(self) -> Axis:
        """Down-cast to an :class:`Axis` for single-coil helper functions."""
        return Axis(
            name=self.name,
            wire_mm=self.wire_mm,
            loop_mm=self.loop_mm,
            turns=self.turns,
            pulse_power_w=self.pulse_power_w,
        )


@dataclass
class Target:
    """
    Stimulation target in 3-D space.

    Attributes
    ----------
    name : str
        Anatomical label (e.g. ``"STN_left"``).
    position_m : tuple[float, float, float]
        (x, y, z) in metres relative to head origin.
    """

    name: str
    position_m: tuple[float, float, float]


@dataclass
class DepthGateResult:
    """Outcome of a single depth-gate evaluation."""

    safe: bool
    per_coil: list[dict]
    """
    Each dict:
        coil          – coil name
        E_surface_vm  – estimated cortical E-field (V/m)
        limit_vm      – threshold (V/m)
        passed        – bool
        current_A     – coil current (A)
    """


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def euclidean_distance(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3-D points (m)."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))))



# ---------------------------------------------------------------------------
# Single-coil B-field in 3-D (with orientation)
# ---------------------------------------------------------------------------
def B_field_at_point(
    coil: Coil,
    I: float,
    point_m: tuple[float, float, float],
) -> np.ndarray:
    """
    B-field vector (T) at an arbitrary 3-D point from a single coil.

    Decomposes the coil-to-point vector into an axial distance (along the
    coil normal) and a radial offset, applies the on-axis Biot–Savart
    formula with an off-axis exponential decay factor, and rotates the
    result into global coordinates using the coil's normal vector.

    Parameters
    ----------
    coil : Coil
    I : float   – current (A)
    point_m : 3-tuple of float  – target coordinates (m)

    Returns
    -------
    B_vec : ndarray, shape (3,)  – (Bx, By, Bz) in Tesla
    """
    R_loop = coil.loop_mm / 2000.0
    N = coil.turns

    r_vec = np.asarray(point_m, dtype=float) - np.asarray(coil.position_m, dtype=float)
    n_hat = np.asarray(coil.normal, dtype=float)

    # Axial (along coil normal) and radial components
    z_axial = float(np.dot(r_vec, n_hat))
    r_radial_vec = r_vec - z_axial * n_hat
    rho = float(np.linalg.norm(r_radial_vec))

    # On-axis B magnitude (Biot–Savart for N-turn loop)
    z_abs = abs(z_axial)
    B_axial_mag = MU0 * I * R_loop ** 2 / (2.0 * (R_loop ** 2 + z_abs ** 2) ** 1.5) * N

    # Off-axis decay (exponential approximation, scale length ~ 2R)
    scale = np.exp(-rho / (R_loop * 2.0)) if rho > 0 else 1.0
    B_mag = B_axial_mag * scale

    # Split into axial and radial components
    angle = np.arctan2(rho, z_abs + 1e-15)
    B_axial = B_mag * np.cos(angle) * (1.0 if z_axial >= 0 else -1.0)
    B_radial = B_mag * np.sin(angle)

    # Build global vector
    B_vec = B_axial * n_hat
    if rho > 1e-15:
        B_vec = B_vec + B_radial * (r_radial_vec / rho)

    return B_vec


# ---------------------------------------------------------------------------
# Distance, cosine, and weight computations
# ---------------------------------------------------------------------------
def compute_coil_distances(coils: list[Coil], target: Target) -> np.ndarray:
    """Euclidean distances (m) from each coil centre to the target."""
    return np.array([euclidean_distance(c.position_m, target.position_m) for c in coils])


def compute_cosine_factors(coils: list[Coil], target: Target) -> np.ndarray:
    r"""
    Cosine alignment factor per coil: :math:`|\cos\theta_i|`.

    ``1.0`` when the coil normal aims directly at the target;
    ``0.0`` when perpendicular.
    """
    factors = np.empty(len(coils))
    for i, coil in enumerate(coils):
        r_vec = np.asarray(target.position_m) - np.asarray(coil.position_m)
        r_mag = float(np.linalg.norm(r_vec))
        if r_mag < 1e-15:
            factors[i] = 1.0
            continue
        factors[i] = abs(float(np.dot(np.asarray(coil.normal), r_vec / r_mag)))
    return factors


def compute_distance_weights(
    coils: list[Coil],
    target: Target,
    *,
    include_cosine: bool = True,
) -> np.ndarray:
    """
    Amplitude weights that equalise each coil's contribution at the target.

    The weight for the *closest, best-aligned* coil is normalised to 1.0;
    farther / more misaligned coils receive weights > 1.

    Physics
    -------
    ``B ∝ R² / (R² + z²)^{3/2} ≈ 1/z³`` for ``z >> R``.
    Since ``B ∝ √P / d³`` (current from power as ``I = √(P/R)``), equalising
    B at the target requires ``P ∝ d^6``.
    Weight ∝ ``d^6`` (distance compensation) ÷ ``cos θ`` (tilt correction).
    """
    distances = compute_coil_distances(coils, target)
    weights = distances ** 6

    if include_cosine:
        cos_f = np.clip(compute_cosine_factors(coils, target), 0.1, 1.0)
        weights = weights / cos_f

    wmin = weights.min()
    if wmin > 0:
        weights = weights / wmin
    return weights


def compute_weighted_powers(
    coils: list[Coil],
    target: Target,
    *,
    include_cosine: bool = True,
) -> np.ndarray:
    """
    Distance-compensated pulse power (W) for every coil.

    Multiplies each coil's *base* ``pulse_power_w`` by its distance weight.
    """
    weights = compute_distance_weights(coils, target, include_cosine=include_cosine)
    return np.array([c.pulse_power_w for c in coils]) * weights


# ---------------------------------------------------------------------------
# Induced E-field estimation & Depth Gate
# ---------------------------------------------------------------------------
def E_field_at_surface(
    coil: Coil,
    I: float,
    pulse_width: float,
    surface_distance_m: float = SCALP_TO_CORTEX_M,
) -> float:
    r"""
    Peak induced E-field (V/m) at the cortical surface under a coil.

    Approximation (half-sine pulse, from Faraday's law for a circular path):

    .. math::

        E \approx \frac{R_{\text{coil}}}{2} \; \frac{\mathrm{d}B}{\mathrm{d}t}
        \qquad
        \frac{\mathrm{d}B}{\mathrm{d}t} \approx \frac{\pi\,B_{\text{peak}}}
        {\tau_{\text{pulse}}}

    Parameters
    ----------
    coil : Coil
    I : float – peak current (A)
    pulse_width : float – pulse duration (s)
    surface_distance_m : float – distance coil → cortex (m)
    """
    R_loop = coil.loop_mm / 2000.0
    N = coil.turns
    z = surface_distance_m

    B_surface = MU0 * I * R_loop ** 2 / (2.0 * (R_loop ** 2 + z ** 2) ** 1.5) * N
    dBdt = B_surface * np.pi / pulse_width   # peak dB/dt for a half-sine pulse
    return float((R_loop / 2.0) * dBdt)     # Faraday: E = (r/2) × dB/dt


def check_depth_gate(
    coils: list[Coil],
    weighted_powers: np.ndarray,
    temperatures_c: np.ndarray,
    pulse_width: float,
    cortical_max_vm: float = E_FIELD_CORTICAL_MAX_VM,
    surface_distance_m: float = SCALP_TO_CORTEX_M,
) -> DepthGateResult:
    """
    **Depth Gate** — blocks the pulse if *any* coil would exceed the
    cortical E-field safety limit.

    Returns a :class:`DepthGateResult` with a per-coil breakdown.
    """
    per_coil: list[dict] = []
    all_safe = True

    for i, coil in enumerate(coils):
        ax = coil.to_axis()
        _, L, A, _, _ = coil_geom(ax)
        R_ohm = resistance(L, A, float(temperatures_c[i]))
        Pin = float(weighted_powers[i])

        if Pin <= 0:
            per_coil.append(dict(
                coil=coil.name, E_surface_vm=0.0,
                limit_vm=cortical_max_vm, passed=True, current_A=0.0,
            ))
            continue

        I = np.sqrt(Pin / R_ohm)
        E_surf = E_field_at_surface(coil, I, pulse_width, surface_distance_m)
        passed = E_surf <= cortical_max_vm

        if not passed:
            all_safe = False

        per_coil.append(dict(
            coil=coil.name, E_surface_vm=E_surf,
            limit_vm=cortical_max_vm, passed=passed, current_A=I,
        ))

    return DepthGateResult(safe=all_safe, per_coil=per_coil)


# ---------------------------------------------------------------------------
# Superposed field at the target
# ---------------------------------------------------------------------------
def superposed_B_at_target(
    coils: list[Coil],
    weighted_powers: np.ndarray,
    temperatures_c: np.ndarray,
    target: Target,
) -> tuple[np.ndarray, float]:
    """
    Total B-field vector & magnitude (T) at the target from all coils.

    Each coil's field (at its weighted power and current temperature) is
    computed via :func:`B_field_at_point` and the vectors are summed.
    """
    B_total = np.zeros(3)
    for i, coil in enumerate(coils):
        Pin = float(weighted_powers[i])
        if Pin <= 0:
            continue
        ax = coil.to_axis()
        _, L, A, _, _ = coil_geom(ax)
        R_ohm = resistance(L, A, float(temperatures_c[i]))
        I = np.sqrt(Pin / R_ohm)
        B_total += B_field_at_point(coil, I, target.position_m)

    return B_total, float(np.linalg.norm(B_total))


def superposed_B_on_grid(
    coils: list[Coil],
    weighted_powers: np.ndarray,
    temperatures_c: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Superposed |B| on a 3-D meshgrid from all coils.

    Uses **vector superposition** — the B-field vectors from every coil are
    summed at each grid point *before* taking the magnitude.  This correctly
    captures constructive / destructive interference and shows the true
    focal-point convergence.

    Returns an array the same shape as *X* (and *Y*, *Z*).
    """
    shape = X.shape
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    n_pts = pts.shape[0]

    # Accumulate B-field VECTORS, not magnitudes
    B_vec_total = np.zeros((n_pts, 3))

    for i, coil in enumerate(coils):
        Pin = float(weighted_powers[i])
        if Pin <= 0:
            continue
        ax = coil.to_axis()
        _, L, A, _, _ = coil_geom(ax)
        R_ohm = resistance(L, A, float(temperatures_c[i]))
        I = np.sqrt(Pin / R_ohm)

        for j in range(n_pts):
            B_vec_total[j] += B_field_at_point(coil, I, tuple(pts[j]))

    # Magnitude of the vector sum
    B_mag = np.linalg.norm(B_vec_total, axis=1)
    return B_mag.reshape(shape)


# ---------------------------------------------------------------------------
# Surface-to-Deep ratio
# ---------------------------------------------------------------------------
def surface_to_deep_ratio(
    coils: list[Coil],
    weighted_powers: np.ndarray,
    temperatures_c: np.ndarray,
    target: Target,
    pulse_width: float,
    surface_distance_m: float = SCALP_TO_CORTEX_M,
) -> dict:
    """
    The Surface-to-Deep Ratio — the biggest safety metric for deep
    stimulation.

    Returns
    -------
    dict with keys:
        max_surface_E_vm  – worst-case cortical E (V/m)
        target_B_T        – combined B magnitude at target (T)
        target_E_vm       – estimated E at target (V/m)
        ratio             – surface / deep  (lower is better)
        worst_coil        – name of the coil with highest surface E
    """
    max_E = 0.0
    worst = ""

    for i, coil in enumerate(coils):
        Pin = float(weighted_powers[i])
        if Pin <= 0:
            continue
        ax = coil.to_axis()
        _, L, A, _, _ = coil_geom(ax)
        R_ohm = resistance(L, A, float(temperatures_c[i]))
        I = np.sqrt(Pin / R_ohm)
        E_surf = E_field_at_surface(coil, I, pulse_width, surface_distance_m)
        if E_surf > max_E:
            max_E = E_surf
            worst = coil.name

    _, B_mag = superposed_B_at_target(coils, weighted_powers, temperatures_c, target)
    R_avg = np.mean([c.loop_mm / 2000.0 for c in coils])
    E_target = R_avg * B_mag * 2.0 / pulse_width
    ratio = max_E / E_target if E_target > 0 else float("inf")

    return dict(
        max_surface_E_vm=max_E,
        target_B_T=B_mag,
        target_E_vm=E_target,
        ratio=ratio,
        worst_coil=worst,
    )


# ---------------------------------------------------------------------------
# MulticoilArray — top-level manager
# ---------------------------------------------------------------------------
class MulticoilArray:
    """
    Manages a set of coils targeting a single deep-brain point.

    On construction it pre-computes distances, cosine factors, weights, and
    weighted powers.  Call :meth:`check_safety` before every pulse and
    :meth:`B_at_target` to evaluate stimulation efficacy.
    """

    def __init__(
        self,
        coils: list[Coil],
        target: Target,
        cortical_max_vm: float = E_FIELD_CORTICAL_MAX_VM,
        surface_distance_m: float = SCALP_TO_CORTEX_M,
    ):
        self.coils = coils
        self.target = target
        self.cortical_max_vm = cortical_max_vm
        self.surface_distance_m = surface_distance_m
        self.n_coils = len(coils)

        # Pre-computed static geometry
        self.distances = compute_coil_distances(coils, target)
        self.cosine_factors = compute_cosine_factors(coils, target)
        self.weights = compute_distance_weights(coils, target, include_cosine=True)
        self.weighted_powers = compute_weighted_powers(coils, target, include_cosine=True)

    # ---- convenience methods ------------------------------------------------

    def get_weighted_power(self, idx: int) -> float:
        return float(self.weighted_powers[idx])

    def check_safety(
        self,
        temperatures_c: np.ndarray,
        pulse_width: float,
    ) -> DepthGateResult:
        """Depth-gate safety check at the given coil temperatures."""
        return check_depth_gate(
            self.coils,
            self.weighted_powers,
            temperatures_c,
            pulse_width,
            self.cortical_max_vm,
            self.surface_distance_m,
        )

    def B_at_target(
        self,
        temperatures_c: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Superposed B-field vector and magnitude at the target.

        *active_mask* is a boolean array the same length as ``coils``; when a
        coil is ``False`` it is excluded (disabled or thermally gated).
        """
        powers = self.weighted_powers.copy()
        if active_mask is not None:
            powers *= np.asarray(active_mask, dtype=float)
        return superposed_B_at_target(self.coils, powers, temperatures_c, self.target)

    def get_surface_to_deep_ratio(
        self,
        temperatures_c: np.ndarray,
        pulse_width: float,
    ) -> dict:
        return surface_to_deep_ratio(
            self.coils,
            self.weighted_powers,
            temperatures_c,
            self.target,
            pulse_width,
            self.surface_distance_m,
        )

    # ---- pretty-print -------------------------------------------------------

    def summary(self, temperatures_c: np.ndarray, pulse_width: float) -> str:
        """Human-readable configuration and safety summary."""
        lines: list[str] = []
        w = 74
        lines.append("=" * w)
        lines.append("MULTICOIL ARRAY — CONFIGURATION & PRE-FLIGHT CHECK")
        lines.append("=" * w)
        t = self.target
        lines.append(
            f"Target : {t.name}  at  "
            f"({t.position_m[0]*100:+.2f}, {t.position_m[1]*100:+.2f}, "
            f"{t.position_m[2]*100:+.2f}) cm"
        )
        lines.append(f"Cortical E-field limit    : {self.cortical_max_vm:.0f} V/m")
        lines.append(
            f"Scalp-to-cortex distance  : {self.surface_distance_m*100:.1f} cm"
        )
        lines.append("")

        hdr = (
            f"{'Coil':<6} {'Dist':>7} {'cos θ':>7} {'Wt':>7} "
            f"{'Base P':>8} {'Wt P':>8} {'E_surf':>9} {'Gate':>6}"
        )
        lines.append(hdr)
        unit = (
            f"{'':6} {'(cm)':>7} {'':>7} {'':>7} "
            f"{'(W)':>8} {'(W)':>8} {'(V/m)':>9} {'':>6}"
        )
        lines.append(unit)
        lines.append("-" * w)

        gate = self.check_safety(temperatures_c, pulse_width)

        for i, coil in enumerate(self.coils):
            r = gate.per_coil[i]
            flag = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"{coil.name:<6} {self.distances[i]*100:7.2f} "
                f"{self.cosine_factors[i]:7.3f} {self.weights[i]:7.2f} "
                f"{coil.pulse_power_w:8.1f} {self.weighted_powers[i]:8.1f} "
                f"{r['E_surface_vm']:9.2f} {flag:>6}"
            )

        lines.append("-" * w)

        B_vec, B_mag = self.B_at_target(temperatures_c)
        lines.append(
            f"Combined B at target : {B_mag*1000:.4f} mT  "
            f"({B_vec[0]*1000:+.4f}, {B_vec[1]*1000:+.4f}, "
            f"{B_vec[2]*1000:+.4f}) mT"
        )

        sdr = self.get_surface_to_deep_ratio(temperatures_c, pulse_width)
        lines.append(
            f"Surface-to-Deep Ratio: {sdr['ratio']:.2f}  "
            f"(worst coil: {sdr['worst_coil']})"
        )

        status = "PASS" if gate.safe else "FAIL — PULSE BLOCKED"
        lines.append(f"\nDepth Gate : {status}")
        lines.append("=" * w)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation loops (batch & stepwise)
# ---------------------------------------------------------------------------
def _mc_sim_params(config: dict) -> dict:
    """Extract simulation parameters from a multicoil config dict."""
    return dict(
        sim_time=config["sim_time"],
        dt=config["dt"],
        pulse_freq=config["pulse_freq"],
        pulse_width=config["pulse_width"],
        t_amb_c=config["t_amb_c"],
        h_conv=config["h_conv"],
        limit_c=f_to_c(config["temp_limit_f"]),
        hyst_c=f_to_c(config["temp_limit_f"] - config["hyst_f"]),
        b_threshold_t=config.get("b_threshold_t", B_THRESHOLD_T),
    )


def run_multicoil_simulation(
    array: MulticoilArray,
    config: dict,
) -> dict:
    """
    Run a full multicoil simulation (batch mode).

    Parameters
    ----------
    array : MulticoilArray
        Fully initialised coil array.
    config : dict
        Flat config dict (from :func:`load_multicoil_config`).

    Returns
    -------
    results : dict with keys
        t            – (N,) time array (s)
        T            – (N, n_coils) coil temperatures (°C)
        P            – (N, n_coils) applied power (W)
        B_target     – (N,) combined |B| at target (T)
        E_surface    – (N, n_coils) cortical E per coil (V/m)
        depth_gate   – (N,) bool, True = safe
    """
    p = _mc_sim_params(config)
    nc = array.n_coils

    t = np.arange(0, p["sim_time"], p["dt"])
    n = len(t)

    T = np.full((n, nc), p["t_amb_c"])
    P = np.zeros((n, nc))
    B_target = np.zeros(n)
    E_surface = np.zeros((n, nc))
    depth_gate = np.ones(n, dtype=bool)

    # Per-coil geometry
    axes = [c.to_axis() for c in array.coils]
    geom = [coil_geom(a) for a in axes]
    Cth = [g[4] * CP_CU for g in geom]
    gated_off = [False] * nc

    for k in range(1, n):
        pulse_on = (t[k] % (1 / p["pulse_freq"])) < p["pulse_width"]

        # --- thermal gating per coil ---
        for i in range(nc):
            gated_off[i] = thermal_gate_update(
                T[k - 1, i], gated_off[i],
                limit_c=p["limit_c"], hyst_c=p["hyst_c"],
            )

        # Active mask: pulse on AND not thermally gated
        active = np.array([pulse_on and not gated_off[i] for i in range(nc)])

        # Weighted powers for active coils
        w_powers = array.weighted_powers * active.astype(float)

        # --- depth gate ---
        if np.any(active):
            gate = check_depth_gate(
                array.coils, w_powers, T[k - 1],
                p["pulse_width"], array.cortical_max_vm,
                array.surface_distance_m,
            )
            depth_gate[k] = gate.safe
            for i in range(nc):
                E_surface[k, i] = gate.per_coil[i]["E_surface_vm"]

            if not gate.safe:
                # Block the entire pulse
                w_powers[:] = 0.0
                active[:] = False
        else:
            depth_gate[k] = True

        # --- apply power & update temperature ---
        for i in range(nc):
            R_loop, L, A, S, m = geom[i]
            Pin = float(w_powers[i])
            P[k, i] = Pin

            Pcool = cooling_power(
                T[k - 1, i], S,
                h_conv=p["h_conv"], t_amb_c=p["t_amb_c"],
            )
            T[k, i] = temp_step(T[k - 1, i], Pin, Pcool, Cth[i], p["dt"])

        # --- superposed B at target ---
        if np.any(active) and depth_gate[k]:
            _, B_mag = superposed_B_at_target(
                array.coils, w_powers, T[k - 1], array.target,
            )
            B_target[k] = B_mag

    return dict(t=t, T=T, P=P, B_target=B_target,
                E_surface=E_surface, depth_gate=depth_gate)


def run_multicoil_simulation_stepwise(
    array: MulticoilArray,
    config: dict,
    coil_enabled: list[bool] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> Generator:
    """
    Stepwise generator for interactive multicoil runs.

    Yields
    ------
    ("step", k, t_k, T_k, P_k, B_target_k, E_surface_k, gate_ok)
        … per time-step.
    ("final", results_dict)
        … after the last step.

    *results_dict* has the same keys as :func:`run_multicoil_simulation`.
    """
    p = _mc_sim_params(config)
    nc = array.n_coils

    if coil_enabled is None:
        coil_enabled = [True] * nc
    coil_enabled = list(coil_enabled)[:nc]
    while len(coil_enabled) < nc:
        coil_enabled.append(True)

    t = np.arange(0, p["sim_time"], p["dt"])
    n = len(t)

    T = np.full((n, nc), p["t_amb_c"])
    P = np.zeros((n, nc))
    B_target = np.zeros(n)
    E_surface = np.zeros((n, nc))
    depth_gate = np.ones(n, dtype=bool)

    axes = [c.to_axis() for c in array.coils]
    geom = [coil_geom(a) for a in axes]
    Cth = [g[4] * CP_CU for g in geom]
    gated_off = [False] * nc

    last_k = 0
    for k in range(1, n):
        if stop_check and stop_check():
            last_k = k
            break

        pulse_on = (t[k] % (1 / p["pulse_freq"])) < p["pulse_width"]

        for i in range(nc):
            if not coil_enabled[i]:
                continue
            gated_off[i] = thermal_gate_update(
                T[k - 1, i], gated_off[i],
                limit_c=p["limit_c"], hyst_c=p["hyst_c"],
            )

        active = np.array([
            pulse_on and coil_enabled[i] and not gated_off[i]
            for i in range(nc)
        ])
        w_powers = array.weighted_powers * active.astype(float)

        gate_ok = True
        if np.any(active):
            gate = check_depth_gate(
                array.coils, w_powers, T[k - 1],
                p["pulse_width"], array.cortical_max_vm,
                array.surface_distance_m,
            )
            gate_ok = gate.safe
            for i in range(nc):
                E_surface[k, i] = gate.per_coil[i]["E_surface_vm"]
            if not gate_ok:
                w_powers[:] = 0.0
                active[:] = False
        depth_gate[k] = gate_ok

        for i in range(nc):
            R_loop, L, A, S, m = geom[i]
            if not coil_enabled[i]:
                Pin = 0.0
            else:
                Pin = float(w_powers[i])
            P[k, i] = Pin
            Pcool = cooling_power(
                T[k - 1, i], S,
                h_conv=p["h_conv"], t_amb_c=p["t_amb_c"],
            )
            T[k, i] = temp_step(T[k - 1, i], Pin, Pcool, Cth[i], p["dt"])

        if np.any(active) and gate_ok:
            _, B_mag = superposed_B_at_target(
                array.coils, w_powers, T[k - 1], array.target,
            )
            B_target[k] = B_mag

        last_k = k
        yield (
            "step", k, t[k],
            T[k].copy(), P[k].copy(),
            B_target[k], E_surface[k].copy(), gate_ok,
        )

    K = last_k + 1
    yield (
        "final",
        dict(t=t[:K], T=T[:K], P=P[:K], B_target=B_target[:K],
             E_surface=E_surface[:K], depth_gate=depth_gate[:K]),
    )
