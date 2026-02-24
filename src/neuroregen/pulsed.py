"""
Capacitor-discharge pulsed coil physics for deep-brain magnetic stimulation.

Physics model (underdamped LC discharge, coil resistance << sqrt(L/C)):
  - Peak current : I_peak = V_charge × sqrt(C / L)
  - Pulse width  : τ = π × sqrt(L × C)   (half-period of LC oscillation)
  - Waveform     : I(t) = I_peak × sin(π t / τ)   for 0 ≤ t ≤ τ
  - Energy/pulse : E_pulse = ½ C V²
  - Heat/pulse   : Q = R_ohm × I_peak² × τ / 2   (exact for sine waveform)

Wall-power constraint:
  - Total average power = sum(E_pulse_i) × f_rep ≤ P_supply × η
  - Maximum rep rate   : f_max = P_supply × η / E_pulse_total

Induced E-field at a point (Faraday's law, circular tissue path radius r):
  - dB/dt|_peak = π × B_peak / τ
  - E_peak = (r / 2) × dB/dt|_peak = r × π × B_peak / (2 τ)

Coil inductance (Neumann formula for a single circular loop):
  - L₁ = μ₀ R (ln(8R/a) − 2)   where R = loop radius, a = wire radius
  - N tightly-wound turns: L_N ≈ N² × L₁
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .constants import MU0, RHO_CU, DENSITY_CU, CP_CU
from .coil import Axis, coil_geom, resistance
from .multicoil import Coil, Target, B_field_at_point


# ---------------------------------------------------------------------------
# Coil inductance
# ---------------------------------------------------------------------------
def coil_inductance(R_m: float, N: int, wire_radius_m: float) -> float:
    """
    Self-inductance (H) of an N-turn circular coil (Neumann formula).

    Parameters
    ----------
    R_m : float
        Loop radius (m).
    N : int
        Number of turns.
    wire_radius_m : float
        Conductor radius (m).  Use half the wire/bundle diameter.

    Returns
    -------
    float
        Inductance in Henries.
    """
    L1 = MU0 * R_m * (np.log(8.0 * R_m / wire_radius_m) - 2.0)
    return float(N ** 2 * L1)


# ---------------------------------------------------------------------------
# Capacitor discharge
# ---------------------------------------------------------------------------
def discharge_params(C: float, V: float, L: float) -> dict:
    """
    Compute LC-discharge parameters for a single capacitor pulse.

    Parameters
    ----------
    C : float   Capacitance (F).
    V : float   Charge voltage (V).
    L : float   Coil inductance (H).

    Returns
    -------
    dict with keys:
        I_peak_A   – peak coil current (A)
        tau_s      – half-sine pulse width (s)
        E_pulse_J  – energy stored / delivered per pulse (J)
    """
    I_peak = V * np.sqrt(C / L)
    tau_s = np.pi * np.sqrt(L * C)
    E_pulse = 0.5 * C * V ** 2
    return dict(I_peak_A=float(I_peak), tau_s=float(tau_s), E_pulse_J=float(E_pulse))


def max_rep_rate(E_pulse_total_J: float, P_supply_w: float, efficiency: float = 0.85) -> float:
    """
    Maximum safe repetition rate (Hz) given total energy per pulse event and wall supply.

    Parameters
    ----------
    E_pulse_total_J : float
        Sum of energy across ALL coils per firing event (J).
    P_supply_w : float
        Available supply power (W).  110 V × 20 A = 2 200 W.
    efficiency : float
        Charger efficiency (default 0.85).

    Returns
    -------
    float   Maximum repetition rate (Hz).
    """
    return P_supply_w * efficiency / E_pulse_total_J


# ---------------------------------------------------------------------------
# Field & induced E
# ---------------------------------------------------------------------------
def B_peak_on_axis(I_peak: float, R_coil: float, z: float, N: int) -> float:
    """
    Peak on-axis B-field (T) at axial distance z from a circular N-turn coil.

    Uses the exact Biot-Savart on-axis formula.
    """
    return float(MU0 * I_peak * R_coil ** 2 * N / (2.0 * (R_coil ** 2 + z ** 2) ** 1.5))


def E_induced_peak(B_peak: float, r_tissue: float, tau_s: float) -> float:
    """
    Peak induced E-field (V/m) in a circular tissue path of radius *r_tissue*.

    From Faraday's law:
      E × 2πr = πr² × dB/dt  →  E = (r/2) × dB/dt
    Peak dB/dt for a half-sine pulse: π × B_peak / τ

    Parameters
    ----------
    B_peak    : float   Peak B-field magnitude (T).
    r_tissue  : float   Effective tissue loop radius (m).
    tau_s     : float   Pulse half-width (s).
    """
    return float(r_tissue * np.pi * B_peak / (2.0 * tau_s))


def heat_per_pulse(R_ohm: float, I_peak: float, tau_s: float) -> float:
    """
    Joule heat (J) deposited in coil resistance per pulse.

    For I(t) = I_peak sin(πt/τ):   ∫₀^τ I²R dt = R × I_peak² × τ/2
    """
    return float(R_ohm * I_peak ** 2 * tau_s / 2.0)


# ---------------------------------------------------------------------------
# Superposed field at the target (3-D vector sum)
# ---------------------------------------------------------------------------
def superposed_B_peak(
    coils: list[Coil],
    I_peaks: list[float],
) -> tuple[np.ndarray, float]:
    """
    Peak superposed B-field vector and magnitude (T) at each coil's target.

    Assumes all coils fire simultaneously with their respective peak currents.

    Parameters
    ----------
    coils   : list of Coil (carries position, normal, target via array.target)
    I_peaks : list of float – peak current per coil (A)

    Returns
    -------
    (B_vec, B_mag) : ndarray shape (3,), float
        B-field vector (T) and magnitude (T) at the first coil's target.
        (All coils in an array share the same target by convention.)
    """
    target_pos = coils[0].position_m   # placeholder; caller passes target explicitly


def superposed_B_peak_at(
    coils: list[Coil],
    I_peaks: list[float],
    target: Target,
    T_c: float = 22.0,
) -> tuple[np.ndarray, float]:
    """
    Peak superposed B-field vector and magnitude (T) at *target*.

    Parameters
    ----------
    coils   : list[Coil]
    I_peaks : list[float]  – peak current for each coil (A) at their respective V_charge
    target  : Target
    T_c     : float  – coil temperature for resistance calc (°C); default ambient

    Returns
    -------
    (B_vec, B_mag) – ndarray(3,), float
    """
    B_total = np.zeros(3)
    for coil, I in zip(coils, I_peaks):
        B_total += B_field_at_point(coil, I, target.position_m)
    return B_total, float(np.linalg.norm(B_total))


# ---------------------------------------------------------------------------
# Pulsed thermal simulation
# ---------------------------------------------------------------------------
@dataclass
class PulsedSimResult:
    """Results from :func:`run_pulsed_thermal_sim`."""
    t_s: np.ndarray            # time axis (s), one point per pulse
    T_c: np.ndarray            # coil temperature (°C) [n_pulses, n_coils]
    B_target_T: np.ndarray     # peak |B| at target per pulse (T)
    E_surface_vm: np.ndarray   # peak cortical E per coil per pulse (V/m) [n_pulses, n_coils]
    E_target_vm: np.ndarray    # peak induced E at target per pulse (V/m)
    Q_pulse_J: np.ndarray      # heat deposited per coil per pulse (J) [n_coils]
    I_peaks_A: np.ndarray      # peak current per coil (A) [n_coils]
    tau_s: float               # pulse half-width (s)
    discharge: dict            # discharge_params result (per coil, all identical if same V)


def run_pulsed_thermal_sim(
    coils: list[Coil],
    target: Target,
    config: dict,
) -> PulsedSimResult:
    """
    Simulate thermal evolution of pulsed coils over *n_pulses* firings.

    Each firing deposits heat Q = R × I_peak² × τ/2 instantaneously in each
    coil.  Between firings, Newton's-law cooling is applied in small time steps.

    Parameters
    ----------
    coils  : list[Coil]
    target : Target
    config : dict  (from :func:`load_pulsed_config`)

    Returns
    -------
    PulsedSimResult
    """
    nc = len(coils)
    C = config["capacitance_f"]
    V = config["charge_voltage_v"]
    f_rep = config["pulse_freq"]
    n_pulses = config["n_pulses"]
    dt = config["dt_between_pulses"]
    T_amb = config["t_amb_c"]
    h_conv = config["h_conv"]
    r_tissue = config["r_tissue_m"]
    scalp_d = config["scalp_to_cortex_m"]

    # Per-coil geometry
    axes = [c.to_axis() for c in coils]
    geoms = [coil_geom(a) for a in axes]
    R_loops = [g[0] for g in geoms]
    L_wires = [g[1] for g in geoms]
    A_wires = [g[2] for g in geoms]
    S_surfs = [g[3] for g in geoms]
    masses = [g[4] for g in geoms]

    # Inductance and discharge parameters per coil (all coils same geometry → same L, I_peak, τ)
    L_coils = [
        coil_inductance(
            coils[i].loop_mm / 2000.0,  # loop radius in m
            coils[i].turns,
            coils[i].wire_mm / 2000.0,  # wire radius in m
        )
        for i in range(nc)
    ]
    discharges = [discharge_params(C, V, L_coils[i]) for i in range(nc)]
    I_peaks = [d["I_peak_A"] for d in discharges]
    taus = [d["tau_s"] for d in discharges]
    E_pulses = [d["E_pulse_J"] for d in discharges]

    # Joule heat per pulse per coil (at ambient T; R variation negligible at these currents)
    R_ohms = [resistance(L_wires[i], A_wires[i], T_amb) for i in range(nc)]
    Q_pulses = np.array([heat_per_pulse(R_ohms[i], I_peaks[i], taus[i]) for i in range(nc)])

    # Thermal masses
    Cth = np.array([masses[i] * CP_CU for i in range(nc)])

    # Cortical E per coil (computed once at ambient T — doesn't change much thermally)
    E_surface = np.zeros(nc)
    for i, coil in enumerate(coils):
        B_cort = B_peak_on_axis(I_peaks[i], R_loops[i], scalp_d, coils[i].turns)
        E_surface[i] = E_induced_peak(B_cort, R_loops[i], taus[i])

    # Output arrays
    T_arr = np.zeros((n_pulses, nc))
    B_arr = np.zeros(n_pulses)
    E_target_arr = np.zeros(n_pulses)
    t_axis = np.zeros(n_pulses)

    T = np.full(nc, T_amb, dtype=float)
    dt_interval = 1.0 / f_rep
    n_cool = max(1, int(round(dt_interval / dt)))
    dt_actual = dt_interval / n_cool

    for p in range(n_pulses):
        t_axis[p] = p / f_rep

        # Instantaneous heating from pulse
        T += Q_pulses / Cth

        # Superposed B at target (using current temperature for resistance — minor effect)
        B_vec, B_mag = superposed_B_peak_at(coils, I_peaks, target, T_amb)
        B_arr[p] = B_mag

        # Induced E at target
        tau_mean = float(np.mean(taus))
        E_target_arr[p] = E_induced_peak(B_mag, r_tissue, tau_mean)

        T_arr[p] = T.copy()

        # Newton's-law cooling over the inter-pulse interval
        for _ in range(n_cool):
            Pcool = h_conv * np.array(S_surfs) * (T - T_amb)
            T -= Pcool * dt_actual / Cth

    return PulsedSimResult(
        t_s=t_axis,
        T_c=T_arr,
        B_target_T=B_arr,
        E_surface_vm=E_surface,
        E_target_vm=E_target_arr,
        Q_pulse_J=Q_pulses,
        I_peaks_A=np.array(I_peaks),
        tau_s=float(np.mean(taus)),
        discharge=discharges[0],   # representative (all coils same geometry)
    )
