#!/usr/bin/env python3
"""
Capacitor-discharge pulsed TMS simulation.

Answers: "What level of B field does that produce at basal ganglia depth?"

Loads config/pulsed_tms.yaml (6-turn / 150 mm / 10 mm wire coils,
C = 200 µF charged to 2 500 V) and prints a full physics summary plus
thermal evolution over N pulses.

Usage:
    python scripts/run_pulsed.py
    python scripts/run_pulsed.py --config path/to/pulsed.yaml
    python scripts/run_pulsed.py --plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.neuroregen.config_loader import load_pulsed_config
from src.neuroregen.multicoil import Coil, Target
from src.neuroregen.pulsed import (
    B_peak_on_axis,
    E_induced_peak,
    coil_inductance,
    discharge_params,
    heat_per_pulse,
    max_rep_rate,
    run_pulsed_thermal_sim,
)
from src.neuroregen.coil import resistance, coil_geom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72


def _print_section(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * (len(title)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pulsed TMS capacitor-discharge simulation."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pulsed YAML config (default: config/pulsed_tms.yaml)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show thermal evolution plots after simulation",
    )
    args = parser.parse_args()

    # Load config
    cfg_path = args.config or os.path.join(ROOT, "config", "pulsed_tms.yaml")
    config = load_pulsed_config(cfg_path)

    # Build simulation objects
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

    C = config["capacitance_f"]
    V = config["charge_voltage_v"]
    n_coils = len(coils)

    # -----------------------------------------------------------------------
    print("\n" + SEP)
    print("  PULSED TMS  —  Capacitor-Discharge Physics Summary")
    print(SEP)
    print(f"  Target : {target.name}  at {tuple(f'{v*100:.1f} cm' for v in target.position_m)}")
    print(f"  Coils  : {n_coils}  ×  {coils[0].turns}-turn / "
          f"{coils[0].loop_mm:.0f} mm loop / "
          f"{coils[0].wire_mm:.0f} mm wire")
    print(f"  Capacitor: C = {C*1e6:.0f} µF  charged to  V = {V:.0f} V")

    # -----------------------------------------------------------------------
    _print_section("Per-Coil Discharge Parameters")

    Ls, taus, I_peaks, E_pulses, Q_pulses, R_ohms = [], [], [], [], [], []
    for coil in coils:
        R_m = coil.loop_mm / 2000.0        # loop radius (m)
        a_m = coil.wire_mm / 2000.0        # wire radius (m)
        L = coil_inductance(R_m, coil.turns, a_m)
        d = discharge_params(C, V, L)
        geom = coil_geom(coil.to_axis())
        R_wire, L_wire, A_wire, S_surf, mass = geom
        R_ohm = resistance(L_wire, A_wire, config["t_amb_c"])
        Q = heat_per_pulse(R_ohm, d["I_peak_A"], d["tau_s"])

        Ls.append(L)
        taus.append(d["tau_s"])
        I_peaks.append(d["I_peak_A"])
        E_pulses.append(d["E_pulse_J"])
        Q_pulses.append(Q)
        R_ohms.append(R_ohm)

        print(f"\n  Coil {coil.name}:")
        print(f"    Inductance   L = {L*1e6:.2f} µH")
        print(f"    Peak current I = {d['I_peak_A']:,.0f} A")
        print(f"    Pulse width  τ = {d['tau_s']*1e6:.0f} µs")
        print(f"    Energy/pulse E = {d['E_pulse_J']:.1f} J")
        print(f"    Coil resist  R = {R_ohm*1e3:.2f} mΩ")
        print(f"    Heat/pulse   Q = {Q:.2f} J  (ΔT ≈ {Q/(mass*385):.2f} °C/pulse)")

    # -----------------------------------------------------------------------
    _print_section("Wall-Power Budget")

    E_total = sum(E_pulses)
    f_max = max_rep_rate(E_total, 2200.0, efficiency=0.85)
    print(f"  Energy per event (all coils): {E_total:.0f} J")
    print(f"  Supply limit: 110 V × 20 A = 2 200 W  (η = 85%)")
    print(f"  Maximum safe repetition rate: {f_max:.3f} Hz  "
          f"({'EXCEEDS' if config['pulse_freq'] > f_max else 'within'} "
          f"config {config['pulse_freq']:.1f} Hz)")

    # -----------------------------------------------------------------------
    _print_section("B-Field at Target (Superposed)")

    print("  Running field calculation...")
    from src.neuroregen.pulsed import superposed_B_peak_at
    B_vec, B_mag = superposed_B_peak_at(coils, I_peaks, target)
    print(f"  B vector at STN: [{B_vec[0]*1000:.2f}, {B_vec[1]*1000:.2f}, {B_vec[2]*1000:.2f}] mT")
    print(f"  |B| at STN     : {B_mag*1000:.2f} mT  ({B_mag*1e4:.2f} Gauss)")
    print()
    print("  Context:")
    print(f"    Clinical TMS (cortex, 0 depth) :  100–200 mT peak")
    print(f"    DBS electrode (implanted)       :  ~0.1–1 mT")
    print(f"    This system at STN (~4 cm deep) :  {B_mag*1000:.2f} mT")

    # -----------------------------------------------------------------------
    _print_section("Induced E-Field at Target")

    tau_mean = float(np.mean(taus))
    r_tissue = config["r_tissue_m"]
    for r_mm, label in [(5, "5 mm (neuron scale)"), (10, "10 mm"), (20, "20 mm (region scale)")]:
        E = E_induced_peak(B_mag, r_mm / 1000.0, tau_mean)
        print(f"  r_tissue = {r_mm:2d} mm : E_target = {E:.2f} V/m")

    print()
    print("  Clinical threshold for neural activation: ~10–50 V/m")

    # -----------------------------------------------------------------------
    _print_section("Cortical E-Field Safety Check")

    scalp_d = config["scalp_to_cortex_m"]
    limit = config["cortical_max_vm"]
    any_exceeded = False
    for i, coil in enumerate(coils):
        R_m = coil.loop_mm / 2000.0
        B_cort = B_peak_on_axis(I_peaks[i], R_m, scalp_d, coil.turns)
        E_cort = E_induced_peak(B_cort, R_m, taus[i])
        exceeded = E_cort > limit
        if exceeded:
            any_exceeded = True
        status = f"*** EXCEEDS {limit:.0f} V/m limit ***" if exceeded else f"within {limit:.0f} V/m limit"
        print(f"  Coil {coil.name}: B_cortex = {B_cort*1000:.1f} mT  →  "
              f"E_cortex = {E_cort:.1f} V/m  [{status}]")

    if any_exceeded:
        print()
        print("  NOTE: At these currents the cortical E-field exceeds the safety")
        print("  gate threshold. The depth gate in run_multicoil.py would suppress")
        print("  all pulses. To operate, either lower V_charge, reduce turns, or")
        print("  accept that this is a single-pulse exploratory calculation only.")

    # -----------------------------------------------------------------------
    _print_section("Thermal Simulation")

    print(f"  Simulating {config['n_pulses']} pulses at {config['pulse_freq']:.1f} Hz...")
    result = run_pulsed_thermal_sim(coils, target, config)

    T_final = result.T_c[-1]
    T_peak = result.T_c.max(axis=0)
    print(f"  Temperature after {config['n_pulses']} pulses:")
    for i, coil in enumerate(coils):
        print(f"    Coil {coil.name}: start {result.T_c[0,i]:.2f} °C  →  "
              f"final {T_final[i]:.2f} °C  (peak {T_peak[i]:.2f} °C)")

    print(f"\n  Heat per pulse per coil: {result.Q_pulse_J}")

    # -----------------------------------------------------------------------
    print("\n" + SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Peak |B| at STN ({target.name}): {B_mag*1000:.2f} mT")
    print(f"  Induced E at STN (r = {r_tissue*1000:.0f} mm):  "
          f"{E_induced_peak(B_mag, r_tissue, tau_mean):.2f} V/m")
    print(f"  Maximum safe rep rate:  {f_max:.3f} Hz  (110 V / 20 A supply)")
    print(f"  Coil temp after {config['n_pulses']} pulses:  {T_final.max():.2f} °C")
    print(SEP)

    # -----------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes_plt = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            t = result.t_s
            for i, coil in enumerate(coils):
                axes_plt[0].plot(t, result.T_c[:, i], label=f"Coil {coil.name}")
            axes_plt[0].axhline(
                config["t_amb_c"], color="k", linestyle="--", linewidth=0.8, label="Ambient"
            )
            axes_plt[0].set_ylabel("Temperature (°C)")
            axes_plt[0].set_title("Pulsed TMS — Coil Temperature vs Time")
            axes_plt[0].legend()
            axes_plt[0].grid(True, alpha=0.3)

            axes_plt[1].plot(t, result.B_target_T * 1000, color="steelblue")
            axes_plt[1].set_ylabel("|B| at STN (mT)")
            axes_plt[1].set_xlabel("Time (s)")
            axes_plt[1].set_title("Peak B-field Magnitude at STN")
            axes_plt[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out_dir = os.path.join(ROOT, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "pulsed_thermal.png")
            plt.savefig(out_path, dpi=150)
            print(f"\n  Plot saved to {out_path}")
            plt.show()
        except ImportError:
            print("\n  matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
