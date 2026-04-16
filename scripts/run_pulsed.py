#!/usr/bin/env python3
"""
Capacitor-discharge pulsed TMS simulation.

Loads config/pulsed_tms.yaml (3-coil / 6-turn / 150 mm / 10 mm wire,
C = 200 µF charged to 798 V) and prints a full physics summary plus
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
    parser = argparse.ArgumentParser(description="Pulsed TMS capacitor-discharge simulation.")
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
    parser.add_argument(
        "--prefix",
        default="pulsed",
        help="Output filename prefix for plots (default: 'pulsed')",
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
    print(f"  Target : {target.name}  at {tuple(f'{v * 100:.1f} cm' for v in target.position_m)}")
    print(
        f"  Coils  : {n_coils}  ×  {coils[0].turns}-turn / "
        f"{coils[0].loop_mm:.0f} mm loop / "
        f"{coils[0].wire_mm:.0f} mm wire"
    )
    print(f"  Capacitor: C = {C * 1e6:.0f} µF  charged to  V = {V:.0f} V")

    # -----------------------------------------------------------------------
    _print_section("Per-Coil Discharge Parameters")

    taus, I_peaks, E_pulses, Q_pulses, R_ohms = [], [], [], [], []
    for coil in coils:
        R_m = coil.loop_mm / 2000.0  # loop radius (m)
        a_m = coil.wire_mm / 2000.0  # wire radius (m)
        L = coil_inductance(R_m, coil.turns, a_m)
        d = discharge_params(C, V, L)
        geom = coil_geom(coil.to_axis())
        R_wire, L_wire, A_wire, S_surf, mass = geom
        R_ohm = resistance(L_wire, A_wire, config["t_amb_c"])
        Q = heat_per_pulse(R_ohm, d["I_peak_A"], d["tau_s"])

        taus.append(d["tau_s"])
        I_peaks.append(d["I_peak_A"])
        E_pulses.append(d["E_pulse_J"])
        Q_pulses.append(Q)
        R_ohms.append(R_ohm)

        print(f"\n  Coil {coil.name}:")
        print(f"    Inductance   L = {L * 1e6:.2f} µH")
        print(f"    Peak current I = {d['I_peak_A']:,.0f} A")
        print(f"    Pulse width  τ = {d['tau_s'] * 1e6:.0f} µs")
        print(f"    Energy/pulse E = {d['E_pulse_J']:.1f} J")
        print(f"    Coil resist  R = {R_ohm * 1e3:.2f} mΩ")
        print(f"    Heat/pulse   Q = {Q:.2f} J  (ΔT ≈ {Q / (mass * 385):.2f} °C/pulse)")

    # -----------------------------------------------------------------------
    _print_section("Wall-Power Budget")

    E_total = sum(E_pulses)
    f_max = max_rep_rate(E_total, 2200.0, efficiency=0.85)
    print(f"  Energy per event (all coils): {E_total:.0f} J")
    print("  Supply limit: 110 V × 20 A = 2 200 W  (η = 85%)")
    print(
        f"  Maximum safe repetition rate: {f_max:.3f} Hz  "
        f"({'EXCEEDS' if config['pulse_freq'] > f_max else 'within'} "
        f"config {config['pulse_freq']:.1f} Hz)"
    )

    # -----------------------------------------------------------------------
    _print_section("B-Field at Target (Superposed)")

    print("  Running field calculation...")
    from src.neuroregen.pulsed import superposed_B_peak_at

    B_vec, B_mag = superposed_B_peak_at(coils, I_peaks, target)
    print(
        f"  B vector at STN: [{B_vec[0] * 1000:.2f}, {B_vec[1] * 1000:.2f}, {B_vec[2] * 1000:.2f}] mT"
    )
    print(f"  |B| at STN     : {B_mag * 1000:.2f} mT  ({B_mag * 1e4:.2f} Gauss)")
    print()
    print("  Context:")
    print("    Clinical TMS (cortex, 0 depth) :  100–200 mT peak")
    print("    DBS electrode (implanted)       :  ~0.1–1 mT")
    print(f"    This system at STN (~4 cm deep) :  {B_mag * 1000:.2f} mT")

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
        status = (
            f"*** EXCEEDS {limit:.0f} V/m limit ***"
            if exceeded
            else f"within {limit:.0f} V/m limit"
        )
        print(
            f"  Coil {coil.name}: B_cortex = {B_cort * 1000:.1f} mT  →  "
            f"E_cortex = {E_cort:.1f} V/m  [{status}]"
        )

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
        print(
            f"    Coil {coil.name}: start {result.T_c[0, i]:.2f} °C  →  "
            f"final {T_final[i]:.2f} °C  (peak {T_peak[i]:.2f} °C)"
        )

    print(f"\n  Heat per pulse per coil: {result.Q_pulse_J}")

    # -----------------------------------------------------------------------
    print("\n" + SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Peak |B| at STN ({target.name}): {B_mag * 1000:.2f} mT")
    print(
        f"  Induced E at STN (r = {r_tissue * 1000:.0f} mm):  "
        f"{E_induced_peak(B_mag, r_tissue, tau_mean):.2f} V/m"
    )
    print(f"  Maximum safe rep rate:  {f_max:.3f} Hz  (110 V / 20 A supply)")
    print(f"  Coil temp after {config['n_pulses']} pulses:  {T_final.max():.2f} °C")
    print(SEP)

    # -----------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            out_dir = os.path.join(ROOT, "outputs")
            os.makedirs(out_dir, exist_ok=True)

            base_colors = [
                "#e05c5c",
                "#5c8fe0",
                "#5cbe7a",
                "#e0a85c",
                "#9b59b6",
                "#1abc9c",
                "#e67e22",
                "#34495e",
                "#c0392b",
                "#2980b9",
                "#7f8c8d",
                "#f39c12",
            ]
            coil_colors = (base_colors * ((len(coils) // len(base_colors)) + 1))[: len(coils)]

            # ---------------------------------------------------------------
            # Figure 1: Voltage sweep — E_cortex and E_STN vs charge voltage
            # ---------------------------------------------------------------
            fig1, ax1 = plt.subplots(figsize=(10, 6))

            voltages = np.linspace(100, float(V), 300)
            L_ref = coil_inductance(
                coils[0].loop_mm / 2000.0, coils[0].turns, coils[0].wire_mm / 2000.0
            )
            R_m_ref = coils[0].loop_mm / 2000.0

            e_cortex_vals = []
            e_stn_5mm = []
            e_stn_10mm = []
            e_stn_20mm = []
            b_stn_vals = []

            for v in voltages:
                d = discharge_params(C, v, L_ref)
                ip = d["I_peak_A"]
                tau = d["tau_s"]
                b_cort = B_peak_on_axis(ip, R_m_ref, config["scalp_to_cortex_m"], coils[0].turns)
                e_cortex_vals.append(E_induced_peak(b_cort, R_m_ref, tau))
                _, b_stn = superposed_B_peak_at(coils, [ip] * len(coils), target)
                b_stn_vals.append(b_stn)
                e_stn_5mm.append(E_induced_peak(b_stn, 0.005, tau))
                e_stn_10mm.append(E_induced_peak(b_stn, 0.010, tau))
                e_stn_20mm.append(E_induced_peak(b_stn, 0.020, tau))

            # Left axis: cortex E-field
            ax1_r = ax1.twinx()

            ax1.plot(
                voltages,
                e_cortex_vals,
                color="#e05c5c",
                linewidth=2.5,
                label="Cortex E-field (coil A, left axis)",
            )
            ax1.axhline(
                150,
                color="#e05c5c",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="Safety limit 150 V/m (left)",
            )

            # Right axis: STN E-field — separate scale so the small values are readable
            ax1_r.plot(
                voltages,
                e_stn_5mm,
                color="#5c8fe0",
                linewidth=2,
                linestyle="-",
                label="STN E-field r=5 mm (right)",
            )
            ax1_r.plot(
                voltages,
                e_stn_10mm,
                color="#5cbe7a",
                linewidth=2,
                linestyle="--",
                label="STN E-field r=10 mm (right)",
            )
            ax1_r.plot(
                voltages,
                e_stn_20mm,
                color="#e0a85c",
                linewidth=2,
                linestyle=":",
                label="STN E-field r=20 mm (right)",
            )
            ax1_r.axhline(
                10,
                color="green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Activation threshold ~10 V/m (right)",
            )
            ax1_r.set_ylabel("Induced E-field at STN (V/m)", color="#5c8fe0")
            ax1_r.tick_params(axis="y", labelcolor="#5c8fe0")
            ax1_r.set_ylim(bottom=0)

            # Crossover voltage
            import bisect

            cross_idx = bisect.bisect_left(e_cortex_vals, 150.0)
            if 0 < cross_idx < len(voltages):
                v_cross = voltages[cross_idx]
                ax1.axvline(v_cross, color="gray", linestyle=":", linewidth=1.5)
                ax1.annotate(
                    f"Cortex limit ≈ {v_cross:.0f} V",
                    xy=(v_cross, 150),
                    xytext=(v_cross - 180, 120),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=9,
                    color="gray",
                )

            ax1.set_xlabel("Charge Voltage (V)")
            ax1.set_ylabel("Induced E-field at Cortex (V/m)", color="#e05c5c")
            ax1.tick_params(axis="y", labelcolor="#e05c5c")
            ax1.set_title(
                f"Pulsed TMS — E-Field vs Charge Voltage\n"
                f"({n_coils}-coil superposition at STN | cortex = single coil)"
            )
            ax1.set_ylim(bottom=0)

            # Combined legend from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_r.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
            ax1.grid(True, alpha=0.3)

            fig1.tight_layout()
            path1 = os.path.join(out_dir, f"{args.prefix}_voltage_sweep.png")
            fig1.savefig(path1, dpi=150)
            print(f"\n  Plot saved → {path1}")
            plt.close(fig1)

            # ---------------------------------------------------------------
            # Figure 2: Field decay with depth (per coil + superposed)
            # ---------------------------------------------------------------
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

            depths = np.linspace(0.001, 0.14, 400)
            tau_ref = float(np.mean(taus))
            ip_mean = float(np.mean(I_peaks))

            # Per-coil depth curves — collapse to a single gray group for readability
            _first_indiv = True
            for i, coil in enumerate(coils):
                R_ci = coil.loop_mm / 2000.0
                b_vals = [B_peak_on_axis(ip_mean, R_ci, z, coil.turns) * 1000 for z in depths]
                ax2a.plot(
                    depths * 100,
                    b_vals,
                    color="gray",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.45,
                    label=f"Individual coils (×{n_coils})" if _first_indiv else "_nolegend_",
                )
                _first_indiv = False

            # Superposed B along axis from skull centroid toward STN target
            from src.neuroregen.pulsed import superposed_B_peak_at as _sbp

            stn_pos = np.array(target.position_m)
            # Centroid of all coil positions
            coil_centroid = np.mean([np.array(c.position_m) for c in coils], axis=0)
            axis_dir = stn_pos - coil_centroid
            axis_dir = axis_dir / np.linalg.norm(axis_dir)
            b_super = []
            for z in depths:
                probe_pos = coil_centroid + axis_dir * z
                _, bm = _sbp(coils, I_peaks, Target(name="probe", position_m=probe_pos))
                b_super.append(bm * 1000)
            # Mark actual STN distance along this axis
            stn_dist = float(np.linalg.norm(stn_pos - coil_centroid))
            ax2a.plot(
                depths * 100,
                b_super,
                color="black",
                linewidth=2.5,
                label="Superposed |B| (centroid→STN axis)",
            )
            ax2a.axvline(
                stn_dist * 100,
                color="#5c8fe0",
                linestyle=":",
                linewidth=1.5,
                label=f"STN ({stn_dist * 100:.1f} cm along axis)",
            )

            cortex_d_cm = config["scalp_to_cortex_m"] * 100
            ax2a.axvline(
                cortex_d_cm,
                color="#e05c5c",
                linestyle=":",
                linewidth=1.5,
                label=f"Cortex ({cortex_d_cm:.1f} cm)",
            )
            ax2a.set_xlabel("Depth from Coil Surface (cm)")
            ax2a.set_ylabel("|B| (mT)")
            ax2a.set_title("B-Field Decay with Depth")
            ax2a.legend(fontsize=9)
            ax2a.grid(True, alpha=0.3)
            ax2a.set_yscale("log")

            # E-field at depth for multiple tissue radii
            for r_mm, ls in [(5, "-"), (10, "--"), (20, ":")]:
                e_vals = [E_induced_peak(b / 1000, r_mm / 1000, tau_ref) for b in b_super]
                ax2b.plot(
                    depths * 100,
                    e_vals,
                    color="#5c8fe0",
                    linewidth=2,
                    linestyle=ls,
                    label=f"r = {r_mm} mm",
                )

            ax2b.axhline(
                150, color="#e05c5c", linestyle="--", linewidth=1.2, label="Safety limit (150 V/m)"
            )
            ax2b.axhline(
                10,
                color="green",
                linestyle="--",
                linewidth=1.2,
                label="Activation threshold (~10 V/m)",
            )
            ax2b.axvline(cortex_d_cm, color="#e05c5c", linestyle=":", linewidth=1.5)
            ax2b.axvline(
                stn_dist * 100,
                color="#5c8fe0",
                linestyle=":",
                linewidth=1.5,
                label=f"STN ({stn_dist * 100:.1f} cm)",
            )
            ax2b.set_xlabel("Depth from Coil Surface (cm)")
            ax2b.set_ylabel("Induced E-field (V/m)")
            ax2b.set_title("Induced E-Field vs Depth (Superposed)")
            ax2b.legend(fontsize=9)
            ax2b.grid(True, alpha=0.3)
            ax2b.set_yscale("log")

            fig2.suptitle(
                f"Field Decay Profile — V = {V:.0f} V | I_peak (mean) = {ip_mean:,.0f} A | {n_coils} coils",
                fontsize=11,
            )
            fig2.tight_layout()
            path2 = os.path.join(out_dir, f"{args.prefix}_depth_profile.png")
            fig2.savefig(path2, dpi=150)
            print(f"  Plot saved → {path2}")
            plt.close(fig2)

            # ---------------------------------------------------------------
            # Figure 3: LC discharge waveform + thermal + 3-D geometry
            # ---------------------------------------------------------------
            fig3 = plt.figure(figsize=(14, 9))
            gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.4, wspace=0.35)

            # (0,0) LC discharge waveform
            ax3a = fig3.add_subplot(gs[0, 0])
            tau_us = tau_mean * 1e6
            t_wf = np.linspace(0, tau_mean, 300)
            i_wf = I_peaks[0] * np.sin(np.pi * t_wf / tau_mean)
            ax3a.fill_between(t_wf * 1e6, i_wf / 1000, alpha=0.25, color="#5c8fe0")
            ax3a.plot(t_wf * 1e6, i_wf / 1000, color="#5c8fe0", linewidth=2)
            ax3a.set_xlabel("Time (µs)")
            ax3a.set_ylabel("Current (kA)")
            i_mean_ka = float(np.mean(I_peaks)) / 1000
            ax3a.set_title(
                f"LC Discharge Waveform\nT/2 = {tau_us:.0f} µs  |  I_peak = {i_mean_ka:.1f} kA"
            )
            ax3a.grid(True, alpha=0.3)

            # (0,1) Coil temperature vs time — tight y-axis so small rise is visible
            ax3b = fig3.add_subplot(gs[0, 1])
            t_sim = result.t_s
            _first_coil = True
            for i, coil in enumerate(coils):
                lbl = (
                    f"Coil {coil.name}"
                    if n_coils <= 5
                    else (
                        f"Coil {coil.name}" if _first_coil or i == len(coils) - 1 else "_nolegend_"
                    )
                )
                ax3b.plot(t_sim, result.T_c[:, i], color=coil_colors[i], linewidth=2, label=lbl)
                _first_coil = False
            T_all = result.T_c
            t_min, t_max = T_all.min(), T_all.max()
            delta_T = t_max - t_min
            pad = max(delta_T * 0.2, 0.05)
            ax3b.set_ylim(t_min - pad, t_max + pad)
            delta_label = f"ΔT = {delta_T:.3f} °C" if delta_T < 1 else f"ΔT = {delta_T:.2f} °C"
            ax3b.annotate(
                delta_label,
                xy=(0.97, 0.05),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="darkred", alpha=0.8),
            )
            ax3b.set_xlabel("Time (s)")
            ax3b.set_ylabel("Temperature (°C)")
            ax3b.set_title("Coil Temperature vs Time")
            ax3b.legend(fontsize=8)
            ax3b.grid(True, alpha=0.3)

            # (0,2) E-field at STN vs time — autoscale y so signal is always visible
            ax3c = fig3.add_subplot(gs[0, 2])
            e_vals_time = result.E_target_vm
            e_max = float(np.max(e_vals_time))
            ax3c.plot(t_sim, e_vals_time, color="#5c8fe0", linewidth=2, label="E at STN")
            # Draw threshold lines only if they'd be in a reasonable range of the data
            y_top = max(e_max * 1.5, 12)
            if 10 <= y_top * 2:
                ax3c.axhline(
                    10, color="green", linestyle="--", linewidth=1.2, label="~10 V/m threshold"
                )
            if 100 <= y_top * 3:
                ax3c.axhline(
                    100,
                    color="#e05c5c",
                    linestyle="--",
                    linewidth=1.2,
                    label="~100 V/m (depolarize)",
                )
            ax3c.set_ylim(bottom=0, top=y_top)
            ax3c.annotate(
                f"Peak = {e_max:.2f} V/m",
                xy=(0.97, 0.92),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                color="#5c8fe0",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#5c8fe0", alpha=0.8),
            )
            ax3c.set_xlabel("Time (s)")
            ax3c.set_ylabel("E-field at STN (V/m)")
            ax3c.set_title("Induced E-Field at STN vs Time")
            ax3c.legend(fontsize=9)
            ax3c.grid(True, alpha=0.3)

            # (1,0-1) B vector at STN bar chart
            ax3d = fig3.add_subplot(gs[1, :2])
            components = ["Bx", "By", "Bz", "|B|"]
            b_vec, b_mag = superposed_B_peak_at(coils, I_peaks, target)
            values = [b_vec[0] * 1000, b_vec[1] * 1000, b_vec[2] * 1000, b_mag * 1000]
            bar_colors = ["#5c8fe0", "#5cbe7a", "#e0a85c", "#e05c5c"]
            bars = ax3d.bar(components, values, color=bar_colors, width=0.5, edgecolor="white")
            for bar, val in zip(bars, values):
                ax3d.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{val:.1f} mT",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax3d.set_ylabel("|B| component (mT)")
            ax3d.set_title("Superposed B-Field at STN (Vector Components)")
            ax3d.axhline(0, color="k", linewidth=0.8)
            ax3d.grid(True, alpha=0.3, axis="y")

            # (1,2) 3-D geometry
            ax3e = fig3.add_subplot(gs[1, 2], projection="3d")
            skull_r = 0.09
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            xs = skull_r * np.outer(np.cos(u), np.sin(v))
            ys = skull_r * np.outer(np.sin(u), np.sin(v))
            zs = skull_r * np.outer(np.ones_like(u), np.cos(v))
            ax3e.plot_surface(xs * 100, ys * 100, zs * 100, alpha=0.07, color="wheat")

            for i, coil in enumerate(coils):
                pos = np.array(coil.position_m) * 100
                nrm = np.array(coil.normal)
                nrm = nrm / np.linalg.norm(nrm)
                ax3e.scatter(*pos, color=coil_colors[i], s=60, zorder=5)
                ax3e.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    nrm[0] * 3,
                    nrm[1] * 3,
                    nrm[2] * 3,
                    color=coil_colors[i],
                    linewidth=1.5,
                    arrow_length_ratio=0.3,
                )
                ax3e.text(
                    pos[0], pos[1], pos[2] + 1, f" {coil.name}", fontsize=8, color=coil_colors[i]
                )

            stn = np.array(target.position_m) * 100
            ax3e.scatter(*stn, color="gold", s=100, marker="*", zorder=6, label="STN")
            ax3e.text(stn[0], stn[1], stn[2] - 2, "STN", fontsize=8, color="goldenrod")
            ax3e.set_xlabel("X (cm)")
            ax3e.set_ylabel("Y (cm)")
            ax3e.set_zlabel("Z (cm)")
            ax3e.set_title("Coil Geometry & STN Target")

            fig3.suptitle(
                f"Pulsed TMS Summary — {n_coils} coils | V = {V:.0f} V | "
                f"I_peak = {float(np.mean(I_peaks)) / 1000:.1f} kA | T/2 = {tau_mean * 1e6:.0f} µs",
                fontsize=12,
                fontweight="bold",
            )
            path3 = os.path.join(out_dir, f"{args.prefix}_summary.png")
            fig3.savefig(path3, dpi=150)
            print(f"  Plot saved → {path3}")
            plt.close(fig3)

            print(f"\n  All plots saved to {out_dir}/")

        except ImportError:
            print("\n  matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
