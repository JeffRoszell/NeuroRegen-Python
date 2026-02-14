#!/usr/bin/env python3
"""
Run the multicoil deep-brain stimulation simulation.

Workflow
--------
1. Load ``config/multicoil.yaml`` (or a custom path).
2. Build the coil array and print a pre-flight safety summary.
3. Run the 30-minute pulsed simulation with:
   - per-coil thermal gating
   - depth-gate safety check every pulse
   - superposed B-field at the STN target
4. Generate plots:
   - temperature per coil
   - weighted power per coil
   - combined B at target vs. time
   - cortical E-field per coil vs. time (with safety limit)
5. Optionally (``--field-map``): 2-D contour of superposed |B| at target
   depth showing the focal-point convergence.

Usage
-----
    python scripts/run_multicoil.py
    python scripts/run_multicoil.py -c path/to/custom.yaml
    python scripts/run_multicoil.py --field-map
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.neuroregen.config_loader import load_multicoil_config, build_multicoil_objects
from src.neuroregen.multicoil import (
    run_multicoil_simulation,
    superposed_B_on_grid,
)
from src.neuroregen.coil import c_to_f


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _colours(n: int) -> list[str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def plot_multicoil_results(
    results: dict,
    array,
    config: dict,
    output_dir: str,
    show: bool = True,
):
    """Create and save 4-panel multicoil result plots."""
    import matplotlib.pyplot as plt

    t_min = results["t"] / 60
    nc = results["T"].shape[1]
    names = [c.name for c in array.coils]
    colors = _colours(nc)
    limit_f = config["temp_limit_f"]
    resume_f = limit_f - config["hyst_f"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Temperature per coil ----
    ax = axes[0, 0]
    for i in range(nc):
        ax.plot(t_min, c_to_f(results["T"][:, i]),
                label=f"Coil {names[i]}", color=colors[i], linewidth=1.5)
    ax.axhline(limit_f, ls="--", color="red", linewidth=1.5, label=f"Limit {limit_f}°F")
    ax.axhline(resume_f, ls=":", color="orange", linewidth=1, label=f"Resume {resume_f:.1f}°F")
    ax.set_ylabel("Temperature (°F)")
    ax.set_title("Per-Coil Temperature", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Weighted power per coil ----
    ax = axes[0, 1]
    for i in range(nc):
        ax.step(t_min, results["P"][:, i],
                where="post", label=f"Coil {names[i]}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("Weighted Power (W)")
    ax.set_title("Per-Coil Weighted Power (distance-compensated)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Combined B at target ----
    ax = axes[1, 0]
    ax.plot(t_min, results["B_target"] * 1e4, color="#2ca02c", linewidth=1.5)
    b_thresh = config.get("b_threshold_t", 1e-4)
    ax.axhline(b_thresh * 1e4, ls="--", color="red", linewidth=1.5,
               label=f"Threshold {b_thresh*1e4:.2f} mT")
    ax.set_ylabel("|B| at target (mT)")
    ax.set_xlabel("Time (min)")
    ax.set_title(
        f"Superposed B-Field at {array.target.name}", fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Cortical E per coil ----
    ax = axes[1, 1]
    for i in range(nc):
        ax.plot(t_min, results["E_surface"][:, i],
                label=f"Coil {names[i]}", color=colors[i], linewidth=1.5)
    ax.axhline(config["cortical_max_vm"], ls="--", color="red", linewidth=2,
               label=f"Safety limit {config['cortical_max_vm']:.0f} V/m")
    ax.set_ylabel("Cortical E-field (V/m)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Cortical Surface E-Field (Depth Gate)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark depth-gate failures (if any)
    gate_fail = ~results["depth_gate"]
    if np.any(gate_fail):
        fail_t = t_min[gate_fail]
        for a in axes.flat:
            for ft in fail_t[::max(1, len(fail_t) // 20)]:
                a.axvline(ft, color="red", alpha=0.15, linewidth=0.5)

    plt.suptitle(
        f"Multicoil Stimulation — {nc} coils → {array.target.name}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "multicoil_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_field_focus(
    array,
    config: dict,
    output_dir: str,
    show: bool = True,
):
    """
    2-D contour of superposed |B| at the target depth.

    Shows how the three coils' fields converge at the focal point.
    """
    import matplotlib.pyplot as plt

    nc = array.n_coils
    target = array.target
    tz = target.position_m[2]

    # Create a grid in the X-Y plane at the target's Z depth
    span = 0.06  # ±6 cm
    tx, ty = target.position_m[0], target.position_m[1]
    nx, ny = 80, 80
    x = np.linspace(tx - span, tx + span, nx)
    y = np.linspace(ty - span, ty + span, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full_like(X, tz)

    temps = np.full(nc, config["t_amb_c"])

    print("  Computing superposed field at target depth (this may take a moment)...")
    B_mag = superposed_B_on_grid(
        array.coils, array.weighted_powers, temps, X, Y, Z,
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    B_mT = B_mag * 1e4

    vmax = B_mT.max()
    vmin = max(B_mT[B_mT > 0].min(), vmax * 1e-3) if np.any(B_mT > 0) else 1e-6
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 25)

    cs = ax.contourf(
        (X - tx) * 1000, (Y - ty) * 1000, B_mT,
        levels=levels, cmap="inferno",
    )
    plt.colorbar(cs, ax=ax, label="|B| (mT)")

    # Threshold contour
    b_thresh = config.get("b_threshold_t", 1e-4) * 1e4
    above = B_mT >= b_thresh
    if np.any(above):
        ax.contour(
            (X - tx) * 1000, (Y - ty) * 1000, above.astype(float),
            levels=[0.5], colors="cyan", linewidths=2, linestyles="--",
        )

    # Mark target
    ax.plot(0, 0, "w+", markersize=14, markeredgewidth=2)
    ax.annotate(
        target.name, (0, 0), textcoords="offset points",
        xytext=(10, 10), color="white", fontsize=11, fontweight="bold",
    )

    # Mark coil projections
    for coil in array.coils:
        cx = (coil.position_m[0] - tx) * 1000
        cy = (coil.position_m[1] - ty) * 1000
        if abs(cx) < span * 1000 and abs(cy) < span * 1000:
            ax.plot(cx, cy, "wo", markersize=8, markeredgewidth=1.5, fillstyle="none")
            ax.annotate(
                coil.name, (cx, cy), textcoords="offset points",
                xytext=(6, 6), color="white", fontsize=9,
            )

    ax.set_xlabel("X offset from target (mm)")
    ax.set_ylabel("Y offset from target (mm)")
    ax.set_title(
        f"Superposed |B| at target depth (z = {tz*100:.1f} cm)\n"
        f"Cyan contour = effective threshold ({b_thresh:.2f} mT)",
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, color="white")

    path = os.path.join(output_dir, "multicoil_field_focus.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_geometry(
    array,
    config: dict,
    output_dir: str,
    show: bool = True,
):
    """
    Interactive 3-D Plotly visualisation of the multicoil array.

    Layers (each toggleable in the legend):
      1. Skull — translucent mesh sphere
      2. Focal volume — isosurface of superposed |B| (top 50 % of peak)
      3. Coils — position markers + cone arrows for normals
      4. Target (STN) — red star-diamond marker
      5. Beam lines — dashed lines from each coil to the target

    The figure is saved as an interactive ``.html`` file (and a static
    ``.png`` if the ``kaleido`` package is installed).  When *show* is
    ``True`` it opens automatically in the default browser.
    """
    import plotly.graph_objects as go

    nc = array.n_coils
    target = array.target
    skull_r = config.get("skull_radius_m", 0.09)
    t_amb = config["t_amb_c"]
    temps = np.full(nc, t_amb)
    b_thresh = config.get("b_threshold_t", 1e-4)
    coil_colors = _colours(nc)

    # ---- 3-D field grid centred on the target ----------------------------
    span = 0.04  # ±4 cm
    npts = 35
    tx, ty, tz = target.position_m
    xv = np.linspace(tx - span, tx + span, npts)
    yv = np.linspace(ty - span, ty + span, npts)
    zv = np.linspace(tz - span, tz + span, npts)
    Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing="ij")

    # Mask to skull interior
    R_grid = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
    outside = R_grid > skull_r

    print("  Computing 3-D superposed field volume (this may take a moment)...")
    B_mag = superposed_B_on_grid(
        array.coils, array.weighted_powers, temps, Xg, Yg, Zg,
    )
    B_mag[outside] = 0.0  # zero out anything outside skull

    B_peak = B_mag.max()
    _, B_at_tgt = array.B_at_target(temps)

    traces: list[go.BaseTraceType] = []

    # ---- 1. Skull mesh sphere --------------------------------------------
    n_u, n_v = 30, 20
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    su = np.outer(np.cos(u), np.sin(v)).ravel() * skull_r * 100
    sv = np.outer(np.sin(u), np.sin(v)).ravel() * skull_r * 100
    sw = np.outer(np.ones(n_u), np.cos(v)).ravel() * skull_r * 100

    # Build triangle indices for the sphere mesh
    tri_i, tri_j, tri_k = [], [], []
    for ui in range(n_u - 1):
        for vi in range(n_v - 1):
            p0 = ui * n_v + vi
            p1 = p0 + 1
            p2 = (ui + 1) * n_v + vi
            p3 = p2 + 1
            tri_i += [p0, p0]
            tri_j += [p1, p2]
            tri_k += [p2, p3]

    traces.append(go.Mesh3d(
        x=su, y=sv, z=sw,
        i=tri_i, j=tri_j, k=tri_k,
        color="lightblue", opacity=0.07,
        name="Skull", legendgroup="skull", showlegend=True,
        hoverinfo="skip",
    ))

    # ---- 2. Focal field volume (coloured point cloud) ----------------------
    B_mT = B_mag * 1e4  # convert to mT for display
    B_at_tgt_mT = B_at_tgt * 1e4

    # Show points above 25 % of peak; subsample to keep it responsive
    floor_frac = 0.25
    floor_mT = B_peak * 1e4 * floor_frac
    mask = (B_mT >= floor_mT) & (~outside)

    if np.any(mask):
        xs = Xg[mask].ravel() * 100
        ys = Yg[mask].ravel() * 100
        zs = Zg[mask].ravel() * 100
        bs = B_mT[mask].ravel()

        # Cap at ~8 000 points for smooth interactivity
        max_pts = 8000
        if len(bs) > max_pts:
            idx = np.random.default_rng(42).choice(len(bs), max_pts, replace=False)
            xs, ys, zs, bs = xs[idx], ys[idx], zs[idx], bs[idx]

        # Marker size proportional to field strength (3–8 px range)
        sz = 3 + 5 * (bs - bs.min()) / (bs.max() - bs.min() + 1e-9)

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(
                size=sz,
                color=bs,
                colorscale="Hot",
                cmin=floor_mT,
                cmax=B_peak * 1e4,
                opacity=0.6,
                colorbar=dict(title="|B| (mT)", x=1.02),
            ),
            name=f"|B| field (>= {floor_mT:.1f} mT)",
            legendgroup="field",
            showlegend=True,
            hovertemplate=(
                "|B| = %{marker.color:.2f} mT<br>"
                "x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f} cm"
                "<extra></extra>"
            ),
        ))

    # ---- 3. Coils — markers + cone arrows --------------------------------
    arrow_len = 3.0  # cm
    for i, coil in enumerate(array.coils):
        cx, cy, cz = [p * 100 for p in coil.position_m]
        nx, ny, nz = coil.normal
        col = coil_colors[i]

        hover_text = (
            f"<b>Coil {coil.name}</b><br>"
            f"Distance to target: {array.distances[i]*100:.1f} cm<br>"
            f"cos(theta): {array.cosine_factors[i]:.3f}<br>"
            f"Weight: {array.weights[i]:.2f}<br>"
            f"Weighted power: {array.weighted_powers[i]:.1f} W"
        )

        # Coil position marker
        traces.append(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text",
            marker=dict(size=10, color=col, line=dict(width=2, color="black")),
            text=[coil.name],
            textposition="top center",
            textfont=dict(size=14, color=col, family="Arial Black"),
            name=f"Coil {coil.name}",
            legendgroup=f"coil_{coil.name}",
            showlegend=True,
            hovertext=hover_text,
            hoverinfo="text",
        ))

        # Normal arrow (cone)
        tip_x = cx + nx * arrow_len
        tip_y = cy + ny * arrow_len
        tip_z = cz + nz * arrow_len
        traces.append(go.Cone(
            x=[tip_x], y=[tip_y], z=[tip_z],
            u=[nx], v=[ny], w=[nz],
            sizemode="absolute", sizeref=1.5,
            anchor="tip", showscale=False,
            colorscale=[[0, col], [1, col]],
            name=f"Coil {coil.name} normal",
            legendgroup=f"coil_{coil.name}",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Shaft of the arrow (line from coil to cone tip)
        traces.append(go.Scatter3d(
            x=[cx, tip_x], y=[cy, tip_y], z=[cz, tip_z],
            mode="lines",
            line=dict(color=col, width=5),
            legendgroup=f"coil_{coil.name}",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- 4. Target marker ------------------------------------------------
    ttx, tty, ttz = [p * 100 for p in target.position_m]
    target_hover = (
        f"<b>{target.name}</b><br>"
        f"Position: ({ttx:.1f}, {tty:.1f}, {ttz:.1f}) cm<br>"
        f"Combined |B|: {B_at_tgt*1e4:.2f} mT"
    )
    traces.append(go.Scatter3d(
        x=[ttx], y=[tty], z=[ttz],
        mode="markers+text",
        marker=dict(
            size=12, color="red", symbol="diamond",
            line=dict(width=2, color="darkred"),
        ),
        text=[target.name],
        textposition="bottom center",
        textfont=dict(size=14, color="red", family="Arial Black"),
        name=target.name,
        legendgroup="target",
        showlegend=True,
        hovertext=target_hover,
        hoverinfo="text",
    ))

    # ---- 5. Beam lines (coil → target) -----------------------------------
    for i, coil in enumerate(array.coils):
        cx, cy, cz = [p * 100 for p in coil.position_m]
        col = coil_colors[i]
        traces.append(go.Scatter3d(
            x=[cx, ttx], y=[cy, tty], z=[cz, ttz],
            mode="lines",
            line=dict(color=col, width=3, dash="dash"),
            name=f"{coil.name} → target",
            legendgroup=f"beam_{coil.name}",
            showlegend=True,
            hoverinfo="skip",
        ))

    # ---- Layout ----------------------------------------------------------
    lim = skull_r * 100 * 1.15
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"Multicoil Array — {nc} coils → {target.name}<br>"
                f"<sup>Skull Ø {skull_r*200:.0f} cm  |  "
                f"Combined B at target: {B_at_tgt*1e4:.2f} mT</sup>"
            ),
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(title="X (cm)  left(-) / right(+)", range=[-lim, lim]),
            yaxis=dict(title="Y (cm)  post(-) / ant(+)", range=[-lim, lim]),
            zaxis=dict(title="Z (cm)  inf(-) / sup(+)", range=[-lim, lim]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
        ),
        legend=dict(
            yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1100,
        height=850,
    )

    # ---- Save & show -----------------------------------------------------
    html_path = os.path.join(output_dir, "multicoil_3d.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"  Saved: {html_path}")

    # Optional static PNG (requires kaleido)
    try:
        png_path = os.path.join(output_dir, "multicoil_3d.png")
        fig.write_image(png_path, width=1100, height=850, scale=2)
        print(f"  Saved: {png_path}")
    except Exception:
        pass  # kaleido not installed — HTML is sufficient

    if show:
        fig.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run NeuroRegen multicoil deep-brain stimulation simulation.",
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Path to multicoil YAML config (default: config/multicoil.yaml)",
    )
    parser.add_argument(
        "--field-map", action="store_true",
        help="Generate 2-D superposed field focus plot at target depth",
    )
    parser.add_argument(
        "--3d", dest="plot_3d", action="store_true",
        help="Generate 3-D visualisation of coil array, skull, and field volume",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots but don't display them (for headless / CI runs)",
    )
    args = parser.parse_args()
    show = not args.no_show

    # ---- load config ----
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(ROOT, "config", "multicoil.yaml")
    config = load_multicoil_config(config_path)
    array, config = build_multicoil_objects(config)

    # ---- pre-flight summary ----
    temps_ambient = np.full(array.n_coils, config["t_amb_c"])
    print(array.summary(temps_ambient, config["pulse_width"]))

    gate = array.check_safety(temps_ambient, config["pulse_width"])
    if not gate.safe:
        print("\n  WARNING: Depth Gate FAILED at ambient temperature!")
        print("  The configured powers are too high for the cortical safety limit.")
        print("  Reduce pulse_power_w or increase scalp_to_cortex_m in the config.")
        print("  Simulation will proceed — pulses will be blocked by the gate.\n")

    # ---- run simulation ----
    print("\nRunning multicoil simulation...")
    print(f"  {array.n_coils} coils, {config['sim_time']:.0f}s, "
          f"dt={config['dt']}s, {config['pulse_freq']} Hz, "
          f"pulse_width={config['pulse_width']*1000:.0f} ms")
    results = run_multicoil_simulation(array, config)
    print("  Done.")

    # ---- statistics ----
    gate_failures = int(np.sum(~results["depth_gate"]))
    total_steps = len(results["t"])
    B_peak = results["B_target"].max()
    E_peak = results["E_surface"].max()
    print(f"\n  Peak |B| at target    : {B_peak*1e4:.4f} mT")
    print(f"  Peak cortical E-field : {E_peak:.2f} V/m")
    print(f"  Depth-gate failures   : {gate_failures}/{total_steps} steps "
          f"({gate_failures/total_steps*100:.1f}%)")

    # ---- plots ----
    output_dir = os.path.join(ROOT, "outputs", "multicoil")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating plots → {output_dir}/")

    plot_multicoil_results(results, array, config, output_dir, show=show)

    if args.field_map:
        plot_field_focus(array, config, output_dir, show=show)

    if args.plot_3d:
        plot_3d_geometry(array, config, output_dir, show=show)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
