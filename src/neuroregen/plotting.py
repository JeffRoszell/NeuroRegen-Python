"""
Build and save the three standard plots (temperature, power, depth).
"""

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .constants import TEMP_LIMIT_F, HYST_F, Z_MAX_M
from .coil import c_to_f


def plot_and_save(
    t,
    T,
    P,
    depth,
    output_dir: str = "outputs",
    show: bool = True,
    temp_limit_f: float | None = None,
    hyst_f: float | None = None,
    z_max_m: float | None = None,
    animate: bool = False,
    animation_duration: float = 5.0,
):
    """
    Create temperature, power, and depth plots with safety limit indicators;
    save to output_dir; optionally show.
    t in s, T in °C [n,3], P in W [n,3], depth in cm [n,3].
    temp_limit_f: temperature limit (°F); if None, use constants default.
    hyst_f: hysteresis band (°F); if None, use constants default.
    z_max_m: depth limit (m); if None, use constants default.
    animate: if True, create animated plots that loop through the data
    animation_duration: duration of one animation loop in seconds (default: 5.0)
    """
    os.makedirs(output_dir, exist_ok=True)
    t_min = t / 60
    limit_f = TEMP_LIMIT_F if temp_limit_f is None else temp_limit_f
    hyst = HYST_F if hyst_f is None else hyst_f
    z_max = Z_MAX_M if z_max_m is None else z_max_m
    resume_f = limit_f - hyst

    if animate:
        _plot_animated(t, T, P, depth, output_dir, limit_f, resume_f, z_max, animation_duration, show)
    else:
        _plot_static(t_min, T, P, depth, output_dir, limit_f, resume_f, z_max, show)


def _plot_static(
    t_min, T, P, depth, output_dir, limit_f, resume_f, z_max, show
):
    """Create static plots (original behavior)."""
    axis_names = ["X", "Y", "Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for i, (ax, name, color) in enumerate(zip(axes, axis_names, colors)):
        ax.plot(t_min, c_to_f(T[:, i]), label=f"{name}-axis", linewidth=1.5, color=color)
        ax.axhspan(resume_f, limit_f, alpha=0.2, color="orange", label="Hysteresis band")
        ax.axhline(limit_f, ls="--", color="red", linewidth=2, label=f"Thermal limit: {limit_f}°F")
        ax.axhline(resume_f, ls=":", color="orange", linewidth=1.5, label=f"Resume: {resume_f}°F")
        ax.set_ylabel("Temperature (°F)", fontsize=11)
        ax.set_title(f"{name}-axis Temperature with Thermal Safety Limits", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature.png"), dpi=150)
    if not show:
        plt.close(fig)

    # Power plot - subplots for each axis
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i, (ax, name, color) in enumerate(zip(axes, axis_names, colors)):
        ax.step(t_min, P[:, i], label=f"{name} pulses", linewidth=1.5, color=color, where="post")
        ax.set_ylabel("Applied Power (W)", fontsize=11)
        ax.set_title(f"{name}-axis Power Pattern", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power.png"), dpi=150)
    if not show:
        plt.close(fig)

    # Depth plot with depth safety limit - subplots for each axis
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    depth_limit_cm = z_max * 100
    
    for i, (ax, name, color) in enumerate(zip(axes, axis_names, colors)):
        ax.plot(t_min, depth[:, i], label=f"{name} depth", linewidth=1.5, color=color)
        ax.axhline(depth_limit_cm, ls="--", color="red", linewidth=2, label=f"Depth limit: {depth_limit_cm:.1f} cm")
        ax.fill_between(t_min, depth_limit_cm, depth_limit_cm + 0.1, alpha=0.2, color="red", label="Safety limit zone")
        ax.set_ylabel("Depth (cm)", fontsize=11)
        ax.set_title(f"{name}-axis Penetration Depth with Safety Limit", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, depth_limit_cm * 1.1)
    
    axes[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "depth.png"), dpi=150)
    if not show:
        plt.close(fig)

    if show:
        plt.show()
    else:
        plt.close("all")


def _plot_animated(
    t, T, P, depth, output_dir, limit_f, resume_f, z_max, animation_duration, show
):
    """Create animated plots that loop through the simulation data."""
    t_min = t / 60
    n = len(t)
    axis_names = ["X", "Y", "Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    depth_limit_cm = z_max * 100
    
    # Calculate frame interval (ms) to complete loop in animation_duration seconds
    interval_ms = (animation_duration * 1000) / n
    
    # Temperature animation
    fig_temp, axes_temp = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    temp_lines = []
    for i, (ax, name, color) in enumerate(zip(axes_temp, axis_names, colors)):
        ax.axhspan(resume_f, limit_f, alpha=0.2, color="orange", label="Hysteresis band")
        ax.axhline(limit_f, ls="--", color="red", linewidth=2, label=f"Thermal limit: {limit_f}°F")
        ax.axhline(resume_f, ls=":", color="orange", linewidth=1.5, label=f"Resume: {resume_f}°F")
        line, = ax.plot([], [], label=f"{name}-axis", linewidth=1.5, color=color)
        temp_lines.append(line)
        ax.set_ylabel("Temperature (°F)", fontsize=11)
        ax.set_title(f"{name}-axis Temperature with Thermal Safety Limits", fontsize=12, fontweight="bold")
        ax.set_xlim(0, t_min[-1])
        ax.set_ylim(c_to_f(T.min()) - 2, c_to_f(T.max()) + 2)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes_temp[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    
    def update_temp(frame):
        idx = frame % n
        for i, line in enumerate(temp_lines):
            line.set_data(t_min[:idx+1], c_to_f(T[:idx+1, i]))
        return temp_lines
    
    anim_temp = FuncAnimation(fig_temp, update_temp, frames=n, interval=interval_ms, blit=True, repeat=True)
    
    # Power animation
    fig_power, axes_power = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    power_lines = []
    for i, (ax, name, color) in enumerate(zip(axes_power, axis_names, colors)):
        line, = ax.plot([], [], label=f"{name} pulses", linewidth=1.5, color=color, drawstyle="steps-post")
        power_lines.append(line)
        ax.set_ylabel("Applied Power (W)", fontsize=11)
        ax.set_title(f"{name}-axis Power Pattern", fontsize=12, fontweight="bold")
        ax.set_xlim(0, t_min[-1])
        ax.set_ylim(0, P.max() * 1.1)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes_power[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    
    def update_power(frame):
        idx = frame % n
        for i, line in enumerate(power_lines):
            line.set_data(t_min[:idx+1], P[:idx+1, i])
        return power_lines
    
    anim_power = FuncAnimation(fig_power, update_power, frames=n, interval=interval_ms, blit=True, repeat=True)
    
    # Depth animation
    fig_depth, axes_depth = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    depth_lines = []
    for i, (ax, name, color) in enumerate(zip(axes_depth, axis_names, colors)):
        ax.axhline(depth_limit_cm, ls="--", color="red", linewidth=2, label=f"Depth limit: {depth_limit_cm:.1f} cm")
        ax.fill_between(t_min, depth_limit_cm, depth_limit_cm + 0.1, alpha=0.2, color="red", label="Safety limit zone")
        line, = ax.plot([], [], label=f"{name} depth", linewidth=1.5, color=color)
        depth_lines.append(line)
        ax.set_ylabel("Depth (cm)", fontsize=11)
        ax.set_title(f"{name}-axis Penetration Depth with Safety Limit", fontsize=12, fontweight="bold")
        ax.set_xlim(0, t_min[-1])
        ax.set_ylim(0, depth_limit_cm * 1.1)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes_depth[-1].set_xlabel("Time (min)", fontsize=12)
    plt.tight_layout()
    
    def update_depth(frame):
        idx = frame % n
        for i, line in enumerate(depth_lines):
            line.set_data(t_min[:idx+1], depth[:idx+1, i])
        return depth_lines
    
    anim_depth = FuncAnimation(fig_depth, update_depth, frames=n, interval=interval_ms, blit=True, repeat=True)
    
    if show:
        plt.show()
    else:
        # Save static snapshots
        for i, line in enumerate(temp_lines):
            line.set_data(t_min, c_to_f(T[:, i]))
        plt.savefig(os.path.join(output_dir, "temperature.png"), dpi=150)
        plt.close(fig_temp)
        
        for i, line in enumerate(power_lines):
            line.set_data(t_min, P[:, i])
        plt.savefig(os.path.join(output_dir, "power.png"), dpi=150)
        plt.close(fig_power)
        
        for i, line in enumerate(depth_lines):
            line.set_data(t_min, depth[:, i])
        plt.savefig(os.path.join(output_dir, "depth.png"), dpi=150)
        plt.close(fig_depth)
