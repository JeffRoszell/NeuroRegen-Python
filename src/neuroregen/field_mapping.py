"""
Spatial B-field mapping and visualization for coil targeting.

Calculates magnetic field strength at 3D spatial points for each axis,
creates contour plots showing field distribution, and identifies
effective stimulation regions (where B-field exceeds threshold).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

from .constants import MU0, B_THRESHOLD_T
from .coil import Axis, coil_geom, resistance, B_loop


def B_field_3d_loop(
    I: float,
    R: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    N: int,
    axis_orientation: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate B-field components (Bx, By, Bz) at 3D points for a circular loop coil.

    For a loop in the xy-plane (axis_orientation="z"), calculates field at points (x, y, z).
    Uses analytical approximation: on-axis uses exact formula, off-axis uses dipole
    approximation for points far from loop, and interpolates for intermediate regions.

    Parameters
    ----------
    I : float
        Current (A)
    R : float
        Loop radius (m)
    x, y, z : np.ndarray
        Spatial coordinates (m) - can be arrays or scalars
    N : int
        Number of turns
    axis_orientation : str
        "x", "y", or "z" - which axis the coil loop is perpendicular to.
        "z" means loop in xy-plane (default, coil along z-axis).

    Returns
    -------
    Bx, By, Bz : np.ndarray
        B-field components (T) at each point. Shape matches input arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if axis_orientation == "z":
        # Loop in xy-plane, field along z-axis
        # Distance from z-axis (radial)
        rho = np.sqrt(x**2 + y**2)
        # Use on-axis formula with distance scaling for off-axis points
        # This is approximate but reasonable for visualization
        z_eff = np.abs(z)
        # On-axis B magnitude
        B_on_axis = MU0 * I * R**2 / (2 * (R**2 + z_eff**2) ** 1.5) * N
        # Scale down for off-axis points: field decreases with radial distance
        # Use exponential decay factor based on distance from axis
        scale_factor = np.exp(-rho / (R * 2))  # Decay over ~2R
        B_mag = B_on_axis * scale_factor
        # Direction: primarily along z near axis, radial component grows off-axis
        Bz = B_mag * np.cos(np.arctan2(rho, z_eff + 1e-10))
        Bx = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, z_eff + 1e-10)) * x / rho, 0)
        By = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, z_eff + 1e-10)) * y / rho, 0)
        # Flip z component if z < 0
        Bz = np.where(z < 0, -Bz, Bz)
    elif axis_orientation == "x":
        # Loop in yz-plane, field along x-axis
        rho = np.sqrt(y**2 + z**2)
        x_eff = np.abs(x)
        B_on_axis = MU0 * I * R**2 / (2 * (R**2 + x_eff**2) ** 1.5) * N
        scale_factor = np.exp(-rho / (R * 2))
        B_mag = B_on_axis * scale_factor
        Bx = B_mag * np.cos(np.arctan2(rho, x_eff + 1e-10))
        By = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, x_eff + 1e-10)) * y / rho, 0)
        Bz = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, x_eff + 1e-10)) * z / rho, 0)
        Bx = np.where(x < 0, -Bx, Bx)
    else:  # axis_orientation == "y"
        # Loop in xz-plane, field along y-axis
        rho = np.sqrt(x**2 + z**2)
        y_eff = np.abs(y)
        B_on_axis = MU0 * I * R**2 / (2 * (R**2 + y_eff**2) ** 1.5) * N
        scale_factor = np.exp(-rho / (R * 2))
        B_mag = B_on_axis * scale_factor
        By = B_mag * np.cos(np.arctan2(rho, y_eff + 1e-10))
        Bx = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, y_eff + 1e-10)) * x / rho, 0)
        Bz = np.where(rho > 0, B_mag * np.sin(np.arctan2(rho, y_eff + 1e-10)) * z / rho, 0)
        By = np.where(y < 0, -By, By)

    return Bx, By, Bz


def B_magnitude_3d(
    I: float,
    R: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    N: int,
    axis_orientation: str = "z",
) -> np.ndarray:
    """
    Calculate B-field magnitude |B| at 3D points.

    Parameters
    ----------
    I : float
        Current (A)
    R : float
        Loop radius (m)
    x, y, z : np.ndarray
        Spatial coordinates (m)
    N : int
        Number of turns
    axis_orientation : str
        "x", "y", or "z" - coil axis orientation

    Returns
    -------
    B_mag : np.ndarray
        |B| (T) at each point. Shape matches input arrays.
    """
    Bx, By, Bz = B_field_3d_loop(I, R, x, y, z, N, axis_orientation)
    return np.sqrt(Bx**2 + By**2 + Bz**2)


def create_spatial_grid(
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    z_range: tuple[float, float] = (0, 0.03),
    nx: int = 50,
    ny: int = 50,
    nz: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D spatial grid for field mapping.

    Parameters
    ----------
    x_range, y_range, z_range : tuple[float, float]
        Min and max coordinates (m) for each axis
    nx, ny, nz : int
        Number of grid points along each axis

    Returns
    -------
    x, y, z : np.ndarray
        3D meshgrid arrays (m)
    grid_points : np.ndarray
        Flattened (N, 3) array of all grid points for vectorized calculation
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return X, Y, Z, grid_points


def calculate_field_map(
    axis: Axis,
    Pin: float,
    T_c: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    axis_orientation: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate B-field magnitude map for a given axis at specified power and temperature.

    Parameters
    ----------
    axis : Axis
        Coil axis configuration
    Pin : float
        Input power (W)
    T_c : float
        Coil temperature (°C)
    x, y, z : np.ndarray
        Spatial grid coordinates (m) - must be same shape
    axis_orientation : str, optional
        "x", "y", or "z". If None, inferred from axis.name.

    Returns
    -------
    B_mag : np.ndarray
        B-field magnitude (T) at each grid point. Same shape as x, y, z.
    """
    if Pin <= 0:
        return np.zeros_like(x)
    R, L, A, S, m = coil_geom(axis)
    R_ohm = resistance(L, A, T_c)
    I = np.sqrt(Pin / R_ohm)
    if axis_orientation is None:
        axis_orientation = axis.name.lower()
    return B_magnitude_3d(I, R, x, y, z, axis.turns, axis_orientation)


def plot_field_contours_2d(
    B_mag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_slice: float,
    axis_name: str,
    threshold: float = B_THRESHOLD_T,
    output_path: Optional[str] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Create 2D contour plot of B-field magnitude at a fixed z-slice.

    Shows field strength contours and highlights effective stimulation region
    (where B >= threshold). Uses fixed color scale (vmin/vmax) to show field
    degradation across depths.

    Parameters
    ----------
    B_mag : np.ndarray
        B-field magnitude (T) at grid points. Shape (nx, ny, nz).
    x, y : np.ndarray
        Grid coordinates (m) - 1D arrays
    z_slice : float
        Z-coordinate of the slice to plot (m)
    axis_name : str
        Name of the axis (for title/labels)
    threshold : float
        B-field threshold (T) for effective stimulation
    output_path : str, optional
        Path to save figure. If None, not saved.
    show : bool
        Whether to display the plot
    vmin, vmax : float, optional
        Color scale limits (T). If None, auto-scales to slice min/max.
        Use fixed values across slices to compare degradation.
    """
    # Find closest z-index
    z_coords = np.linspace(0, 0.03, B_mag.shape[2])
    z_idx = np.argmin(np.abs(z_coords - z_slice))
    B_slice = B_mag[:, :, z_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Contour levels - use fixed scale if provided, else auto-scale
    if vmin is None:
        vmin = B_slice[B_slice > 0].min() + 1e-6
    if vmax is None:
        vmax = B_slice.max() + 1e-6
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
    cs = ax.contourf(
        X * 1000, Y * 1000, B_slice, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(cs, ax=ax, label="B-field magnitude (T)")

    # Highlight threshold region
    threshold_region = B_slice >= threshold
    if np.any(threshold_region):
        ax.contour(
            X * 1000,
            Y * 1000,
            threshold_region.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=2,
            linestyles="--",
            label=f"Effective region (B ≥ {threshold*1e4:.2f} mT)",
        )

    ax.set_xlabel("X position (mm)", fontsize=12)
    ax.set_ylabel("Y position (mm)", fontsize=12)
    ax.set_title(
        f"{axis_name}-axis B-field at z = {z_slice*1000:.1f} mm\n"
        f"Effective stimulation region (B ≥ {threshold*1e4:.2f} mT)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_field_interactive_slice(
    B_mag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_coords: np.ndarray,
    axis_name: str,
    threshold: float = B_THRESHOLD_T,
    output_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Create interactive 2D contour plot with slider to change z-slice depth.

    Uses fixed color scale (vmin/vmax) to show field degradation across depths,
    same as static contour plots. This allows visual comparison of field strength
    at different depths.

    Parameters
    ----------
    B_mag : np.ndarray
        B-field magnitude (T) at grid points. Shape (nx, ny, nz).
    x, y : np.ndarray
        Grid coordinates (m) - 1D arrays
    z_coords : np.ndarray
        Z-coordinates (m) - 1D array of length nz
    axis_name : str
        Name of the axis
    threshold : float
        B-field threshold (T)
    output_path : str, optional
        Path to save initial figure (interactive state not saved)
    vmin, vmax : float, optional
        Fixed color scale limits (T). If None, auto-scales to current slice.
        Use fixed values to compare degradation across depths.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)

    X, Y = np.meshgrid(x, y, indexing="ij")
    z_idx = len(z_coords) // 2
    B_slice = B_mag[:, :, z_idx]

    # Use fixed scale if provided, else auto-scale to current slice
    if vmin is None:
        vmin = B_slice[B_slice > 0].min() + 1e-6
    if vmax is None:
        vmax = B_slice.max() + 1e-6
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
    cs = ax.contourf(
        X * 1000, Y * 1000, B_slice, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(cs, ax=ax, label="B-field magnitude (T)")
    threshold_contour = ax.contour(
        X * 1000,
        Y * 1000,
        (B_slice >= threshold).astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2,
        linestyles="--",
        label=f"Effective region (B ≥ {threshold*1e4:.2f} mT)",
    )

    ax.set_xlabel("X position (mm)", fontsize=12)
    ax.set_ylabel("Y position (mm)", fontsize=12)
    title = ax.set_title(
        f"{axis_name}-axis B-field at z = {z_coords[z_idx]*1000:.1f} mm",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Depth (mm)",
        z_coords[0] * 1000,
        z_coords[-1] * 1000,
        valinit=z_coords[z_idx] * 1000,
        valstep=(z_coords[-1] - z_coords[0]) * 1000 / len(z_coords),
    )

    def update(val):
        nonlocal cbar
        z_val = slider.val / 1000
        z_idx = np.argmin(np.abs(z_coords - z_val))
        B_slice = B_mag[:, :, z_idx]
        
        # Remove old colorbar before clearing axes
        try:
            if cbar is not None:
                cbar.remove()
        except:
            pass
        
        # Use fixed scale (vmin/vmax from outer scope) to show degradation
        ax.clear()
        cs = ax.contourf(
            X * 1000, Y * 1000, B_slice, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
        )
        # Recreate colorbar with fixed scale
        cbar = plt.colorbar(cs, ax=ax, label="B-field magnitude (T)")
        
        ax.contour(
            X * 1000,
            Y * 1000,
            (B_slice >= threshold).astype(float),
            levels=[0.5],
            colors="red",
            linewidths=2,
            linestyles="--",
            label=f"Effective region (B ≥ {threshold*1e4:.2f} mT)",
        )
        ax.legend()
        ax.set_xlabel("X position (mm)", fontsize=12)
        ax.set_ylabel("Y position (mm)", fontsize=12)
        # Recreate title after clear()
        ax.set_title(f"{axis_name}-axis B-field at z = {z_coords[z_idx]*1000:.1f} mm", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_targeting_volume(
    B_mag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_coords: np.ndarray,
    axis_name: str,
    threshold: float = B_THRESHOLD_T,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize 3D effective stimulation volume (where B >= threshold).

    Creates an isosurface plot showing the spatial extent of effective stimulation.

    Parameters
    ----------
    B_mag : np.ndarray
        B-field magnitude (T) at grid points. Shape (nx, ny, nz).
    x, y : np.ndarray
        Grid coordinates (m) - 1D arrays
    z_coords : np.ndarray
        Z-coordinates (m) - 1D array
    axis_name : str
        Name of the axis
    threshold : float
        B-field threshold (T)
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = np.meshgrid(x * 1000, y * 1000, z_coords * 1000, indexing="ij")
    effective = B_mag >= threshold

    if np.any(effective):
        # Plot isosurface points
        x_eff = X[effective]
        y_eff = Y[effective]
        z_eff = Z[effective]
        B_eff = B_mag[effective]

        scatter = ax.scatter(
            x_eff,
            y_eff,
            z_eff,
            c=B_eff,
            cmap="hot",
            s=10,
            alpha=0.6,
            label=f"Effective region (B ≥ {threshold*1e4:.2f} mT)",
        )
        plt.colorbar(scatter, ax=ax, label="B-field magnitude (T)")

    ax.set_xlabel("X position (mm)", fontsize=12)
    ax.set_ylabel("Y position (mm)", fontsize=12)
    ax.set_zlabel("Depth (mm)", fontsize=12)
    ax.set_title(
        f"{axis_name}-axis Effective Stimulation Volume\n"
        f"(B-field ≥ {threshold*1e4:.2f} mT threshold)",
        fontsize=14,
    )
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
