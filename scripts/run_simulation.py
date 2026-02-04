#!/usr/bin/env python3
"""
Run the 3-axis pulsed 30-minute simulation and save plots to outputs/.
Uses config/default.yaml if present; otherwise uses built-in defaults.
"""

import argparse
import sys
import os

# Allow running from repo root or from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.neuroregen import run_simulation, plot_and_save, default_axes
from src.neuroregen.constants import T_AMB_C, B_THRESHOLD_T

try:
    from src.neuroregen.config_loader import load_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

try:
    import numpy as np
    from src.neuroregen.field_mapping import (
        create_spatial_grid,
        calculate_field_map,
        plot_field_contours_2d,
        plot_field_interactive_slice,
        plot_targeting_volume,
    )
    HAS_FIELD_MAPS = True
except ImportError:
    HAS_FIELD_MAPS = False


def main():
    parser = argparse.ArgumentParser(description="Run NeuroRegen 3-axis pulsed simulation.")
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--field-maps",
        action="store_true",
        help="Also generate spatial B-field maps and targeting visualizations",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive field map plots (faster, saves only static images)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animated plots that loop through the simulation (5 second loop)",
    )
    args = parser.parse_args()

    config = None
    if HAS_CONFIG:
        if args.config is not None:
            try:
                config = load_config(args.config)
            except (ImportError, FileNotFoundError) as e:
                if isinstance(e, ImportError):
                    print("Warning: PyYAML not installed. Using default parameters.")
                else:
                    print(f"Warning: Config file not found: {args.config}. Using default parameters.")
                config = None
        else:
            config_path = os.path.join(ROOT, "config", "default.yaml")
            if os.path.isfile(config_path):
                try:
                    config = load_config(config_path)
                except ImportError:
                    print("Warning: PyYAML not installed. Using default parameters.")
                    config = None
    elif args.config is not None:
        print("Warning: PyYAML not installed. Cannot load config file. Using default parameters.")

    output_dir = os.path.join(ROOT, "outputs")
    t, T, P, depth = run_simulation(config=config)
    plot_and_save(
        t, T, P, depth,
        output_dir=output_dir,
        show=True,
        temp_limit_f=config["temp_limit_f"] if config else None,
        hyst_f=config["hyst_f"] if config else None,
        z_max_m=config["z_max_m"] if config else None,
        animate=args.animate,
    )
    print(f"Plots saved to {output_dir}/")

    # Generate field maps if requested
    if args.field_maps:
        if not HAS_FIELD_MAPS:
            print("Warning: Field mapping not available. Install required dependencies.")
        else:
            print("\n" + "="*60)
            print("Generating spatial B-field maps...")
            print("="*60)
            
            # Get parameters from config or use defaults
            if config:
                axes = config["axes"]
                Pin = axes[0].pulse_power_w
                T_c = config["t_amb_c"]
                threshold = config.get("b_threshold_t", B_THRESHOLD_T)
            else:
                axes = default_axes()
                Pin = axes[0].pulse_power_w
                T_c = T_AMB_C
                threshold = B_THRESHOLD_T
            
            field_maps_dir = os.path.join(output_dir, "field_maps")
            os.makedirs(field_maps_dir, exist_ok=True)
            
            # Create spatial grid
            X, Y, Z, _ = create_spatial_grid(
                x_range=(-0.05, 0.05),
                y_range=(-0.05, 0.05),
                z_range=(0, 0.03),
                nx=50,
                ny=50,
                nz=30,
            )
            x_coords = np.linspace(-0.05, 0.05, 50)
            y_coords = np.linspace(-0.05, 0.05, 50)
            z_coords = np.linspace(0, 0.03, 30)
            z_slices_mm = [5, 10, 15, 20, 25]
            
            for axis in axes:
                print(f"\nProcessing {axis.name}-axis...")
                print(f"  Calculating B-field map at {Pin} W, {T_c:.1f}°C...")
                
                # Calculate field map
                B_mag = calculate_field_map(axis, Pin, T_c, X, Y, Z)
                
                print(f"  B-field range: {B_mag.min()*1e4:.3f} - {B_mag.max()*1e4:.3f} mT")
                
                # Calculate global B-field bounds for fixed color scale
                B_min = B_mag[B_mag > 0].min() + 1e-6
                B_max = B_mag.max() + 1e-6
                
                # 2D contour plots at multiple depths
                print(f"  Generating 2D contour plots at {len(z_slices_mm)} depths...")
                for z_mm in z_slices_mm:
                    z_m = z_mm / 1000
                    if z_m > z_coords[-1]:
                        continue
                    plot_field_contours_2d(
                        B_mag,
                        x_coords,
                        y_coords,
                        z_m,
                        axis.name,
                        threshold=threshold,
                        output_path=os.path.join(
                            field_maps_dir, f"{axis.name.lower()}_contour_z{z_mm}mm.png"
                        ),
                        show=False,
                        vmin=B_min,
                        vmax=B_max,
                    )
                
                # Interactive depth-slice plot
                if not args.no_interactive:
                    print(f"  Generating interactive depth-slice plot...")
                    plot_field_interactive_slice(
                        B_mag,
                        x_coords,
                        y_coords,
                        z_coords,
                        axis.name,
                        threshold=threshold,
                        output_path=os.path.join(
                            field_maps_dir, f"{axis.name.lower()}_interactive.png"
                        ),
                        vmin=B_min,
                        vmax=B_max,
                    )
                
                # 3D targeting volume
                print(f"  Generating 3D targeting volume visualization...")
                plot_targeting_volume(
                    B_mag,
                    x_coords,
                    y_coords,
                    z_coords,
                    axis.name,
                    threshold=threshold,
                    output_path=os.path.join(
                        field_maps_dir, f"{axis.name.lower()}_volume_3d.png"
                    ),
                    show=False,
                )
            
            print(f"\nAll field maps saved to {field_maps_dir}/")


if __name__ == "__main__":
    main()
