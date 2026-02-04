#!/usr/bin/env python3
"""
Menu-driven command interface for the NeuroRegen 3-axis coil controller.
States: OFF, ARMED, FIRING, FAULT. Supports axis enable, live plot, CSV logging.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.neuroregen.config_loader import load_config
except ImportError:
    load_config = None

from src.neuroregen.controller import Controller
from src.neuroregen.state_machine import ControllerState


def load_config_or_default():
    config_path = os.path.join(ROOT, "config", "default.yaml")
    if load_config and os.path.isfile(config_path):
        return load_config(config_path)
    # Minimal default config
    from src.neuroregen.simulation import default_axes

    return {
        "sim_time": 1800.0,
        "dt": 0.1,
        "pulse_freq": 5.0,
        "pulse_width": 0.02,
        "axes": default_axes(),
        "t_amb_c": 22.0,
        "h_conv": 10.0,
        "temp_limit_f": 75.0,
        "hyst_f": 0.7,
        "z_max_m": 0.03,
        "z_points": 300,
        "b_threshold_t": 1e-4,
    }


def print_status(ctrl: Controller):
    """Print current state, axis enable, and (if available) last temps."""
    state = ctrl.state.value
    fault = f" [{ctrl.fault_reason}]" if ctrl.fault_reason else ""
    print(f"\n  State: {state}{fault}")
    en = ctrl.axis_enabled
    print(f"  Axes enabled: X={en[0]}, Y={en[1]}, Z={en[2]}")
    if ctrl.csv_path:
        print(f"  Last log: {ctrl.csv_path}")


def print_pulse_config(ctrl: Controller):
    """Display pulse definition and axis power from config."""
    c = ctrl.config
    print(f"\n  Pulse: {c['pulse_freq']} Hz, width {c['pulse_width'] * 1000:.0f} ms")
    print("  Axis power (W):", end="")
    for a in c["axes"]:
        print(f"  {a.name}={a.pulse_power_w}", end="")
    print()


def toggle_axis(ctrl: Controller, i: int):
    if 0 <= i < 3:
        ctrl.axis_enabled[i] = not ctrl.axis_enabled[i]
        names = "XYZ"
        print(f"  {names[i]}-axis now {'enabled' if ctrl.axis_enabled[i] else 'disabled'}.")


def run_firing_with_live_plot(ctrl: Controller, live: bool):
    try:
        import matplotlib.pyplot as plt
        from src.neuroregen.plotting import live_plot_init
    except ImportError:
        live = False
    if live:
        limit_f = ctrl.config["temp_limit_f"]
        resume_f = limit_f - ctrl.config["hyst_f"]
        z_max_cm = ctrl.config["z_max_m"] * 100
        fig, update_fn, _ = live_plot_init(limit_f=limit_f, resume_f=resume_f, z_max_cm=z_max_cm)
        ctrl.set_live_plot_update(update_fn)
        plt.show(block=False)
    t, T, P, depth = ctrl.run_firing_loop()
    ctrl.set_live_plot_update(None)
    if live and t.size > 0:
        print("  Run finished. Close the plot window to continue.")
        plt.show(block=True)
    return t, T, P, depth


def main():
    config = load_config_or_default()
    csv_dir = os.path.join(ROOT, "outputs", "logs")
    ctrl = Controller(config=config, csv_dir=csv_dir, log_run=True)

    print("\n" + "=" * 60)
    print("  NeuroRegen — Menu-driven controller (OFF / ARMED / FIRING / FAULT)")
    print("=" * 60)

    live_plot_next = False
    while True:
        print_status(ctrl)
        live_hint = " [live plot ON for next Start]" if live_plot_next else ""
        print("\n  Commands:")
        print("    A = Arm (OFF -> ARMED)   S = Start (ARMED -> FIRING, run simulation)")
        print("    P = Stop (FIRING -> OFF) C = Clear fault (FAULT -> OFF)   Q = Quit")
        print(
            "    E = Toggle axis (X/Y/Z)  D = Display pulse/axis config   L = Toggle live plot for next run"
            + live_hint
        )
        try:
            line = input("\n  Choice: ").strip().upper() or " "
            cmd = line[0]
        except (EOFError, KeyboardInterrupt):
            print("\n  Quit.")
            break

        if cmd == "Q":
            print("  Quit.")
            break
        if cmd == "A":
            if ctrl.arm():
                print("  Armed.")
            else:
                print("  Cannot arm from current state (must be OFF).")
        elif cmd == "S":
            if ctrl.state != ControllerState.ARMED:
                print("  Must be ARMED to start. Use A to arm.")
                continue
            if not ctrl.start():
                print("  Start failed.")
                continue
            live = live_plot_next
            live_plot_next = False
            print("  Running... (wait for end, fault, or Ctrl+C)")
            run_firing_with_live_plot(ctrl, live=live)
            if ctrl.state == ControllerState.FAULT:
                print(f"  FAULT: {ctrl.fault_reason}")
            else:
                print("  Run complete.")
        elif cmd == "P":
            ctrl.stop_requested = True
            print("  Stop requested (will take effect at next step).")
        elif cmd == "C":
            if ctrl.clear_fault():
                print("  Fault cleared.")
            else:
                print("  No fault to clear (or not in FAULT state).")
        elif cmd == "E":
            sub = line[1:].strip() or input("  Toggle which axis? (X/Y/Z): ").strip().upper()
            idx = {"X": 0, "Y": 1, "Z": 2}.get(sub)
            if idx is not None:
                toggle_axis(ctrl, idx)
            else:
                print("  Use X, Y, or Z.")
        elif cmd == "D":
            print_pulse_config(ctrl)
        elif cmd == "L":
            live_plot_next = not live_plot_next
            print(
                "  Live plot for next Start: ON."
                if live_plot_next
                else "  Live plot for next Start: OFF."
            )
        else:
            print("  Unknown command.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
