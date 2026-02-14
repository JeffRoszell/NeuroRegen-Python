# NeuroRegen — Pulsed Coil Stimulation Simulation

This project has **two modes**:

1. **Single-coil mode** — Simulates a 3-axis pulsed coil system (the original setup). Models pulse power, coil temperature, and penetration depth for each axis independently.

2. **Multicoil mode** — Simulates multiple coils positioned around the skull, all aimed at a deep brain target (the Subthalamic Nucleus). The coils work together so their magnetic fields add up at the target while keeping the field at the brain surface within safe limits.

---

## What you need

- **Python 3** (3.9 or newer).
  - **Mac / Linux:** Open Terminal and run `python3 --version`.
  - **Windows:** Open Command Prompt or PowerShell and run `py -3 --version` or `python --version`.

---

## One-time setup

### Mac / Linux

1. **Open Terminal** (on Mac: Applications → Utilities → Terminal).

2. **Go to the project folder**  
   Replace the path below with the actual location of the `Python Code` folder on your computer:
   ```bash
   cd "/Users/jeff/Documents/UND BME/NeuroRegen/Python Code"
   ```

3. **Create a virtual environment** (keeps this project's packages separate):
   ```bash
   python3 -m venv venv
   ```

4. **Turn the environment on** (do this each time you open a new Terminal to run the simulation):
   ```bash
   source venv/bin/activate
   ```
   When it's on, you'll see `(venv)` at the start of the line.

5. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   Wait until it finishes without errors.

### Windows

1. **Open Command Prompt or PowerShell**  
   (Press Win + R, type `cmd` or `powershell`, press Enter.)

2. **Go to the project folder**  
   Replace the path below with the actual location of the `Python Code` folder on your computer (use your own username and drive letter):
   ```cmd
   cd "C:\Users\YourName\Documents\UND BME\NeuroRegen\Python Code"
   ```
   In PowerShell you can use the same command, or forward slashes: `cd "C:/Users/YourName/Documents/UND BME/NeuroRegen/Python Code"`.

3. **Create a virtual environment**:
   ```cmd
   py -3 -m venv venv
   ```
   If `py` doesn't work, try: `python -m venv venv`.

4. **Turn the environment on** (do this each time you open a new window to run the simulation):
   - **Command Prompt:**
     ```cmd
     venv\Scripts\activate
     ```
   - **PowerShell:** you may need to allow scripts first (one time):
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
     Then:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   When it's on, you'll see `(venv)` at the start of the line.

5. **Install required packages**:
   ```cmd
   pip install -r requirements.txt
   ```
   Wait until it finishes without errors.

---

---

# Single-Coil Mode (Original 3-Axis System)

This is the original simulation. Each of the three coil axes (X, Y, Z) operates independently.

## Run the single-coil simulation

With your Terminal open, in the project folder, with `(venv)` active:

```bash
python scripts/run_simulation.py
```

On Windows, if `python` doesn't work, try: `py -3 scripts/run_simulation.py`.

This runs a 30-minute simulation and saves three plots to the `outputs/` folder:

1. **Temperature** — Coil temperature (°F) over time for each axis, with a 75°F safety limit line. Pulses are automatically skipped when a coil hits the limit and resume when it cools.
2. **Power** — Applied power (W) over time. You can see when pulses are on or off.
3. **Depth** — Estimated penetration depth (cm, up to 3 cm) over time.

## Interactive controller (single-coil)

The interactive controller lets you manually Arm, Start, Stop, and manage faults through a menu.

```bash
python scripts/run_interactive.py
```

**Quick start:**
1. Type **A** then Enter to **Arm** the system.
2. Type **S** then Enter to **Start** the simulation. It runs for 30 minutes (simulated).
3. When it finishes, you're back to the menu. Type **Q** to quit.

**All commands:**

| Key | What it does |
|-----|-------------|
| **A** | Arm the system (gets it ready to fire) |
| **S** | Start the simulation (must be Armed first) |
| **P** | Stop a running simulation |
| **C** | Clear a fault (if the system tripped a safety limit) |
| **E** + **X**/**Y**/**Z** | Turn an individual axis on or off |
| **D** | Show the current pulse and power settings |
| **L** | Turn on live plotting for the next run (shows graphs updating in real time) |
| **Q** | Quit |

**What are the states?**

| State | What it means |
|-------|--------------|
| **OFF** | Idle. Nothing is happening. |
| **ARMED** | Ready to go. Press S to start. |
| **FIRING** | Simulation is running. |
| **FAULT** | A safety limit was hit (e.g. coil got too hot). Press C to clear it. |

CSV logs are saved to `outputs/logs/` after each run.

---

---

# Multicoil Mode (Deep Brain Targeting)

This mode simulates multiple coils positioned around the skull, all aimed at a single deep target — the **Subthalamic Nucleus (STN)** — which is relevant to conditions like Parkinson's disease.

## Why multiple coils?

A single coil sitting on the scalp produces a strong field right under it (at the brain surface) but a very weak field deep inside. To stimulate a structure like the STN (which is about 5-8 cm below the skull surface), you need multiple coils working together. Each coil contributes a small amount of field at the target, and when their fields **add up (superpose)**, the combined field at the target becomes strong enough to be effective.

## How it works — the key concepts

### 1. Distance compensation (inverse-cube law)

The magnetic field from a coil drops off very quickly with distance — roughly as 1/r^3. If Coil A is 9 cm from the target and Coil B is 13 cm away, Coil B needs much more power to deliver the same field strength at the target. The script automatically calculates a **weight** for each coil based on its distance. Close coils get their power turned *down*; far coils get their power turned *up*. This ensures all coils contribute equally at the focal point.

### 2. The cosine effect (coil tilt)

It's not just about distance — the angle matters too. Each coil has a **normal vector** (the direction it "points"). If a coil is aimed directly at the target, its full field strength contributes. If it's tilted away, only a fraction contributes (proportional to the cosine of the angle). The script measures this alignment and compensates for it automatically.

### 3. Depth Gate — the surface safety check

This is the most important safety feature. To get enough field to the deep target, the coils produce a strong field at the brain surface (the cortex) right underneath them. If that surface field gets too strong, it could cause unwanted stimulation. The script calculates the **induced electric field (E-field) at the cortex** under every coil before each pulse. If any coil exceeds the safety limit (default: 150 V/m), the **entire pulse is blocked**. This is called a **Depth Gate Failure**.

### 4. Surface-to-Deep Ratio

This number tells you how much stronger the field is at the brain surface compared to the deep target. A ratio of 10 means the surface field is 10x stronger than the target field. **Lower is better** — it means the coils are focusing more efficiently. With 3 coils, typical ratios are 8-15. Adding more coils (5-8) can improve this.

## Run the multicoil simulation

With your Terminal open, in the project folder, with `(venv)` active:

```bash
python scripts/run_multicoil.py
```

This will:
1. Print a **pre-flight safety summary** showing each coil's distance, alignment, weight, power, and whether it passes the Depth Gate.
2. Run a 30-minute pulsed simulation.
3. Print peak field values and safety statistics.
4. Save a 4-panel results plot to `outputs/multicoil/`.

### What the pre-flight summary tells you

```
Coil      Dist   cos θ      Wt   Base P     Wt P    E_surf   Gate
A        12.81   1.000    2.61     60.0    156.3      0.05   PASS
B        11.70   1.000    1.99     60.0    119.2      0.04   PASS
C         9.31   1.000    1.00     60.0     60.0      0.03   PASS
```

Reading this table:
- **Dist** — How far the coil is from the target (cm). Coil C is closest (9.31 cm), Coil A is farthest (12.81 cm).
- **cos θ** — How well the coil is aimed at the target. 1.000 = perfectly aimed.
- **Wt** — The distance-compensation weight. Coil C (closest) gets weight 1.00; Coil A (farthest) gets 2.61 — meaning it needs 2.61x the power to match.
- **Base P** — The starting power you set in the config (W).
- **Wt P** — The actual power after distance compensation (W). Coil A gets boosted to 156.3 W.
- **E_surf** — The estimated electric field at the cortex under this coil (V/m). Must be below 150 V/m to pass.
- **Gate** — PASS or FAIL. If any coil says FAIL, pulses will be blocked.

### What the results plots show

The simulation saves a 4-panel plot (`outputs/multicoil/multicoil_results.png`):

1. **Temperature** (top left) — Each coil's temperature over time, with the 75°F safety limit.
2. **Weighted Power** (top right) — The distance-compensated power delivered to each coil. Notice the farther coils get more power.
3. **Combined B at Target** (bottom left) — The total magnetic field strength at the STN from all coils combined, over time. The red dashed line shows the minimum threshold for effective stimulation.
4. **Cortical E-field** (bottom right) — The induced electric field at the brain surface under each coil. The red line is the 150 V/m safety limit. If any coil crosses this line, pulses are blocked.

## Extra visualizations

### 2D field focus map

Shows a cross-section of the combined magnetic field at the target depth. This lets you see whether the field is focused on the target or drifting toward one coil:

```bash
python scripts/run_multicoil.py --field-map
```

Saved to `outputs/multicoil/multicoil_field_focus.png`.

### Interactive 3D visualization

Opens an interactive 3D view in your web browser where you can rotate, zoom, and hover for values:

```bash
python scripts/run_multicoil.py --3d
```

This shows:
- The **skull** as a translucent sphere
- Each **coil** as a colored dot with an arrow showing which way it points
- **Dashed lines** from each coil to the target
- The **STN target** as a red diamond
- A **colored point cloud** showing where the magnetic field is strong (yellow = strongest, dark red = weaker)

**How to use the 3D view:**
- **Rotate** — Click and drag
- **Zoom** — Scroll wheel
- **Pan** — Right-click and drag
- **Hover** — Mouse over any point to see its field value
- **Toggle layers** — Click items in the legend to show/hide them

The file is saved as `outputs/multicoil/multicoil_3d.html` — you can reopen it anytime by double-clicking the file.

### Run everything at once

```bash
python scripts/run_multicoil.py --field-map --3d
```

---

## Changing settings (config files)

You can change simulation settings **without editing code** by editing the config files with any text editor (TextEdit, Notepad, VS Code, etc.).

### Single-coil config: `config/default.yaml`

Controls the 3-axis single-coil simulation:
- `sim_time` — How long the simulation runs (seconds). Default: 1800 (30 minutes).
- `pulse_freq` — How many pulses per second (Hz). Default: 5.
- `pulse_width` — How long each pulse lasts (seconds). Default: 0.02 (20 ms).
- `temp_limit_f` — Temperature safety limit (°F). Default: 75.
- Each axis's `wire_mm`, `loop_mm`, `turns`, and `pulse_power_w`.

### Multicoil config: `config/multicoil.yaml`

Controls the multicoil deep-brain targeting simulation:
- **`target`** — The (x, y, z) position of the brain target in metres. Default is the left STN.
- **`safety`** — `cortical_max_vm` is the maximum allowed E-field at the cortex (V/m). `scalp_to_cortex_m` is the distance from the coil to the cortex (scalp + skull thickness).
- **`coils`** — Each coil has:
  - `position_m` — Where it sits on the skull (x, y, z in metres).
  - `normal` — Which direction it points (a unit vector aimed at the target).
  - `pulse_power_w` — The base power before distance compensation.
  - `wire_mm`, `loop_mm`, `turns` — Physical coil properties.
- **`simulation`**, **`thermal`**, **`depth`** — Same timing and safety settings as the single-coil config.

**To use a custom config file:**

```bash
python scripts/run_multicoil.py -c path/to/my_config.yaml
```

---

## Field mapping (single-coil)

Generate spatial B-field maps showing field strength contours and effective stimulation regions for the single-coil system:

```bash
python scripts/run_simulation.py --field-maps
```

This creates 2D contour plots at multiple depths, interactive depth-slice plots, and 3D targeting volume visualizations. Saved to `outputs/field_maps/`.

---

## If something goes wrong

- **"python3: command not found" (Mac/Linux)** or **"python is not recognized" (Windows)** — Python isn't installed or isn't on your path. Install Python 3 from [python.org](https://www.python.org/downloads/) (on Windows, check "Add Python to PATH" during setup). On Windows you can also try `py -3` instead of `python`.
- **"No such file or directory"** — The `cd` path is wrong. Check the path to the `Python Code` folder and fix it in the `cd` command.
- **PowerShell won't run Activate.ps1** — Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then try activating again.
- **"No module named plotly"** — Run `pip install -r requirements.txt` again with your virtual environment active.
- **Errors when running the script** — Make sure you ran the one-time setup steps (venv, activate, pip install) in the project folder.

---

## Optional: run the tests

If you want to check that the code is working correctly:

```bash
python -m pytest tests/ -v
```

You should see all tests listed with "passed" next to each one.
