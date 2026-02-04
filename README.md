# NeuroRegen — 3-Axis Pulsed Coil Simulation

This project simulates a **30-minute pulsed stimulation session** for a 3-axis coil system. It models pulse power, coil temperature (with a 75°F safety limit), and estimated penetration depth, and saves three plots to an `outputs` folder.

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

3. **Create a virtual environment** (keeps this project’s packages separate):
   ```bash
   python3 -m venv venv
   ```

4. **Turn the environment on** (do this each time you open a new Terminal to run the simulation):
   ```bash
   source venv/bin/activate
   ```
   When it’s on, you’ll see `(venv)` at the start of the line.

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
   If `py` doesn’t work, try: `python -m venv venv`.

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
   When it’s on, you’ll see `(venv)` at the start of the line.

5. **Install required packages**:
   ```cmd
   pip install -r requirements.txt
   ```
   Wait until it finishes without errors.

---

## Run the simulation

With the same Terminal (Mac/Linux) or Command Prompt/PowerShell (Windows) window, still in the project folder and with `(venv)` active:

```bash
python scripts/run_simulation.py
```

On Windows, if `python` doesn’t work, try: `py -3 scripts/run_simulation.py`.

### Menu-driven controller (interactive)

For a **state-machine–driven interface** with OFF → ARMED → FIRING → FAULT, axis enable/disable, **live plotting**, and **CSV data logging**:

```bash
python scripts/run_interactive.py
```

- **A** = Arm (OFF → ARMED)  
- **S** = Start (run simulation; optionally enable **L** first for live plot)  
- **P** = Stop (request stop during run)  
- **C** = Clear fault (FAULT → OFF)  
- **E** then **X**/**Y**/**Z** = Toggle that axis on/off  
- **D** = Display pulse and axis power config  
- **L** = Toggle live plot for the next run  
- **Q** = Quit  

CSV logs are written to `outputs/logs/run_YYYYMMDD_HHMMSS.csv` (time, temperatures, power, depth, state).

- The simulation runs for about 30–60 seconds.
- When it finishes, **three plot windows** will open (Temperature, Power, Depth).
- The same three plots are also saved as images in the **`outputs`** folder:
  - `temperature.png`
  - `power.png`
  - `depth.png`

You can close the plot windows when you’re done; the files in `outputs` stay.

**Using a different config file:**  
`python scripts/run_simulation.py -c path/to/my_config.yaml`

---

## Field mapping and spatial visualization

Generate spatial B-field maps showing field strength contours and effective stimulation regions:

```bash
python scripts/plot_field_maps.py
```

This creates:
- **2D contour plots** at multiple depths (5, 10, 15, 20, 25 mm) showing B-field magnitude
- **Interactive depth-slice plots** with a slider to explore different depths
- **3D targeting volume visualizations** showing where B-field exceeds the threshold

Options:
- `-p POWER` — Power per axis (W). Default: from config or 45 W
- `-t TEMP` — Coil temperature (°C). Default: from config or 22°C
- `-o OUTPUT_DIR` — Output directory. Default: `outputs/field_maps/`
- `--no-interactive` — Skip interactive plots (faster, saves only static images)

Example:
```bash
python scripts/plot_field_maps.py -p 50 -t 25 --no-interactive
```

Plots are saved to `outputs/field_maps/` with names like `x_contour_z5mm.png`, `z_interactive.png`, `y_volume_3d.png`, etc.

---

## Changing run parameters (config)

You can change simulation settings **without editing code** by editing the config file:

- **`config/default.yaml`** — simulation time, pulse frequency/width, thermal limit, coil geometry (wire/loop size, turns, power per axis), depth limits, etc.

Edit the YAML file with any text editor, save, then run the simulation again. The script uses `config/default.yaml` by default if that file exists.

---

## What the plots show

1. **Temperature** — Coil temperature (°F) over the 30-minute session for each axis (X, Y, Z), with a 75°F limit line. Pulses are skipped when temperature hits the limit and resume when it cools.
2. **Power** — Applied power (W) over time for each axis; you see when pulses are on or off.
3. **Depth** — Estimated penetration depth (cm, up to 3 cm) over time for each axis.

---

## If something goes wrong

- **“python3: command not found” (Mac/Linux)** or **“python is not recognized” (Windows)** — Python isn’t installed or isn’t on your path. Install Python 3 from [python.org](https://www.python.org/downloads/) (on Windows, check “Add Python to PATH” during setup) or ask your team for the recommended installer. On Windows you can also try `py -3` instead of `python`.
- **“No such file or directory”** — The `cd` path is wrong. Check the path to the `Python Code` folder and fix it in the `cd` command. On Windows, paths often look like `C:\Users\Name\...` and need quotes if there are spaces.
- **PowerShell won’t run Activate.ps1** — Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then try activating again.
- **Errors when running the script** — Make sure you ran steps 3–5 (venv, activate, `pip install -r requirements.txt`) in the same project folder.

---

## Optional: run the tests

If you want to check that the code is working as intended:

```bash
python -m pytest tests/ -v
```

(On Windows, use `python` or `py -3` as above.) You should see several tests listed and “passed” for each.
