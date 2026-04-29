# NeuroRegen — Pulsed Coil Stimulation Simulation

Three simulations of a 3-axis multi-coil TMS device targeting the Subthalamic Nucleus (STN).

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Simulations

### 1. Single-Coil (original 3-axis system)

Models three small orthogonal coil axes (X/Y/Z) independently — pulse power, thermal gating, and penetration depth.

```bash
python scripts/run_simulation.py                   # 30-min simulation + plots
python scripts/run_simulation.py --field-maps      # + spatial B-field maps
python scripts/run_interactive.py                  # menu-driven FSM controller
```

Config: `config/default.yaml`

Outputs saved to `outputs/`.

---

### 2. Multicoil (3-coil deep-brain targeting)

Three coils on the skull surface aimed at the STN. Distance compensation ensures equal field contribution at the target. Includes thermal gating and a depth-gate safety check (cortical E-field limit) per pulse.

```bash
python scripts/run_multicoil.py                    # 30-min simulation + plots
python scripts/run_multicoil.py --field-map        # + 2D field focus map
python scripts/run_multicoil.py --3d               # + interactive 3D HTML
```

Config: `config/multicoil.yaml`

Outputs saved to `outputs/multicoil/`.

---

### 3. Pulsed Multicoil (capacitor-discharge, matches ANSYS AEDT)

Capacitor-discharge physics for the 3-coil array (6-turn / 150 mm / 10 mm wire, 200 µF at 798 V). Computes LC-discharge parameters, peak current, cortical E-field safety check, B-field at STN, and thermal evolution over N pulses.

```bash
python scripts/run_pulsed.py                       # physics summary
python scripts/run_pulsed.py --plot                # + voltage sweep, depth profile, waveform plots
```

Config: `config/pulsed_tms.yaml`

Outputs saved to `outputs/`.

---

## Configuration

All tunable parameters live in the YAML config files — no code changes needed.

| File | Used by |
|---|---|
| `config/default.yaml` | `run_simulation.py` |
| `config/multicoil.yaml` | `run_multicoil.py` |
| `config/pulsed_tms.yaml` | `run_pulsed.py` |

Pass a custom config with `-c path/to/config.yaml`.

---

## Configuration Reference

A full breakdown of every setting across all three simulations — including the rationale for the multicoil per-coil power values and the neural activation context — is in:

```
docs/simulation-settings-breakdown.md
```

## ANSYS Integration

See `docs/ansys-integration-guide.md` for exporting B/E field CSVs from ANSYS Maxwell and activating them in `multicoil.yaml`.

To generate synthetic ANSYS-format test CSVs from the analytical model:

```bash
python scripts/generate_ansys_sample.py
```

---

## ANSYS Alternative — Pure-Python B-field Simulator

Self-contained subproject in [`ansys_alternative/`](ansys_alternative/) that replaces ANSYS Maxwell for **outside-the-head** magnetic-field calculation and 3-D visualization of the 3-coil array. Cortical / deep-brain E-fields still go through SimNIBS.

Stack: [magpylib](https://github.com/magpylib/magpylib) (analytical Biot–Savart), [PyVista](https://pyvista.org/) (VTK rendering), trimesh, PyYAML. Default coil block matches the validated Maxwell build (6 turns × 150 mm × 10 mm wire, 3 674 A peak).

```bash
cd ansys_alternative
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python simulate_bfield.py                          # interactive 3-D scene + field grid
python simulate_bfield.py --no-show \              # headless export
    --save-html outputs/scene.html \
    --save-vti  outputs/bfield.vti
python simulate_bfield.py --check-only             # coil/scalp standoff sanity check only
```

Keep its venv isolated from the main project's — different dependency set (magpylib, pyvista, trimesh).

Docs: [`ansys_alternative/README.md`](ansys_alternative/README.md) · [QUICKSTART](ansys_alternative/docs/QUICKSTART.md) · [MANUAL](ansys_alternative/docs/MANUAL.md) · [VS_ANSYS](ansys_alternative/docs/VS_ANSYS.md)

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## FSM / Interactive Controller

The interactive controller (`run_interactive.py`) exercises the state machine directly:

| Key | Action |
|-----|--------|
| A | Arm |
| S | Start (must be Armed) |
| P | Stop |
| C | Clear fault |
| E + X/Y/Z | Toggle axis |
| Q | Quit |

States: `OFF → ARMED → FIRING → FAULT`
