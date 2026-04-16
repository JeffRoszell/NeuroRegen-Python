# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Run
```bash
python scripts/run_simulation.py                    # Single-coil 3-axis (30 min)
python scripts/run_simulation.py --field-maps       # + spatial B-field maps
python scripts/run_interactive.py                   # Menu-driven FSM controller
python scripts/run_multicoil.py                     # Multicoil deep-brain targeting
python scripts/run_multicoil.py --field-map --3d    # + 2D focus map + 3D HTML
python scripts/run_pulsed.py                        # Pulsed capacitor-discharge sim
python scripts/run_pulsed.py --plot                 # + voltage sweep / depth / waveform plots
python scripts/generate_ansys_sample.py             # Generate synthetic ANSYS CSV exports
# Any script accepts -c path/to/cfg.yaml for a custom config
```

### Test & Lint
```bash
python -m pytest tests/ -v
python -m pytest tests/test_coil.py::test_name -v  # single test
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/
```

### Pre-commit
Hooks run automatically on `git commit`: ruff (lint + format), codespell, pytest, conventional commit validation.

### Commits
[Conventional Commits](https://www.conventionalcommits.org/) enforced by hook.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Architecture

NeuroRegen simulates magnetic coil stimulation targeting the Subthalamic Nucleus (STN). Three simulations driven by YAML configs in `config/`.

### The Three Simulations

| Script | Config | Purpose |
|---|---|---|
| `run_simulation.py` | `default.yaml` | Single-coil 3-axis, continuous-wave, thermal gating |
| `run_multicoil.py` | `multicoil.yaml` | 3-coil array, distance-compensated targeting, depth-gate safety |
| `run_pulsed.py` | `pulsed_tms.yaml` | Capacitor-discharge physics, matches ANSYS AEDT build |

### Current Hardware / ANSYS Build

All three configs now use the same physical coil geometry (matching `3AxisTeslaCoil_6Turns_10mmWire_150mmLargest_SeparateLeads_300mm_Apr15.aedt`):
- **6 turns**, **150 mm loop diameter**, **10 mm wire**
- **200 µF** capacitor bank, **798 V** charge voltage (pulsed mode)
- Peak current: ~3,674 A, pulse half-period: ~137 µs

### Modules (`src/neuroregen/`)
- **Physics**: `coil.py`, `thermal.py`, `pulsed.py` (LC discharge), `constants.py`
- **Simulation**: `simulation.py` (single-coil 3-axis), `multicoil.py` (multicoil targeting + field)
- **Control**: `state_machine.py` (OFF→ARMED→FIRING→FAULT FSM), `controller.py`
- **Config**: `config_loader.py` (YAML parsing for all three modes)
- **Output**: `plotting.py`, `field_mapping.py`, `csv_logger.py`
- **ANSYS**: `ansys_field_map.py` (load CSV exports), `ansys_connection.py` (live pyaedt)

### Key Design Rules
- All tunable parameters in YAML configs — no hardcoded sim values in source
- Thermal gating with hysteresis: skip pulses at 75°F limit, resume at 74.3°F
- Multicoil depth gate: block entire pulse if any coil exceeds 150 V/m cortical E-field
- Distance compensation: weights ∝ distance⁶ so all coils deliver equal field at target
- ANSYS field maps are optional; falls back to analytical Biot-Savart when absent
- Ruff: line length 100, Python 3.9 target, ignores `E741` (allows `I` for current) and `F841`

### Known Issue
In `pulsed.py`, cortical E-field uses the coil loop radius (75 mm) as the tissue loop radius — this is conservative (overestimates cortical E by ~4–15×). The reported value won't match SimNIBS or ANSYS output directly. Target E-field at STN uses the configurable `r_tissue_m` (default 5 mm), which is more realistic.

## CI
GitHub Actions (`.github/workflows/ci.yml`) runs pytest and ruff on Python 3.9/3.10/3.11.

## Agents

Custom agents in `.claude/agents/`.

| Agent | Role |
|-------|------|
| `orchestrator-builder` | Default — builds features, fixes bugs |
| `code-reviewer` | Runs ruff, pytest, manual review (read-only) |
| `code-simplifier` | Simplifies recently written code |
| `neuro-research-lookup` | TMS/STN academic references |
| `ansys-simnibs-expert` | PyAEDT / SimNIBS integration guidance |
