# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Run simulations
```bash
python scripts/run_simulation.py                    # Single-coil 3-axis
python scripts/run_simulation.py --field-maps       # With spatial field maps
python scripts/run_interactive.py                   # Menu-driven interactive controller
python scripts/run_multicoil.py                     # Multicoil deep-brain targeting
python scripts/run_multicoil.py --field-map --3d    # With 2D map and interactive 3D HTML
```

### Test
```bash
python -m pytest tests/ -v                                            # All tests
python -m pytest tests/test_coil.py -v                               # Single file
python -m pytest tests/test_thermal.py::test_gate_off_at_limit -v    # Single test
```

### Lint
```bash
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/
```

Ruff config is in `pyproject.toml`: line length 100, target Python 3.9, ignores `E741` (allows `I` for current) and `F841`.

## Architecture

NeuroRegen simulates magnetic coil stimulation for deep-brain neuromodulation research, targeting the Subthalamic Nucleus. It has two operating modes driven by YAML config files in `config/`.

### Module layers

**Physics** (`src/neuroregen/`):
- `coil.py` — coil geometry, temperature-dependent resistance, on-axis B-field (Biot-Savart), penetration depth
- `thermal.py` — heat balance, convective cooling, thermal gating with hysteresis
- `constants.py` — physical constants (μ₀, ρ_Cu, thermal properties)

**Simulation**:
- `simulation.py` — single-coil 3-axis stepwise loop; per-axis thermal gating and depth tracking
- `multicoil.py` (879 lines) — multicoil physics: Coil/Target dataclasses, 3D B-field with rotation matrices, distance-weighted power (inverse-cube), cosine-factor alignment, field superposition, cortical E-field depth-gate safety check

**Control**:
- `state_machine.py` — 4-state FSM: `OFF → ARMED → FIRING → FAULT`
- `controller.py` — wraps FSM, per-axis enable/disable, run loop, CSV logging, live plotting

**Output**:
- `plotting.py` — live matplotlib plots and static result figures
- `field_mapping.py` — 2D/3D spatial field maps, contour plots, interactive HTML slices
- `csv_logger.py` — time-series CSV export

### Data flow
```
YAML config → load_config / load_multicoil_config
           → Axis / Coil / Target objects
           → run_*_simulation_stepwise (per timestep: pulse?, thermal gate?, B-field, ΔT)
           → outputs/ PNG plots + optional HTML 3D
           → (interactive) CSV logs to outputs/logs/run_TIMESTAMP.csv
```

### Key physics

**Single-coil** (per axis):
- B(z) = μ₀ I R² N / (2(R²+z²)^1.5)
- R(T) = ρ L/A × (1 + α(T − 20°C))
- Thermal: dT/dt = (P_in − h·S·(T − T_amb)) / C_thermal

**Multicoil** (novel additions in `multicoil.py`):
- Distance weight: w_i = (d_max / d_i)³
- Cosine alignment: cos θ = n_coil · (target − pos_coil) / |...|
- Weighted power: P_i = w_i × P_base × cos θ_i
- Safety depth gate: if any coil's cortical E-field > 150 V/m, entire pulse is blocked

### Configuration
All tunable parameters live in `config/default.yaml` (single-coil) and `config/multicoil.yaml` (multicoil STN targeting). No hardcoded simulation values in source.

### CI
GitHub Actions (`.github/workflows/ci.yml`) runs pytest and ruff on Python 3.9/3.10/3.11 for pushes to `main` and `feature/menu-interface-state-machine`.
