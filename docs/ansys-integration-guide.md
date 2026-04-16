# ANSYS Maxwell Integration

Two ways to feed ANSYS field data into the Python simulation:

| | Method 1: CSV Export | Method 2: Live (pyaedt) |
|---|---|---|
| **How** | Export CSVs from ANSYS, drop in folder | Python drives ANSYS directly |
| **Requires** | ANSYS Maxwell (any version) | ANSYS Maxwell + `pip install pyaedt` |

---

## Method 1: CSV Export

**In ANSYS Maxwell:** Run one solve per coil (A, B, C) at a known reference current (1 A recommended). Export on a rectangular grid: *Fields → Export Fields to File → On Grid*. Select `Bx`, `By`, `Bz` (and optionally `Ex`, `Ey`, `Ez`). Units: metres + Tesla (or mT/Gauss — auto-converted).

**CSV format:**
```
# reference_current_A: 1.0
x_m,y_m,z_m,Bx_T,By_T,Bz_T,Ex_Vm,Ey_Vm,Ez_Vm
-0.06,-0.06,-0.10,1.23e-06,2.34e-07,5.67e-06,...
```
Accepted column names: `x_m`/`x_mm`, `Bx_T`/`Bx_mT`/`Bx_G`, `Ex_Vm`/`Ex_V/m` (case-insensitive).

**Place files:** `ansys_exports/coil_A.csv`, `coil_B.csv`, `coil_C.csv`

**Activate in `config/multicoil.yaml`:**
```yaml
ansys:
  mode: "file"
  field_maps:
    reference_current_a: 1.0
    coil_maps:
      A: ansys_exports/coil_A.csv
      B: ansys_exports/coil_B.csv
      C: ansys_exports/coil_C.csv
```

Then run normally: `python scripts/run_multicoil.py`. Points outside the exported grid fall back to analytical Biot-Savart automatically.

---

## Method 2: Live Connection (pyaedt)

**Activate in `config/multicoil.yaml`:**
```yaml
ansys:
  mode: "live"
  connection:
    version: "2024.1"             # match your installed version
    non_graphical: true
    export_grid_spacing_mm: 3.0
    reference_current_a: 1.0
    head_model:
      scalp_radius_m: 0.095
      skull_radius_m: 0.090
      brain_radius_m: 0.080
```

Then run: `python scripts/run_multicoil.py`. Python launches ANSYS, builds geometry from config, solves, extracts fields, and closes. Takes several minutes.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: ANSYS field map not found` | Check CSV paths in `multicoil.yaml` are relative to project root |
| `missing required column for 'bx'` | Column names don't match — see accepted names above |
| `ImportError: scipy` | `pip install scipy` |
| `ImportError: pyaedt` | `pip install pyaedt` |
| Results look wrong | Verify `reference_current_a` matches current used in ANSYS solve |

---

## Generate Test CSVs

Verify the pipeline with synthetic data before using real ANSYS exports:
```bash
python scripts/generate_ansys_sample.py
```
Creates `ansys_exports/coil_{A,B,C}.csv` from the analytical model.
