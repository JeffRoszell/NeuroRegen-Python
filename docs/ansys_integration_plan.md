# ANSYS Maxwell Build Specs — NeuroRegen 3-Coil Array

Reference for the ANSYS side of the integration. Python side is already built; see `ansys-integration-guide.md` for connecting outputs.

---

## Solution Type

**Eddy Current at 50 Hz** (not Magnetostatic).

Pulse half-period τ = 137 µs → fundamental frequency ≈ 50 Hz. Eddy current accounts for tissue conductivity, skin/proximity effect in the coil, and induced E-field — none of which magnetostatic can compute.

---

## Project Structure

One AEDT project, three design variants (one coil active per solve at 1 A reference):

```
NeuroRegen_Multicoil.aedt
├── Design_CoilA     — Coil A = 1 A,  B = 0 A,  C = 0 A
├── Design_CoilB     — Coil B = 1 A,  A = 0 A,  C = 0 A
└── Design_CoilC     — Coil C = 1 A,  A = 0 A,  B = 0 A
```

Python scales at runtime: `B_actual = B_ansys × (I_actual / 1.0)`.

---

## Coordinate System

Match Python exactly (Tools → Options → Units → metres):

```
Origin : centre of the skull sphere
X      : left(−) / right(+)
Y      : posterior(−) / anterior(+)
Z      : inferior(−) / superior(+)
```

---

## Head Geometry (concentric spherical shells)

| Layer | Inner radius | Outer radius |
|---|---|---|
| Brain | 0 | 75 mm |
| CSF | 75 mm | 77 mm |
| Skull | 77 mm | 87 mm |
| Scalp | 87 mm | 90 mm |
| Air region | 90 mm | 250 mm |

Radiation boundary on outer air sphere surface.

---

## Coil Geometry

All three coils identical. From `config/pulsed_tms.yaml` / AEDT build:

| Parameter | Value |
|---|---|
| Loop diameter | 150 mm |
| Wire diameter | 10 mm |
| Turns | 6 |
| Material | Copper (σ = 5.8×10⁷ S/m) |

**Recommended approach:** model as a toroid (one lumped conductor), assign *Coil Terminal* with `Turns = 6`. Draw a 10 mm circle at (0, 75, 0) mm in the YZ plane, sweep 360° around Z.

**Coil positions and normals** (from `config/pulsed_tms.yaml`):

| Coil | Position (m) | Normal | Description |
|---|---|---|---|
| A | (−0.010, −0.010, 0.088) | (−0.016, −0.023, −1.000) | Vertex |
| B | (0.085, −0.020, 0.025) | (−0.829, 0.060, −0.556) | Right lateral |
| C | (−0.015, −0.088, 0.015) | (0.032, 0.806, −0.591) | Posterior |

Apply: *Move* to position, then *Rotate* to align coil Z-axis with normal vector.

---

## Material Properties at 50 Hz

| Material | σ (S/m) | εᵣ | Reference |
|---|---|---|---|
| Scalp | 0.44 | 1100 | Gabriel (1996) |
| Skull | 0.020 | 20 | Gabriel (1996) |
| CSF | 1.79 | 109 | Gabriel (1996) |
| Gray matter | 0.30 | 2000 | Gabriel (1996) |
| Copper | 5.8×10⁷ | 1 | Standard |
| Air | 0 | 1 | — |

---

## Mesh Settings

| Region | Max element size |
|---|---|
| Wire cross-section | 0.3 mm |
| Scalp/skull | 2 mm |
| CSF | 0.5 mm |
| Brain near STN | 3 mm |
| Brain bulk | 8 mm |
| Air region | 20 mm |

Adaptive refinement: max 12 passes, ΔEnergy < 1%, refine 30%/pass.

---

## Field Export

Export centred on STN target `(−1.2, −1.3, −4.0) cm`. Recommended grid:

| Axis | Range | Step | Points |
|---|---|---|---|
| X | −9 to +9 cm | 3 mm | 61 |
| Y | −9 to +9 cm | 3 mm | 61 |
| Z | −9 to +3 cm | 3 mm | 41 |

*Fields → Export Fields to File → On Grid.* Select `Bx`, `By`, `Bz` (and `Ex`, `Ey`, `Ez`). Units: metres + Tesla.

Post-process: rename columns and prepend metadata header (see `ansys-integration-guide.md` for format). Save as `ansys_exports/coil_A.csv`, `coil_B.csv`, `coil_C.csv`.

---

## Validation

1. Generate analytical baseline: `python scripts/generate_ansys_sample.py`
2. Run with ANSYS maps active, then again without (`ansys:` block commented out)
3. Compare `B_target` time series — expect 10–30% lower with ANSYS (tissue attenuation at 50 Hz)
4. If divergence > 50%: check coordinate alignment, `reference_current_a`, grid coverage of STN point, column names
