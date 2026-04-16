# NeuroRegen — Simulation Settings Breakdown

_Last updated: 2026-04-15. Intended audience: project group members reviewing configuration choices._

---

## Overview

Three independent simulations cover different aspects of the same physical hardware. Each is driven by its own YAML config file; no values are hardcoded in source.

| Simulation | Script | Config | Purpose |
|---|---|---|---|
| Single-coil 3-axis | `run_simulation.py` | `config/default.yaml` | Compact cortical coil, per-axis thermal & depth |
| Multicoil deep-brain | `run_multicoil.py` | `config/multicoil.yaml` | 3-coil array focused on STN, cortical safety gate |
| Pulsed capacitor-discharge | `run_pulsed.py` | `config/pulsed_tms.yaml` | LC-discharge physics, matches ANSYS AEDT build |

The **multicoil** and **pulsed** simulations use the same physical coil geometry. The **single-coil** simulation models a separate, smaller compact coil from published literature.

---

## 1. Single-Coil 3-Axis — `config/default.yaml`

### What it models
Three small orthogonal coil axes (X, Y, Z) on a compact TMS headset. Based on the Navarro de Lara et al. 2021 design (NeuroImage 224, 117355).

### Coil geometry
| Parameter | X axis | Y axis | Z axis | Source |
|---|---|---|---|---|
| Wire diameter | 2 mm Litz | 2 mm Litz | 2 mm Litz | Navarro de Lara 2021, §2.1 |
| Loop diameter | 42.85 mm | 42.85 mm | 38.0 mm | Fig 1A (mean inner/outer) |
| Turns | 18 | 22 | 18 | §2.1 |
| Helix height | 8.0 mm | 8.0 mm | 19.6 mm | Fig 1A |
| Base pulse power | 1 000 W | 1 000 W | 1 000 W | — |

- X/Y loop: mean of inner 36 mm / outer 49.70 mm from Fig 1A
- Z loop: mean of inner 26 mm / outer 50 mm from Fig 1A
- Y has more turns (22) to match inductance of the X element

### Simulation timing
| Parameter | Value | Notes |
|---|---|---|
| Sim duration | 1 800 s (30 min) | — |
| Time step (dt) | 0.1 s | — |
| Pulse frequency | 5 Hz | Standard facilitatory rTMS |
| Pulse width | 326 µs | Bi-phasic half-width, Navarro de Lara 2021 Fig 2B |

### Thermal limits
| Parameter | Value | Notes |
|---|---|---|
| Ambient temperature | 22 °C | — |
| Convective coefficient | 10 W/m²/K | Natural convection at scalp |
| Thermal gate ON | 105 °F (40.6 °C) | IEC 60601-1 skin-contact limit |
| Thermal gate OFF (resume) | 103.5 °F (39.7 °C) | 1.5 °F hysteresis |

### Depth evaluation
| Parameter | Value |
|---|---|
| Max depth evaluated | 3 cm |
| Grid points | 300 |
| B-field penetration threshold | 0.1 mT |

---

## 2. Multicoil Deep-Brain — `config/multicoil.yaml`

### What it models
Three large coils placed on the skull surface aimed at the left STN. Distance compensation (d⁶ weighting) ensures each coil delivers equal B-field at the target. A cortical depth gate blocks pulses if any coil's surface E-field would exceed the safety limit.

### Target
| Parameter | Value |
|---|---|
| Target name | STN_left |
| Position (head-centred) | x = −12 mm, y = −13 mm, z = −40 mm |
| Approximate depth from skull | ~4 cm |

Coordinate convention: origin at geometric skull centre; +x right, +y anterior, +z superior.

### Safety thresholds
| Parameter | Value | Notes |
|---|---|---|
| Cortical E-field limit | 150 V/m | Depth gate blocks entire pulse if exceeded |
| Scalp-to-cortex distance | 15 mm | Scalp + skull combined thickness |

### Coil geometry (all three identical hardware)
| Parameter | Value | Notes |
|---|---|---|
| Wire diameter | 10 mm | Matches ANSYS AEDT build |
| Loop diameter | 150 mm | Matches ANSYS AEDT build |
| Turns | 6 | Matches ANSYS AEDT build |

### Coil positions, orientations, and power
Each coil sits on the skull surface with its normal vector aimed toward the STN target.

| Coil | Position (m) | Dist. to STN | d⁶ weight | Base power | Weighted power | Cortical E |
|---|---|---|---|---|---|---|
| A (vertex) | (−0.010, −0.010, +0.088) | 12.81 cm | 6.79 | **1 200 W** | 8 148 W | ~149 V/m |
| B (lateral R) | (+0.085, −0.020, +0.025) | 11.70 cm | 3.95 | **2 075 W** | 8 196 W | ~149 V/m |
| C (posterior) | (−0.015, −0.088, +0.015) | 9.31 cm | 1.00 | **8 100 W** | 8 100 W | ~149 V/m |

**Why base powers differ:** the d⁶ weight amplifies each coil's base power to equalize B-field at the target. With equal base powers, the farthest coil (A, weight 6.79×) would hit 272 V/m — far above the safety limit. By setting each coil's base power to `max_safe_weighted_power / weight`, all three arrive at ~149 V/m simultaneously. The maximum achievable weighted power at 150 V/m is ~8 200 W for this coil geometry.

### Simulation timing
| Parameter | Value | Notes |
|---|---|---|
| Sim duration | 1 800 s (30 min) | — |
| Time step (dt) | 0.1 s | — |
| Pulse frequency | 5 Hz | — |
| Pulse width | 137 µs | LC half-period of the hardware capacitor bank (π√LC, 200 µF / 9.5 µH) |

### Thermal limits (same as single-coil)
| Parameter | Value |
|---|---|
| Ambient temperature | 22 °C |
| Convective coefficient | 10 W/m²/K |
| Thermal gate ON | 105 °F |
| Thermal gate OFF | 103.5 °F |

### Depth evaluation
| Parameter | Value |
|---|---|
| Max depth | 12 cm (deep target) |
| Grid points | 500 |
| B threshold | 0.1 mT |

### Simulation results at optimal config
| Metric | Value |
|---|---|
| Peak \|B\| at STN target | **79.8 mT** |
| Depth-gate failures | 0 / 18 000 steps (0%) |
| Max cortical E-field | 149.4 V/m |
| Peak coil temperature at 30 min | < 80 °F (well under 105 °F limit) |
| Average thermal power per coil | ~11 W (pulse_fraction = 137 µs / 100 ms = 0.00137) |

### Note on neural activation threshold
79.8 mT at the STN corresponds to a peak dB/dt of ~582 T/s and an induced E-field of ~1.5 V/m at the target (using Faraday's law, r_tissue = 5 mm). Direct neural activation typically requires 50–150 V/m. This simulation therefore demonstrates **geometric focusing and cortical safety compliance**, not suprathreshold deep activation. Whether sub-threshold fields produce modulatory effects is an open research question.

---

## 3. Pulsed Capacitor-Discharge — `config/pulsed_tms.yaml`

### What it models
Full LC-discharge physics for the same 3-coil hardware. Intended to match the ANSYS AEDT build file (`3AxisTeslaCoil_6Turns_10mmWire_150mmLargest_SeparateLeads_300mm_Apr15.aedt`).

### Capacitor bank
| Parameter | Value | Notes |
|---|---|---|
| Capacitance | 200 µF (per coil) | Independently switched |
| Charge voltage | 798 V | Sweet spot: cortex at 150 V/m limit |

### Derived LC parameters
| Parameter | Value | Formula |
|---|---|---|
| Inductance (L) | ~9.5 µH | Neumann formula; single loop ≈ 0.26 µH × N² = × 36 |
| Peak current (I_peak) | ~3 674 A | V√(C/L) |
| Pulse half-period (τ) | ~137 µs | π√(LC) |
| Energy per pulse | ~63.8 J | ½CV² per coil |

### Simulation timing
| Parameter | Value | Notes |
|---|---|---|
| Number of pulses | 600 | 60 s at 10 Hz |
| Pulse frequency | 10 Hz | Standard facilitatory rTMS |
| Cooling integration step | 1 ms | Between pulses |

### Power budget check
| Parameter | Value |
|---|---|
| Max frequency (1 outlet) | ~9.8 Hz (3 coils × 63.8 J / (2 200 W × 0.85 eff.)) |
| 10 Hz operation | Fills one 110V/20A outlet exactly |

### Thermal limits
Same as multicoil: 105 °F gate ON, 103.5 °F resume (IEC 60601-1).

### Safety limits
| Parameter | Value |
|---|---|
| Cortical E-field limit | 150 V/m |
| Scalp-to-cortex distance | 15 mm |
| Tissue loop radius (STN E calc.) | 5 mm |

### Coil geometry
Same hardware as multicoil: 150 mm / 6-turn / 10 mm wire. Positions and normals identical.

---

## Shared Design Rules

1. **All parameters in YAML** — no hardcoded simulation values in source code.
2. **Thermal gating with hysteresis** — pulses are individually skipped (not the full session halted) when a coil exceeds the temperature limit; they resume automatically when the coil cools below (limit − 1.5 °F).
3. **Pulse-fraction thermal fix** — thermal energy deposited per timestep is scaled by `min(pulse_width, dt) / dt` to avoid overcounting when `dt >> pulse_width`. Peak power (for B-field / E-field calculations) uses the full pulse power.
4. **Cortical depth gate** — in multicoil and pulsed modes, the entire pulse is blocked if *any* coil exceeds the cortical E-field limit, not just that coil.
5. **Distance compensation (multicoil)** — weights ∝ d⁶ so B ∝ √P/d³ is equal from all coils at the target.
6. **ANSYS field maps optional** — when absent the simulation falls back to analytical Biot-Savart approximation throughout.

---

## Known Approximation / Limitation

In `pulsed.py`, the cortical E-field uses the coil loop radius (75 mm) as the tissue loop radius. This is conservative and overestimates cortical E by ~4–15×. The STN E-field uses the configurable `r_tissue_m` (default 5 mm), which is more realistic. Neither matches SimNIBS or ANSYS output directly — they are analytical free-space estimates only.
