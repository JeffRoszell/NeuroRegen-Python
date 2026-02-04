#!/usr/bin/env python3
"""
3‑Axis Tesla Coil — High‑Pulse 30‑Minute Cycle (Educational Simulation)
=====================================================================

Purpose
-------
This script models a **30‑minute (1800 s) stimulation cycle** using
**many short pulses** (high pulse count) while respecting a **75°F temperature ceiling**.

Key behavior
------------
• Power per axis is CONSTANT during each pulse
• Pulses repeat at a fixed frequency (many pulses over 30 min)
• If temperature reaches 75°F, pulses are temporarily skipped
• When temperature cools below the hysteresis band, pulsing resumes

This mirrors how real neuromodulation systems:
- deliver repeated pulses
- maintain thermal safety
- preserve constant pulse amplitude

Outputs
-------
1) Temperature (°F) vs time (X/Y/Z + average)
2) Applied pulsed power (W) vs time
3) Estimated effective depth (cm) vs time (≤ 3 cm)

Run
---
python tesla_3axis_pulsed_30min.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------- Constants --------------------
MU0 = 4e-7 * np.pi
RHO_CU = 1.68e-8
ALPHA_CU = 0.0039
DENSITY_CU = 8960
CP_CU = 385

H_CONV = 10.0
T_AMB_C = 22.0

TEMP_LIMIT_F = 75.0
HYST_F = 0.7

Z_MAX_M = 0.03
Z_POINTS = 300
B_THRESHOLD_T = 1e-4

# -------------------- Parameters --------------------
SIM_TIME = 1800.0        # 30 minutes
DT = 0.1                # time step (s)

PULSE_FREQ = 5.0        # pulses per second (5 Hz → 9000 pulses)
PULSE_WIDTH = 0.02      # seconds ON per pulse

# --------------------------------------------------

@dataclass
class Axis:
    name: str
    wire_mm: float
    loop_mm: float
    turns: int
    pulse_power_w: float


def c_to_f(tc): return tc * 9/5 + 32
def f_to_c(tf): return (tf - 32) * 5/9


def coil_geom(axis):
    R = axis.loop_mm / 2000
    L = axis.turns * 2 * np.pi * R
    A = np.pi * (axis.wire_mm/2000)**2
    S = L * np.pi * (axis.wire_mm/1000)
    m = DENSITY_CU * L * A
    return R, L, A, S, m


def resistance(L, A, T):
    R0 = RHO_CU * L / A
    return R0 * (1 + ALPHA_CU*(T-20))


def B_loop(I, R, z, N):
    return MU0 * I * R**2 / (2*(R**2+z**2)**1.5) * N


# -------------------- Simulation --------------------
t = np.arange(0, SIM_TIME, DT)
z = np.linspace(0, Z_MAX_M, Z_POINTS)

axes = [
    Axis("X", 1.2, 80, 20, 45),
    Axis("Y", 1.2, 80, 20, 45),
    Axis("Z", 1.2, 80, 20, 45),
]

T = np.full((len(t),3), T_AMB_C)
P = np.zeros((len(t),3))
depth = np.zeros((len(t),3))

limit_c = f_to_c(TEMP_LIMIT_F)
hyst_c = f_to_c(TEMP_LIMIT_F - HYST_F)

geom = [coil_geom(a) for a in axes]
Cth = [g[4]*CP_CU for g in geom]

for k in range(1,len(t)):
    pulse_on = ( (t[k] % (1/PULSE_FREQ)) < PULSE_WIDTH )

    for i,a in enumerate(axes):
        R,L,A,S,m = geom[i]
        on = pulse_on and (T[k-1,i] < limit_c or T[k-1,i] < hyst_c)

        Pin = a.pulse_power_w if on else 0
        P[k,i] = Pin

        Pcool = H_CONV * S * (T[k-1,i]-T_AMB_C)
        T[k,i] = T[k-1,i] + (Pin-Pcool)/Cth[i]*DT

        I = np.sqrt(Pin / resistance(L,A,T[k-1,i])) if Pin>0 else 0
        Bz = B_loop(I,R,z,a.turns)
        depth[k,i] = z[Bz>=B_THRESHOLD_T][-1]*100 if np.any(Bz>=B_THRESHOLD_T) else 0

# -------------------- Plots --------------------
plt.figure()
plt.plot(t/60, c_to_f(T[:,0]), label="X")
plt.plot(t/60, c_to_f(T[:,1]), label="Y")
plt.plot(t/60, c_to_f(T[:,2]), label="Z")
plt.axhline(TEMP_LIMIT_F, ls="--", label="75°F limit")
plt.xlabel("Time (min)")
plt.ylabel("Temperature (°F)")
plt.title("Temperature vs Time (30‑min Pulsed Session)")
plt.legend(); plt.grid(True)

plt.figure()
plt.step(t/60, P[:,0], label="X pulses")
plt.step(t/60, P[:,1], label="Y pulses")
plt.step(t/60, P[:,2], label="Z pulses")
plt.xlabel("Time (min)")
plt.ylabel("Applied Power (W)")
plt.title("High‑Pulse Power Pattern (30‑min)")
plt.legend(); plt.grid(True)

plt.figure()
plt.plot(t/60, depth[:,0], label="X depth")
plt.plot(t/60, depth[:,1], label="Y depth")
plt.plot(t/60, depth[:,2], label="Z depth")
plt.xlabel("Time (min)")
plt.ylabel("Depth (cm)")
plt.title("Estimated Penetration Depth (≤3 cm)")
plt.legend(); plt.grid(True)

plt.show()
