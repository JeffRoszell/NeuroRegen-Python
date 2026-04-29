# Quickstart

Five minutes from clone to a 3-D B-field plot.

## 1. Install

```bash
cd "Ansys Alternative"
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.10–3.12, Linux / macOS / Windows + WSL2. PyVista's
interactive window needs a working OpenGL context — on a headless box use
`--no-show --save-html` instead.

## 2. Smoke test (no head mesh)

```bash
python simulate_bfield.py --check-only
```

You'll see:

```
[config] .../config.yaml
[target] STN_left at (-0.012, -0.013, -0.04) m
[coils ] 3: A(stacked), B(stacked), C(stacked)
[dI/dt] declared peak (for downstream E-field scaling):
  A: 84.30 A/μs   (I_peak = 3674 A)
  B: 84.30 A/μs   (I_peak = 3674 A)
  C: 84.30 A/μs   (I_peak = 3674 A)
[mesh] WARN: '.../head_models/skin.stl' not found — skipping head mesh.

[standoff check]
  coil      status     min |d|    target  verts in
  A        NO_MESH       n/a       4.0 mm        0
  B        NO_MESH       n/a       4.0 mm        0
  C        NO_MESH       n/a       4.0 mm        0
```

Exit code 0 = config parsed, coils built, no mesh available (expected on
first run).

## 3. Drop in a head mesh

Export the scalp surface from your SimNIBS m2m folder:

```bash
mkdir -p "Ansys Alternative/head_models"
cp ~/SimNIBS-4.x/m2m_<subject>/skin.stl \
   "Ansys Alternative/head_models/skin.stl"
```

If your mesh is in millimetres (SimNIBS default), set
`mesh.scale_to_m: 0.001` in `config.yaml`. SI-metres meshes need no change.

Re-run `--check-only` and you should see real distances:

```
[standoff check]
  coil      status     min |d|    target  verts in
  A             OK     4.21 mm   4.0 mm        0
  B             OK     3.87 mm   4.0 mm        0
  C       TOO_CLOSE    1.05 mm   4.0 mm        0
```

`TOO_CLOSE` / `TOO_FAR` / `INSIDE` flags use the SimNIBS-style 4 mm hair-gap
default ([SimNIBS discussion #348][simnibs348]). Adjust
`safety.skin_offset_m` in the config if your CAD coordinates already sit on
the scalp surface.

## 4. Compute fields + render

```bash
python simulate_bfield.py
```

A PyVista window opens with:

- semi-transparent scalp
- three coloured coil tubes (one polyline per turn)
- magnetic field streamlines threading each coil, coloured by `|B|` (log)
- three orthogonal `|B|` heat-map slices through the target
- a red sphere at `STN_left`

Console reports the superposed B-vector at the target:

```
[target] B at STN_left = (+0.142, -0.038, -1.875) mT  |B|=1.879 mT
```

## 5. Save outputs

```bash
python simulate_bfield.py \
    --no-show \
    --save-html outputs/scene.html \
    --save-vti  outputs/bfield.vti
```

- `scene.html` — single-file interactive viewer (works in any browser, share
  with collaborators without a Python install)
- `bfield.vti` — ParaView-readable structured grid containing the full
  B-vector field and `|B|` magnitude

## 6. Common tweaks

| Goal | How |
|---|---|
| Higher resolution near the target | `--spacing 0.003` (3 mm voxels, ≈ 1 M points) |
| True 3-D helical wire path | `--geometry helix` (slower, useful for ANSYS visual diff) |
| Different mesh file | `--mesh /path/to/your.stl` |
| Tighten / relax standoff envelope | edit `safety.tolerance_m` in `config.yaml` |
| Disable the skin-distance check entirely | remove the head mesh — the run still completes |

[simnibs348]: https://github.com/simnibs/simnibs/discussions/348
