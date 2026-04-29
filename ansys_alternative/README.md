# NeuroRegen — External B-field Simulator

ANSYS Maxwell replacement for **outside-the-head** magnetic-field calculation
and 3-D visualisation of the NeuroRegen 3-coil TMS array. Internal cortical /
deep-brain E-fields are still solved by SimNIBS.

## Documentation

| File | Audience |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Get a B-field plot in 5 minutes. |
| [`docs/MANUAL.md`](docs/MANUAL.md) | Physics primer, pipeline walk-through, design decisions, citations. |
| [`docs/VS_ANSYS.md`](docs/VS_ANSYS.md) | Pros / cons vs ANSYS Maxwell, recommended workflow. |

## Stack

- **magpylib** — analytical Biot–Savart for arbitrary 3-D current paths
- **PyVista** — VTK rendering, ImageData wrapping, streamlines, slice planes
- **trimesh / numpy** — mesh ops + vectorised grid math
- **PyYAML** — single config file mirroring the validated Maxwell3D build

## Setup

```bash
cd "Ansys Alternative"
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Default: load config.yaml + scalp mesh, render full 3-D scene
python simulate_bfield.py

# Override the mesh path from the SimNIBS export folder
python simulate_bfield.py --mesh ../SimNIBS/m2m_subject1/skin.stl

# Higher-resolution field grid (3 mm voxels — slower)
python simulate_bfield.py --spacing 0.003

# Force true-helix Polyline geometry (slower, matches Maxwell wire path
# visually). Default is stacked Circle loops — magpylib idiomatic, faster,
# and uses the closed-form Ortner-2022 current-loop formula.
python simulate_bfield.py --geometry helix

# Headless: save scene to standalone HTML + grid to ParaView VTI
python simulate_bfield.py --no-show \
    --save-html outputs/scene.html \
    --save-vti  outputs/bfield.vti

# Just sanity-check coil/scalp standoff (no field calc, no plot)
python simulate_bfield.py --check-only
```

## Configuration (`config.yaml`)

| Section | What it controls |
|---|---|
| `mesh`   | Path to head .stl/.msh, units, optional translation |
| `target` | Anatomical target (default: STN_left, head-centred frame) |
| `coils`  | Per-coil geometry (`stacked` / `helix`), position, normal, peak current, optional `peak_dIdt_a_per_us` |
| `grid`   | Bounding box + voxel spacing for the observation cube |
| `viz`    | Streamline density, slice planes, opacity, log scale |
| `safety` | `skin_offset_m` (SimNIBS default 4 mm), `tolerance_m`, sample count |

The default coil block matches the validated Maxwell build
(`3AxisTeslaCoil_6Turns_10mmWire_150mmLargest_SeparateLeads_300mm_Apr15.aedt`):
6 turns × 150 mm loop × 10 mm wire, 3 674 A peak, helix height 35.7 mm.

## Outputs

- **Interactive PyVista window** — rotate / zoom / inspect the scene
- `--save-html`  — single self-contained file (works in any browser)
- `--save-vti`   — ParaView-compatible field grid (B-vectors + \|B\|)

The scene contains:
- Semi-transparent scalp mesh
- 3-D helix coils (colour-coded, labelled)
- Streamlines through each coil (tubes coloured by \|B\|)
- Three orthogonal \|B\| heat-map slices through the target
- Red sphere marker at the target

## Notes

- All units are SI (m, A, T) end-to-end. If the SimNIBS mesh is in mm,
  set `mesh.scale_to_m: 0.001` in the config.
- The B-field calculation is one vectorised `Collection.getB(observers)` call
  — fast enough that 3 mm voxels (~1 M points) finish in a few seconds.
- **Coil geometry: `stacked` is the default and follows magpylib's own
  recommendation** ([magpylib coils example](https://magpylib.readthedocs.io/en/stable/_pages/user_guide/examples/examples_app_coils.html)).
  Each turn becomes a `magpylib.current.Circle` with the closed-form
  Ortner-2022 field — fast and singularity-free outside the wire. Use
  `geometry: helix` (or `--geometry helix`) when you specifically need the
  true 3-D wire path for a Maxwell-equivalent visual.
- **Standoff check** follows the SimNIBS convention
  ([discussion #348](https://github.com/simnibs/simnibs/discussions/348)):
  default `skin_offset_m = 0.004` (4 mm hair gap). Coils are flagged
  `INSIDE`, `TOO_CLOSE`, `OK`, or `TOO_FAR` based on the signed distance
  from sampled winding vertices to the scalp surface.
- `peak_dIdt_a_per_us` per coil is metadata only — printed with the run for
  downstream E-field reports (per the
  [Database of 25 validated TMS coils](https://www.brainstimjrnl.com/article/S1935-861X(22)00077-8/fulltext)
  convention). The B-field plot itself is linear in `current_a`.

## Citations

If you use this tool in a publication, please cite both the underlying
physics libraries and the TMS-modeling literature it relies on. Full
inline citations are in [`docs/MANUAL.md#references`](docs/MANUAL.md#references)
and [`docs/VS_ANSYS.md#references`](docs/VS_ANSYS.md#references); the
canonical entries:

- **Magpylib** (B-field engine):
  Ortner, M., & Coliado Bandeira, L. G. (2020). "Magpylib: A free Python
  package for magnetic field computation." *SoftwareX*, 11, 100466.
  <https://www.sciencedirect.com/science/article/pii/S2352711020300170>

- **Magpylib current-loop formula**:
  Ortner, M., Helbig, T., et al. (2022). "Numerically stable and
  computationally efficient expression for the magnetic field of a current
  loop." Used by `magpylib.current.Circle`.
  <https://magpylib.readthedocs.io/en/stable/_pages/user_guide/guide_resources_01_physics.html>

- **PyVista** (3-D rendering):
  Sullivan, C. B., & Kaszynski, A. (2019). "PyVista: 3D plotting and mesh
  analysis through a streamlined interface for the Visualization Toolkit
  (VTK)." *Journal of Open Source Software*, 4(37), 1450.
  <https://joss.theoj.org/papers/10.21105/joss.01450>

- **SimNIBS** (downstream E-field handoff):
  Thielscher, A., Antunes, A., & Saturnino, G. B. (2015). "Field modeling
  for transcranial magnetic stimulation: A useful tool to understand the
  physiological effects of TMS?" *37th IEEE EMBC*.
  <https://simnibs.github.io/simnibs/>

- **Validated TMS coil convention** (peak dI/dt normalisation):
  Drakaki, M., Mathiesen, C., Siebner, H. R., Madsen, K., & Thielscher, A.
  (2022). "Database of 25 validated coil models for electric field
  simulations for TMS." *Brain Stimulation*, 15(3), 697–706.
  <https://www.brainstimjrnl.com/article/S1935-861X(22)00077-8/fulltext>

- **Coil-placement verification** (standoff failure modes):
  Wang, B., Aberra, A. S., Grill, W. M., & Peterchev, A. V. (2022). "TAP:
  targeting and analysis pipeline for optimization and verification of
  coil placement in TMS." *Journal of Neural Engineering*.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9131512/>

- **Skin-distance default**:
  SimNIBS Discussion #348 — *Skin Distance Parameter*.
  <https://github.com/simnibs/simnibs/discussions/348>
