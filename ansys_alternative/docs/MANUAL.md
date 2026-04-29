# NeuroRegen External B-field Simulator — Manual

A walk-through of the physics, the code, and every design choice — with
citations to the literature and library documentation that drove each
decision.

## Contents

1. [Scope and division of labour](#scope-and-division-of-labour)
2. [Physics primer](#physics-primer)
3. [Pipeline stages](#pipeline-stages)
4. [Configuration reference](#configuration-reference)
5. [Design decisions](#design-decisions)
6. [Extending the simulator](#extending-the-simulator)
7. [References](#references)

---

## Scope and division of labour

This tool computes the **external magnetic flux density** \(\mathbf{B}\) of
a multi-coil TMS array everywhere outside the head and visualises it. It is
the open-source replacement for the ANSYS Maxwell stage of the NeuroRegen
pipeline. It deliberately does **not** compute the induced electric field
inside conductive tissue — that step is handed off to SimNIBS [^thielscher2015].

| Stage | Tool | What it does |
|---|---|---|
| External \(\mathbf{B}\) field, coil placement, dosimetry preview | **this** | analytical Biot–Savart, vectorised over a 3-D grid, interactive 3-D viewer |
| Induced \(\mathbf{E}\) field in cortex / deep tissue | SimNIBS | finite-element solve on subject-specific head model |
| Time-domain pulse physics, thermal gating, safety state machine | upstream NeuroRegen package | LC-discharge, capacitor model, depth gate |

The split mirrors the validated coil-modeling pipeline used in TMS research,
where the coil-side magnetic field is treated separately from the
volume-conductor problem in the head [^drakaki2022] [^gomez2018].

---

## Physics primer

### Biot–Savart for a current loop

For a single circular loop of radius \(R\) carrying steady current \(I\),
the magnetic flux density at a field point \(\mathbf{r}\) is
\[
\mathbf{B}(\mathbf{r}) = \frac{\mu_0 I}{4\pi}
\oint \frac{d\mathbf{l}\times\hat{\mathbf{r}}'}{|\mathbf{r}'|^2}
\]
where \(\mathbf{r}' = \mathbf{r} - \mathbf{r}_\text{wire}\). For a single
loop the integral has a closed form in terms of the complete elliptic
integrals \(K\) and \(E\), which Magpylib's `current.Circle` evaluates with
the numerically stable expression of Ortner et al. [^ortner2022]. For an
arbitrary 3-D wire path, Magpylib's `current.Polyline` evaluates the
discrete Biot–Savart sum over straight segments [^ortner2020].

### Why analytical instead of FEM

Outside the head, tissue conductivity has a negligible effect on
\(\mathbf{B}\) at TMS frequencies (~kHz). The wavelength is enormous
relative to the head, the displacement-current term is irrelevant, and the
permeability is \(\mu_0\) almost everywhere. Under those conditions the
analytical free-space Biot–Savart field is essentially exact, while the FEM
solve in ANSYS Maxwell mostly absorbs computational expense for an answer
that does not change. See `docs/VS_ANSYS.md` for the trade table.

### Why \(\mathbf{B}\) is enough for visualisation

Stimulation efficacy depends on the induced electric field
\(\mathbf{E} = -\partial\mathbf{A}/\partial t\), which scales as
\(\mathbf{E} \propto dI/dt\). Because the system is linear in \(I\), a
\(\mathbf{B}\)-field rendered at peak current and a peak \(dI/dt\) reported
alongside it together carry all the information needed for downstream
\(\mathbf{E}\) reporting. This is the convention used in the validated
TMS coil database [^drakaki2022], where coil libraries are normalised to a
reference \(dI/dt = 1\)\,A/μs so that any operating point can be recovered
by a single multiplication.

The config field `peak_dIdt_a_per_us` per coil makes this link explicit.

---

## Pipeline stages

```
config.yaml ──► CoilSpec dataclasses
                       │
                       ▼
            build_coil_sources()
                stacked: N × Circle  ◄── default, idiomatic magpylib
                helix:   1 × Polyline
                       │
                       ▼
            magpy.Collection ────► Collection.getB(observers)  ←─ vectorised
                       │                          │
                       │                          ▼
                       │             pyvista.ImageData("B", "B_mag")
                       │                          │
                       ▼                          ▼
            coil polylines        ┌──────── streamlines ────────┐
                       │           │   slice_orthogonal()       │
                       ▼           │   target marker            │
                pyvista.PolyData ──┴──── pyvista.Plotter() ─────┘
                       ▲
                       │
            head mesh ─┴──► standoff check (signed-distance vs skin offset)
```

### 1. Geometry construction (`build_coil_sources`)

Each `CoilSpec` produces:

- **`geometry: stacked`** (default) — \(N\) `magpy.current.Circle`
  loops distributed along the coil normal, each oriented with the
  Rodrigues rotation that maps the local +z axis to `normal`. This is
  Magpylib's recommended idiom for solenoidal geometry [^magpy_coils].
- **`geometry: helix`** — one `magpy.current.Polyline` of
  `turns × segments_per_turn + 1` vertices following an exact helix.
  Slower; chosen when matching the visual wire path of an ANSYS Maxwell
  build is the priority.

Both produce identical far-field results to within the elliptic-integral
precision; differences appear only when the field point sits within
roughly one wire radius of the conductor.

### 2. Mesh ingestion (`load_head_mesh`)

PyVista [^sullivan2019] reads `.stl`, `.ply`, `.vtk`, `.msh`, and `.obj`.
SimNIBS exports the scalp surface as `m2m_<subject>/skin.stl` in
millimetres; we apply `mesh.scale_to_m` and an optional rigid translation
to bring the mesh into the simulator's metres-and-head-centred frame.

### 3. Standoff check (`check_coil_standoff`)

For each coil we sample winding vertices, query
`select_enclosed_points` (inside / outside test) and
`compute_implicit_distance` (unsigned distance), and combine them into a
signed distance. The minimum is compared against the SimNIBS-style
`skin_offset_m` (default 4 mm, accounting for hair) [^simnibs348], with
results classified as `INSIDE` / `TOO_CLOSE` / `OK` / `TOO_FAR`. The
four-class scheme follows the failure modes flagged by the TAP coil-placement
verification pipeline [^wang2022].

### 4. Field computation (`compute_B_on_grid`)

`pyvista.ImageData` builds the structured voxel grid; `img.points` gives the
flat `(N, 3)` observer array. A single `Collection.getB(observers)` call
evaluates Biot–Savart at every point, summing the contributions of every
source in the collection. For 5 mm voxels in a 26 cm cube (~150 k points)
this finishes in roughly a second on a modern CPU.

### 5. Rendering (`render_scene`)

- Scalp mesh: semi-transparent (`opacity=0.25`).
- Coil tubes: one `PolyData` per coil with N polyline cells (one per turn
  for `stacked`, one for `helix`); rendered as tubes with
  `render_lines_as_tubes=True`.
- Streamlines: seeded from per-coil disks (random points in a disk in the
  coil plane), integrated bidirectionally via
  `streamlines_from_source`, rendered as tubes coloured by `|B|` (log
  colour map).
- Slices: three orthogonal `|B|` heat-maps through the target, generated
  by `slice_orthogonal(x, y, z)`.
- Target: red sphere + label.

---

## Configuration reference

See `config.yaml` for the canonical example. Every block in detail:

### `mesh`
| Field | Type | Default | Notes |
|---|---|---|---|
| `path` | str | `head_models/skin.stl` | Relative to config file or absolute |
| `scale_to_m` | float | `1.0` | Use `0.001` for SimNIBS mm meshes |
| `translate_m` | `[x,y,z]` | `[0,0,0]` | Applied after scaling |

### `target`
| Field | Type | Notes |
|---|---|---|
| `name` | str | Anatomical label |
| `position_m` | `[x,y,z]` | Head-centred frame, metres |

### `coils[]`
| Field | Type | Notes |
|---|---|---|
| `loop_diameter_m` | float | Outer loop diameter |
| `wire_diameter_m` | float | Cross-section, currently used only for visualisation tube radius |
| `turns` | int | Number of windings |
| `helix_height_m` | float | Total axial length |
| `segments_per_turn` | int | Polyline tessellation |
| `current_a` | float | Operating current (peak for pulsed mode) |
| `peak_dIdt_a_per_us` | float | Optional metadata for downstream E reports [^drakaki2022] |
| `geometry` | `"stacked"` \| `"helix"` | See [Design decisions](#design-decisions) |
| `position_m` | `[x,y,z]` | Coil centre |
| `normal` | `[nx,ny,nz]` | Auto-normalised; +z of local frame maps to this vector |

### `grid`
Bounding box (`xmin … zmax`) and isotropic `spacing_m`. Smaller spacing →
quadratic memory and cubic field-eval cost.

### `viz`
Cosmetics only — colours, opacity, streamline density, log scale toggle.

### `safety`
| Field | Default | Notes |
|---|---|---|
| `skin_offset_m` | `0.004` | SimNIBS hair-gap default [^simnibs348] |
| `tolerance_m` | `0.003` | Half-width of the OK band |
| `intersection_sample_points` | `256` | Sub-sampling of helix vertices |

---

## Design decisions

### Why stacked Circles instead of a true 3-D helix

Magpylib's documentation [^magpy_coils] explicitly demonstrates the
solenoid-as-stack-of-Circles pattern. The reasons:

- **Numerical stability.** `Circle` uses the closed-form expression of
  Ortner et al. [^ortner2022], which is well-conditioned across the entire
  half-space outside the wire. A helical Polyline introduces many segment
  endpoints, each of which contributes a small singularity in the
  Biot–Savart sum at points close to the wire.
- **Speed.** A 6-turn coil at 64 segments/turn is 384 segments;
  the Circle stack is 6. For a 150 k-point observer grid that's a 60×
  reduction in floating-point work.
- **Idiom.** Future readers of the code recognise the stack-of-Circles
  pattern immediately.

The `helix` mode remains available because matching the visual wire path
of an ANSYS Maxwell solid is sometimes more important than speed —
specifically when generating side-by-side renders for publication or
pre-flight cross-checking against a Maxwell solve.

### Why the four-state standoff check

The TAP coil placement pipeline [^wang2022] documents two failure modes that
both produce silently wrong simulations: a coil floating too far from the
scalp (under-stimulation, plausible-looking but quantitatively wrong field
at depth) and a coil whose CAD coordinate has drifted inside the scalp
mesh (geometry violates the model assumption that the conductor is in
free space). A binary "intersects?" flag catches only the second; the
four-state scheme (`INSIDE` / `TOO_CLOSE` / `OK` / `TOO_FAR`) catches both
and adds an explicit `OK` band centred on the SimNIBS hair-gap default of
4 mm [^simnibs348].

### Why `peak_dIdt_a_per_us` is metadata, not a driver

\(\mathbf{B}\) is linear in \(I\), so simulating at peak current and
declaring \(dI/dt_\text{peak}\) separately preserves every degree of
freedom needed for downstream \(\mathbf{E}\) reporting without
double-bookkeeping. This matches the "1 A/μs reference" convention used
when distributing pre-computed coil libraries [^drakaki2022], where users
multiply the stored field by their device's actual peak \(dI/dt\) at
analysis time.

### Why pure SI throughout

The upstream NeuroRegen package was already SI-internal. Magpylib v5
defaults to SI [^magpy_coils]. Mixing units (mm in some places, m in
others) is a documented source of TMS coil-modeling errors [^drakaki2022].
Mesh-side mm↔m conversion happens at exactly one place: `load_head_mesh`.

---

## Extending the simulator

| Want | Where to start |
|---|---|
| Different coil shape (figure-8, butterfly, H-coil) | `build_coil_sources` — add a new branch returning a list of `Circle` / `Polyline` sources |
| Permanent magnets / other sources | Add to the `magpy.Collection` returned by `build_collection` |
| Time-varying current (transient) | Wrap the `compute_B_on_grid` call in a loop over time samples; rescale current per step |
| Compare against ANSYS export | Generate `--save-vti` and load both into ParaView — same VTI format |
| Hand the field grid to SimNIBS as a primary | Export VTI then convert to SimNIBS' magnetic vector potential format (downstream module) |

---

## References

[^thielscher2015]: Thielscher, A., Antunes, A., & Saturnino, G. B. (2015).
"Field modeling for transcranial magnetic stimulation: A useful tool to
understand the physiological effects of TMS?" *37th Annual International
Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*.
SimNIBS framework. <https://simnibs.github.io/simnibs/>

[^ortner2020]: Ortner, M., & Coliado Bandeira, L. G. (2020). "Magpylib:
A free Python package for magnetic field computation." *SoftwareX*, 11,
100466. <https://www.sciencedirect.com/science/article/pii/S2352711020300170>

[^ortner2022]: Ortner, M., Helbig, T., et al. (2022). "Numerically stable
and computationally efficient expression for the magnetic field of a
current loop." Closed-form formula used by `magpylib.current.Circle`.
Referenced from the Magpylib physics guide:
<https://magpylib.readthedocs.io/en/stable/_pages/user_guide/guide_resources_01_physics.html>

[^magpy_coils]: Magpylib documentation, "Coils" example.
<https://magpylib.readthedocs.io/en/stable/_pages/user_guide/examples/examples_app_coils.html>

[^drakaki2022]: Drakaki, M., Mathiesen, C., Siebner, H. R., Madsen, K., &
Thielscher, A. (2022). "Database of 25 validated coil models for electric
field simulations for TMS." *Brain Stimulation*, 15(3), 697–706.
<https://www.brainstimjrnl.com/article/S1935-861X(22)00077-8/fulltext>

[^gomez2018]: Gomez, L. J., Goetz, S. M., & Peterchev, A. V. (2018). "How
much detail is needed in modeling a transcranial magnetic stimulation
figure-8 coil: Measurements and brain simulations." *Brain Stimulation*.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC5480865/>

[^wang2022]: Wang, B., Aberra, A. S., Grill, W. M., & Peterchev, A. V.
(2022). "TAP: targeting and analysis pipeline for optimization and
verification of coil placement in transcranial magnetic stimulation."
*Journal of Neural Engineering*.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9131512/>

[^simnibs348]: SimNIBS GitHub — "Skin Distance Parameter," Discussion #348
(documents the 4 mm default and what it represents).
<https://github.com/simnibs/simnibs/discussions/348>

[^sullivan2019]: Sullivan, C. B., & Kaszynski, A. (2019). "PyVista: 3D
plotting and mesh analysis through a streamlined interface for the
Visualization Toolkit (VTK)." *Journal of Open Source Software*, 4(37),
1450. <https://joss.theoj.org/papers/10.21105/joss.01450>

### Foundational

- Barker, A. T., Jalinous, R., & Freeston, I. L. (1985). "Non-invasive
  magnetic stimulation of human motor cortex." *The Lancet*, 325(8437),
  1106–1107. The original TMS paper.
- Jackson, J. D. (1998). *Classical Electrodynamics* (3rd ed.), Wiley.
  Biot–Savart, vector potentials, magnetostatics. Chapter 5.
- Schroeder, W., Martin, K., & Lorensen, B. (2006). *The Visualization
  Toolkit* (4th ed.), Kitware. The library underneath PyVista.
