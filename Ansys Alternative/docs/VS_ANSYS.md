# magpylib + PyVista vs ANSYS Maxwell

A frank comparison for the NeuroRegen use case (multi-coil TMS array,
external magnetic field, non-invasive head model). TL;DR: **use both** —
ANSYS for final-design validation and any problem involving conductive
volumes other than tissue (shields, bobbins, eddy currents in metal),
this stack for everything else.

## At a glance

| | magpylib + PyVista (this) | ANSYS Maxwell |
|---|---|---|
| Solver type | Analytical Biot–Savart (closed form for `Circle`, segment sum for `Polyline`) [^ortner2022] [^ortner2020] | Finite-element method, full Maxwell equations |
| Time-domain coverage | Snapshot at peak \(I\); transients by scripted re-runs | Native transient solver, eddy-current AC, magnetostatic |
| Geometry input | Analytical primitives (loops, polylines) + arbitrary triangulated meshes for visualisation | CAD-quality solids, tet/hex mesh |
| Material model | Vacuum / free space only (\(\mu = \mu_0\) everywhere) | Linear, nonlinear B-H, anisotropic, eddy currents in conductors |
| Typical solve time, 3-coil 5 mm grid | ~1 second | minutes to hours |
| Hardware | Any laptop; vectorised single CPU is fine | Maxwell licence, often a workstation or HPC node |
| Cost | Free, MIT / BSD-style open source | Commercial licence (per-seat, often $10k–$30k/yr) |
| Reproducibility | YAML + Python, fully git-trackable | `.aedt` binary project file |
| Headless / CI | Trivial (`--no-show`) | PyAEDT exists but is heavy |
| Visualisation | PyVista interactive, HTML export, ParaView VTI | Built-in field plotter, in-app only |
| TMS-specific validation | Uses the same coil-loop closed form as the validated TMS coil database [^drakaki2022] | Independently validated [^gomez2018] |

## What ANSYS Maxwell still does better

1. **Conductive volumes that aren't tissue.** A real coil has an aluminium
   bobbin, a copper backplate, sometimes an iron shield. Eddy currents
   induced in those volumes back-react on \(\mathbf{B}\) at TMS
   frequencies. magpylib treats them as transparent.
2. **Non-circular wire cross-section / proximity effect.** At 3.7 kA peak
   the current redistributes within a 10 mm wire (skin and proximity
   effects). magpylib treats every conductor as a filament.
3. **Transient EM with switching events.** Capacitor-discharge transients,
   mutual inductance during the pulse, induced voltages on adjacent coils
   — Maxwell's transient solver handles all of these in one project.
4. **Coupled physics.** Joule heating of the conductor, mechanical
   forces on the windings, structural deformation under Lorentz load —
   each is one workbench tab away.
5. **Manufacturing-grade analysis.** Validation runs that have to survive
   regulatory review benefit from a single tool with a long compliance
   pedigree.

## What this stack does better

1. **Speed.** A 1 M-point grid in seconds vs. a multi-hour Maxwell solve.
   Iterating coil placement is genuinely interactive; in Maxwell each
   "what if I move it 5 mm" is a coffee break.
2. **Accuracy where it matters for TMS.** The B-field outside the head
   doesn't depend on tissue conductivity at TMS frequencies — the
   analytical free-space answer is the answer Maxwell converges to,
   modulo discretisation error in Maxwell's mesh. For coil placement
   and dosimetry preview the analytical result is *more* accurate than a
   coarse FEM solve, not less.
3. **Reproducibility.** A 50-line YAML config + git hash uniquely
   reproduces the simulation. ANSYS `.aedt` files are binary, version-
   sensitive, and not diffable.
4. **Headless and CI-native.** Drop into a Docker container, run on any
   GitHub Actions runner, save HTML plots to artifacts. ANSYS
   licence-server requirements make this clumsy at best.
5. **Open scientific publishing.** Reviewers and readers can clone the
   repo, install requirements, and reproduce every figure. Maxwell
   results are reproducible only by other Maxwell licensees.
6. **Composable with SimNIBS.** Both are open Python tools with the same
   units convention; gluing them together is an afternoon.
7. **Web-shareable visualisation.** `--save-html` produces a single file
   that opens in any browser — no licence dance for collaborators.

## Validation status

The `Circle` field formula in magpylib is the same closed-form expression
used in the SimNIBS coil-modeling pipeline and in the validated-coil
database paper [^drakaki2022], where the figure-8 coil B-field measured by
a Hall probe matches the analytical model to within a few percent across
the relevant volume. Independent FEM validation of TMS coil models has
been published [^gomez2018]; the consensus is that for a TMS coil in air,
B-field models converge across solvers and the differences are
sub-percent. ANSYS Maxwell is one of those validated solvers; this stack
implements the same physics with a different (closed-form rather than
FEM) numerical backend.

## Recommended workflow

```
            ┌────────────────────────┐
            │  Coil concept / sweep  │
            └──────────┬─────────────┘
                       ▼
        magpylib + PyVista (this)
        ▶ rapid iteration
        ▶ external B field
        ▶ standoff check
        ▶ pre-flight visualisation
                       │
            converged design
                       ▼
         ┌─────────────┴──────────────┐
         ▼                            ▼
   ANSYS Maxwell                  SimNIBS
   ▶ final-design validation       ▶ subject-specific
   ▶ shields, eddy currents          induced E in cortex
   ▶ thermal / mechanical            and deep targets
   ▶ regulatory dossier
```

Use this tool to iterate quickly, ANSYS to certify the final hardware,
and SimNIBS to translate the certified coil into a per-subject E-field
prediction.

## When NOT to use this stack

- You need eddy currents in a conductive shield. → ANSYS.
- You need induced E in the head with realistic anatomy. → SimNIBS.
- You need transient capacitor-discharge co-simulation with ferromagnetic
  saturation in a core. → ANSYS.
- You're submitting to a regulatory body that requires a specific
  commercial solver. → ANSYS, plus this stack as a sanity-check.

## When TO use this stack

- Coil placement iteration before committing to a Maxwell run.
- Publishing reproducible figures with a free, scriptable pipeline.
- Continuous integration / regression tests on field magnitudes.
- Headless cluster runs sweeping coil parameters.
- Teaching: students can install in 30 seconds and see the field.
- Embedding in a larger Python research codebase (this case).

---

## References

[^ortner2020]: Ortner, M., & Coliado Bandeira, L. G. (2020). "Magpylib:
A free Python package for magnetic field computation." *SoftwareX*, 11,
100466. <https://www.sciencedirect.com/science/article/pii/S2352711020300170>

[^ortner2022]: Ortner, M., Helbig, T., et al. (2022). "Numerically stable
and computationally efficient expression for the magnetic field of a
current loop." Used by `magpylib.current.Circle`. Referenced in the
Magpylib physics guide:
<https://magpylib.readthedocs.io/en/stable/_pages/user_guide/guide_resources_01_physics.html>

[^drakaki2022]: Drakaki, M., Mathiesen, C., Siebner, H. R., Madsen, K., &
Thielscher, A. (2022). "Database of 25 validated coil models for electric
field simulations for TMS." *Brain Stimulation*, 15(3), 697–706.
<https://www.brainstimjrnl.com/article/S1935-861X(22)00077-8/fulltext>

[^gomez2018]: Gomez, L. J., Goetz, S. M., & Peterchev, A. V. (2018). "How
much detail is needed in modeling a transcranial magnetic stimulation
figure-8 coil." *Brain Stimulation*.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC5480865/>
