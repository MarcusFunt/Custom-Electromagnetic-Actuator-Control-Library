# Physics validation

This note records how the host physics engine's accuracy is *measured* -- not asserted --
and the numbers those measurements currently return. Everything here is reproducible from
the repo: the FEMM-free checks run in the normal test suite, and the FEMM cross-checks run
automatically wherever FEMM is installed.

The point of this document is to let you decide whether a given model is accurate enough for
your work, with a number rather than a hand-wave.

## What is validated, and against what

| Model | Validated against | Where |
|---|---|---|
| Analytic coil-magnet coupling (`coil_design.winding_averaged_force_per_amp`, used by the plant's `k_a` **and** `AnalyticReferenceBackend`) | Real FEMM axisymmetric magnetostatic solve | `fem/validate.py`, `tests/test_fem_femm_backend.py`, `tests/test_fem_validate.py` |
| Mechanical integrators (`plant.step`, `linear_plant.step`) | Self-convergence order + energy conservation | `tests/test_physics_validation.py`, `tests/test_numerical_accuracy.py` |
| Motional back-EMF `e = (dF/di)·v` | Central-difference `dF/di`; electrical/mechanical power identity | `tests/test_physics_validation.py` |
| Event interpolation (`numerics.hermite_event_fraction`) | Exact for constant acceleration | `tests/test_numerical_accuracy.py` |

## Analytic coupling vs real FEMM

The analytic force law is a closed-form Biot-Savart integral of an idealized uniformly
magnetized cylinder in vacuum (no iron, no saturation, `mu_r = 1` everywhere), averaged over
the winding cross-section. FEMM solves the actual axisymmetric field on a mesh, including the
NdFeB `mu_r = 1.05`. Over a full (offset, current) sweep, the two agree to:

| Geometry (turns, coil L×radial, magnet r×L) | peak \|F\| | max rel. err | mean rel. err |
|---|---|---|---|
| 400, 10×6 mm, 5×10 mm | 5.68 N | 1.8 % | 0.7 % |
| 400, 30×20 mm, 6×20 mm | 2.99 N | 1.2 % | 0.6 % |
| 800, 50×30 mm, 6×20 mm | 2.96 N | 1.2 % | 0.4 % |

(Relative error is normalized by the peak force and scored only where either model exceeds
5 % of peak, so the physically negligible far-field tail doesn't dominate the summary.)

**Takeaway:** for a bare-PM-slug / air-core-coil actuator, the fast analytic model is within
~2 % of a real field solve. The residual is the magnet's `mu_r = 1.05` reluctance term (a
sub-percent `i²` component the analytic linear-in-current model omits) plus discretization.
A swept FEMM LUT is only worth its cost if you add iron to the slug or the coil, where the
analytic vacuum assumption breaks.

Run it for your own geometry:

```python
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.validate import compare_analytic_to_femm   # needs FEMM installed

coil = CoilWindingGeometry(0.0, turns=400, coil_length_m=0.03, radial_thickness_m=0.02)
slug = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.2)
print(compare_analytic_to_femm(coil, slug).report())
```

### FEMM force extraction (important)

Getting the FEMM number *right* was itself a finding. The force on the slug must be read as
the **axial Lorentz force on the coil** (FEMM block integral type 12), not the weighted
Maxwell stress tensor over the magnet block (type 19). The stress tensor integrated over a
bare permanent-magnet block does **not** converge under mesh refinement here -- it was up to
~2× wrong and flipped sign in the far field -- because the magnet's own large internal B/H
dominates the tensor and the auto-meshed air band around it is too coarse for the weighting
function. The coil sits in linear, non-magnetic copper/air carrying a known current density,
so `∫ J×B` over it is mesh-robust and, by Newton's third law, is the total force on the slug.
`tests/test_fem_femm_backend.py` now pins magnitude agreement, mesh convergence, and current
linearity -- the checks that would have caught the original extraction bug (the old test only
compared force *sign*, which happened to be right in the near field).

## Integrator accuracy

Both mechanical steppers are damped velocity-Verlet (kick-drift-kick with exact exponential
half-step damping). Measured properties:

- **Second-order in dt.** Reference-free grid-convergence order (from steps 4h, 2h, h) is
  2.0 ± 0.2 for both the pendulum and the linear stepper.
- **Energy-conserving.** An undamped, undriven large-amplitude pendulum holds mechanical
  energy to < 1e-4 relative over 5 periods, with drift ~1e-7 over 20 s (symplectic, no
  secular growth).
- **Exact linear damping.** The damping-only velocity update is the exact exponential factor.

## Energy conservation of the electrical model

The linear stepper's `"rl"` current loop includes the motional back-EMF `e = (dF/di)·v`.
`coil_force_gradient` is verified to equal the true `dF/di` of the force law (Maxwell
reciprocity, central-difference checked for both the PM and reluctance branches). For the
default pure-PM slug this makes the back-EMF power `e·i` equal the mechanical power `F·v`
exactly, tick by tick -- mechanical work is drawn from the electrical source rather than
created from nothing.

## Tooling for a large FEM run

A big FEM sweep produces hundreds of expensive, black-box force tables. Three tools make
that trustworthy instead of hopeful:

**Before the run -- pick a mesh and estimate the cost** (`emac-femcheck`, needs FEMM):

```powershell
emac-femcheck --config examples/configs/linear_stepper_5coil_fem.toml --n-geometries 100
```

Solves one representative point at a ladder of mesh sizes, reports whether the force has
converged, recommends the coarsest safe mesh, and projects the whole sweep's wall-clock from
timed sample solves. (On the corrected backend a representative coil converges to <1% by the
default mesh; the old extraction never did -- that is the check that was missing.)

**After the run -- QC every table** (`emac-femqc`, no FEMM needed):

```powershell
emac-femqc studies/femm_trends/study/luts   # scans every .npz, exits non-zero if any suspect
```

Scores each `ForceLUT` against the physical invariants a real coil-magnet coupling must
satisfy (finite; zero-current and centered nulls; far-field decay; restoring sign; odd
symmetry; current linearity; a monotone tail with no far-field bumps) and prints only the
suspect ones. Every one of these invariants was violated by the old stress-tensor extraction,
so this would have caught that bug automatically. `check_lut` / `check_backend` are the
library entry points; pass `--reluctance-slug` to skip the (deliberately) inapplicable
current-linearity check for an iron slug.

**Division of labour:** `emac-femqc` catches *shape* errors (sign, symmetry, non-monotone
tails, nonlinearity) from a single table; a uniform *magnitude* error (e.g. the old
extraction's ~2x over-estimate) looks self-consistent in isolation and is caught instead by
`fem/validate.py`'s cross-check against the analytic reference. Run both.

## Known limitations (accuracy caveats)

- **Slug iron / saturation -- supported, but not validated to a number.** Every measurement
  in this document is for a *bare PM* slug. A ferromagnetic slug (`slug_type="reluctance"`,
  the plant's `CoilStation.Cmag > 0` branch) *is* covered by the FEM pipeline -- the FEMM
  backend solves it as nonlinear-B-H steel, so saturation is captured -- but it has no
  analytic-vs-FEMM accuracy table like the one above, because the analytic side is
  deliberately coarse there: `coil_design.reluctance_force_model` is a coenergy estimate
  routed through the synthetic `q_shape` lobe, not a shape-accurate field model. Read that
  as: the PM analytic model is validated to ~2%; the reluctance analytic model is a
  sanity-checked approximation whose only accuracy reference is FEMM.
  `tests/test_reluctance_mode.py` pins its qualitative invariants (attract-only, even in
  current, saturating past `i_sat`), not its accuracy.
- **Coupling shape (quantified).** The analytic *plant* still uses a Gaussian `q_shape` lobe
  anchored to the physically-correct peak height (`k_a`) and location (`x_c`). The peak now
  matches the winding-averaged reference exactly, but the *tails* do not: the Gaussian decays
  super-exponentially while the real coil-magnet coupling has a fat, dipole-like (~1/r³) tail.
  For a representative coil the analytic force is within ~20 % of the reference across the
  peak region but falls to ~15 % of it roughly two coupling scales out (e.g. 0.005 N vs
  0.037 N at 50 mm). Consequences: the analytic plant *under*-represents the sustained thrust
  a slug feels from coils it is not yet centered on, which for a tightly-packed multi-coil
  stepper can *under*-state exit speed by tens of percent -- the opposite direction from the
  old single-point `k_a` *over*-statement, so the two errors partly cancelled and hid each
  other. Only a swept FEM/analytic LUT (`emac-femgen`, `--force-law fem_reference`) captures
  the full curve; use it when absolute speed matters, not just relative design ranking.
- **Pendulum electrical dynamics.** The pendulum plant commands current instantaneously (no
  RL lag, no back-EMF, no thermal state) -- an intentional idealization for controller
  validation, not a driver-limited hardware model. The linear stepper has the full electrical
  model. See `docs/PHYSICS_ENGINE_ANALYSIS.md`.
- **Unmodeled effects.** Inter-coil mutual inductance, eddy-current braking in a conductive
  tube, bus-voltage droop, and end-stop contact are not modeled. See the roadmap in
  `docs/PHYSICS_ENGINE_ANALYSIS.md`.
