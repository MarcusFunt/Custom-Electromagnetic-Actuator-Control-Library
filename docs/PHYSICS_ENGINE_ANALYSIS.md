# Physics engine analysis and accuracy roadmap

This note reviews the current host-side physics engine and records the first accuracy
upgrade implemented in this branch.

## Scope of the current engine

The repository currently has two concrete plant families rather than a single inherited
engine:

- `plant.py` + `sim.py`: nonlinear magnetic pendulum with one bottom coil and one bottom
  photogate.  The pendulum model uses the full `sin(theta)` gravity term, a linear viscous
  damping torque derived from `Q`, and a separable magnetic torque
  `tau_mag(theta, i) = q_shape(theta) * f_current(i)`.
- `linear_plant.py` + `linear_sim.py`: translational N-coil actuator.  It reuses the same
  odd coupling lobe and current laws, but anchors a lobe at each coil position and sums the
  force contributions.

That split is right.  The pendulum and linear stepper share low-level force/current helper
functions, but their estimators and supervisors are different enough that forcing a common
base class would hide important design decisions.

## Existing model quality

### Good parts

1. **Full nonlinear pendulum gravity.** The pendulum does not use the small-angle
   approximation in the plant.  That matters once amplitudes move above roughly a few tens
   of degrees, because the real period stretches with amplitude.
2. **Energy-domain state reporting.** The pendulum logger records true mechanical energy and
   converts it to amplitude with the exact large-angle turning-point formula.
3. **Attract-only soft-iron behavior.** The soft-iron branch correctly models force scale as
   non-negative and roughly quadratic with current, with a saturation term.
4. **Signed permanent-magnet branch.** The linear actuator has a signed PM term, which is
   needed for future push/pull or repel-braking experiments.
5. **Exact first-order electrical update.** `rl_current_step()` solves the RL circuit in
   closed form for piecewise-constant applied voltage.  That is much better than explicit
   Euler for coil dynamics.
6. **Linear-sim CFL guard.** `linear_sim.py` already subdivides mechanical integration when
   a fast slug would otherwise cross too much of a coupling lobe in one nominal tick.

### Main limitations before this branch

1. **Mechanical integration was first-order.** Both plants used semi-implicit Euler.  That is
   far better than explicit Euler for oscillators, but it is still first-order accurate in
   time, so timestep sensitivity shows up as phase and energy error.
2. **Crossing interpolation used constant-speed/linear position interpolation.** The
   simulator already had both endpoint positions and velocities, but event timestamps were
   estimated from endpoint positions only.  That throws away information, especially around
   narrow gates, high accelerations, or coarse smoke-test timesteps.
3. **The magnetic lobe is still synthetic by default.** `q_shape()` is an analytic
   placeholder, useful for controller development but not a real coupling profile. A fitted
   lookup table now exists as an opt-in alternative: `docs/FEM_PIPELINE.md`'s
   `emac-femgen` pipeline sweeps a real axisymmetric field solve (FEMM) or a shape-accurate
   analytic-reference backend into a `ForceLUT`, which `linear_plant.net_force` uses
   directly for any coil with `force_lut_path` set -- `q_shape` remains the default for
   every config that doesn't opt in. Calibration-data-fitted tables (vs. simulated) are
   still not supported. The reluctance (soft-iron) branch is also not yet covered by this
   pipeline -- see `docs/FEM_PIPELINE.md`'s known limitations.
4. **Bus droop is still absent; winding self-heating is now modeled.** `plant.
   resistance_at_temperature` / `plant.thermal_step` track each linear-stepper coil's
   temperature from its own i^2*R dissipation (one-node thermal model, opt-in via
   `LinearActuatorParams.thermal_model` -- see `docs/DESIGN_LINEAR.md` section 2.4) and
   feed the resulting resistance back into the `"rl"` current loop; `optimize_design.py`'s
   search runs with it on for every candidate. Driver current limits and supply-voltage
   droop under load are still not modeled, and the pendulum side has no thermal state at
   all yet.
5. **No contact/end-stop mechanics.** The linear actuator has an end-of-travel mode in the
   params, but the plant itself still has no collision/contact model.

## Implemented accuracy upgrade

### 1. Damped velocity-Verlet mechanical step

`plant.step()` and `linear_plant.step()` now use a kick-drift-kick velocity-Verlet core for
position-dependent forces.  Linear viscous damping is split out and applied as exact
half-step exponential factors:

```text
v <- exp(-gamma dt/2) v
v <- v + 0.5 a(x) dt
x <- x + v dt
v <- v + 0.5 a(x_new) dt
v <- exp(-gamma dt/2) v
```

For the undamped conservative part this is second-order and time-symmetric.  With damping,
it is still a cheap deterministic split method and the damping-only velocity decay is exact.
Current is still treated as piecewise-constant within one tick, matching the supervisor and
current-controller abstraction.

### 2. Cubic Hermite event interpolation

A new `numerics.hermite_event_fraction()` helper estimates crossing time from both endpoint
positions and endpoint velocities.  The pendulum bottom crossing and the linear gate crossing
now both use this shared helper.  For a constant-acceleration segment the event time is exact;
for a real nonlinear segment it is still a better local approximation than linear position
interpolation.

### 3. Accuracy regression tests

`tests/test_numerical_accuracy.py` pins three properties:

- Hermite interpolation is exact for a constant-acceleration crossing.
- The linear plant's damping-only velocity update is exactly exponential.
- The undriven, negligibly damped nonlinear pendulum keeps mechanical energy nearly constant
  over several periods.

## Recommended next improvements

1. **Fitted force-map tables.** Replace `q_shape()` with a calibrated table supporting
   interpolation and optional derivative lookup.  Start with a static `q(theta)`/`q(x)` table
   per coil, then add current-dependent saturation if measured data requires it.
2. **Coupled electrical/mechanical back-EMF.** For a moving permanent-magnet slug, the coil
   voltage equation should eventually include speed-dependent induced voltage.  The current
   state already exists in `linear_sim.py`, so this can be added without changing the
   high-level supervisor API.
3. **Adaptive or event-aligned substepping.** For offline reference runs, optionally cut a
   mechanical step exactly at predicted sensor events and pulse boundary times.  That would
   reduce small timing errors in controller/plant interaction without changing firmware-like
   fixed-tick behavior by default.
4. **Thermal resistance model -- implemented for the linear stepper.** Copper temperature
   now tracks each coil's own i^2*R dissipation and feeds back into `"rl"` current
   dynamics (`docs/DESIGN_LINEAR.md` section 2.4); the design optimizer runs with it on.
   Not yet extended to the pendulum, and `thermal_resistance_k_per_w` has no geometry-
   derived calibration (unlike `thermal_mass_j_per_k`, which comes from the winding's own
   copper volume) -- still a placeholder pending a real measurement. Bus-voltage droop
   under load and driver current limits remain unmodeled.
5. **End-stop/contact model for the linear actuator.** Add configurable hard stops with
   restitution/damping, or soft bumpers if the physical design uses compliant stops.
6. **Reference integrator mode.** Keep the current deterministic fixed-step engine for
   firmware parity, but add an offline high-accuracy reference mode for calibration sweeps and
   regression comparisons.
