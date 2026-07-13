# EMAC simulator — bug review

Focus: the simulation / physics / FEM core (the linear-stepper path exercised by the FEMM
study). All three FEMM bugs below were found by *actually running* the real FEMM backend
end-to-end — none are reachable by the existing test suite, which skips FEMM entirely
(`pytest.importorskip("femm")`) and marks `FemmBackend.solve` `# pragma: no cover`.

## Severity summary

| # | Bug | File | Severity | Caught by tests? |
|---|-----|------|----------|------------------|
| 1 | FEMM force sign is inverted vs. the reference backend & plant convention | `fem/femm_backend.py` | **High** | No |
| 2 | FEMM air domain doesn't grow with offset → phantom far-field force | `fem/femm_backend.py` | **High** | No |
| 3 | `mesh_size_m` is a silent no-op (`automesh=1` overrides it) | `fem/femm_backend.py` | Medium | No |
| 4 | `max_current_a` config field parsed but never used | `config.py` | Low | n/a |

All three FEMM bugs hit the **documented default command** `emac-femgen --backend femm`
(`fem/cli.py` sweeps `default_sweep_ranges` — ±5× coupling scale — through the stock
`FemmBackend`). A user following `docs/FEM_PIPELINE.md` would get a table that is
mesh-uncontrollable, sign-inverted, and tail-corrupted; feeding it into `emac-sim` yields a
slug driven backward → **0 m/s exit speed for every design**.

---

## Bug 1 — FEMM force sign inverted  (HIGH)

`fem/femm_backend.py`, `FemmBackend.solve` last line:
```python
force_n = femm.mo_blockintegral(19)   # axial weighted stress tensor
return ForcePoint(force_n=float(force_n))
```
The module docstring claims the slug placement "matches reference_backend's convention," but
the returned axial Maxwell-stress force has the **opposite sign** to both
`fem/reference_backend.py` and `linear_plant.net_force`'s expectation.

**Evidence** (same coil/slug geometry, I = +22.5 A):

| offset (m) | FEMM (shipped) | reference backend |
|---|---|---|
| +0.047 | **+3.13 N** | **−2.49 N** |
| −0.047 | **−3.00 N** | **+2.49 N** |

Running the linear stepper with a FEMM LUT gives **0.000 m/s** for every design (slug pushed
backward, never clears a gate). Negating the FEMM force makes it agree in sign with the
reference backend and yields sensible speeds (~6.6 m/s vs 7.8 m/s reference — the real ~16%
FEMM-vs-analytic divergence `FEM_PIPELINE.md` predicts).

**Fix:** `return ForcePoint(force_n=-float(force_n))` (and correct the docstring). A proper
regression test would assert FEMM and the reference backend agree in sign at a few
(offset, current) points when `femm` is importable.

## Bug 2 — Air domain doesn't grow with offset → phantom far-field force  (HIGH)

```python
half_extent = _AIR_MARGIN_FACTOR * max(coil.coil_length_m, slug.magnet_length_m)
# slug is then placed at z = -offset_m, which can exceed half_extent
```
The air rectangle is sized from part dimensions only, but the slug is drawn at `z=-offset`.
`sweep.py` sweeps offsets out to `default_sweep_ranges`' 5× coupling scale, which for the
**default config coil** is 0.20 m while `half_extent` is only 0.12 m — so the outer ~30% of
every swept LUT places the slug **on/outside the FEMM boundary**, returning garbage.

**Evidence:** at a far offset the shipped backend returned **0.93 N** where the true
(reference) value is 0.07 N — a ~30%-of-peak "phantom" force on coils the slug has already
passed. This is the exact failure `tests/test_fem_analysis_regression.py` guards against for
the *reference* backend, but that test never exercises FEMM. Enlarging the domain to enclose
the slug (`half_extent += abs(offset_m)`) drops the far-offset force to 0.21 N (→ ~0 after
the domain is generous), matching the reference tail.

**Fix:** `half_extent = _AIR_MARGIN_FACTOR*max(coil_length, magnet_length) + abs(offset_m)`.

## Bug 3 — `mesh_size_m` is a silent no-op  (MEDIUM)

```python
mesh = self.mesh_size_m or (0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m))
femm.mi_setblockprop("NdFeB",  1, mesh, "<None>", 90, _SLUG_GROUP, 0)   # <-- automesh=1
femm.mi_setblockprop("Copper", 1, mesh, circuit,  0, _COIL_GROUP, coil.turns)  # automesh=1
```
FEMM's `mi_setblockprop(..., automesh, meshsize, ...)` **ignores `meshsize` when
`automesh=1`**. So the constructor's `mesh_size_m` parameter (and the intended
`0.15·min(...)` formula) never take effect — the mesh is always FEMM's automatic choice.

**Evidence:** sweeping `mesh_size_m` across a 4× range produced byte-identical forces and
timings. Passing `automesh=0` makes the intended mesh take effect and cuts the
offset-symmetry error on a representative coil from **~3.3% to ~0.6%** (a real accuracy gain
that is currently unreachable). Note: naive *further* refinement below the intended value is
non-monotonic (stress-tensor noise from the coarse air mesh dominates), so the intended
`0.15·min` value is genuinely the sweet spot — it just isn't being applied.

**Fix:** pass `automesh=0` in both `mi_setblockprop` calls.

## Bug 4 — dead config field  (LOW)

`config.py`: `LinearCoilConfig.max_current_a` (and `CoilConfig.max_current_a`) are parsed by
`_linear_coils`/`_coils` but never consumed — `to_actuator_params`/`to_pendulum_params`
don't pass them anywhere (the current cap is `LinearControllerConfig.i_max_a`). Harmless, but
a user setting per-coil `max_current_a` in TOML would see no effect. Either wire it into the
supervisor's per-coil clamp or drop the field.

---

## What I checked and found clean
`plant.py` (odd q-shape, saturating reluctance inverse, exact RL/thermal integration),
`linear_plant.py` (net_force LUT dispatch, velocity-Verlet with exact damping split,
rail-limited current solve), `linear_sim.py` (CFL sub-stepping, multi-gate `while`,
Hermite event detection), `numerics.py`, `linear_estimator.py` / `estimator.py`
(gate-width→speed recovery, damped-oscillator reconstruction), `supervisor.py` /
`linear_supervisor.py` (envelope averages, pump/departure/end-of-travel timing),
`config.py` parsing. No correctness bugs found outside the FEMM backend.

## Testing gap
The FEMM backend is the one component with real solver logic and essentially no test
coverage: the only solve test asserts `force_n == force_n` (not NaN). A FEMM-gated test that
checks sign + magnitude agreement against the reference backend would have caught bugs 1 & 2.
