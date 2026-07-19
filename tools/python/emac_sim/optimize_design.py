"""Design-space optimizer for the linear stepper (docs/DESIGN_OPTIMIZER.md).

Searches over driver, winding, magnet, and topology knobs to maximize the slug's exit
speed, using coil_design.py's physical parametrization (turns/dimensions -> resistance/
inductance/thrust-constant) so the search has a real trade-off to explore -- without it,
"more turns" or "more coils" would have no modeled downside and the search would trivially
diverge to whatever bound you set. Uses `current_loop="rl"` (real per-coil electrical
dynamics) for every evaluation: this is the ONLY mode sensitive to resistance/inductance at
all, so running the search under "ideal" mode would make the turns-vs-copper-loss and
current-loop-topology knobs meaningless (see coil_design.py and linear_sim.py).

Knobs (11 total, mixed continuous/integer/categorical):
    bus_voltage_v, driver_bipolar (single-ended vs H-bridge), pump_envelope
    (rcos/trapezoid/square), n_coils, turns, coil_length_m, radial_thickness_m,
    magnet_radius_m, magnet_length_m, remanence_t, i_max_a.

The objective is always "exit speed with the velocity governor effectively disabled"
(v_tgt set far above anything achievable) -- this is a pure speed-maximization search, not
a tracking/efficiency one. FAULTed or stalled designs score 0.0, and any design whose total
tube length (n_coils * coil_length_m) exceeds `Bounds.max_tube_length_m` is rejected the
same way -- both push the search away from that region rather than crashing it.

Every bound below is a placeholder you should replace with your actual constraints (driver
voltage/current rating, available space, magnet grades on hand, etc.) -- see
docs/DESIGN_OPTIMIZER.md for what each one means physically and why it's not enforced any
more precisely than this (no duty-cycle or cost model).

Each candidate's per-coil winding self-heating IS modeled (`LinearActuatorParams.
thermal_model=True`, see linear_plant.py / docs/DESIGN_LINEAR.md section 2.4): resistance
rises with each coil's own i^2*R dissipation over the run, fed back into the "rl" current
loop, so a design can no longer look fast purely because it never pays a heating penalty --
more turns/current/voltage than a winding could sustain now shows up as a real, if partial,
speed penalty within the run itself (the model has no duty-cycle notion of repeated runs
back-to-back, only self-heating within one).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from scipy.optimize import differential_evolution

from . import coil_design
from .fem.femm_backend import FemmNotAvailableError
from .fem.geometry import CoilWindingGeometry, SlugGeometry, default_sweep_ranges
from .fem.lut import ForceLUT
from .fem.reference_backend import AnalyticReferenceBackend
from .fem.sweep import sweep_coil
from .linear_estimator import LinearStepperEstimator
from .linear_plant import CoilStation, GateStation, LinearActuatorParams
from .linear_sim import LinearSimulator
from .linear_supervisor import FAULT, StepperSupervisor

PUMP_ENVELOPES = ("rcos", "trapezoid", "square")
# "analytic" (default): coil_design.build_coil_station's k_a/x_c estimate, evaluated
# through plant.q_shape's Gaussian-lobe SHAPE at simulation time -- fast, and what every
# search/sweep here has always used. "fem_reference": each coil's force law comes directly
# from fem.reference_backend.AnalyticReferenceBackend instead -- the coil's REAL (non-
# Gaussian) coupling shape, evaluated in closed form per query (no LUT file needed; it's
# cheap enough -- ~0.1 ms/call -- to call live during a search). NEITHER of the above is a
# real FEM solve -- both are closed-form analytic approximations (no iron, no saturation,
# vacuum permeability everywhere); see reference_backend.py's own module docstring.
# "femm": each coil's force law comes from an actual FEMM axisymmetric magnetostatic solve
# (fem.femm_backend.FemmBackend), via ONE sweep per design shared across every coil (see
# _femm_force_lut) -- too slow to drive the search loop itself (seconds per solve), so it's
# only for simulate_design/build_params single-design calls and optimize()'s post-search
# verification pass (see verify_with_femm). Requires FEMM installed (femm.info).
# All three use the SAME winding electrical/thermal properties (coil_design.wind_coil) --
# only the FORCE law differs. See docs/FEM_PIPELINE.md.
FORCE_LAWS = ("analytic", "fem_reference", "femm")
V_TGT_FULL_THRUST = 100.0   # m/s -- unreachable, so the velocity governor never throttles back

# Coarser than emac-femgen's LUT-file default (31x11): this grid backs optimize()'s
# post-search verification pass and single-design force_law="femm" calls, where wall-clock
# time (real FEMM solves, seconds each) matters more than interpolation precision. The full
# 31x11 default in fem.geometry.default_sweep_ranges is unchanged for emac-femgen itself.
_FEMM_VERIFY_N_OFFSETS = 15
_FEMM_VERIFY_N_CURRENTS = 5

# Memoized per unique coil/slug geometry (see _femm_geometry_key) so repeated calls with
# the same knobs (re-verification, sensitivity sweeps) don't re-run a real FEMM sweep.
_femm_lut_cache: dict[tuple, ForceLUT] = {}


@dataclass(frozen=True)
class DesignKnobs:
    bus_voltage_v: float
    driver_bipolar: bool
    pump_envelope: str
    n_coils: int
    turns: int
    coil_length_m: float
    radial_thickness_m: float
    magnet_radius_m: float
    magnet_length_m: float
    remanence_t: float
    i_max_a: float


@dataclass(frozen=True)
class Bounds:
    bus_voltage_v: tuple[float, float] = (3.0, 400.0)
    n_coils: tuple[int, int] = (2, 30)
    turns: tuple[int, int] = (10, 1500)
    coil_length_m: tuple[float, float] = (0.005, 0.08)
    radial_thickness_m: tuple[float, float] = (0.002, 0.04)
    magnet_radius_m: tuple[float, float] = (0.002, 0.025)
    magnet_length_m: tuple[float, float] = (0.005, 0.08)
    remanence_t: tuple[float, float] = (0.3, 1.42)      # ferrite .. N52 NdFeB
    i_max_a: tuple[float, float] = (1.0, 150.0)
    max_tube_length_m: float = 1.0    # n_coils * coil_length_m must not exceed this


def _bounds_list(b: Bounds) -> list[tuple[float, float]]:
    return [
        b.bus_voltage_v,
        (0, 1),                          # driver_bipolar, integer-coded
        (0, len(PUMP_ENVELOPES) - 1),     # pump_envelope, integer-coded
        b.n_coils,
        b.turns,
        b.coil_length_m,
        b.radial_thickness_m,
        b.magnet_radius_m,
        b.magnet_length_m,
        b.remanence_t,
        b.i_max_a,
    ]


_INTEGRALITY = [False, True, True, True, True, False, False, False, False, False, False]


def decode(x: Sequence[float]) -> DesignKnobs:
    (bus_voltage_v, driver_bipolar_code, pump_envelope_code, n_coils, turns,
     coil_length_m, radial_thickness_m, magnet_radius_m, magnet_length_m,
     remanence_t, i_max_a) = x
    return DesignKnobs(
        bus_voltage_v=float(bus_voltage_v),
        driver_bipolar=bool(round(driver_bipolar_code)),
        pump_envelope=PUMP_ENVELOPES[int(round(pump_envelope_code))],
        n_coils=int(round(n_coils)),
        turns=int(round(turns)),
        coil_length_m=float(coil_length_m),
        radial_thickness_m=float(radial_thickness_m),
        magnet_radius_m=float(magnet_radius_m),
        magnet_length_m=float(magnet_length_m),
        remanence_t=float(remanence_t),
        i_max_a=float(i_max_a),
    )


def _fem_reference_coil(position_m: float, knobs: DesignKnobs,
                         backend: AnalyticReferenceBackend) -> CoilStation:
    """A CoilStation whose force law is fem.reference_backend.AnalyticReferenceBackend
    itself (bound as a closure -- see FORCE_LAWS above), rather than
    coil_design.build_coil_station's k_a/x_c estimate. Winding electrical/thermal
    properties still come from coil_design.wind_coil -- only the force law differs, so a
    force_law="fem_reference" run stays comparable to "analytic" on everything else."""
    slug = SlugGeometry(magnet_radius_m=knobs.magnet_radius_m,
                         magnet_length_m=knobs.magnet_length_m, remanence_t=knobs.remanence_t)
    coil_geometry = CoilWindingGeometry(position_m=position_m, turns=knobs.turns,
                                         coil_length_m=knobs.coil_length_m,
                                         radial_thickness_m=knobs.radial_thickness_m)
    winding = coil_design.wind_coil(knobs.turns, knobs.coil_length_m, knobs.radial_thickness_m,
                                     coil_geometry.bore_radius_m(slug))

    def force_lut(offset_m: float, current_a: float,
                  _coil: CoilWindingGeometry = coil_geometry, _slug: SlugGeometry = slug) -> float:
        return backend.solve(_coil, _slug, offset_m, current_a).force_n

    return CoilStation(position_m=position_m, resistance_ohm=winding.resistance_ohm,
                        inductance_h=winding.inductance_h,
                        thermal_mass_j_per_k=winding.thermal_mass_j_per_k,
                        force_lut=force_lut)


def _femm_geometry_key(knobs: DesignKnobs) -> tuple:
    """The subset of `knobs` that determines a coil/slug's FEMM force law -- everything
    `_femm_force_lut` builds `CoilWindingGeometry`/`SlugGeometry` from, PLUS `i_max_a`
    (determines the swept current span -- see _femm_force_lut). `position_m` and `n_coils`
    are deliberately excluded: every coil in one design shares the same winding/magnet
    geometry (only position differs), and FEMBackend.solve() never reads coil.position_m
    (see backend.py/femm_backend.py) -- one sweep covers every coil."""
    return (knobs.turns, knobs.coil_length_m, knobs.radial_thickness_m,
            knobs.magnet_radius_m, knobs.magnet_length_m, knobs.remanence_t, knobs.i_max_a)


def _femm_force_lut(knobs: DesignKnobs) -> ForceLUT:
    """One real FEMM sweep for this design's coil/slug geometry, memoized in
    `_femm_lut_cache` so identical knobs (re-verification, sensitivity sweeps) don't pay
    for a second sweep. Raises FemmNotAvailableError (propagated, not swallowed -- see
    femm_backend.py) if FEMM isn't installed; callers decide whether that's fatal."""
    key = _femm_geometry_key(knobs)
    cached = _femm_lut_cache.get(key)
    if cached is not None:
        return cached

    slug = SlugGeometry(magnet_radius_m=knobs.magnet_radius_m,
                         magnet_length_m=knobs.magnet_length_m, remanence_t=knobs.remanence_t)
    coil_geometry = CoilWindingGeometry(position_m=0.0, turns=knobs.turns,
                                         coil_length_m=knobs.coil_length_m,
                                         radial_thickness_m=knobs.radial_thickness_m)
    offsets, currents = default_sweep_ranges(
        coil_geometry, slug, n_offsets=_FEMM_VERIFY_N_OFFSETS, n_currents=_FEMM_VERIFY_N_CURRENTS,
        max_current_a=knobs.i_max_a,
    )

    from .fem.femm_backend import FemmBackend  # imported lazily: optional dependency

    with FemmBackend() as backend:
        lut = sweep_coil(coil_geometry, slug, backend, offsets_m=offsets, currents_a=currents)
    _femm_lut_cache[key] = lut
    return lut


def _femm_coil(position_m: float, lut: ForceLUT) -> CoilStation:
    """A CoilStation whose force law is a real FEMM-swept ForceLUT (see _femm_force_lut) --
    the SAME `lut` instance shared by every coil in the design (force_lut is called as
    force_lut(x - coil.position_m, current); the LUT itself has no notion of which coil it
    belongs to). Winding electrical/thermal properties still come from coil_design.wind_coil,
    matching _fem_reference_coil's pattern."""
    winding = coil_design.wind_coil(
        int(lut.metadata["turns"]), float(lut.metadata["coil_length_m"]),
        float(lut.metadata["radial_thickness_m"]),
        float(lut.metadata["magnet_radius_m"]) + float(lut.metadata["bore_clearance_m"]),
    )
    return CoilStation(position_m=position_m, resistance_ohm=winding.resistance_ohm,
                        inductance_h=winding.inductance_h,
                        thermal_mass_j_per_k=winding.thermal_mass_j_per_k, force_lut=lut)


def build_params(knobs: DesignKnobs, force_law: str = "analytic") -> LinearActuatorParams:
    """Assemble a full LinearActuatorParams from a design vector. Coils are packed
    edge-to-edge (pitch == coil_length_m, no inter-coil gap) -- a simplification, real
    builds need some clearance/former thickness between windings. `force_law` selects
    between the default analytic coupling estimate and the FEM reference backend (see
    FORCE_LAWS above) -- it changes NOTHING else about the design (mass, electrical/
    thermal properties, driver, gates all stay identical), so results are comparable."""
    if force_law not in FORCE_LAWS:
        raise ValueError(f"unknown force_law: {force_law!r} (expected one of {FORCE_LAWS})")

    pitch = knobs.coil_length_m
    if force_law == "analytic":
        coils = tuple(
            coil_design.build_coil_station(
                position_m=k * pitch, turns=knobs.turns, coil_length_m=knobs.coil_length_m,
                radial_thickness_m=knobs.radial_thickness_m, magnet_radius_m=knobs.magnet_radius_m,
                magnet_length_m=knobs.magnet_length_m, remanence_t=knobs.remanence_t,
            )
            for k in range(knobs.n_coils)
        )
    elif force_law == "fem_reference":
        backend = AnalyticReferenceBackend()
        coils = tuple(_fem_reference_coil(k * pitch, knobs, backend) for k in range(knobs.n_coils))
    else:  # "femm" -- one real FEMM sweep shared across every coil, see _femm_force_lut
        lut = _femm_force_lut(knobs)
        coils = tuple(_femm_coil(k * pitch, lut) for k in range(knobs.n_coils))
    gate_positions = [-0.5 * pitch] + [(k + 0.5) * pitch for k in range(knobs.n_coils - 1)]
    gates = tuple(GateStation(position_m=x, w_eff=0.002) for x in gate_positions)
    return LinearActuatorParams(
        mass_kg=coil_design.magnet_mass_kg(knobs.magnet_radius_m, knobs.magnet_length_m),
        coils=coils, gates=gates,
        current_loop="rl", bus_voltage_v=knobs.bus_voltage_v,
        driver_bipolar=knobs.driver_bipolar,
        # Self-heating is on for the search itself (not just available as an opt-in) --
        # see module docstring and docs/DESIGN_LINEAR.md section 2.4. ambient_temperature_c
        # matches coil_design.build_coil_station's own build-time reference temperature
        # (20 C, its default), so a coil starts the run exactly at its as-wound resistance.
        thermal_model=True,
        ambient_temperature_c=20.0,
    )


def simulate_design(knobs: DesignKnobs, dt: float = 2e-4, t_end: float = 3.0,
                     bootstrap_timeout_s: float = 0.05, force_law: str = "analytic",
                     push_pull: bool = False) -> float:
    """Exit speed (m/s) at the last gate, or 0.0 if the run FAULTed or never cleared a
    single gate (an infeasible or too-weak design) -- pushes the search away from that
    region the same way an explicit penalty would, without a separate constraint mechanism.
    `bootstrap_timeout_s` defaults short (vs. StepperSupervisor's own 0.20s) so hopeless
    candidates fail fast during a large search; pass the default back in for a final,
    more patient verification run. `force_law`: see build_params / FORCE_LAWS.
    `push_pull` (default False): opt into two-coil push-pull commutation (a departing coil
    repels from behind WHILE the next coil attracts from ahead) -- see
    StepperSupervisor.push_pull. Off by default, so the search behaves exactly as before."""
    p = build_params(knobs, force_law=force_law)
    pitch = knobs.coil_length_m
    x0 = -0.5 * pitch - 0.001
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, i_max=knobs.i_max_a, pm_envelope=knobs.pump_envelope,
                            bootstrap_timeout_s=bootstrap_timeout_s, push_pull=push_pull)
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=1_000_000)
    log = sim.run(x0=x0, v0=0.0, v_tgt=V_TGT_FULL_THRUST, t_end=t_end)
    if sup.mode == FAULT or not log.gate_t:
        return 0.0
    return log.gate_v[-1]


def _objective(x: Sequence[float], bounds: Bounds, dt: float, t_end: float,
                force_law: str = "analytic") -> float:
    knobs = decode(x)
    if knobs.n_coils * knobs.coil_length_m > bounds.max_tube_length_m:
        return 0.0    # infeasible: over the tube-length budget: same as a 0-speed design
    try:
        v = simulate_design(knobs, dt=dt, t_end=t_end, force_law=force_law)
    except (ValueError, ZeroDivisionError):
        v = 0.0
    return -v   # differential_evolution MINIMIZES


def optimize(bounds: Bounds = Bounds(), maxiter: int = 15, popsize: int = 12,
             seed: int = 0, dt: float = 2e-4, t_end: float = 3.0, workers: int = 1,
             force_law: str = "analytic", verify_with_femm: bool | None = None):
    """Search, then re-verify the winner. `force_law` drives the SEARCH itself (analytic or
    fem_reference only -- 'femm' is rejected here, see below) and the analytic re-verify
    step. `verify_with_femm` controls a SEPARATE, additional re-simulation of the winning
    design under real FEMM (force_law="femm", see _femm_force_lut): None (default) auto-
    attempts it and falls back to `femm_speed=None` with a printed note if FEMM isn't
    installed; True requires it (raises FemmNotAvailableError if FEMM is missing); False
    skips it entirely. This never substitutes the FEMM number for the analytic one -- both
    are returned, distinctly, so nothing here silently reports an approximation as if it
    were a real FEM result."""
    if force_law == "femm":
        raise ValueError(
            "force_law='femm' cannot drive the search itself -- each FEMM solve takes "
            "seconds and the search calls the force law millions of times. Search with "
            "'analytic' or 'fem_reference' (force_law) and pass verify_with_femm=True to "
            "verify the winning design against real FEMM afterward instead."
        )
    result = differential_evolution(
        _objective, bounds=_bounds_list(bounds), integrality=_INTEGRALITY,
        args=(bounds, dt, t_end, force_law), maxiter=maxiter, popsize=popsize, seed=seed,
        polish=False, workers=workers, updating="deferred" if workers != 1 else "immediate",
    )
    best_knobs = decode(result.x)
    # Re-verify at higher fidelity + the supervisor's normal (more patient) bootstrap
    # timeout -- the search itself uses a shorter one and a coarser dt to stay fast.
    best_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end, bootstrap_timeout_s=0.20,
                                  force_law=force_law)

    femm_speed: float | None = None
    if verify_with_femm is not False:
        try:
            femm_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end,
                                          bootstrap_timeout_s=0.20, force_law="femm")
        except FemmNotAvailableError:
            if verify_with_femm is True:
                raise
            print("FEMM not installed -- reported speed is analytic-only (see "
                  "docs/FEM_PIPELINE.md to install FEMM for a real verification pass)")
    return best_knobs, best_speed, result, femm_speed


def _print_design(knobs: DesignKnobs, speed: float, force_law: str = "analytic") -> None:
    print(f"exit speed: {speed:.4f} m/s\n")
    print("driver:")
    print(f"  bus_voltage_v   = {knobs.bus_voltage_v:.1f}")
    print(f"  driver_bipolar  = {knobs.driver_bipolar}  ({'H-bridge' if knobs.driver_bipolar else 'single half-bridge'})")
    print(f"  pump_envelope   = {knobs.pump_envelope}")
    print(f"  i_max_a         = {knobs.i_max_a:.1f}")
    print("topology:")
    print(f"  n_coils         = {knobs.n_coils}")
    print(f"  tube_length_m   = {knobs.n_coils * knobs.coil_length_m:.4f}")
    print("coil winding:")
    print(f"  turns           = {knobs.turns}")
    print(f"  coil_length_m   = {knobs.coil_length_m:.5f}")
    print(f"  radial_thick_m  = {knobs.radial_thickness_m:.5f}")
    print("slug / magnet:")
    print(f"  magnet_radius_m = {knobs.magnet_radius_m:.5f}")
    print(f"  magnet_length_m = {knobs.magnet_length_m:.5f}")
    print(f"  remanence_t     = {knobs.remanence_t:.3f}")
    p = build_params(knobs, force_law=force_law)
    print(f"  slug_mass_kg    = {p.mass_kg:.5f}")
    print(f"  force_law       = {force_law}")
    w = coil_design.wind_coil(knobs.turns, knobs.coil_length_m, knobs.radial_thickness_m,
                              knobs.magnet_radius_m + 0.0015)
    print(f"  per-coil R={w.resistance_ohm:.3f} ohm, L={w.inductance_h*1e3:.4f} mH, "
          f"wire_dia={w.wire_diameter_m*1e3:.3f} mm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search driver/winding/magnet/topology knobs to maximize slug exit speed."
    )
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for the search (1 = sequential).")
    parser.add_argument("--dt", type=float, default=2e-4, help="Search-phase simulation step (s).")
    parser.add_argument("--t-end", type=float, default=3.0, help="Per-evaluation simulated duration (s).")
    parser.add_argument("--max-tube-length-m", type=float, default=1.0)
    parser.add_argument("--force-law", choices=("analytic", "fem_reference"), default="analytic",
                        help="'analytic' (default): coil_design's k_a/x_c estimate through "
                             "the synthetic q_shape lobe. 'fem_reference': each coil's real "
                             "coupling shape via fem.reference_backend, evaluated live "
                             "(no LUT file needed) -- see docs/FEM_PIPELINE.md. Neither is a "
                             "real FEM solve (see --verify-femm for that); 'femm' can't drive "
                             "the search itself -- each solve is too slow.")
    parser.add_argument("--verify-femm", choices=("auto", "yes", "no"), default="auto",
                        help="After the search, re-simulate the winning design under a real "
                             "FEMM solve (fem.femm_backend.FemmBackend) and report it "
                             "separately from the search's analytic estimate. 'auto' "
                             "(default): verify if FEMM is installed, skip with a note "
                             "otherwise. 'yes': require FEMM, error if missing. 'no': skip.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bounds = Bounds(max_tube_length_m=args.max_tube_length_m)
    verify_with_femm = {"auto": None, "yes": True, "no": False}[args.verify_femm]
    print(f"searching: maxiter={args.maxiter} popsize={args.popsize} "
          f"force_law={args.force_law} verify_femm={args.verify_femm} "
          f"(~{args.maxiter * args.popsize * 11} evaluations worst case)")
    best_knobs, best_speed, result, femm_speed = optimize(
        bounds=bounds, maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
        dt=args.dt, t_end=args.t_end, workers=args.workers, force_law=args.force_law,
        verify_with_femm=verify_with_femm,
    )
    print(f"\nsearch reported {-result.fun:.4f} m/s at low fidelity; "
          f"re-verified (analytic) at high fidelity: {best_speed:.4f} m/s")
    if femm_speed is not None:
        print(f"FEMM-verified exit speed: {femm_speed:.4f} m/s\n")
    else:
        print()
    _print_design(best_knobs, best_speed, force_law=args.force_law)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
