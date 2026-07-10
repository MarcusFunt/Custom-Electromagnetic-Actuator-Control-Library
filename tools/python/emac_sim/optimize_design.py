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

The objective uses an explicit full-thrust controller mode -- no finite velocity sentinel
can accidentally throttle a fast candidate. This is a pure speed-maximization search, not
a tracking/efficiency one. Incomplete, FAULTed or stalled designs score 0.0, and any design whose total
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
import math
from dataclasses import dataclass, replace
from typing import Sequence

from scipy.optimize import differential_evolution

from . import coil_design
from .linear_estimator import LinearStepperEstimator
from .linear_plant import GateStation, LinearActuatorParams
from .linear_sim import LinearSimulator
from .linear_supervisor import FAULT, StepperSupervisor

PUMP_ENVELOPES = ("rcos", "trapezoid", "square")


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


def build_params(knobs: DesignKnobs) -> LinearActuatorParams:
    """Assemble a full LinearActuatorParams from a design vector. Coils are packed
    edge-to-edge (pitch == coil_length_m, no inter-coil gap) -- a simplification, real
    builds need some clearance/former thickness between windings."""
    pitch = knobs.coil_length_m
    # Every station shares one winding geometry.  Build its expensive winding-volume
    # coupling table once, then relocate the immutable station template.
    coil_template = coil_design.build_coil_station(
        position_m=0.0, turns=knobs.turns, coil_length_m=knobs.coil_length_m,
        radial_thickness_m=knobs.radial_thickness_m, magnet_radius_m=knobs.magnet_radius_m,
        magnet_length_m=knobs.magnet_length_m, remanence_t=knobs.remanence_t,
    )
    coils = tuple(replace(coil_template, position_m=k * pitch) for k in range(knobs.n_coils))
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
                     bootstrap_timeout_s: float = 0.05) -> float:
    """Speed (m/s) at the physical exit plane, or 0.0 unless every ordered gate and
    the exit plane were crossed (an infeasible, stalled or too-weak design).
    `bootstrap_timeout_s` defaults short (vs. StepperSupervisor's own 0.20s) so hopeless
    candidates fail fast during a large search; pass the default back in for a final,
    more patient verification run."""
    p = build_params(knobs)
    pitch = knobs.coil_length_m
    x0 = -0.5 * pitch - 0.001
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(
        p, i_max=knobs.i_max_a, pm_envelope=knobs.pump_envelope,
        bootstrap_timeout_s=bootstrap_timeout_s, full_thrust=True,
    )
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=1_000_000)
    log = sim.run(x0=x0, v0=0.0, v_tgt=None, t_end=t_end)
    completed = (
        sup.mode != FAULT
        and est.cleared_last_gate()
        and len(log.gate_t) == len(p.gates)
        and log.exit_v is not None
        and math.isfinite(log.exit_v)
        and log.exit_v > 0.0
    )
    if not completed:
        return 0.0
    return log.exit_v


def _objective(x: Sequence[float], bounds: Bounds, dt: float, t_end: float) -> float:
    knobs = decode(x)
    if knobs.n_coils * knobs.coil_length_m > bounds.max_tube_length_m:
        return 0.0    # infeasible: over the tube-length budget: same as a 0-speed design
    try:
        v = simulate_design(knobs, dt=dt, t_end=t_end)
    except (ValueError, ZeroDivisionError):
        v = 0.0
    return -v   # differential_evolution MINIMIZES


def optimize(bounds: Bounds = Bounds(), maxiter: int = 15, popsize: int = 12,
             seed: int = 0, dt: float = 2e-4, t_end: float = 3.0, workers: int = 1):
    result = differential_evolution(
        _objective, bounds=_bounds_list(bounds), integrality=_INTEGRALITY,
        args=(bounds, dt, t_end), maxiter=maxiter, popsize=popsize, seed=seed,
        polish=False, workers=workers, updating="deferred" if workers != 1 else "immediate",
    )
    best_knobs = decode(result.x)
    # Re-verify at higher fidelity + the supervisor's normal (more patient) bootstrap
    # timeout -- the search itself uses a shorter one and a coarser dt to stay fast.
    best_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end, bootstrap_timeout_s=0.20)
    return best_knobs, best_speed, result


def _print_design(knobs: DesignKnobs, speed: float) -> None:
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
    p = build_params(knobs)
    print(f"  slug_mass_kg    = {p.mass_kg:.5f}")
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bounds = Bounds(max_tube_length_m=args.max_tube_length_m)
    print(f"searching: maxiter={args.maxiter} popsize={args.popsize} "
          f"(~{args.maxiter * args.popsize * 11} evaluations worst case)")
    best_knobs, best_speed, result = optimize(
        bounds=bounds, maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
        dt=args.dt, t_end=args.t_end, workers=args.workers,
    )
    print(f"\nsearch reported {-result.fun:.4f} m/s at low fidelity; "
          f"re-verified at high fidelity: {best_speed:.4f} m/s\n")
    _print_design(best_knobs, best_speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
