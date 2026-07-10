"""MCP server exposing the design-space optimizer (docs/DESIGN_OPTIMIZER.md) as tools for an
LLM client, with live progress and per-generation fault-rate reporting so a search that is
stuck or badly bounded is visible long before it finishes -- not just at the end.

Run via `emac-mcp` (registered entry point) or `python -m emac_sim.mcp_server`. Add it to a
client with e.g. `claude mcp add emac -- emac-mcp` (Claude Code) once the `mcp` optional
dependency group is installed (`pip install -e ".[mcp]"`).

Every `run_optimization` call writes a JSON snapshot to `build/optimize_results/latest.json`
after EVERY generation, not just at the end -- reload that file at any point in the "EMAC
Optimizer Dashboard" artifact to see the convergence curve, fault rate, and current best
design of a search that is still running.

Deliberately does not expose `optimize_design.optimize()`'s `workers` option: that path uses
a multiprocessing pool, and the per-evaluation fault/best-so-far instrumentation here uses
in-process shared state (a lock-guarded closure) that would not survive being pickled into
worker processes. Single-process search is slower per wall-clock second but this server's
whole purpose is visibility into that time, not raw throughput.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP
from scipy.optimize import differential_evolution

from . import design_sensitivity, optimize_design
from .linear_estimator import LinearStepperEstimator
from .linear_sim import LinearSimulator
from .linear_supervisor import FAULT, StepperSupervisor
from .optimize_design import Bounds, DesignKnobs, build_params, decode, simulate_design

RESULTS_DIR = Path(__file__).resolve().parents[3] / "build" / "optimize_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = RESULTS_DIR / "latest.json"

mcp = FastMCP("emac-optimizer")


def _bounds_from_overrides(overrides: Optional[dict[str, Any]]) -> Bounds:
    b = Bounds()
    if not overrides:
        return b
    kwargs: dict[str, Any] = {}
    valid = {f.name for f in dataclasses.fields(b)}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(f"unknown bound {key!r}; valid keys: {sorted(valid)}")
        current = getattr(b, key)
        kwargs[key] = tuple(value) if isinstance(current, tuple) else value
    return dataclasses.replace(b, **kwargs)


def _knobs_dict(knobs: DesignKnobs) -> dict[str, Any]:
    return dataclasses.asdict(knobs)


def _load_latest_knobs() -> DesignKnobs:
    if LATEST_PATH.exists():
        data = json.loads(LATEST_PATH.read_text())
        best = data.get("best_knobs")
        if best:
            return DesignKnobs(**best)
    # A reasonable starting point when nothing has been optimized yet.
    return DesignKnobs(
        bus_voltage_v=48.0, driver_bipolar=True, pump_envelope="rcos", n_coils=8,
        turns=180, coil_length_m=0.02, radial_thickness_m=0.01, magnet_radius_m=0.008,
        magnet_length_m=0.02, remanence_t=1.2, i_max_a=20.0,
    )


@dataclasses.dataclass
class _SearchState:
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    start_time: float = dataclasses.field(default_factory=time.time)
    evals: int = 0
    faults: int = 0
    gen_evals: int = 0
    gen_faults: int = 0
    generation: int = 0
    best_speed: float = 0.0
    best_knobs: Optional[dict[str, Any]] = None
    history: list = dataclasses.field(default_factory=list)
    done: bool = False
    error: Optional[str] = None


def _run_search(state: _SearchState, bounds: Bounds, maxiter: int, popsize: int, seed: int,
                 dt: float, t_end: float, result_path: Path) -> None:
    def objective(x: Any) -> float:
        knobs = decode(x)
        if knobs.n_coils * knobs.coil_length_m > bounds.max_tube_length_m:
            v = 0.0
        else:
            try:
                v = simulate_design(knobs, dt=dt, t_end=t_end)
            except (ValueError, ZeroDivisionError):
                v = 0.0
        with state.lock:
            state.evals += 1
            state.gen_evals += 1
            if v <= 0.0:
                state.faults += 1
                state.gen_faults += 1
            if v > state.best_speed:
                state.best_speed = v
                state.best_knobs = _knobs_dict(knobs)
        return -v

    def callback(xk: Any, convergence: float) -> bool:
        with state.lock:
            state.generation += 1
            fault_frac = state.gen_faults / state.gen_evals if state.gen_evals else 0.0
            elapsed = time.time() - state.start_time
            eta_s = (elapsed / state.generation) * max(0, maxiter - state.generation)
            state.history.append({
                "generation": state.generation,
                "evals_total": state.evals,
                "evals_this_gen": state.gen_evals,
                "fault_fraction_this_gen": fault_frac,
                "best_speed_so_far": state.best_speed,
                "convergence": float(convergence),
                "elapsed_s": elapsed,
            })
            state.gen_evals = 0
            state.gen_faults = 0
            snapshot = {
                "status": "running", "generation": state.generation, "maxiter": maxiter,
                "popsize": popsize, "seed": seed, "evals_total": state.evals,
                "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
                "best_speed_m_s": state.best_speed, "best_knobs": state.best_knobs,
                "elapsed_s": elapsed, "eta_s": eta_s, "history": state.history,
            }
        result_path.write_text(json.dumps(snapshot, indent=2))
        return False  # never request an early stop

    try:
        result = differential_evolution(
            objective, bounds=optimize_design._bounds_list(bounds),
            integrality=optimize_design._INTEGRALITY, maxiter=maxiter, popsize=popsize,
            seed=seed, polish=False, workers=1, updating="immediate", callback=callback,
        )
        best_knobs = decode(result.x)
        best_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end, bootstrap_timeout_s=0.20)
        with state.lock:
            state.best_speed = best_speed
            state.best_knobs = _knobs_dict(best_knobs)
            state.done = True
            elapsed = time.time() - state.start_time
            final = {
                "status": "done", "generation": state.generation, "maxiter": maxiter,
                "popsize": popsize, "seed": seed, "evals_total": state.evals,
                "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
                "best_speed_m_s": best_speed,
                "best_speed_search_estimate_m_s": -result.fun,
                "best_knobs": state.best_knobs, "elapsed_s": elapsed, "eta_s": 0.0,
                "history": state.history,
            }
        result_path.write_text(json.dumps(final, indent=2))
    except Exception as exc:  # surfaced to the polling coroutine below
        with state.lock:
            state.error = f"{type(exc).__name__}: {exc}"
            state.done = True


@mcp.tool()
async def run_optimization(
    ctx: Context,
    maxiter: int = 15,
    popsize: int = 12,
    seed: int = 0,
    dt: float = 2e-4,
    t_end: float = 3.0,
    bounds_overrides: Optional[dict[str, Any]] = None,
    fault_warning_threshold: float = 0.9,
) -> dict[str, Any]:
    """Run the design-space optimizer (differential evolution over the 11 knobs in
    docs/DESIGN_OPTIMIZER.md) to maximize slug exit speed. Reports live progress through the
    MCP progress channel and emits an explicit warning as soon as a generation's fault rate
    (candidates that FAULTed or never cleared a gate) crosses `fault_warning_threshold`, so a
    badly-bounded search is visible within the first generation or two rather than only at
    the end of a multi-minute run.

    Also writes a JSON snapshot to build/optimize_results/latest.json after EVERY generation
    (not just at the end) -- reload that file in the "EMAC Optimizer Dashboard" artifact at
    any time, including mid-run, to see the convergence curve, fault rate, and current best
    design.

    bounds_overrides: optional dict of Bounds field name -> [min, max] (or a bare number for
    max_tube_length_m), e.g. {"bus_voltage_v": [3, 60], "i_max_a": [1, 30]} to cap the search
    to hardware you can actually source. Unlisted fields keep optimize_design.Bounds' defaults.
    """
    bounds = _bounds_from_overrides(bounds_overrides)
    state = _SearchState()
    total_evals_estimate = max(1, maxiter * popsize * 11)

    thread = threading.Thread(
        target=_run_search,
        args=(state, bounds, maxiter, popsize, seed, dt, t_end, LATEST_PATH),
        daemon=True,
    )
    thread.start()

    last_reported_gen = -1
    while thread.is_alive():
        await asyncio.sleep(0.5)
        with state.lock:
            evals, gen, best, done, error = (
                state.evals, state.generation, state.best_speed, state.done, state.error,
            )
            history = list(state.history)
        await ctx.report_progress(
            min(evals, total_evals_estimate), total_evals_estimate,
            f"generation {gen}/{maxiter}, best {best:.3f} m/s ({evals} evaluations)",
        )
        if history and history[-1]["generation"] != last_reported_gen:
            last_reported_gen = history[-1]["generation"]
            frac = history[-1]["fault_fraction_this_gen"]
            if frac >= fault_warning_threshold:
                await ctx.warning(
                    f"generation {last_reported_gen}: {frac * 100:.0f}% of candidates FAULTed "
                    f"or scored 0 m/s -- bounds may be infeasible (tube length, current/voltage "
                    f"too low to bootstrap, etc.)"
                )
            elif best <= 0.0 and last_reported_gen >= 3:
                await ctx.warning(
                    f"generation {last_reported_gen}: no feasible design found yet after "
                    f"{evals} evaluations -- consider widening bounds_overrides"
                )
    thread.join()

    with state.lock:
        if state.error:
            raise RuntimeError(state.error)
        summary = {
            "generations": state.generation, "evals_total": state.evals,
            "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
            "best_speed_m_s": state.best_speed, "best_knobs": state.best_knobs,
            "results_file": str(LATEST_PATH),
        }
    await ctx.report_progress(total_evals_estimate, total_evals_estimate, "done")
    return summary


@mcp.tool()
def get_latest_result() -> dict[str, Any]:
    """Return the most recent optimization snapshot from build/optimize_results/latest.json
    (may belong to a search that is still running -- check its "status" field). Useful for
    reloading the dashboard artifact's state without re-running anything."""
    if not LATEST_PATH.exists():
        return {"status": "no_results_yet"}
    return json.loads(LATEST_PATH.read_text())


@mcp.tool()
def simulate_design_detailed(
    knobs: dict[str, Any], dt: float = 2e-5, t_end: float = 3.0,
    bootstrap_timeout_s: float = 0.20, max_samples: int = 2000,
) -> dict[str, Any]:
    """Run one closed-loop simulation for a fully-specified design (the same shape as
    run_optimization's/get_latest_result's "best_knobs") and return the full position/
    velocity/current/temperature time series plus gate-crossing events, downsampled to at
    most max_samples points, for the time-series view of the EMAC Optimizer Dashboard
    artifact. Use this on the winning design from a search to see *how* it reaches its exit
    speed, not just the final number."""
    dk = DesignKnobs(**knobs)
    p = build_params(dk)
    pitch = dk.coil_length_m
    x0 = -0.5 * pitch - 0.001
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, i_max=dk.i_max_a, pm_envelope=dk.pump_envelope,
                             bootstrap_timeout_s=bootstrap_timeout_s, full_thrust=True)
    expected_steps = max(1, int(t_end / dt))
    sample_every = max(1, expected_steps // max_samples)
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=sample_every)
    log = sim.run(x0=x0, v0=0.0, v_tgt=None, t_end=t_end)
    return {
        "fault": sup.mode == FAULT,
        "t": log.t, "x": log.x, "v": log.v,
        "active_current": log.active_current, "active_temperature_c": log.active_temperature_c,
        "gate_t": log.gate_t, "gate_v": log.gate_v, "gate_index": log.gate_index,
        "exit_t": log.exit_t, "exit_position_m": log.exit_position_m,
        "exit_speed_m_s": log.exit_v if log.exit_v is not None else 0.0,
        "bus_energy_j": log.bus_energy_j, "copper_loss_j": log.copper_loss_j,
        "magnetic_energy_j": log.magnetic_energy_j,
        "mechanical_em_work_j": log.mechanical_em_work_j,
        "energy_residual_j": log.energy_residual_j,
        "knobs": _knobs_dict(dk),
    }


@mcp.tool()
def sensitivity_sweep(
    knob: str, baseline: Optional[dict[str, Any]] = None,
    bounds_overrides: Optional[dict[str, Any]] = None,
    n_points: int = 9, dt: float = 2e-4, t_end: float = 3.0,
) -> dict[str, Any]:
    """One-at-a-time sensitivity sweep of a single knob (see docs/DESIGN_OPTIMIZER.md
    section 6 for the knob list) around a baseline design, holding every other knob fixed.
    baseline defaults to the latest run_optimization result's best_knobs. Returns
    {"knob", "baseline", "points": [{"value", "speed"}, ...]} -- load it in the EMAC
    Optimizer Dashboard artifact to see the main-effect curve."""
    bounds = _bounds_from_overrides(bounds_overrides)
    base = DesignKnobs(**baseline) if baseline else _load_latest_knobs()
    points = design_sensitivity.sweep_knob(knob, base, bounds, n_points=n_points, dt=dt, t_end=t_end)
    return {"knob": knob, "baseline": _knobs_dict(base), "points": points}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
