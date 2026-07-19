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

from . import coil_design, design_sensitivity, optimize_design
from .fem.femm_backend import FemmBackend, FemmNotAvailableError
from .fem.geometry import FINE_SPAN_FACTOR, CoilWindingGeometry, SlugGeometry, coupling_scale_m
from .fem.reference_backend import AnalyticReferenceBackend
from .linear_estimator import LinearStepperEstimator
from .linear_sim import LinearSimulator
from .linear_supervisor import FAULT, StepperSupervisor
from .optimize_design import Bounds, DesignKnobs, build_params, decode, simulate_design
from .plant import f_current, f_current_pm, q_shape

RESULTS_DIR = Path(__file__).resolve().parents[3] / "build" / "optimize_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = RESULTS_DIR / "latest.json"

FEM_RESULTS_DIR = Path(__file__).resolve().parents[3] / "build" / "fem_lut"
FEM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEM_LATEST_PATH = FEM_RESULTS_DIR / "latest_analysis.json"

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


def _fem_coupling_analysis(dk: DesignKnobs, coil_index: int = 0, n_offsets: int = 121,
                            n_currents: int = 4, field_lines: bool = False,
                            field_line_offset_m: Optional[float] = None,
                            field_line_current_a: Optional[float] = None) -> dict[str, Any]:
    """Force-vs-offset curves for ONE coil of `dk`, at `n_currents` evenly-spaced current
    levels up to dk.i_max_a, under BOTH force laws (see optimize_design.FORCE_LAWS):
    "analytic" via coil_design.build_coil_station's k_a/x_c through plant.q_shape's
    Gaussian-lobe shape (what every simulation uses by default), and "fem_reference" via
    fem.reference_backend.AnalyticReferenceBackend evaluated directly at each offset (the
    coil's real, non-Gaussian coupling shape). Geometry is built with coil position_m=0.0
    since everything here is a function of OFFSET from the coil's own center, not absolute
    position -- see fem/geometry.py's docstrings for why the peak sits away from offset=0.
    coil_index only affects the returned "coil_index"/"n_coils" metadata (every coil in a
    design_optimizer design shares the same geometry -- see optimize_design.build_params).

    field_lines: opt-in (default False, keeps this tool's existing fast/FEMM-free behavior
    unchanged) real magnetic field lines via FemmBackend.field_lines -- see
    docs/FEM_PIPELINE.md. Traced at ONE operating point: field_line_offset_m (default 0.0,
    slug centered on the coil -- the strongest-coupling point) and field_line_current_a
    (default dk.i_max_a). Requires FEMM installed; if it isn't, "field_lines" is None and
    "field_lines_note" explains why instead of raising."""
    if not (0 <= coil_index < dk.n_coils):
        raise ValueError(f"coil_index {coil_index} out of range for n_coils={dk.n_coils}")
    if n_offsets < 2:
        raise ValueError("n_offsets must be >= 2")
    if n_currents < 1:
        raise ValueError("n_currents must be >= 1")

    slug_geom = SlugGeometry(magnet_radius_m=dk.magnet_radius_m,
                              magnet_length_m=dk.magnet_length_m, remanence_t=dk.remanence_t)
    coil_geom = CoilWindingGeometry(position_m=0.0, turns=dk.turns,
                                     coil_length_m=dk.coil_length_m,
                                     radial_thickness_m=dk.radial_thickness_m)

    # 2x FINE_SPAN_FACTOR's own span: covers the peak/falloff (FINE_SPAN_FACTOR alone) PLUS
    # enough of the smooth tail beyond it to visually confirm the curve is actually decaying
    # -- see the FEM_PIPELINE.md phantom-force writeup this exists to make visible.
    scale_m = coupling_scale_m(coil_geom, slug_geom)
    span_m = 2.0 * FINE_SPAN_FACTOR * scale_m
    offsets_m = [-span_m + 2.0 * span_m * k / (n_offsets - 1) for k in range(n_offsets)]
    currents_a = [dk.i_max_a * (k + 1) / n_currents for k in range(n_currents)]

    analytic_coil = coil_design.build_coil_station(
        position_m=0.0, turns=dk.turns, coil_length_m=dk.coil_length_m,
        radial_thickness_m=dk.radial_thickness_m, magnet_radius_m=dk.magnet_radius_m,
        magnet_length_m=dk.magnet_length_m, remanence_t=dk.remanence_t,
    )
    backend = AnalyticReferenceBackend()

    force_analytic_n = []
    force_fem_reference_n = []
    for i_a in currents_a:
        analytic_row = []
        fem_row = []
        for offset_m in offsets_m:
            q = q_shape(offset_m, analytic_coil.x_c)
            analytic_row.append(q * (f_current(i_a, analytic_coil) + f_current_pm(i_a, analytic_coil.k_a)))
            fem_row.append(backend.solve(coil_geom, slug_geom, offset_m, i_a).force_n)
        force_analytic_n.append(analytic_row)
        force_fem_reference_n.append(fem_row)

    peak_analytic_n = max((abs(v) for row in force_analytic_n for v in row), default=0.0)
    peak_fem_reference_n = max((abs(v) for row in force_fem_reference_n for v in row), default=0.0)
    peak_relative_difference = (
        (peak_fem_reference_n - peak_analytic_n) / peak_analytic_n if peak_analytic_n else None
    )

    resolved_field_line_offset_m = 0.0 if field_line_offset_m is None else field_line_offset_m
    resolved_field_line_current_a = dk.i_max_a if field_line_current_a is None else field_line_current_a
    field_lines_result: Optional[list] = None
    field_lines_note: Optional[str] = None
    if field_lines:
        try:
            with FemmBackend() as backend:
                field_lines_result = backend.field_lines(
                    coil_geom, slug_geom, resolved_field_line_offset_m,
                    resolved_field_line_current_a,
                )
        except FemmNotAvailableError:
            field_lines_note = ("FEMM not installed -- field lines require a real FEMM "
                                 "solve (see docs/FEM_PIPELINE.md)")

    return {
        "kind": "fem_coupling",
        "coil_index": coil_index,
        "n_coils": dk.n_coils,
        "knobs": _knobs_dict(dk),
        "geometry": {
            "turns": dk.turns,
            "coil_length_m": dk.coil_length_m,
            "radial_thickness_m": dk.radial_thickness_m,
            "bore_clearance_m": coil_geom.bore_clearance_m,
            "bore_radius_m": coil_geom.bore_radius_m(slug_geom),
            "mean_radius_m": coil_geom.mean_radius_m(slug_geom),
            "outer_radius_m": coil_geom.outer_radius_m(slug_geom),
            "magnet_radius_m": dk.magnet_radius_m,
            "magnet_length_m": dk.magnet_length_m,
            "remanence_t": dk.remanence_t,
        },
        "currents_a": currents_a,
        "offsets_m": offsets_m,
        "force_analytic_n": force_analytic_n,
        "force_fem_reference_n": force_fem_reference_n,
        "peak_analytic_n": peak_analytic_n,
        "peak_fem_reference_n": peak_fem_reference_n,
        "peak_relative_difference": peak_relative_difference,
        "field_lines": field_lines_result,
        "field_lines_note": field_lines_note,
        "field_line_offset_m": resolved_field_line_offset_m if field_lines else None,
        "field_line_current_a": resolved_field_line_current_a if field_lines else None,
    }


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
    femm_speed: Optional[float] = None
    femm_note: Optional[str] = None
    history: list = dataclasses.field(default_factory=list)
    done: bool = False
    error: Optional[str] = None
    last_write_time: float = 0.0       # throttles the interim (mid-generation) file writes
    last_gen_evals: Optional[int] = None   # previous generation's eval count, for ETA interpolation


_INTERIM_WRITE_INTERVAL_S = 2.0   # how often objective() refreshes the file mid-generation


def _run_search(state: _SearchState, bounds: Bounds, maxiter: int, popsize: int, seed: int,
                 dt: float, t_end: float, result_path: Path, force_law: str = "analytic",
                 verify_with_femm: bool | None = None) -> None:
    def objective(x: Any) -> float:
        knobs = decode(x)
        if knobs.n_coils * knobs.coil_length_m > bounds.max_tube_length_m:
            v = 0.0
        else:
            try:
                v = simulate_design(knobs, dt=dt, t_end=t_end, force_law=force_law)
            except (ValueError, ZeroDivisionError):
                v = 0.0
        snapshot = None
        with state.lock:
            state.evals += 1
            state.gen_evals += 1
            if v <= 0.0:
                state.faults += 1
                state.gen_faults += 1
            if v > state.best_speed:
                state.best_speed = v
                state.best_knobs = _knobs_dict(knobs)

            # A generation typically takes long enough (popsize * len(knobs) evaluations,
            # single-process by design -- see the module docstring) that only writing at
            # generation boundaries (callback(), below) leaves the file -- and anything
            # watching it, like tools/web/optimizer_dashboard.html -- looking frozen for
            # the whole generation. Throttled to _INTERIM_WRITE_INTERVAL_S so this doesn't
            # turn into a write-every-eval I/O cost on fast candidates.
            now = time.time()
            if now - state.last_write_time >= _INTERIM_WRITE_INTERVAL_S:
                state.last_write_time = now
                elapsed = now - state.start_time
                expected_gen_evals = state.last_gen_evals or (popsize * 11)
                gen_fraction = min(1.0, state.gen_evals / expected_gen_evals) if expected_gen_evals else 0.0
                progress = state.generation + gen_fraction
                eta_s = (elapsed / progress) * max(0.0, maxiter - progress) if progress > 0 else None
                snapshot = {
                    "status": "running", "generation": state.generation, "maxiter": maxiter,
                    "popsize": popsize, "seed": seed, "force_law": force_law,
                    "evals_total": state.evals,
                    "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
                    "best_speed_m_s": state.best_speed, "best_knobs": state.best_knobs,
                    "elapsed_s": elapsed, "eta_s": eta_s, "history": state.history,
                    "generation_in_progress": state.generation + 1,
                    "evals_this_gen_so_far": state.gen_evals,
                    "evals_this_gen_expected": expected_gen_evals,
                }
        if snapshot is not None:
            result_path.write_text(json.dumps(snapshot, indent=2))
        return -v

    def callback(xk: Any, convergence: float) -> bool:
        with state.lock:
            state.generation += 1
            fault_frac = state.gen_faults / state.gen_evals if state.gen_evals else 0.0
            elapsed = time.time() - state.start_time
            eta_s = (elapsed / state.generation) * max(0, maxiter - state.generation)
            state.last_gen_evals = state.gen_evals
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
            state.last_write_time = time.time()
            snapshot = {
                "status": "running", "generation": state.generation, "maxiter": maxiter,
                "popsize": popsize, "seed": seed, "force_law": force_law,
                "evals_total": state.evals,
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
        best_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end, bootstrap_timeout_s=0.20,
                                      force_law=force_law)

        # Real-FEMM verification of the winner ONLY -- never the search itself (see
        # optimize_design.optimize's docstring: each FEMM solve is seconds, the search calls
        # the force law millions of times). None (default) auto-attempts and falls back to
        # femm_speed=None if FEMM isn't installed; True requires it and re-raises.
        femm_speed: float | None = None
        femm_note: str | None = None
        if verify_with_femm is not False:
            try:
                femm_speed = simulate_design(best_knobs, dt=2e-5, t_end=t_end,
                                              bootstrap_timeout_s=0.20, force_law="femm")
            except FemmNotAvailableError:
                if verify_with_femm is True:
                    raise
                femm_note = "FEMM not installed -- best_speed_m_s is analytic-only"

        with state.lock:
            state.best_speed = best_speed
            state.best_knobs = _knobs_dict(best_knobs)
            state.femm_speed = femm_speed
            state.femm_note = femm_note
            state.done = True
            elapsed = time.time() - state.start_time
            final = {
                "status": "done", "generation": state.generation, "maxiter": maxiter,
                "popsize": popsize, "seed": seed, "force_law": force_law,
                "evals_total": state.evals,
                "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
                "best_speed_m_s": best_speed,
                "best_speed_search_estimate_m_s": -result.fun,
                "femm_speed_m_s": femm_speed, "femm_note": femm_note,
                "best_knobs": state.best_knobs, "elapsed_s": elapsed, "eta_s": 0.0,
                "history": state.history,
            }
        result_path.write_text(json.dumps(final, indent=2))
    except Exception as exc:
        # Surfaced two ways: raised to the polling coroutine below (for a live MCP tool
        # caller), AND written to result_path here -- anything watching the FILE instead
        # (tools/web/optimizer_dashboard.html, or a client re-attaching via get_latest_result)
        # has no other way to learn the run died; without this it would just show the last
        # "running" snapshot forever, indistinguishable from a run that's still healthy.
        error_msg = f"{type(exc).__name__}: {exc}"
        with state.lock:
            state.error = error_msg
            state.done = True
            elapsed = time.time() - state.start_time
            error_snapshot = {
                "status": "error", "error": error_msg, "generation": state.generation,
                "maxiter": maxiter, "popsize": popsize, "seed": seed, "force_law": force_law,
                "evals_total": state.evals,
                "fault_fraction_overall": state.faults / state.evals if state.evals else 0.0,
                "best_speed_m_s": state.best_speed, "best_knobs": state.best_knobs,
                "elapsed_s": elapsed, "eta_s": None, "history": state.history,
            }
        result_path.write_text(json.dumps(error_snapshot, indent=2))


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
    force_law: str = "analytic",
    verify_with_femm: Optional[bool] = None,
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

    force_law: "analytic" (default) or "fem_reference" -- drives the SEARCH itself. Neither
    is a real FEM solve (see optimize_design.FORCE_LAWS / docs/FEM_PIPELINE.md); a live
    FEMM-in-the-loop search is computationally infeasible (each FEMM solve is seconds, the
    search calls the force law millions of times). "fem_reference" runs meaningfully slower
    per evaluation than "analytic" -- budget maxiter/popsize accordingly.

    verify_with_femm: after the search, re-simulate ONLY the winning design under a real
    FEMM solve (fem.femm_backend.FemmBackend, one sweep shared across every coil) and report
    it as "femm_speed_m_s" -- distinct from "best_speed_m_s" (always the analytic/fem_reference
    number), never substituted for it. None (default): auto-verify if FEMM is installed,
    leave femm_speed_m_s null with a "femm_note" otherwise. True: require FEMM, raise if
    missing. False: skip verification entirely.
    """
    bounds = _bounds_from_overrides(bounds_overrides)
    state = _SearchState()
    total_evals_estimate = max(1, maxiter * popsize * 11)

    thread = threading.Thread(
        target=_run_search,
        args=(state, bounds, maxiter, popsize, seed, dt, t_end, LATEST_PATH, force_law,
              verify_with_femm),
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
            "femm_speed_m_s": state.femm_speed, "femm_note": state.femm_note,
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
    force_law: str = "analytic",
) -> dict[str, Any]:
    """Run one closed-loop simulation for a fully-specified design (the same shape as
    run_optimization's/get_latest_result's "best_knobs") and return the full position/
    velocity/current/temperature time series plus gate-crossing events, downsampled to at
    most max_samples points, for the time-series view of the EMAC Optimizer Dashboard
    artifact -- including the "Slug animation" view, which replays this ACTUAL simulated
    trajectory (not a canned/illustrative one) against the design's real coil/gate layout.
    Use this on the winning design from a search to see *how* it reaches its exit speed, not
    just the final number. force_law: "analytic" (default), "fem_reference" (fast analytic
    approximation, not real FEM), or "femm" (a real FEMM sweep, needs FEMM installed -- slower
    but this IS a real FEM solve) -- see optimize_design.FORCE_LAWS / docs/FEM_PIPELINE.md."""
    dk = DesignKnobs(**knobs)
    p = build_params(dk, force_law=force_law)
    pitch = dk.coil_length_m
    x0 = -0.5 * pitch - 0.001
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, i_max=dk.i_max_a, pm_envelope=dk.pump_envelope,
                             bootstrap_timeout_s=bootstrap_timeout_s)
    expected_steps = max(1, int(t_end / dt))
    sample_every = max(1, expected_steps // max_samples)
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=sample_every)
    log = sim.run(x0=x0, v0=0.0, v_tgt=optimize_design.V_TGT_FULL_THRUST, t_end=t_end)
    return {
        "fault": sup.mode == FAULT,
        "t": log.t, "x": log.x, "v": log.v,
        "active_current": log.active_current, "active_temperature_c": log.active_temperature_c,
        "active_coil": log.active_coil,
        "gate_t": log.gate_t, "gate_v": log.gate_v, "gate_index": log.gate_index,
        "exit_speed_m_s": log.gate_v[-1] if log.gate_t else 0.0,
        "knobs": _knobs_dict(dk),
        "force_law": force_law,
        # Geometry the animation/track view draws -- derived from the SAME LinearActuatorParams
        # the simulation actually ran against (build_params packs coils edge-to-edge, pitch ==
        # coil_length_m), not re-derived from knobs client-side, so it can never drift from what
        # the slug was actually simulated moving through.
        "coil_positions_m": [c.position_m for c in p.coils],
        "gate_positions_m": [g.position_m for g in p.gates],
        "coil_length_m": dk.coil_length_m,
    }


@mcp.tool()
def sensitivity_sweep(
    knob: str, baseline: Optional[dict[str, Any]] = None,
    bounds_overrides: Optional[dict[str, Any]] = None,
    n_points: int = 9, dt: float = 2e-4, t_end: float = 3.0,
    force_law: str = "analytic",
) -> dict[str, Any]:
    """One-at-a-time sensitivity sweep of a single knob (see docs/DESIGN_OPTIMIZER.md
    section 6 for the knob list) around a baseline design, holding every other knob fixed.
    baseline defaults to the latest run_optimization result's best_knobs. Returns
    {"knob", "baseline", "points": [{"value", "speed"}, ...]} -- load it in the EMAC
    Optimizer Dashboard artifact to see the main-effect curve. force_law: "analytic"
    (default), "fem_reference", or "femm" (real FEM, needs FEMM installed, much slower per
    point) -- see optimize_design.FORCE_LAWS / docs/FEM_PIPELINE.md."""
    bounds = _bounds_from_overrides(bounds_overrides)
    base = DesignKnobs(**baseline) if baseline else _load_latest_knobs()
    points = design_sensitivity.sweep_knob(knob, base, bounds, n_points=n_points, dt=dt, t_end=t_end,
                                            force_law=force_law)
    return {"knob": knob, "baseline": _knobs_dict(base), "points": points, "force_law": force_law}


@mcp.tool()
def fem_coupling_analysis(
    knobs: Optional[dict[str, Any]] = None, coil_index: int = 0,
    n_offsets: int = 121, n_currents: int = 4, field_lines: bool = False,
    field_line_offset_m: Optional[float] = None, field_line_current_a: Optional[float] = None,
) -> dict[str, Any]:
    """Compare the analytic coupling-shape estimate (coil_design.build_coil_station's k_a/
    x_c through plant.q_shape's Gaussian lobe -- what every simulation uses unless it opts
    into force_law="fem_reference") against fem.reference_backend's real, non-Gaussian
    coupling shape: force vs. slug offset, at n_currents evenly-spaced current levels up to
    the design's i_max_a, for one coil. This is the same divergence that changes both the
    reported exit speed AND the winning design when a search runs under force_law=
    "fem_reference" instead of "analytic" -- see docs/FEM_PIPELINE.md's "Using it in the
    design optimizer and sensitivity sweeps" for measured numbers.

    knobs defaults to the latest run_optimization result's best_knobs (same fallback
    simulate_design_detailed/sensitivity_sweep use). Also writes a JSON snapshot to
    build/fem_lut/latest_analysis.json (same load-a-file pattern as run_optimization) for
    the "FEM coupling curve" view of the EMAC Optimizer Dashboard artifact.

    field_lines: opt-in (default False -- no added FEMM cost unless requested) real
    magnetic field lines traced through a real FEMM solve at one operating point
    (field_line_offset_m, default 0.0 = slug centered on the coil; field_line_current_a,
    default the design's i_max_a) -- useful both to see what the FEM solve actually looks
    like and for building intuition when designing a control scheme for the electromagnets
    driving the PM slug. Needs FEMM installed; if it isn't, the result's "field_lines" is
    null and "field_lines_note" explains why, rather than raising. See
    docs/FEM_PIPELINE.md."""
    dk = DesignKnobs(**knobs) if knobs else _load_latest_knobs()
    result = _fem_coupling_analysis(dk, coil_index=coil_index, n_offsets=n_offsets,
                                     n_currents=n_currents, field_lines=field_lines,
                                     field_line_offset_m=field_line_offset_m,
                                     field_line_current_a=field_line_current_a)
    FEM_LATEST_PATH.write_text(json.dumps(result, indent=2))
    result["results_file"] = str(FEM_LATEST_PATH)
    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
