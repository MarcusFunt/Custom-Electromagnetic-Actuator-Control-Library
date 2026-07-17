"""Tests for the FEMM Bayesian-optimization driver (studies/femm_trends/bo_search.py).

The loop, warm-start, resume, objectives, and snapshot are all exercised with the ANALYTIC
reference backend (--backend reference) -- fast, no FEMM, CI-safe. A real-FEMM smoke test is
gated on `pytest.importorskip("femm")` and only runs 1 evaluation when FEMM is present.

Note: bo_search runs real FEMM only via a subprocess it spawns; these reference tests never
touch FEMM, so they are safe to run while a separate FEMM sweep is in progress.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

bo = pytest.importorskip("bo_search")
pytest.importorskip("skopt")

# A moderate, feasible geometry that launches the slug under the reference force law.
GEOM = {"turns": 450, "coil_length_m": 0.02, "radial_thickness_m": 0.014,
        "magnet_radius_m": 0.008, "magnet_length_m": 0.022, "remanence_t": 1.25}

# A cheap 2-combo driver set so the inner sweep is fast in tests (patched in per-test).
SMALL_DRIVERS = {"bus_voltage_v": [110.0, 260.0], "driver_bipolar": [True],
                 "pump_envelope": ["rcos"], "i_max_a": [70.0]}


@pytest.fixture
def small_drivers(monkeypatch):
    monkeypatch.setattr(bo, "DRIVER_FACTORS", SMALL_DRIVERS)


# ----------------------------- pure helpers -----------------------------------
def test_objective_value_variants():
    assert bo.objective_value(10.0, 2.0, "speed") == 10.0
    assert bo.objective_value(10.0, 2.0, "momentum") == 20.0
    assert bo.objective_value(10.0, 2.0, "energy") == pytest.approx(0.5 * 2.0 * 100.0)
    with pytest.raises(ValueError):
        bo.objective_value(1.0, 1.0, "nope")


def test_geom_x_roundtrip():
    x = bo.x_from_geom(GEOM)
    g = bo.geom_from_x(x)
    assert g["turns"] == GEOM["turns"] and isinstance(g["turns"], int)
    for k in bo.GEOM_KEYS:
        assert g[k] == pytest.approx(GEOM[k])


def test_driver_combos_count():
    # default is the full 32; product of the axis lengths
    n = 1
    for v in bo.DRIVER_FACTORS.values():
        n *= len(v)
    assert len(bo.driver_combos()) == n


def test_space_bounds():
    dims = bo.build_space()
    assert len(dims) == len(bo.GEOM_KEYS)
    assert dims[0].name == "turns"


# ----------------------------- one geometry evaluation ------------------------
def test_evaluate_geometry_reference_launches(small_drivers):
    be = bo.make_backend("reference")
    res = bo.evaluate_geometry(GEOM, n_coils=5, objective="speed", backend=be, sim_t_end=0.5)
    assert res["error"] is None
    assert res["value"] > 0.0                 # the reference LUT should launch the slug
    assert res["driver"] is not None
    assert res["n_evaluated"] == len(bo.driver_combos())


def test_evaluate_geometry_tube_too_long():
    be = bo.make_backend("reference")
    res = bo.evaluate_geometry(GEOM, n_coils=1000, objective="speed", backend=be)
    assert res["value"] == 0.0
    assert res["error"] == "tube_too_long"


def test_energy_objective_scales_with_mass(small_drivers):
    be = bo.make_backend("reference")
    r_speed = bo.evaluate_geometry(GEOM, 5, "speed", be, sim_t_end=0.5)
    r_energy = bo.evaluate_geometry(GEOM, 5, "energy", be, sim_t_end=0.5)
    # energy = 1/2 m v^2 with the SAME winning speed's mass -> strictly larger number here
    assert r_energy["value"] > 0.0
    assert r_energy["speed_mps"] > 0.0
    assert r_speed["value"] == pytest.approx(r_speed["speed_mps"])


# ----------------------------- the loop, snapshot, log ------------------------
def test_run_bo_reference_loop(tmp_path, small_drivers):
    outdir = tmp_path / "bo"
    snap = tmp_path / "latest.json"
    ev = bo.inprocess_evaluator(5, "speed", "reference", sim_t_end=0.5)
    best = bo.run_bo(ev, n_calls=5, objective="speed", n_coils=5, outdir=outdir,
                     snapshot_path=snap, warmstart_dirs=None, n_initial=3, seed=0,
                     log=lambda *a: None)
    assert best["value"] > 0.0
    assert best["knobs"] is not None
    # log has exactly n_calls lines
    lines = (outdir / "bo_eval_log.jsonl").read_text().splitlines()
    assert len([ln for ln in lines if ln.strip()]) == 5
    # snapshot has the GUI-compatible keys
    s = json.loads(snap.read_text())
    for k in ("best_speed_mps", "best_value", "best_knobs", "history", "generation"):
        assert k in s
    assert s["generation"] == 5
    assert len(s["history"]) == 5
    # best-so-far is monotonic non-decreasing
    bests = [h["best"] for h in s["history"]]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))


def test_run_bo_resume_replays_log(tmp_path, small_drivers):
    outdir = tmp_path / "bo"
    ev = bo.inprocess_evaluator(5, "speed", "reference", sim_t_end=0.5)
    bo.run_bo(ev, n_calls=4, objective="speed", n_coils=5, outdir=outdir,
              snapshot_path=None, warmstart_dirs=None, n_initial=3, seed=0, log=lambda *a: None)
    first = [ln for ln in (outdir / "bo_eval_log.jsonl").read_text().splitlines() if ln.strip()]
    assert len(first) == 4
    # resume with a larger budget: replays 4, adds 3 more -> 7 total
    bo.run_bo(ev, n_calls=7, objective="speed", n_coils=5, outdir=outdir,
              snapshot_path=None, warmstart_dirs=None, n_initial=3, seed=0, log=lambda *a: None)
    second = [ln for ln in (outdir / "bo_eval_log.jsonl").read_text().splitlines() if ln.strip()]
    assert len(second) == 7
    assert second[:4] == first          # earlier evaluations are preserved verbatim


# ----------------------------- warm start -------------------------------------
def _write_cell(dirpath: Path, cell_id: int, geom: dict, speed_by_driver):
    dirpath.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (bus, speed) in enumerate(speed_by_driver):
        rows.append({"cell_id": cell_id, **geom, "bus_voltage_v": bus, "driver_bipolar": True,
                     "pump_envelope": "rcos", "i_max_a": 70.0, "n_coils": 5,
                     "force_law": "femm", "exit_speed_mps": speed, "sim_error": None})
        # an analytic-law row that must be IGNORED by warm-start
        rows.append({"cell_id": cell_id, **geom, "bus_voltage_v": bus, "driver_bipolar": True,
                     "pump_envelope": "rcos", "i_max_a": 70.0, "n_coils": 5,
                     "force_law": "analytic", "exit_speed_mps": speed * 10, "sim_error": None})
    (dirpath / f"cell_{cell_id:04d}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_load_warmstart_reduces_to_per_geometry_best(tmp_path):
    d = tmp_path / "results"
    _write_cell(d, 1, GEOM, [(110.0, 4.0), (260.0, 9.0)])       # best femm speed = 9.0
    other = dict(GEOM, turns=200)
    _write_cell(d, 2, other, [(110.0, 2.0), (260.0, 3.0)])       # best = 3.0
    recs, n_rows = bo.load_warmstart([d], "speed")
    assert n_rows == 4                                            # only the 4 femm rows counted
    by_geom = {tuple(bo.x_from_geom(r["geom"])): r for r in recs}
    assert by_geom[tuple(bo.x_from_geom(GEOM))]["value"] == pytest.approx(9.0)
    assert by_geom[tuple(bo.x_from_geom(other))]["value"] == pytest.approx(3.0)
    # the winning driver row is captured (the 260 V row won GEOM at 9.0 m/s)
    assert by_geom[tuple(bo.x_from_geom(GEOM))]["driver"]["bus_voltage_v"] == 260.0


def test_warmstart_seeds_best(tmp_path, small_drivers):
    # A warm-start point with an absurdly high objective should dominate the reported best,
    # proving the priors are actually told to the optimizer.
    d = tmp_path / "results"
    _write_cell(d, 1, GEOM, [(260.0, 999.0)])
    outdir = tmp_path / "bo"
    ev = bo.inprocess_evaluator(5, "speed", "reference", sim_t_end=0.5)
    best = bo.run_bo(ev, n_calls=3, objective="speed", n_coils=5, outdir=outdir,
                     snapshot_path=None, warmstart_dirs=[d], n_initial=2, seed=0,
                     log=lambda *a: None)
    # the warm-start best seeds the reported best AND reconstructs its full knobs from the row
    assert best["value"] >= 999.0
    assert best["knobs"] is not None
    assert best["knobs"]["bus_voltage_v"] == 260.0


# ----------------------------- subprocess plumbing (no FEMM) ------------------
def test_subprocess_evaluator_reference(tmp_path, small_drivers):
    # Exercise the spawn/marshal/timeout path WITHOUT FEMM by running the worker with the
    # analytic reference backend. NOTE: the child process reads bo_search.DRIVER_FACTORS at
    # its own import time, so it uses the full 32-driver set regardless of the patch here.
    ev = bo.subprocess_evaluator(5, "speed", backend_name="reference", timeout_s=120.0,
                                 workdir=tmp_path / "work", sim_t_end=0.3)
    res = ev(GEOM)
    assert res.get("error") is None
    assert res["value"] > 0.0
    assert res["driver"] is not None


def test_subprocess_evaluator_timeout(tmp_path):
    # An impossibly short timeout must be survived: score 0 with an error, not a hang/raise.
    ev = bo.subprocess_evaluator(5, "speed", backend_name="reference", timeout_s=0.001,
                                 workdir=tmp_path / "work")
    res = ev(GEOM)
    assert res["value"] == 0.0
    assert res["error"] == "timeout"


# ----------------------------- real-FEMM smoke (gated) ------------------------
def test_femm_smoke_single_eval(tmp_path):
    """One real-FEMM geometry evaluation, only when FEMM is installed. Needs EXCLUSIVE FEMM;
    skipped in CI/without FEMM. Kept tiny (1 geometry) to stay fast."""
    pytest.importorskip("femm")
    be = bo.make_backend("femm")
    be._ensure_open()
    try:
        res = bo.evaluate_geometry(GEOM, n_coils=5, objective="speed", backend=be)
    finally:
        be.close()
    assert res["error"] is None
    assert res["value"] >= 0.0
