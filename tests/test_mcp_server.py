import json

import pytest

pytest.importorskip("mcp")   # optional dependency (pip install -e .[mcp]) -- skip cleanly without it

from emac_sim import mcp_server as m
from emac_sim.optimize_design import Bounds, DesignKnobs


TIGHT_BOUNDS = Bounds(n_coils=(2, 3), turns=(20, 60), coil_length_m=(0.02, 0.04),
                       radial_thickness_m=(0.004, 0.01), magnet_radius_m=(0.003, 0.008),
                       magnet_length_m=(0.01, 0.02), i_max_a=(5.0, 30.0),
                       max_tube_length_m=0.2)


@pytest.fixture(autouse=True)
def isolated_results_file(tmp_path, monkeypatch):
    """Every test gets its own results file -- never touch the real
    build/optimize_results/latest.json a live run might be using."""
    path = tmp_path / "latest.json"
    monkeypatch.setattr(m, "LATEST_PATH", path)
    fem_path = tmp_path / "fem_latest.json"
    monkeypatch.setattr(m, "FEM_LATEST_PATH", fem_path)
    return path


def test_bounds_from_overrides_converts_lists_to_tuples_for_tuple_fields():
    b = m._bounds_from_overrides({"bus_voltage_v": [3, 60], "i_max_a": [1, 30],
                                   "max_tube_length_m": 0.5})
    assert b.bus_voltage_v == (3, 60)
    assert b.i_max_a == (1, 30)
    assert b.max_tube_length_m == pytest.approx(0.5)


def test_bounds_from_overrides_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown bound"):
        m._bounds_from_overrides({"not_a_real_field": [1, 2]})


def test_bounds_from_overrides_none_or_empty_returns_defaults():
    assert m._bounds_from_overrides(None) == Bounds()
    assert m._bounds_from_overrides({}) == Bounds()


def test_load_latest_knobs_falls_back_to_a_reasonable_default_when_no_file_exists():
    knobs = m._load_latest_knobs()
    assert isinstance(knobs, DesignKnobs)
    assert knobs.n_coils > 0


def test_load_latest_knobs_reads_best_knobs_from_an_existing_snapshot(isolated_results_file):
    knobs = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                         n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                         magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                         i_max_a=15.0)
    isolated_results_file.write_text(json.dumps({"best_knobs": m._knobs_dict(knobs)}))
    loaded = m._load_latest_knobs()
    assert loaded == knobs


def test_get_latest_result_reports_no_results_yet_before_any_run():
    assert m.get_latest_result() == {"status": "no_results_yet"}


def test_run_search_writes_a_done_snapshot_and_populates_state(isolated_results_file):
    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=2, popsize=3, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)

    assert state.done is True
    assert state.error is None
    assert state.evals > 0
    assert state.best_speed >= 0.0
    assert state.best_knobs is not None

    snapshot = json.loads(isolated_results_file.read_text())
    assert snapshot["status"] == "done"
    assert snapshot["best_speed_m_s"] == pytest.approx(state.best_speed)
    assert 0.0 <= snapshot["fault_fraction_overall"] <= 1.0
    assert len(snapshot["history"]) == snapshot["generation"] == 2


def test_run_search_snapshot_is_readable_via_get_latest_result(isolated_results_file):
    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=1, popsize=2, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)
    result = m.get_latest_result()
    assert result["status"] == "done"
    assert result["best_knobs"] == state.best_knobs


def test_run_search_writes_an_error_snapshot_when_the_search_raises(isolated_results_file, monkeypatch):
    """Found by watching a real long run through the GUI: an exception inside _run_search's
    try block used to only set state.error (visible to a live MCP tool caller) without ever
    writing to result_path -- anything watching the FILE instead (the dashboard, or a client
    reattaching via get_latest_result) would see the last "running" snapshot forever, with no
    way to tell the search had actually died. The file must reflect the failure too."""
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure for this test")
    monkeypatch.setattr(m, "differential_evolution", boom)

    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=2, popsize=3, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)

    assert state.done is True
    assert state.error is not None

    snapshot = json.loads(isolated_results_file.read_text())
    assert snapshot["status"] == "error"
    assert "synthetic failure for this test" in snapshot["error"]


def test_run_search_writes_interim_snapshots_mid_generation(isolated_results_file, monkeypatch):
    """A single-process generation (popsize * 11 evaluations) can take minutes -- without a
    mid-generation write, anything watching result_path only sees an update once per
    generation and looks frozen the whole time. Forcing the throttle interval to 0 makes
    every evaluation eligible to write, so with more evaluations than generations we should
    see interim snapshots (carrying generation_in_progress/evals_this_gen_so_far) land in
    the file before the run finishes, not just the per-generation and final ones."""
    monkeypatch.setattr(m, "_INTERIM_WRITE_INTERVAL_S", 0.0)
    seen_interim = []
    real_write_text = type(isolated_results_file).write_text
    def spy_write_text(self, text, *a, **kw):
        data = json.loads(text)
        if "generation_in_progress" in data:
            seen_interim.append(data)
        return real_write_text(self, text, *a, **kw)
    monkeypatch.setattr(type(isolated_results_file), "write_text", spy_write_text)

    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=2, popsize=3, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)

    assert len(seen_interim) > 0
    sample = seen_interim[0]
    assert sample["status"] == "running"
    assert sample["evals_this_gen_so_far"] >= 1
    assert sample["evals_this_gen_expected"] > 0


def test_simulate_design_detailed_returns_full_time_series_and_matches_exit_speed():
    knobs = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                         n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                         magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                         i_max_a=15.0)
    detail = m.simulate_design_detailed(m._knobs_dict(knobs), dt=1e-3, t_end=0.3, max_samples=50)

    assert detail["fault"] is False
    assert len(detail["t"]) == len(detail["x"]) == len(detail["v"]) == len(detail["active_coil"])
    assert len(detail["t"]) <= 50 + 1     # downsampled, small overshoot from the floor div is fine
    if detail["gate_t"]:
        assert detail["exit_speed_m_s"] == pytest.approx(detail["gate_v"][-1])
    else:
        assert detail["exit_speed_m_s"] == 0.0


def test_simulate_design_detailed_includes_the_actual_coil_and_gate_layout_it_simulated():
    """The 'Slug animation' dashboard view needs to draw the REAL coil/gate positions the
    trajectory was simulated against -- derived from the same LinearActuatorParams the sim
    ran with, not re-derived client-side from knobs (which could drift if build_params'
    pitch logic ever changes)."""
    knobs = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                         n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                         magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                         i_max_a=15.0)
    detail = m.simulate_design_detailed(m._knobs_dict(knobs), dt=1e-3, t_end=0.3, max_samples=50)

    assert detail["coil_positions_m"] == [0.0, pytest.approx(0.03), pytest.approx(0.06)]
    assert len(detail["gate_positions_m"]) == 3
    assert detail["gate_positions_m"][0] == pytest.approx(-0.015)
    assert detail["coil_length_m"] == pytest.approx(0.03)


def test_sensitivity_sweep_uses_the_given_baseline_not_the_latest_result():
    baseline = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                            n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                            magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                            i_max_a=15.0)
    result = m.sensitivity_sweep("i_max_a", baseline=m._knobs_dict(baseline),
                                 bounds_overrides={"i_max_a": [5.0, 30.0]},
                                 n_points=3, dt=1e-3, t_end=0.3)
    assert result["knob"] == "i_max_a"
    assert result["baseline"] == m._knobs_dict(baseline)
    assert len(result["points"]) == 3
    assert [p["value"] for p in result["points"]] == [5.0, 17.5, 30.0]


def test_sensitivity_sweep_defaults_baseline_to_latest_result(isolated_results_file):
    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=1, popsize=2, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)
    result = m.sensitivity_sweep("i_max_a", n_points=2, dt=1e-3, t_end=0.3)
    assert result["baseline"] == state.best_knobs


def test_run_search_and_sensitivity_sweep_accept_fem_reference_force_law(isolated_results_file):
    """The dashboard needs to be able to request a FEM-backed run through the same tools a
    default (analytic) run uses -- see docs/FEM_PIPELINE.md."""
    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=1, popsize=2, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file, force_law="fem_reference")
    snapshot = json.loads(isolated_results_file.read_text())
    assert snapshot["force_law"] == "fem_reference"

    baseline = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                            n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                            magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                            i_max_a=15.0)
    result = m.sensitivity_sweep("i_max_a", baseline=m._knobs_dict(baseline),
                                 bounds_overrides={"i_max_a": [5.0, 30.0]},
                                 n_points=2, dt=1e-3, t_end=0.3, force_law="fem_reference")
    assert result["force_law"] == "fem_reference"
    assert len(result["points"]) == 2


DEFAULT_KNOBS = DesignKnobs(bus_voltage_v=36.0, driver_bipolar=True, pump_envelope="rcos",
                             n_coils=5, turns=150, coil_length_m=0.02, radial_thickness_m=0.008,
                             magnet_radius_m=0.006, magnet_length_m=0.018, remanence_t=1.2,
                             i_max_a=10.0)


def test_fem_coupling_analysis_returns_matching_analytic_and_fem_grids():
    result = m.fem_coupling_analysis(knobs=m._knobs_dict(DEFAULT_KNOBS), coil_index=0,
                                      n_offsets=15, n_currents=3)
    assert result["kind"] == "fem_coupling"
    assert result["coil_index"] == 0
    assert result["n_coils"] == 5
    assert len(result["offsets_m"]) == 15
    assert len(result["currents_a"]) == 3
    assert len(result["force_analytic_n"]) == 3
    assert len(result["force_analytic_n"][0]) == 15
    assert len(result["force_fem_reference_n"]) == 3
    assert len(result["force_fem_reference_n"][0]) == 15
    assert result["peak_analytic_n"] > 0.0
    assert result["peak_fem_reference_n"] > 0.0
    assert result["peak_relative_difference"] is not None


def test_fem_coupling_analysis_writes_a_results_file(isolated_results_file):
    result = m.fem_coupling_analysis(knobs=m._knobs_dict(DEFAULT_KNOBS), n_offsets=11, n_currents=2)
    assert m.FEM_LATEST_PATH.exists()
    on_disk = json.loads(m.FEM_LATEST_PATH.read_text())
    assert on_disk["kind"] == "fem_coupling"
    assert result["results_file"] == str(m.FEM_LATEST_PATH)


def test_fem_coupling_analysis_defaults_knobs_to_latest_result(isolated_results_file):
    state = m._SearchState()
    m._run_search(state, TIGHT_BOUNDS, maxiter=1, popsize=2, seed=0, dt=1e-3, t_end=0.3,
                  result_path=isolated_results_file)
    result = m.fem_coupling_analysis(n_offsets=11, n_currents=2)
    assert result["knobs"] == state.best_knobs


def test_fem_coupling_analysis_rejects_out_of_range_coil_index():
    with pytest.raises(ValueError, match="coil_index"):
        m.fem_coupling_analysis(knobs=m._knobs_dict(DEFAULT_KNOBS), coil_index=99)
