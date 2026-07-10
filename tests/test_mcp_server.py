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


def test_simulate_design_detailed_returns_full_time_series_and_matches_exit_speed():
    knobs = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                         n_coils=3, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                         magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                         i_max_a=15.0)
    detail = m.simulate_design_detailed(m._knobs_dict(knobs), dt=1e-3, t_end=0.3, max_samples=50)

    assert detail["fault"] is False
    assert len(detail["t"]) == len(detail["x"]) == len(detail["v"])
    assert len(detail["t"]) <= 50 + 1     # downsampled, small overshoot from the floor div is fine
    if detail["exit_t"] is not None:
        assert detail["exit_speed_m_s"] > 0.0
        assert detail["exit_t"] > detail["gate_t"][-1]
    else:
        assert detail["exit_speed_m_s"] == 0.0


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
