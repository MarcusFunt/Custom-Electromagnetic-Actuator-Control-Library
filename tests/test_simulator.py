import math

import pytest

from emac_sim.cli import run_scenario, steady_state_rms_error
from emac_sim.config import parse_config


def test_phase0_scenario_tracks_all_hold_targets_under_two_percent():
    _, log = run_scenario(t_end=22.0)

    assert steady_state_rms_error(log, 6.5, 8.0) < 2.0
    assert steady_state_rms_error(log, 13.5, 15.0) < 2.0
    assert steady_state_rms_error(log, 20.5, 22.0) < 2.0


def test_simulator_logs_have_consistent_lengths():
    _, log = run_scenario(t_end=4.0)

    sample_len = len(log.t)
    assert len(log.theta) == sample_len
    assert len(log.omega) == sample_len
    assert len(log.i) == sample_len
    assert len(log.E_true) == sample_len
    assert len(log.theta_est) == sample_len

    crossing_len = len(log.cx_t)
    assert crossing_len > 0
    assert len(log.cx_A_peak) == crossing_len
    assert len(log.cx_A_energy) == crossing_len
    assert len(log.cx_A_est) == crossing_len
    assert len(log.cx_E_true) == crossing_len
    assert len(log.cx_E_est) == crossing_len
    assert len(log.cx_A_tgt) == crossing_len
    assert len(log.cx_kind) == crossing_len
    assert len(log.cx_ipeak) == crossing_len

    assert all(math.isfinite(x) for x in log.cx_A_energy)


def test_pendulum_gate_dropout_configuration_affects_sensor_events():
    config = parse_config({"gate": {"dropout_probability": 1.0},
                           "sim": {"duration_s": 2.0, "random_seed": 1}})
    _, log = run_scenario(config=config)
    assert log.cx_t == []
