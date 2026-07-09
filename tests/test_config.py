from pathlib import Path

import pytest

from emac_sim.cli import run_scenario
from emac_sim.config import load_config, parse_config


EXAMPLE_CONFIG = Path("examples/configs/pendulum_softiron_1gate.toml")


def test_example_config_loads_and_maps_to_pendulum_params():
    config = load_config(EXAMPLE_CONFIG)
    params = config.to_pendulum_params()

    assert params.L == pytest.approx(config.pendulum.length_m)
    assert params.m == pytest.approx(config.pendulum.bob_mass_kg)
    assert params.Q == pytest.approx(config.pendulum.quality_factor)
    assert params.theta_c == pytest.approx(config.primary_coil.theta_c_rad)
    assert params.Cmag == pytest.approx(config.primary_coil.c_mag_nm_per_a2)
    assert params.i_sat == pytest.approx(config.primary_coil.i_sat_a)
    assert params.dalpha == pytest.approx(config.primary_gate.angular_width_rad)


def test_config_values_affect_run_scenario():
    config = parse_config(
        {
            "pendulum": {
                "length_m": 0.42,
                "bob_mass_kg": 0.075,
                "quality_factor": 150.0,
                "initial_angle_rad": 0.04,
            },
            "gate": {"angular_width_rad": 0.045},
            "coil": {
                "theta_c_rad": 0.04,
                "c_mag_nm_per_a2": 0.012,
                "i_sat_a": 6.0,
                "max_current_a": 5.0,
            },
            "controller": {
                "target_amplitude_rad": 0.18,
                "k_energy": 0.25,
                "pulse_width_half_period_fraction": 0.22,
            },
            "sim": {"duration_s": 1.5, "dt_s": 0.0005, "sample_every": 5},
        }
    )

    params, log = run_scenario(config=config)

    assert params.L == pytest.approx(0.42)
    assert params.m == pytest.approx(0.075)
    assert params.dalpha == pytest.approx(0.045)
    assert log.t[-1] == pytest.approx(1.5, abs=0.01)
    assert len(log.t) > 0


def test_target_segments_are_sorted_and_used():
    config = parse_config(
        {
            "target": {
                "segments": [
                    {"t_s": 5.0, "amplitude_rad": 0.10},
                    {"t_s": 0.0, "amplitude_rad": 0.30},
                ]
            },
            "sim": {"duration_s": 1.0},
        }
    )

    assert [segment.t_s for segment in config.target_segments] == [0.0, 5.0]
    assert [segment.amplitude_rad for segment in config.target_segments] == [0.30, 0.10]
