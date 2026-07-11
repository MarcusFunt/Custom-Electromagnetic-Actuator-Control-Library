from pathlib import Path

import pytest

from emac_sim.config import (
    LinearSimulationConfig,
    SimulationConfig,
    load_config,
    parse_config,
)


EXAMPLE_PENDULUM_CONFIG = Path("examples/configs/pendulum_softiron_1gate.toml")
EXAMPLE_LINEAR_CONFIG = Path("examples/configs/linear_stepper_5coil.toml")


def test_sim_kind_defaults_to_pendulum_when_absent():
    config = parse_config({"sim": {"duration_s": 1.0}})
    assert isinstance(config, SimulationConfig)


def test_existing_pendulum_config_is_unaffected_by_the_dispatch():
    config = load_config(EXAMPLE_PENDULUM_CONFIG)
    assert isinstance(config, SimulationConfig)
    assert not isinstance(config, LinearSimulationConfig)


def test_unknown_sim_kind_raises():
    with pytest.raises(ValueError):
        parse_config({"sim": {"kind": "not_a_real_kind"}})


def test_linear_example_config_loads_five_coils_and_gates():
    config = load_config(EXAMPLE_LINEAR_CONFIG)

    assert isinstance(config, LinearSimulationConfig)
    assert len(config.coils) == 5
    assert len(config.gates) == 5
    assert config.controller.target_velocity_m_s == pytest.approx(0.5)


def test_linear_config_maps_to_actuator_params():
    config = load_config(EXAMPLE_LINEAR_CONFIG)
    p = config.to_actuator_params()

    assert len(p.coils) == 5
    assert len(p.gates) == 5
    assert p.coils[0].position_m == pytest.approx(config.coils[0].position_m)
    assert p.coils[0].Cmag == pytest.approx(config.coils[0].c_mag_n_per_a2)
    assert p.gates[0].w_eff == pytest.approx(config.gates[0].effective_width_m)
    assert p.end_of_travel == config.actuator.end_of_travel
    assert p.pressure_bias_n == pytest.approx(config.actuator.pressure_bias_n)
    assert p.coils[0].k_a == pytest.approx(config.coils[0].k_a_n_per_a)
    assert p.coils[0].resistance_ohm == pytest.approx(config.coils[0].resistance_ohm)
    assert p.coils[0].inductance_h == pytest.approx(config.coils[0].inductance_h)
    assert p.current_loop == config.driver.current_loop
    assert p.bus_voltage_v == pytest.approx(config.driver.bus_voltage_v)


def test_default_slug_is_pure_pm_no_iron():
    """c_mag_n_per_a2 defaults to 0.0 -- the current assumption is a slug with no iron,
    only a magnet (docs/DESIGN_LINEAR.md section 2.1)."""
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.coils[0].c_mag_n_per_a2 == pytest.approx(0.0)
    assert default.to_actuator_params().coils[0].Cmag == pytest.approx(0.0)


def test_current_loop_defaults_to_ideal_and_is_configurable():
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.to_actuator_params().current_loop == "ideal"

    rl = parse_config(
        {"sim": {"kind": "linear_stepper"}, "driver": {"current_loop": "rl", "bus_voltage_v": 24.0}}
    )
    p = rl.to_actuator_params()
    assert p.current_loop == "rl"
    assert p.bus_voltage_v == pytest.approx(24.0)


def test_k_a_n_per_a_defaults_are_consistent_between_dataclass_and_toml_fallback():
    """Regression guard: config.py has two separate defaults for each field -- the
    dataclass default (used when a whole [[coils]] entry is omitted) and the per-field
    TOML fallback (used when an entry is present but missing this one key). They drifted
    out of sync once already for x_c_m/initial_position_m; pin k_a explicitly."""
    from_missing_section = parse_config({"sim": {"kind": "linear_stepper"}})
    from_partial_entry = parse_config(
        {"sim": {"kind": "linear_stepper"}, "coils": [{"position_m": 0.0}]}
    )
    assert from_missing_section.coils[0].k_a_n_per_a == pytest.approx(0.20)
    assert from_partial_entry.coils[0].k_a_n_per_a == pytest.approx(0.20)


def test_pressure_bias_n_defaults_to_zero_and_is_configurable():
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.actuator.pressure_bias_n == pytest.approx(0.0)
    assert default.to_actuator_params().pressure_bias_n == pytest.approx(0.0)

    biased = parse_config(
        {"sim": {"kind": "linear_stepper"}, "actuator": {"pressure_bias_n": 0.5}}
    )
    assert biased.actuator.pressure_bias_n == pytest.approx(0.5)
    assert biased.to_actuator_params().pressure_bias_n == pytest.approx(0.5)


def test_thermal_model_defaults_off_and_is_configurable():
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.actuator.thermal_model is False
    assert default.actuator.ambient_temperature_c == pytest.approx(20.0)
    assert default.to_actuator_params().thermal_model is False
    assert default.to_actuator_params().ambient_temperature_c == pytest.approx(20.0)

    warm = parse_config(
        {
            "sim": {"kind": "linear_stepper"},
            "actuator": {"thermal_model": True, "ambient_temperature_c": 35.0},
            "coils": [{"thermal_mass_j_per_k": 9.0, "thermal_resistance_k_per_w": 6.0}],
        }
    )
    assert warm.actuator.thermal_model is True
    assert warm.actuator.ambient_temperature_c == pytest.approx(35.0)
    params = warm.to_actuator_params()
    assert params.thermal_model is True
    assert params.ambient_temperature_c == pytest.approx(35.0)
    assert params.coils[0].thermal_mass_j_per_k == pytest.approx(9.0)
    assert params.coils[0].thermal_resistance_k_per_w == pytest.approx(6.0)


def test_coil_geometry_fields_default_and_are_configurable():
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    coil = default.coils[0]
    assert coil.turns == 200
    assert coil.coil_winding_length_m == pytest.approx(0.020)
    assert coil.radial_thickness_m == pytest.approx(0.010)
    assert coil.bore_clearance_m == pytest.approx(0.0015)
    assert coil.packing_factor == pytest.approx(0.8)
    assert coil.winding_temperature_c == pytest.approx(20.0)
    assert coil.force_lut_path is None

    custom = parse_config({
        "sim": {"kind": "linear_stepper"},
        "coils": [{
            "position_m": 0.0, "turns": 150, "coil_winding_length_m": 0.015,
            "radial_thickness_m": 0.008, "bore_clearance_m": 0.001,
            "packing_factor": 0.7, "winding_temperature_c": 45.0,
        }],
    })
    coil = custom.coils[0]
    assert coil.turns == 150
    assert coil.coil_winding_length_m == pytest.approx(0.015)
    assert coil.radial_thickness_m == pytest.approx(0.008)
    assert coil.bore_clearance_m == pytest.approx(0.001)
    assert coil.packing_factor == pytest.approx(0.7)
    assert coil.winding_temperature_c == pytest.approx(45.0)


def test_force_lut_path_defaults_to_none_and_flows_into_actuator_params(tmp_path):
    import numpy as np

    from emac_sim.fem.lut import ForceLUT

    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.coils[0].force_lut_path is None
    assert default.to_actuator_params().coils[0].force_lut is None

    lut_path = tmp_path / "coil_00.npz"
    ForceLUT(
        offsets_m=np.array([-0.01, 0.0, 0.01]),
        currents_a=np.array([-1.0, 1.0]),
        force_n=np.zeros((3, 2)),
    ).save(lut_path)

    with_lut = parse_config({
        "sim": {"kind": "linear_stepper"},
        "coils": [{"position_m": 0.0, "force_lut_path": str(lut_path)}],
    })
    assert with_lut.coils[0].force_lut_path == str(lut_path)
    params = with_lut.to_actuator_params()
    assert params.coils[0].force_lut is not None
    assert params.coils[0].force_lut(0.0, 0.0) == pytest.approx(0.0)


def test_force_lut_path_must_be_a_string():
    with pytest.raises(ValueError):
        parse_config({
            "sim": {"kind": "linear_stepper"},
            "coils": [{"position_m": 0.0, "force_lut_path": 123}],
        })


def test_linear_config_values_affect_run_scenario():
    from emac_sim.linear_cli import run_scenario

    config = parse_config(
        {
            "sim": {"kind": "linear_stepper", "duration_s": 0.5, "dt_s": 0.0005, "sample_every": 5},
            "actuator": {"mass_kg": 0.15, "initial_position_m": -0.02},
            "controller": {"target_velocity_m_s": 0.4},
        }
    )

    assert isinstance(config, LinearSimulationConfig)
    p, sup, log = run_scenario(config=config)

    assert p.mass_kg == pytest.approx(0.15)
    assert log.t[-1] == pytest.approx(0.5, abs=0.01)
