"""Small helpers for displaying simulation config values."""

from __future__ import annotations

from .config import LinearSimulationConfig, SimulationConfig


def config_summary(config: SimulationConfig, source: str | None = None) -> dict:
    """Return JSON-serializable headline values for visual reports."""
    return {
        "source": source or "built-in Phase 0 default",
        "pendulum": {
            "length_m": config.pendulum.length_m,
            "bob_mass_kg": config.pendulum.bob_mass_kg,
            "quality_factor": config.pendulum.quality_factor,
            "initial_angle_rad": config.pendulum.initial_angle_rad,
        },
        "gate0": {
            "kind": config.primary_gate.kind,
            "angle_rad": config.primary_gate.angle_rad,
            "angular_width_rad": config.primary_gate.angular_width_rad,
            "noise_std_s": config.primary_gate.noise_std_s,
            "dropout_probability": config.primary_gate.dropout_probability,
        },
        "coil0": {
            "angle_rad": config.primary_coil.angle_rad,
            "theta_c_rad": config.primary_coil.theta_c_rad,
            "c_mag_nm_per_a2": config.primary_coil.c_mag_nm_per_a2,
            "i_sat_a": config.primary_coil.i_sat_a,
            "max_current_a": config.primary_coil.max_current_a,
        },
        "controller": {
            "kind": config.controller.kind,
            "target_amplitude_rad": config.controller.target_amplitude_rad,
            "k_energy": config.controller.k_energy,
            "pulse_width_half_period_fraction": config.controller.pulse_width_half_period_fraction,
            "hold_deadband_fraction": config.controller.hold_deadband_fraction,
        },
        "sim": {
            "duration_s": config.duration_s,
            "dt_s": config.dt_s,
            "sample_every": config.sample_every,
            "random_seed": config.random_seed,
        },
        "targets": [
            {"t_s": segment.t_s, "amplitude_rad": segment.amplitude_rad}
            for segment in config.target_segments
        ],
    }


def linear_config_summary(config: LinearSimulationConfig, source: str | None = None) -> dict:
    """Return JSON-serializable headline values for the linear stepper's visual reports.
    Sibling of config_summary() -- kept separate rather than branching one function on
    config type, since the two configs share no fields in common."""
    return {
        "source": source or "built-in linear stepper default",
        "actuator": {
            "mass_kg": config.actuator.mass_kg,
            "damping_n_per_mps": config.actuator.damping_n_per_mps,
            "initial_position_m": config.actuator.initial_position_m,
            "end_of_travel": config.actuator.end_of_travel,
            "pressure_bias_n": config.actuator.pressure_bias_n,
            "thermal_model": config.actuator.thermal_model,
            "ambient_temperature_c": config.actuator.ambient_temperature_c,
            "exit_position_m": config.actuator.exit_position_m,
        },
        "gates": [
            {"position_m": g.position_m, "effective_width_m": g.effective_width_m,
             "noise_std_s": g.noise_std_s,
             "dropout_probability": g.dropout_probability}
            for g in config.gates
        ],
        "coils": [
            {
                "position_m": c.position_m,
                "x_c_m": c.x_c_m,
                "c_mag_n_per_a2": c.c_mag_n_per_a2,
                "i_sat_a": c.i_sat_a,
                "k_a_n_per_a": c.k_a_n_per_a,
                "resistance_ohm": c.resistance_ohm,
                "inductance_h": c.inductance_h,
                "thermal_mass_j_per_k": c.thermal_mass_j_per_k,
                "thermal_resistance_k_per_w": c.thermal_resistance_k_per_w,
            }
            for c in config.coils
        ],
        "driver": {
            "bus_voltage_v": config.driver.bus_voltage_v,
            "current_loop": config.driver.current_loop,
            "bipolar": config.driver.bipolar,
        },
        "controller": {
            "kind": config.controller.kind,
            "target_velocity_m_s": config.controller.target_velocity_m_s,
            "k_velocity": config.controller.k_velocity,
            "pulse_width_half_period_fraction": config.controller.pulse_width_half_period_fraction,
            "phase_advance_s": config.controller.phase_advance_s,
            "full_thrust": config.controller.full_thrust,
        },
        "sim": {
            "duration_s": config.duration_s,
            "dt_s": config.dt_s,
            "sample_every": config.sample_every,
            "random_seed": config.random_seed,
        },
    }
