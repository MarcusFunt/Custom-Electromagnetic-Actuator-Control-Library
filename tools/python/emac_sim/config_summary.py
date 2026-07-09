"""Small helpers for displaying simulation config values."""

from __future__ import annotations

from .config import SimulationConfig


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
            "resistance_ohm": config.primary_coil.resistance_ohm,
            "inductance_h": config.primary_coil.inductance_h,
        },
        "driver": {
            "bus_voltage_v": config.driver.bus_voltage_v,
            "pwm_frequency_hz": config.driver.pwm_frequency_hz,
            "current_loop": config.driver.current_loop,
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
        },
        "targets": [
            {"t_s": segment.t_s, "amplitude_rad": segment.amplitude_rad}
            for segment in config.target_segments
        ],
    }
