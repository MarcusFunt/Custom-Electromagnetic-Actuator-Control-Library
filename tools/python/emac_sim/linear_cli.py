"""Command-line support for the linear one-way stepper scenario (docs/DESIGN_LINEAR.md).

Mirrors cli.py's structure (run_scenario / summary / table / plots). Invoked by
cli.main() when the loaded config is a LinearSimulationConfig (config.parse_config's
`[sim] kind` dispatch) -- this is the "one shared entry point" for both geometries:
the same `emac-sim --config foo.toml` command reaches this module transparently.
"""

from __future__ import annotations

import os

from .config import LinearSimulationConfig, default_linear_config
from .config_summary import linear_config_summary
from .linear_estimator import LinearStepperEstimator
from .linear_sim import LinearSimulator
from .linear_supervisor import FAULT, StepperSupervisor


def run_scenario(t_end: float | None = None, config: LinearSimulationConfig | None = None):
    config = config or default_linear_config()
    p = config.to_actuator_params()
    est = LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )
    sup = StepperSupervisor(
        p,
        k_v=config.controller.k_velocity,
        T_p_frac=config.controller.pulse_width_half_period_fraction,
        phase_advance_s=config.controller.phase_advance_s,
        i_max=config.controller.i_max_a,
        bootstrap_dwell_s=config.controller.bootstrap_dwell_s,
        bootstrap_timeout_s=config.controller.bootstrap_timeout_s,
        pm_envelope=config.controller.pump_envelope,
    )
    sim = LinearSimulator(p, est, sup, dt=config.dt_s, sample_every=config.sample_every)
    log = sim.run(
        x0=config.actuator.initial_position_m,
        v0=config.actuator.initial_velocity_m_s,
        v_tgt=config.controller.target_velocity_m_s,
        t_end=config.duration_s if t_end is None else t_end,
    )
    return p, sup, log


def print_config_summary(config: LinearSimulationConfig, source: str | None) -> None:
    summary = linear_config_summary(config, source)
    print(f"linear stepper config: {summary['source']}")
    print(
        "  actuator: "
        f"mass={summary['actuator']['mass_kg']:g} kg, "
        f"damping={summary['actuator']['damping_n_per_mps']:g} N*s/m, "
        f"pressure_bias={summary['actuator']['pressure_bias_n']:g} N, "
        f"end_of_travel={summary['actuator']['end_of_travel']}"
    )
    print(f"  stations: {len(summary['coils'])} coils, {len(summary['gates'])} gates")
    print(
        "  controller: "
        f"target_v={summary['controller']['target_velocity_m_s']:g} m/s, "
        f"k_v={summary['controller']['k_velocity']:g}"
    )


def print_gate_table(log) -> None:
    print(f"{'gate_t':>8} {'gate_idx':>8} {'v_at_gate':>10}")
    for k in range(len(log.gate_t)):
        print(f"{log.gate_t[k]:8.4f} {log.gate_index[k]:8d} {log.gate_v[k]:10.4f}")


def report_outcome(sup: StepperSupervisor, log) -> None:
    print(f"\nsupervisor final mode: {sup.mode}")
    if log.x:
        print(f"final position: {log.x[-1]:.4f} m, final velocity: {log.v[-1]:.4f} m/s")
    if sup.mode == FAULT:
        print("FAULT: bootstrap exhausted every station with no gate response "
              "(no slug in the tube, or it's jammed).")


def write_plots(log, outdir: str) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(log.t, log.x, color="#1f77b4", lw=1.6, label="true position")
    ax1.plot(log.t, log.x_est, "--", color="#d62728", lw=1.2, label="estimated position")
    for gt in log.gate_t:
        ax1.axvline(gt, color="0.7", lw=0.8, ls=":")
    ax1.set_ylabel("position (m)")
    ax1.set_title("Linear stepper: forward commutation through N coil stations")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(log.t, log.active_current, color="#2ca02c", lw=0.8)
    ax2.set_ylabel("active coil current (A)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    position_path = os.path.join(outdir, "linear_position.png")
    fig.savefig(position_path, dpi=110)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(log.t, log.v, color="#9467bd", lw=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_title("Slug velocity vs. time")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    velocity_path = os.path.join(outdir, "linear_velocity.png")
    fig2.savefig(velocity_path, dpi=110)
    plt.close(fig2)

    return position_path, velocity_path


def run(args, config: LinearSimulationConfig) -> int:
    """Entry point delegated to by emac_sim.cli.main() for a linear-stepper config."""
    print_config_summary(config, args.config)
    _, sup, log = run_scenario(t_end=args.t_end, config=config)
    print_gate_table(log)
    report_outcome(sup, log)

    if not args.no_plots:
        position_path, velocity_path = write_plots(log, args.outdir)
        print(f"\nwrote {position_path}")
        print(f"wrote {velocity_path}")

    return 0
