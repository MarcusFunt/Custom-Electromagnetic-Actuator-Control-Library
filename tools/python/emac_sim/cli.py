"""Command-line entrypoint for the Phase 0 host simulator."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

from emac_sim import PendulumParams, Tier1Estimator, EnergySupervisor, Simulator
from emac_sim import plant
from emac_sim.config import LinearSimulationConfig, SimulationConfig, default_config, load_config
from emac_sim.config_summary import config_summary


def load_or_default_config(path: str | None) -> "SimulationConfig | LinearSimulationConfig":
    return load_config(path) if path else default_config()


def make_target(p: PendulumParams, config: SimulationConfig | None = None):
    config = config or default_config()
    segments = [
        (segment.t_s, plant.energy_for_amplitude(segment.amplitude_rad, p))
        for segment in config.target_segments
    ]
    if not segments:
        segments = [(0.0, plant.energy_for_amplitude(config.controller.target_amplitude_rad, p))]

    def target_E(t: float) -> float:
        current = segments[0][1]
        for ts, value in segments:
            if t >= ts:
                current = value
        return current
    return target_E


def run_scenario(t_end: float | None = None, config: SimulationConfig | None = None):
    config = config or default_config()
    p = config.to_pendulum_params()
    est = Tier1Estimator(p)
    sup = EnergySupervisor(
        p,
        k_E=config.controller.k_energy,
        T_p_frac=config.controller.pulse_width_half_period_fraction,
        i_max=config.primary_coil.max_current_a,
        eps_frac=config.controller.hold_deadband_fraction,
    )
    sim = Simulator(p, est, sup, dt=config.dt_s, sample_every=config.sample_every)
    target_E = make_target(p, config)
    log = sim.run(
        theta0=config.pendulum.initial_angle_rad,
        omega0=config.pendulum.initial_omega_rad_s,
        target_E=target_E,
        t_end=config.duration_s if t_end is None else t_end,
    )
    return p, log


def print_config_summary(config: SimulationConfig, source: str | None) -> None:
    summary = config_summary(config, source)
    print(f"simulation config: {summary['source']}")
    print(
        "  pendulum: "
        f"L={summary['pendulum']['length_m']:g} m, "
        f"m={summary['pendulum']['bob_mass_kg']:g} kg, "
        f"Q={summary['pendulum']['quality_factor']:g}"
    )
    print(
        "  gate[0]:   "
        f"angle={summary['gate0']['angle_rad']:g} rad, "
        f"width={summary['gate0']['angular_width_rad']:g} rad"
    )
    print(
        "  coil[0]:   "
        f"theta_c={summary['coil0']['theta_c_rad']:g} rad, "
        f"Cmag={summary['coil0']['c_mag_nm_per_a2']:g} N*m/A^2, "
        f"Imax={summary['coil0']['max_current_a']:g} A"
    )
    print(
        "  controller: "
        f"target={summary['controller']['target_amplitude_rad']:g} rad, "
        f"k_E={summary['controller']['k_energy']:g}, "
        f"T_p_frac={summary['controller']['pulse_width_half_period_fraction']:g}"
    )


def print_convergence_table(log) -> None:
    print(
        f"{'t_cross':>8} {'A_peak':>8} {'A_cross':>8} {'A_est':>8} "
        f"{'A_tgt':>8} {'errA%':>7} {'kind':>6} {'i_peak':>7}"
    )
    for k in range(len(log.cx_t)):
        a_peak = log.cx_A_peak[k]
        a_cross = log.cx_A_energy[k]
        a_tgt = log.cx_A_tgt[k]
        err = 100.0 * (a_cross - a_tgt) / a_tgt if a_tgt > 0 else 0.0
        if k % 2 == 0:
            print(
                f"{log.cx_t[k]:8.3f} {a_peak:8.4f} {a_cross:8.4f} "
                f"{log.cx_A_est[k]:8.4f} {a_tgt:8.4f} {err:7.2f} "
                f"{log.cx_kind[k]:>6} {log.cx_ipeak[k]:7.3f}"
            )


def steady_state_rms_error(log, t_lo: float, t_hi: float) -> float:
    errs = []
    for k in range(len(log.cx_t)):
        if t_lo <= log.cx_t[k] <= t_hi and log.cx_A_tgt[k] > 0:
            errs.append((log.cx_A_energy[k] - log.cx_A_tgt[k]) / log.cx_A_tgt[k])
    if not errs:
        return math.nan
    return math.sqrt(sum(e * e for e in errs) / len(errs)) * 100.0


def report_steady_state(log) -> None:
    print("\nsteady-state amplitude RMS error (last ~1 s of each hold):")
    print(f"  hold @0.35 rad : {steady_state_rms_error(log, 6.5, 8.0):5.2f} %")
    print(f"  hold @0.20 rad : {steady_state_rms_error(log, 13.5, 15.0):5.2f} %")
    print(f"  hold @0.30 rad : {steady_state_rms_error(log, 20.5, 22.0):5.2f} %")


def write_plots(log, outdir: str) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.step(log.cx_t, log.cx_A_tgt, where="post", color="0.6", lw=2, label="target amplitude")
    ax1.plot(log.cx_t, log.cx_A_energy, "o-", ms=3, color="#1f77b4", label="true crossing energy")
    ax1.plot(log.cx_t, log.cx_A_peak, ".", ms=4, color="#9467bd", label="previous physical peak")
    ax1.plot(log.cx_t, log.cx_A_est, "x", ms=4, color="#d62728", label="estimated from pulse width")
    for k in range(len(log.cx_t)):
        if log.cx_kind[k] == "brake":
            ax1.axvspan(log.cx_t[k] - 0.02, log.cx_t[k] + 0.02, color="#ff7f0e", alpha=0.08)
    ax1.set_ylabel("amplitude (rad)")
    ax1.set_title("Phase 0: energy-shaping swing-up / hold / brake (soft-iron bob)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(log.t, log.i, color="#2ca02c", lw=0.7)
    ax2.set_ylabel("coil current (A)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    amplitude_path = os.path.join(outdir, "phase0_amplitude.png")
    fig.savefig(amplitude_path, dpi=110)
    plt.close(fig)

    t0, t1 = 5.0, 8.0
    ts = [x for x in log.t if t0 <= x <= t1]
    th_true = [log.theta[i] for i, x in enumerate(log.t) if t0 <= x <= t1]
    th_est = [log.theta_est[i] for i, x in enumerate(log.t) if t0 <= x <= t1]
    cx = [x for x in log.cx_t if t0 <= x <= t1]

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, th_true, color="#1f77b4", lw=1.6, label="true theta(t)")
    ax.plot(ts, th_est, "--", color="#d62728", lw=1.4, label="estimator dead-reckoning")
    for c in cx:
        ax.axvline(c, color="0.7", lw=0.8, ls=":")
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("theta (rad)")
    ax.set_title("Reconstruction from sparse bottom-gate events (dotted = crossing times)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    reconstruction_path = os.path.join(outdir, "phase0_reconstruction.png")
    fig2.savefig(reconstruction_path, dpi=110)
    plt.close(fig2)

    return amplitude_path, reconstruction_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EMAC Phase 0 host simulator.")
    parser.add_argument("--config", help="TOML file describing fictional hardware and sim settings.")
    parser.add_argument("--outdir", default=os.path.join("tools", "python", "out"))
    parser.add_argument("--t-end", type=float, default=None, help="Override the config simulation duration.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_or_default_config(args.config)

    if isinstance(config, LinearSimulationConfig):
        from . import linear_cli
        return linear_cli.run(args, config)

    print_config_summary(config, args.config)
    _, log = run_scenario(t_end=args.t_end, config=config)
    print_convergence_table(log)
    report_steady_state(log)

    if not args.no_plots:
        amplitude_path, reconstruction_path = write_plots(log, args.outdir)
        print(f"\nwrote {amplitude_path}")
        print(f"wrote {reconstruction_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
