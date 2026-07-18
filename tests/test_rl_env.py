"""Tests for the RL coilgun control environment (emac_sim.rl).

All CI-safe: the env uses the analytic plant / reference force law -- no FEMM, no long training.
gymnasium is required; SB3/torch only for the short optional PPO smoke.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from emac_sim.rl.env import CoilgunEnv          # noqa: E402
from emac_sim.rl.geometry import CoilgunSpec, build_params, build_reference_lut  # noqa: E402

SPEC = CoilgunSpec(n_coils=8)                    # smaller gun -> fast tests
PUSH_PULL = np.array([-1.0, 0.0, 1.0], dtype=np.float32)   # repel behind + attract ahead


# ----------------------------- geometry ---------------------------------------
def test_build_params_shape():
    p = build_params(SPEC)
    assert len(p.coils) == SPEC.n_coils and len(p.gates) == SPEC.n_coils
    assert p.driver_bipolar is True and p.current_loop == "rl"
    assert p.bus_voltage_v == SPEC.bus_voltage_v
    assert p.mass_kg > 0.0


# ----------------------------- gym API ----------------------------------------
def test_env_passes_gym_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(CoilgunEnv(SPEC), skip_render_check=True)


def test_reset_seed_determinism():
    e1, e2 = CoilgunEnv(SPEC), CoilgunEnv(SPEC)
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.allclose(o1, o2)
    assert e1.v0 == e2.v0
    o3, _ = e1.reset(seed=8)
    assert e1.v0 != e2.v0 or not np.allclose(o1, o3)   # different seed -> different start


def test_action_and_obs_spaces():
    e = CoilgunEnv(SPEC, window=3)
    assert e.action_space.shape == (3,)
    assert e.observation_space.shape == (6 + 2 * 3,)


# ----------------------------- dynamics / physics -----------------------------
def _rollout(env, action_fn, seed=0):
    obs, _ = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(action_fn(obs))
        if term or trunc:
            return info


def test_push_pull_launches_slug():
    info = _rollout(CoilgunEnv(SPEC), lambda o: PUSH_PULL)
    assert info["reason"] == "exit"          # bipolar push+pull clears all stages
    assert info["v"] > 5.0                    # and does real work


def test_always_pull_ahead_worse_than_push_pull():
    # a naive always-on ahead coil suffers pull-back -> far slower than gated push+pull
    pull = _rollout(CoilgunEnv(SPEC), lambda o: np.array([0, 0, 1.0], dtype=np.float32))
    fast = _rollout(CoilgunEnv(SPEC), lambda o: PUSH_PULL)
    assert pull["v"] < fast["v"]


def test_energy_accounting_positive_and_efficiency_bounded():
    info = _rollout(CoilgunEnv(SPEC), lambda o: PUSH_PULL)
    assert info["e_in"] > 0.0
    ke = 0.5 * build_params(SPEC).mass_kg * info["v"] ** 2
    assert info["efficiency"] == pytest.approx(ke / info["e_in"], rel=1e-6)
    assert 0.0 < info["efficiency"] < 1.0     # can't beat 100% (energy conservation)


def test_lambda_penalizes_energy_in_reward():
    # same rollout, higher lambda -> lower (more energy-penalized) return
    def total(lam):
        env = CoilgunEnv(SPEC, lam=lam)
        obs, _ = env.reset(seed=0); tot = 0.0
        while True:
            obs, r, term, trunc, info = env.step(PUSH_PULL); tot += r
            if term or trunc:
                return tot
    assert total(0.1) < total(0.0)


def test_reference_lut_env_runs():
    lut = build_reference_lut(SPEC)
    info = _rollout(CoilgunEnv(SPEC, force_lut=lut), lambda o: PUSH_PULL)
    assert info["reason"] == "exit" and info["v"] > 0.0


# ----------------------------- baselines --------------------------------------
def test_hand_tuned_supervisor_baseline():
    from emac_sim.rl import baselines as B
    r = B.run_supervisor(SPEC)
    assert r["v"] > 5.0 and 0.0 < r["efficiency"] < 1.0


# ----------------------------- optional PPO smoke -----------------------------
def test_ppo_learns_smoke():
    pytest.importorskip("stable_baselines3")
    from pathlib import Path
    from emac_sim.rl.train import train_one
    # tiny run: just assert it trains + evaluates without error and produces finite metrics
    _, m = train_one(SPEC, lam=0.0, timesteps=3000, outdir=Path("build/rl_test"),
                     n_envs=4, seed=0, verbose=0)
    assert np.isfinite(m["v"]) and 0.0 <= m["efficiency"] <= 1.0
