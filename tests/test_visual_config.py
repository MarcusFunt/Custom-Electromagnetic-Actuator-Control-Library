from emac_sim.cli import run_scenario
from emac_sim.config import load_config
from emac_sim.config_summary import config_summary
from emac_sim.visual import build_visual_payload


def test_visual_payload_can_include_config_summary():
    config = load_config("examples/configs/pendulum_softiron_1gate.toml")
    p, log = run_scenario(t_end=1.0, config=config)
    payload = build_visual_payload(p, log)
    payload["config"] = config_summary(config, "examples/configs/pendulum_softiron_1gate.toml")

    assert payload["config"]["source"] == "examples/configs/pendulum_softiron_1gate.toml"
    assert payload["config"]["pendulum"]["length_m"] == config.pendulum.length_m
    assert payload["config"]["coil0"]["max_current_a"] == config.primary_coil.max_current_a
