from emac_sim.config import default_config
from emac_sim.config_summary import config_summary


def test_config_summary_is_json_serializable_shape():
    summary = config_summary(default_config(), source="demo.toml")

    assert summary["source"] == "demo.toml"
    assert summary["pendulum"]["length_m"] > 0
    assert summary["gate0"]["kind"] == "photogate"
    assert summary["coil0"]["max_current_a"] > 0
    assert summary["controller"]["kind"] == "energy_supervisor"
    assert summary["targets"]
