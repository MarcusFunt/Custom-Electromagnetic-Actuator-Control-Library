from emac_sim.visual import main


def test_visual_cli_accepts_config(tmp_path):
    assert main([
        "--config",
        "examples/configs/pendulum_softiron_1gate.toml",
        "--outdir",
        str(tmp_path),
        "--t-end",
        "1",
    ]) == 0

    assert (tmp_path / "phase0_visual.html").exists()
