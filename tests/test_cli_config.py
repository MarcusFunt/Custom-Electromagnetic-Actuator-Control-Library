from emac_sim.cli import main


def test_cli_accepts_config_for_smoke_run(capsys):
    result = main([
        "--config",
        "examples/configs/pendulum_softiron_1gate.toml",
        "--t-end",
        "1",
        "--no-plots",
    ])

    captured = capsys.readouterr()
    assert result == 0
    assert "simulation config:" in captured.out
    assert "examples/configs/pendulum_softiron_1gate.toml" in captured.out
