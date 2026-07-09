from emac_sim.cli import run_scenario
from emac_sim.visual import build_visual_payload, main, write_visual_simulator


def test_visual_payload_contains_required_series():
    p, log = run_scenario(t_end=1.0)

    payload = build_visual_payload(p, log)

    assert payload["params"]["L"] == p.L
    assert payload["summary"]["t_end"] > 0
    assert len(payload["series"]["t"]) == len(log.t)
    assert len(payload["series"]["theta"]) == len(log.theta)
    assert len(payload["series"]["theta_est"]) == len(log.theta_est)
    assert len(payload["series"]["current"]) == len(log.i)
    assert len(payload["crossings"]["t"]) == len(log.cx_t)


def test_write_visual_simulator_creates_standalone_html(tmp_path):
    p, log = run_scenario(t_end=1.0)

    html_path = write_visual_simulator(p, log, tmp_path)
    text = html_path.read_text(encoding="utf-8")

    assert html_path.name == "phase0_visual.html"
    assert "<canvas id=\"stageCanvas\"" in text
    assert "<canvas id=\"traceCanvas\"" in text
    assert "const DATA =" in text
    assert "__EMAC_DATA__" not in text
    assert "<script src=" not in text


def test_visual_cli_writes_html(tmp_path):
    assert main(["--outdir", str(tmp_path), "--t-end", "1"]) == 0
    assert (tmp_path / "phase0_visual.html").exists()
