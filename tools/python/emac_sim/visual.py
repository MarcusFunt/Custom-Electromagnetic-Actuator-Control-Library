"""Standalone HTML visual simulator for the Phase 0 pendulum scenario."""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from emac_sim import PendulumParams
from emac_sim.cli import run_scenario, steady_state_rms_error
from emac_sim.config import SimulationConfig, default_config, load_config
from emac_sim.config_summary import config_summary


def build_visual_payload(
    p: PendulumParams,
    log,
    config: SimulationConfig | None = None,
    config_source: str | None = None,
) -> dict:
    """Convert a simulation log into compact JSON-serializable browser data."""
    max_theta = max([abs(x) for x in log.theta] + [abs(x) for x in log.theta_est] + [0.1])
    max_current = max([abs(x) for x in log.i] + [1.0])

    payload = {
        "params": {
            "L": p.L,
            "theta_c": p.theta_c,
            "dalpha": p.dalpha,
        },
        "summary": {
            "t_end": log.t[-1] if log.t else 0.0,
            "max_theta": max_theta,
            "max_current": max_current,
            "rms_035": steady_state_rms_error(log, 6.5, 8.0),
            "rms_020": steady_state_rms_error(log, 13.5, 15.0),
            "rms_030": steady_state_rms_error(log, 20.5, 22.0),
        },
        "series": {
            "t": _round_series(log.t, 4),
            "theta": _round_series(log.theta, 6),
            "theta_est": _round_series(log.theta_est, 6),
            "omega": _round_series(log.omega, 6),
            "current": _round_series(log.i, 5),
            "energy": _round_series(log.E_true, 8),
        },
        "crossings": {
            "t": _round_series(log.cx_t, 4),
            "a_peak": _round_series(log.cx_A_peak, 5),
            "a_energy": _round_series(log.cx_A_energy, 5),
            "a_est": _round_series(log.cx_A_est, 5),
            "a_target": _round_series(log.cx_A_tgt, 5),
            "kind": list(log.cx_kind),
            "i_peak": _round_series(log.cx_ipeak, 5),
        },
    }
    if config is not None:
        payload["config"] = config_summary(config, config_source)
    return payload


def write_visual_simulator(
    p: PendulumParams,
    log,
    outdir: str | os.PathLike[str],
    config: SimulationConfig | None = None,
    config_source: str | None = None,
) -> Path:
    """Write a standalone visual simulator HTML file and return its path."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_path = out_path / "phase0_visual.html"
    payload = json.dumps(build_visual_payload(p, log, config, config_source), separators=(",", ":"))
    html_path.write_text(_HTML_TEMPLATE.replace("__EMAC_DATA__", payload), encoding="utf-8")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the EMAC Phase 0 visual simulator.")
    parser.add_argument("--config", help="TOML file describing fictional hardware and sim settings.")
    parser.add_argument("--outdir", default=os.path.join("build", "visual"))
    parser.add_argument("--t-end", type=float, default=None, help="Override the config simulation duration.")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else default_config()
    p, log = run_scenario(t_end=args.t_end, config=config)
    html_path = write_visual_simulator(p, log, args.outdir, config=config, config_source=args.config)
    print(f"wrote {html_path}")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


def _round_series(values, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EMAC Phase 0 Visual Simulator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f5;
      --panel: #ffffff;
      --panel-2: #eef3f1;
      --ink: #18211f;
      --muted: #66736e;
      --line: #d8e0dd;
      --blue: #1f6feb;
      --red: #cf3f4b;
      --green: #1f8a5b;
      --amber: #c2761b;
      --coil: #6f56c9;
      --shadow: 0 12px 34px rgba(25, 38, 34, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 18px 24px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(10px);
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 680;
    }

    .run-meta {
      display: flex;
      align-items: center;
      gap: 14px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 3px rgba(31, 138, 91, 0.14);
    }

    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 330px;
      grid-template-rows: minmax(360px, 1fr) 250px;
      gap: 16px;
      padding: 16px;
    }

    .stage-wrap,
    .side,
    .timeline {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .stage-wrap {
      min-height: 360px;
      position: relative;
      overflow: hidden;
    }

    #stageCanvas,
    #traceCanvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    .side {
      min-height: 0;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .controls {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 12px;
      align-items: center;
    }

    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      height: 36px;
      font: inherit;
      cursor: pointer;
    }

    .icon-button {
      width: 44px;
      height: 44px;
      font-size: 17px;
      display: grid;
      place-items: center;
    }

    input[type="range"] {
      width: 100%;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .metric {
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }

    .metric .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .metric .value {
      margin-top: 4px;
      font-size: 18px;
      font-weight: 680;
    }

    .legend {
      display: flex;
      flex-direction: column;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .swatch {
      width: 20px;
      height: 3px;
      border-radius: 999px;
      background: var(--blue);
    }

    .timeline {
      grid-column: 1 / -1;
      position: relative;
      overflow: hidden;
    }

    .config-block {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .config-block b {
      color: var(--ink);
      font-weight: 650;
    }

    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
        grid-template-rows: 420px auto 240px;
      }

      .timeline { grid-column: 1; }
    }
  </style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1>EMAC Phase 0 Visual Simulator</h1>
      <div class="run-meta"><span class="status-dot"></span><span id="runMeta">ready</span></div>
    </div>
  </header>
  <main>
    <section class="stage-wrap"><canvas id="stageCanvas"></canvas></section>
    <aside class="side">
      <div class="controls">
        <button id="playPause" class="icon-button">▶</button>
        <input id="scrubber" type="range" min="0" max="1000" value="0">
      </div>
      <div class="metric-grid">
        <div class="metric"><div class="label">time</div><div class="value" id="mTime">0.00 s</div></div>
        <div class="metric"><div class="label">current</div><div class="value" id="mCurrent">0.00 A</div></div>
        <div class="metric"><div class="label">theta</div><div class="value" id="mTheta">0.000 rad</div></div>
        <div class="metric"><div class="label">target</div><div class="value" id="mTarget">0.000 rad</div></div>
      </div>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background: var(--blue)"></span>true angle</div>
        <div class="legend-row"><span class="swatch" style="background: var(--red)"></span>estimated angle</div>
        <div class="legend-row"><span class="swatch" style="background: var(--green)"></span>coil current</div>
        <div class="legend-row"><span class="swatch" style="background: var(--amber)"></span>crossing / pulse plan</div>
      </div>
      <div class="config-block" id="configBlock"></div>
    </aside>
    <section class="timeline"><canvas id="traceCanvas"></canvas></section>
  </main>
</div>
<script>
const DATA = __EMAC_DATA__;
const stageCanvas = document.getElementById('stageCanvas');
const traceCanvas = document.getElementById('traceCanvas');
const playPause = document.getElementById('playPause');
const scrubber = document.getElementById('scrubber');
const runMeta = document.getElementById('runMeta');
const mTime = document.getElementById('mTime');
const mCurrent = document.getElementById('mCurrent');
const mTheta = document.getElementById('mTheta');
const mTarget = document.getElementById('mTarget');
const configBlock = document.getElementById('configBlock');
let playing = false;
let idx = 0;
let lastFrame = performance.now();
const speed = 1.0;

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: rect.width, h: rect.height};
}

function currentTargetAmplitude(t) {
  const cx = DATA.crossings;
  if (!cx.t.length) return 0;
  let best = cx.a_target[0];
  for (let i = 0; i < cx.t.length; i++) {
    if (cx.t[i] <= t) best = cx.a_target[i];
  }
  return best;
}

function drawStage() {
  const {ctx, w, h} = resizeCanvas(stageCanvas);
  ctx.clearRect(0, 0, w, h);
  const theta = DATA.series.theta[idx] || 0;
  const thetaEst = DATA.series.theta_est[idx] || 0;
  const current = DATA.series.current[idx] || 0;
  const cx = w * 0.5;
  const cy = Math.max(70, h * 0.16);
  const len = Math.min(h * 0.62, w * 0.38);
  const bx = cx + Math.sin(theta) * len;
  const by = cy + Math.cos(theta) * len;
  const ex = cx + Math.sin(thetaEst) * len;
  const ey = cy + Math.cos(thetaEst) * len;
  const coilX = cx;
  const coilY = cy + len + 18;

  ctx.strokeStyle = '#d8e0dd';
  ctx.lineWidth = 1;
  for (let a = -0.6; a <= 0.61; a += 0.2) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.sin(a) * len, cy + Math.cos(a) * len);
    ctx.stroke();
  }

  ctx.strokeStyle = '#cf3f4b';
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#1f6feb';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(bx, by);
  ctx.stroke();

  ctx.fillStyle = '#18211f';
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#26322e';
  ctx.beginPath();
  ctx.arc(bx, by, 18, 0, Math.PI * 2);
  ctx.fill();

  const glow = Math.min(1, Math.abs(current) / Math.max(1, DATA.summary.max_current));
  ctx.fillStyle = `rgba(111, 86, 201, ${0.15 + 0.45 * glow})`;
  ctx.beginPath();
  ctx.ellipse(coilX, coilY, 48 + 18 * glow, 16 + 6 * glow, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#6f56c9';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = '#66736e';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText('photogate / coil at bottom', coilX - 70, coilY + 36);
}

function drawTrace() {
  const {ctx, w, h} = resizeCanvas(traceCanvas);
  ctx.clearRect(0, 0, w, h);
  const t = DATA.series.t;
  if (!t.length) return;
  const tMax = t[t.length - 1] || 1;
  const pad = {l: 42, r: 18, t: 18, b: 28};
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const maxTheta = Math.max(DATA.summary.max_theta, 0.1);
  const maxCurrent = Math.max(DATA.summary.max_current, 1.0);

  function xOf(time) { return pad.l + (time / tMax) * plotW; }
  function yTheta(v) { return pad.t + plotH * (0.5 - 0.45 * v / maxTheta); }
  function yCurrent(v) { return pad.t + plotH * (0.95 - 0.35 * v / maxCurrent); }

  ctx.strokeStyle = '#d8e0dd';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, yTheta(0));
  ctx.lineTo(w - pad.r, yTheta(0));
  ctx.stroke();

  for (let i = 0; i < DATA.crossings.t.length; i++) {
    const x = xOf(DATA.crossings.t[i]);
    ctx.strokeStyle = DATA.crossings.kind[i] === 'brake' ? 'rgba(194,118,27,0.28)' : 'rgba(31,111,235,0.18)';
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, h - pad.b);
    ctx.stroke();
  }

  function line(series, yFn, color, dash) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const x = xOf(t[i]);
      const y = yFn(series[i] || 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  line(DATA.series.theta, yTheta, '#1f6feb');
  line(DATA.series.theta_est, yTheta, '#cf3f4b', [5, 4]);
  line(DATA.series.current, yCurrent, '#1f8a5b');

  const xNow = xOf(t[idx] || 0);
  ctx.strokeStyle = '#18211f';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xNow, pad.t);
  ctx.lineTo(xNow, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = '#66736e';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText('time (s)', w - 70, h - 8);
  ctx.fillText('θ / current traces', 12, 16);
}

function renderConfig() {
  const c = DATA.config;
  if (!c) {
    configBlock.innerHTML = '<b>Config</b><br>built-in Phase 0 defaults';
    return;
  }
  configBlock.innerHTML = `
    <b>Config</b><br>${c.source}<br>
    L=${c.pendulum.length_m} m, m=${c.pendulum.bob_mass_kg} kg, Q=${c.pendulum.quality_factor}<br>
    gate width=${c.gate0.angular_width_rad} rad<br>
    coil Cmag=${c.coil0.c_mag_nm_per_a2} N·m/A², Imax=${c.coil0.max_current_a} A<br>
    controller target=${c.controller.target_amplitude_rad} rad, kE=${c.controller.k_energy}
  `;
}

function render() {
  const t = DATA.series.t[idx] || 0;
  mTime.textContent = `${t.toFixed(2)} s`;
  mCurrent.textContent = `${(DATA.series.current[idx] || 0).toFixed(2)} A`;
  mTheta.textContent = `${(DATA.series.theta[idx] || 0).toFixed(3)} rad`;
  mTarget.textContent = `${currentTargetAmplitude(t).toFixed(3)} rad`;
  runMeta.textContent = `${DATA.series.t.length} samples · ${DATA.crossings.t.length} crossings`;
  scrubber.value = String(Math.round((idx / Math.max(1, DATA.series.t.length - 1)) * 1000));
  drawStage();
  drawTrace();
}

playPause.addEventListener('click', () => {
  playing = !playing;
  playPause.textContent = playing ? '⏸' : '▶';
  lastFrame = performance.now();
});

scrubber.addEventListener('input', () => {
  const f = Number(scrubber.value) / 1000;
  idx = Math.round(f * Math.max(0, DATA.series.t.length - 1));
  render();
});

window.addEventListener('resize', render);

function tick(now) {
  if (playing && DATA.series.t.length > 1) {
    const dt = (now - lastFrame) / 1000 * speed;
    const tNow = DATA.series.t[idx] || 0;
    const target = tNow + dt;
    while (idx < DATA.series.t.length - 1 && DATA.series.t[idx] < target) idx++;
    if (idx >= DATA.series.t.length - 1) {
      idx = 0;
    }
    render();
  }
  lastFrame = now;
  requestAnimationFrame(tick);
}

renderConfig();
render();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
