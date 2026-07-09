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


def build_visual_payload(p: PendulumParams, log) -> dict:
    """Convert a simulation log into compact JSON-serializable browser data."""
    max_theta = max([abs(x) for x in log.theta] + [abs(x) for x in log.theta_est] + [0.1])
    max_current = max([abs(x) for x in log.i] + [1.0])

    return {
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


def write_visual_simulator(p: PendulumParams, log, outdir: str | os.PathLike[str]) -> Path:
    """Write a standalone visual simulator HTML file and return its path."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_path = out_path / "phase0_visual.html"
    payload = json.dumps(build_visual_payload(p, log), separators=(",", ":"))
    html_path.write_text(_HTML_TEMPLATE.replace("__EMAC_DATA__", payload), encoding="utf-8")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the EMAC Phase 0 visual simulator.")
    parser.add_argument("--outdir", default=os.path.join("build", "visual"))
    parser.add_argument("--t-end", type=float, default=22.0)
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p, log = run_scenario(t_end=args.t_end)
    html_path = write_visual_simulator(p, log, args.outdir)
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
      display: grid;
      place-items: center;
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }

    .icon-button svg {
      width: 19px;
      height: 19px;
      fill: currentColor;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--blue);
    }

    .speed-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .speed-row button[aria-pressed="true"] {
      border-color: var(--blue);
      color: var(--blue);
      background: #edf4ff;
      font-weight: 650;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--panel-2);
    }

    .metric-label,
    .toggle label,
    .section-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 680;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .metric-value {
      margin-top: 4px;
      font-size: 18px;
      line-height: 1.15;
      font-weight: 720;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .toggle-list {
      display: grid;
      gap: 9px;
    }

    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid var(--line);
      padding-top: 9px;
    }

    .toggle:first-child {
      border-top: 0;
      padding-top: 0;
    }

    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--blue);
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      color: var(--muted);
      font-size: 12px;
    }

    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .swatch {
      width: 14px;
      height: 3px;
      border-radius: 999px;
      background: currentColor;
    }

    .timeline {
      grid-column: 1 / -1;
      min-height: 230px;
      overflow: hidden;
    }

    @media (max-width: 900px) {
      header {
        align-items: flex-start;
        flex-direction: column;
        gap: 8px;
      }

      main {
        grid-template-columns: 1fr;
        grid-template-rows: 420px auto 250px;
      }

      .timeline {
        grid-column: auto;
      }
    }

    @media (max-width: 560px) {
      main {
        padding: 10px;
        gap: 10px;
        grid-template-rows: 360px auto 250px;
      }

      header {
        padding: 14px;
      }

      .metric-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>EMAC Phase 0 Visual Simulator</h1>
      <div class="run-meta"><span class="status-dot"></span><span id="runSummary">host simulation</span></div>
    </header>
    <main>
      <section class="stage-wrap" aria-label="Pendulum visualization">
        <canvas id="stageCanvas"></canvas>
      </section>
      <aside class="side">
        <div class="controls">
          <button class="icon-button" id="playPause" title="Play or pause" aria-label="Play or pause">
            <svg id="playIcon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
          </button>
          <input id="scrubber" type="range" min="0" max="10000" value="0" aria-label="Simulation time">
        </div>
        <div class="speed-row" role="group" aria-label="Playback speed">
          <button data-speed="0.5">0.5x</button>
          <button data-speed="1" aria-pressed="true">1x</button>
          <button data-speed="2">2x</button>
          <button data-speed="4">4x</button>
        </div>
        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Time</div><div class="metric-value" id="timeValue">0.00 s</div></div>
          <div class="metric"><div class="metric-label">Mode</div><div class="metric-value" id="modeValue">coast</div></div>
          <div class="metric"><div class="metric-label">Theta</div><div class="metric-value" id="thetaValue">0.000 rad</div></div>
          <div class="metric"><div class="metric-label">Current</div><div class="metric-value" id="currentValue">0.00 A</div></div>
          <div class="metric"><div class="metric-label">Target amp</div><div class="metric-value" id="targetValue">0.000 rad</div></div>
          <div class="metric"><div class="metric-label">Est. amp</div><div class="metric-value" id="estAmpValue">0.000 rad</div></div>
        </div>
        <div class="toggle-list">
          <div class="toggle"><label for="showTrue">True state</label><input id="showTrue" type="checkbox" checked></div>
          <div class="toggle"><label for="showEstimate">Estimator</label><input id="showEstimate" type="checkbox" checked></div>
          <div class="toggle"><label for="showField">Coil field</label><input id="showField" type="checkbox" checked></div>
        </div>
        <div>
          <div class="section-label">Trace legend</div>
          <div class="legend">
            <span style="color: var(--blue)"><i class="swatch"></i>true theta</span>
            <span style="color: var(--red)"><i class="swatch"></i>estimated theta</span>
            <span style="color: var(--green)"><i class="swatch"></i>coil current</span>
            <span style="color: var(--amber)"><i class="swatch"></i>target amplitude</span>
          </div>
        </div>
      </aside>
      <section class="timeline" aria-label="Simulation traces">
        <canvas id="traceCanvas"></canvas>
      </section>
    </main>
  </div>
  <script>
    const DATA = __EMAC_DATA__;
    const t = DATA.series.t;
    const theta = DATA.series.theta;
    const thetaEst = DATA.series.theta_est;
    const current = DATA.series.current;
    const crossings = DATA.crossings;
    const summary = DATA.summary;

    const stageCanvas = document.getElementById("stageCanvas");
    const traceCanvas = document.getElementById("traceCanvas");
    const playPause = document.getElementById("playPause");
    const playIcon = document.getElementById("playIcon");
    const scrubber = document.getElementById("scrubber");
    const showTrue = document.getElementById("showTrue");
    const showEstimate = document.getElementById("showEstimate");
    const showField = document.getElementById("showField");
    const timeValue = document.getElementById("timeValue");
    const modeValue = document.getElementById("modeValue");
    const thetaValue = document.getElementById("thetaValue");
    const currentValue = document.getElementById("currentValue");
    const targetValue = document.getElementById("targetValue");
    const estAmpValue = document.getElementById("estAmpValue");
    const runSummary = document.getElementById("runSummary");

    let simTime = 0;
    let playing = false;
    let speed = 1;
    let lastFrame = performance.now();
    let traceStatic = null;

    runSummary.textContent = `${summary.t_end.toFixed(1)} s run`;

    function setupCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: rect.width, h: rect.height };
    }

    function nearestIndex(time) {
      let lo = 0;
      let hi = t.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (t[mid] < time) lo = mid + 1;
        else hi = mid;
      }
      return Math.max(0, Math.min(t.length - 1, lo));
    }

    function crossingIndex(time) {
      let idx = -1;
      for (let i = 0; i < crossings.t.length; i++) {
        if (crossings.t[i] <= time) idx = i;
        else break;
      }
      return idx;
    }

    function currentMode(time, idx, amps) {
      if (amps > 0.03) {
        const cx = crossingIndex(time);
        if (cx >= 0) return crossings.kind[cx];
        return "pulse";
      }
      const cx = crossingIndex(time);
      if (cx >= 0) return crossings.kind[cx] || "coast";
      return "search";
    }

    function setPlayIcon() {
      playIcon.innerHTML = playing
        ? '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>'
        : '<path d="M8 5v14l11-7z"/>';
    }

    function drawPendulum(ctx, w, h, angle, estAngle, amps, idx) {
      ctx.clearRect(0, 0, w, h);
      const pivot = { x: w * 0.5, y: Math.max(46, h * 0.13) };
      const arm = Math.min(w * 0.36, h * 0.58);
      const bottom = { x: pivot.x, y: pivot.y + arm };
      const coilY = Math.min(h - 48, bottom.y + 42);

      ctx.fillStyle = "#f9fbfa";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#cfd9d5";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pivot.x - 118, pivot.y);
      ctx.lineTo(pivot.x + 118, pivot.y);
      ctx.moveTo(pivot.x, pivot.y);
      ctx.lineTo(pivot.x, coilY + 18);
      ctx.stroke();

      ctx.fillStyle = "#18211f";
      ctx.beginPath();
      ctx.arc(pivot.x, pivot.y, 7, 0, Math.PI * 2);
      ctx.fill();

      if (showField.checked && amps > 0.03) {
        const strength = Math.min(1, amps / summary.max_current);
        ctx.strokeStyle = `rgba(111, 86, 201, ${0.18 + strength * 0.42})`;
        ctx.lineWidth = 2 + strength * 4;
        for (let r = 24; r <= 82; r += 18) {
          ctx.beginPath();
          ctx.ellipse(bottom.x, coilY - 16, r, r * 0.34, 0, Math.PI, Math.PI * 2);
          ctx.stroke();
        }
      }

      ctx.strokeStyle = "#6f56c9";
      ctx.lineWidth = 5;
      for (let x = -34; x <= 34; x += 17) {
        ctx.beginPath();
        ctx.arc(bottom.x + x, coilY, 13, Math.PI, 0);
        ctx.stroke();
      }
      ctx.fillStyle = "#433779";
      ctx.fillRect(bottom.x - 56, coilY + 12, 112, 10);

      drawGate(ctx, bottom.x, bottom.y, w);

      if (showEstimate.checked) {
        drawBob(ctx, pivot, arm, estAngle, "#cf3f4b", true);
      }
      if (showTrue.checked) {
        drawBob(ctx, pivot, arm, angle, "#1f6feb", false);
      }

      const cx = crossingIndex(t[idx]);
      if (cx >= 0) {
        ctx.fillStyle = "#66736e";
        ctx.font = "12px Inter, system-ui, sans-serif";
        ctx.fillText(`last crossing ${crossings.t[cx].toFixed(2)} s`, 18, h - 22);
      }
    }

    function drawGate(ctx, x, y, w) {
      ctx.strokeStyle = "#1f8a5b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x - 72, y);
      ctx.lineTo(x - 30, y);
      ctx.moveTo(x + 30, y);
      ctx.lineTo(x + 72, y);
      ctx.stroke();

      ctx.fillStyle = "#e8f6ef";
      ctx.strokeStyle = "#b8d9ca";
      ctx.lineWidth = 1;
      ctx.fillRect(x - 84, y - 20, 18, 40);
      ctx.strokeRect(x - 84, y - 20, 18, 40);
      ctx.fillRect(x + 66, y - 20, 18, 40);
      ctx.strokeRect(x + 66, y - 20, 18, 40);
    }

    function drawBob(ctx, pivot, arm, angle, color, dashed) {
      const bob = {
        x: pivot.x + Math.sin(angle) * arm,
        y: pivot.y + Math.cos(angle) * arm
      };
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 2 : 4;
      ctx.globalAlpha = dashed ? 0.72 : 1;
      if (dashed) ctx.setLineDash([7, 7]);
      ctx.beginPath();
      ctx.moveTo(pivot.x, pivot.y);
      ctx.lineTo(bob.x, bob.y);
      ctx.stroke();
      ctx.setLineDash([]);
      const grad = ctx.createRadialGradient(bob.x - 7, bob.y - 9, 4, bob.x, bob.y, 20);
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(0.24, color);
      grad.addColorStop(1, "#14201d");
      ctx.fillStyle = dashed ? "transparent" : grad;
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 3 : 1.5;
      ctx.beginPath();
      ctx.arc(bob.x, bob.y, dashed ? 14 : 18, 0, Math.PI * 2);
      if (!dashed) ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function drawTraceBase() {
      const { ctx, w, h } = setupCanvas(traceCanvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const pad = { l: 48, r: 20, t: 22, b: 30 };
      const plotW = w - pad.l - pad.r;
      const plotH = h - pad.t - pad.b;
      const split = pad.t + plotH * 0.62;
      const xFor = value => pad.l + (value / summary.t_end) * plotW;
      const thetaY = value => pad.t + plotH * 0.31 - (value / summary.max_theta) * plotH * 0.26;
      const currentY = value => split + plotH * 0.30 - (value / summary.max_current) * plotH * 0.24;

      ctx.strokeStyle = "#d8e0dd";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.t + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
      }

      drawSeries(ctx, t, theta, xFor, thetaY, "#1f6feb", 1.5);
      drawSeries(ctx, t, thetaEst, xFor, thetaY, "#cf3f4b", 1.2, [6, 5]);
      drawSeries(ctx, t, current, xFor, currentY, "#1f8a5b", 1.3);

      ctx.strokeStyle = "#c2761b";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < crossings.t.length; i++) {
        const x = xFor(crossings.t[i]);
        const y = thetaY(crossings.a_target[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.strokeStyle = "rgba(31, 111, 235, 0.2)";
      for (const cx of crossings.t) {
        const x = xFor(cx);
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, h - pad.b);
        ctx.stroke();
      }

      ctx.fillStyle = "#66736e";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText("theta / amplitude", 10, pad.t + 12);
      ctx.fillText("current", 10, split + 14);
      ctx.fillText("0 s", pad.l, h - 9);
      ctx.fillText(`${summary.t_end.toFixed(1)} s`, w - pad.r - 44, h - 9);

      traceStatic = { pad, plotW, xFor };
    }

    function drawSeries(ctx, xs, ys, xFor, yFor, color, width, dash) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const x = xFor(xs[i]);
        const y = yFor(ys[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    function drawTraceCursor(time) {
      if (!traceStatic) drawTraceBase();
      const { ctx, w, h } = setupCanvas(traceCanvas);
      if (!traceStatic) return;
      drawTraceBase();
      const x = traceStatic.xFor(time);
      ctx.strokeStyle = "#18211f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 16);
      ctx.lineTo(x, h - 28);
      ctx.stroke();
    }

    function updateReadouts(idx) {
      const cx = crossingIndex(t[idx]);
      const target = cx >= 0 ? crossings.a_target[cx] : 0;
      const estAmp = cx >= 0 ? crossings.a_est[cx] : 0;
      const mode = currentMode(t[idx], idx, current[idx]);
      timeValue.textContent = `${t[idx].toFixed(2)} s`;
      modeValue.textContent = mode;
      thetaValue.textContent = `${theta[idx].toFixed(3)} rad`;
      currentValue.textContent = `${current[idx].toFixed(2)} A`;
      targetValue.textContent = `${target.toFixed(3)} rad`;
      estAmpValue.textContent = `${estAmp.toFixed(3)} rad`;
    }

    function render() {
      const idx = nearestIndex(simTime);
      const stage = setupCanvas(stageCanvas);
      drawPendulum(stage.ctx, stage.w, stage.h, theta[idx], thetaEst[idx], current[idx], idx);
      drawTraceCursor(simTime);
      updateReadouts(idx);
      scrubber.value = String(Math.round((simTime / summary.t_end) * 10000));
    }

    function tick(now) {
      const elapsed = Math.min(0.08, (now - lastFrame) / 1000);
      lastFrame = now;
      if (playing) {
        simTime += elapsed * speed;
        if (simTime > summary.t_end) simTime = 0;
        render();
      }
      requestAnimationFrame(tick);
    }

    playPause.addEventListener("click", () => {
      playing = !playing;
      lastFrame = performance.now();
      setPlayIcon();
    });

    scrubber.addEventListener("input", () => {
      simTime = (Number(scrubber.value) / 10000) * summary.t_end;
      render();
    });

    document.querySelectorAll("[data-speed]").forEach(button => {
      button.addEventListener("click", () => {
        speed = Number(button.dataset.speed);
        document.querySelectorAll("[data-speed]").forEach(node => node.setAttribute("aria-pressed", "false"));
        button.setAttribute("aria-pressed", "true");
      });
    });

    [showTrue, showEstimate, showField].forEach(input => input.addEventListener("change", render));

    window.addEventListener("resize", () => {
      traceStatic = null;
      render();
    });

    setPlayIcon();
    drawTraceBase();
    render();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
