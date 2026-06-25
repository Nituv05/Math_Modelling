#!/usr/bin/env python3
"""Animated browser demo for the pedestrian-flow model.

Run:

    python3 demo.py

By default this writes ``figures/demo_simulation.html``: a self-contained
browser replay of several precomputed sensitivity cases.  The demo is a
controlled replay, not a realtime parameter fitter; changing the selected case
switches the entire simulated trajectory and velocity data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import webbrowser

import numpy as np

from pedestrian import (
    ModelParameters,
    HardBodyModel,
    RemoteActionModel,
    empirical_mean_velocity_near_density,
    fundamental_diagram,
    load_empirical_points,
    rmse_against_empirical,
)


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
DEFAULT_HTML = FIG_DIR / "demo_simulation.html"


@dataclass(frozen=True)
class DemoCase:
    key: str
    label: str
    group: str
    model_name: str
    model_label: str
    n: int
    L: float
    params: ModelParameters
    seed_offset: int
    purpose: str

    @property
    def model_cls(self):
        return HardBodyModel if self.model_name == "hardbody" else RemoteActionModel


CASE_DEFINITIONS = [
    DemoCase(
        key="hard_b0",
        label="Hard body, b = 0",
        group="Baseline b sensitivity",
        model_name="hardbody",
        model_label="Hard body",
        n=21,
        L=17.3,
        params=ModelParameters(a=0.36, b=0.0, tau=0.61),
        seed_offset=0,
        purpose="Fixed required length; useful as the control case.",
    ),
    DemoCase(
        key="hard_b056",
        label="Hard body, b = 0.56",
        group="Baseline b sensitivity",
        model_name="hardbody",
        model_label="Hard body",
        n=21,
        L=17.3,
        params=ModelParameters(a=0.36, b=0.56, tau=0.61),
        seed_offset=100,
        purpose="Velocity-dependent required length; closest macroscopic fit.",
    ),
    DemoCase(
        key="hard_b106",
        label="Hard body, b = 1.06",
        group="Baseline b sensitivity",
        model_name="hardbody",
        model_label="Hard body",
        n=21,
        L=17.3,
        params=ModelParameters(a=0.36, b=1.06, tau=0.61),
        seed_offset=200,
        purpose="Empirically motivated b; decelerates too strongly macroscopically.",
    ),
    DemoCase(
        key="remote_b0_n20",
        label="Remote, b = 0, N = 20",
        group="Density wave region",
        model_name="remote",
        model_label="Remote force",
        n=20,
        L=17.3,
        params=ModelParameters(a=0.36, b=0.0, tau=0.61, e=0.07, f=2.0),
        seed_offset=300,
        purpose="Just below the reported velocity-gap region.",
    ),
    DemoCase(
        key="remote_b0_n21",
        label="Remote, b = 0, N = 21",
        group="Density wave region",
        model_name="remote",
        model_label="Remote force",
        n=21,
        L=17.3,
        params=ModelParameters(a=0.36, b=0.0, tau=0.61, e=0.07, f=2.0),
        seed_offset=400,
        purpose="Near rho = 1.2 1/m, where stop-and-go waves appear.",
    ),
    DemoCase(
        key="remote_b056",
        label="Remote, b = 0.56",
        group="Remote force comparison",
        model_name="remote",
        model_label="Remote force",
        n=21,
        L=17.3,
        params=ModelParameters(a=0.36, b=0.56, tau=0.61, e=0.07, f=2.0),
        seed_offset=500,
        purpose="Remote force has small effect once dynamic space is present.",
    ),
]


FD_DEFINITIONS = [
    ("hard_b0", "Hard body, b = 0", HardBodyModel, ModelParameters(a=0.36, b=0.0, tau=0.61), "#2563eb"),
    ("hard_b056", "Hard body, b = 0.56", HardBodyModel, ModelParameters(a=0.36, b=0.56, tau=0.61), "#0f766e"),
    ("hard_b106", "Hard body, b = 1.06", HardBodyModel, ModelParameters(a=0.36, b=1.06, tau=0.61), "#d97706"),
    ("remote_b0", "Remote, b = 0", RemoteActionModel, ModelParameters(a=0.36, b=0.0, tau=0.61, e=0.07, f=2.0), "#be123c"),
    ("remote_b056", "Remote, b = 0.56", RemoteActionModel, ModelParameters(a=0.36, b=0.56, tau=0.61, e=0.07, f=2.0), "#7c3aed"),
]


def collect_frames(model, frame_count: int, steps_per_frame: int):
    positions = np.empty((frame_count, model.n), dtype=float)
    velocities = np.empty((frame_count, model.n), dtype=float)
    for frame in range(frame_count):
        for _ in range(steps_per_frame):
            model.step()
        positions[frame] = model.x
        velocities[frame] = model.v
    return positions, velocities


def simulate_case(case: DemoCase, args) -> dict:
    model = case.model_cls(n=case.n, L=case.L, params=case.params, seed=args.seed + case.seed_offset)

    for _ in range(args.relax_steps):
        model.step()

    positions, velocities = collect_frames(model, args.frames, args.steps_per_frame)
    sim_time = np.arange(args.frames) * args.steps_per_frame * model.dt
    mean_velocity = velocities.mean(axis=1)
    density = case.n / case.L
    return {
        "key": case.key,
        "label": case.label,
        "group": case.group,
        "purpose": case.purpose,
        "model": case.model_name,
        "modelLabel": case.model_label,
        "a": case.params.a,
        "b": case.params.b,
        "tau": case.params.tau,
        "e": case.params.e if case.model_name == "remote" else None,
        "f": case.params.f if case.model_name == "remote" else None,
        "L": case.L,
        "n": case.n,
        "density": density,
        "intervalMs": args.interval_ms,
        "stepsPerFrame": args.steps_per_frame,
        "positions": np.round(positions, 4).tolist(),
        "velocities": np.round(velocities, 4).tolist(),
        "meanVelocity": np.round(mean_velocity, 4).tolist(),
        "meanOverall": float(np.mean(mean_velocity)),
        "time": np.round(sim_time, 4).tolist(),
    }


def build_fundamental_diagram(args) -> dict:
    counts = [5, 10, 15, 20, 21, 25, 30, 35, 40, 45, 48]
    curves = []
    for index, (key, label, model_cls, params, color) in enumerate(FD_DEFINITIONS):
        print(f"fundamental diagram: {label}")
        results = fundamental_diagram(
            model_cls,
            params,
            L=17.3,
            counts=counts,
            relax_steps=args.fd_relax_steps,
            measure_steps=args.fd_measure_steps,
            seed=args.seed + 10_000 + index * 1000,
            progress=False,
        )
        curves.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "points": [
                    {
                        "density": round(result.density, 4),
                        "velocity": round(result.mean_velocity, 4),
                    }
                    for result in results
                ],
            }
        )

    empirical = load_empirical_points()
    return {
        "curves": curves,
        "empirical": [
            {"density": round(point.density, 4), "velocity": round(point.velocity, 4)}
            for point in empirical
        ],
    }


def build_payload(args) -> dict:
    cases = {}
    for case in CASE_DEFINITIONS:
        print(f"demo case: {case.label}")
        cases[case.key] = simulate_case(case, args)

    return {
        "defaultCase": "remote_b0_n21",
        "cases": cases,
        "caseOrder": [case.key for case in CASE_DEFINITIONS],
        "fundamentalDiagram": build_fundamental_diagram(args),
    }


def render_html(payload) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1D Periodic Pedestrian Flow Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64707d;
      --line: #d7dce2;
      --soft: #f9fafb;
      --track: #e7eaee;
      --track-edge: #bcc5ce;
      --accent: #0f766e;
      --accent-2: #d97706;
      --accent-3: #be123c;
      --blue: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .shell {
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 28px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-end;
      margin-bottom: 16px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 750;
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    .badge {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(360px, 0.95fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.07);
      overflow: hidden;
    }
    .panel + .panel { margin-top: 14px; }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
    }
    .panel-title {
      font-weight: 700;
      font-size: 15px;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      max-width: 560px;
    }
    .canvas-wrap { padding: 10px 12px 14px; }
    canvas {
      display: block;
      width: 100%;
      background: #fbfcfd;
      border: 1px solid #edf0f2;
      border-radius: 6px;
    }
    #ringCanvas { height: 390px; }
    #fdCanvas { height: 285px; }
    #speedCanvas { height: 190px; }
    #spaceCanvas { height: 270px; }
    .side {
      display: grid;
      grid-template-rows: auto auto auto;
      gap: 14px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 9px;
      padding: 14px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 72px;
      background: #fcfdfd;
    }
    .stat label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 8px;
      white-space: nowrap;
    }
    .stat strong {
      display: block;
      font-size: 21px;
      line-height: 1.05;
      overflow-wrap: anywhere;
    }
    .stat .small-value {
      font-size: 15px;
      line-height: 1.2;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1.15fr) auto minmax(170px, 1fr) auto;
      gap: 10px;
      padding: 14px;
      align-items: center;
      border-top: 1px solid var(--line);
    }
    button, select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      height: 38px;
      padding: 0 12px;
      font: inherit;
      cursor: pointer;
      min-width: 0;
    }
    button {
      min-width: 82px;
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
      font-weight: 650;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .case-copy {
      padding: 0 14px 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .legend {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .key {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .swatch {
      width: 18px;
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--blue), var(--accent), var(--accent-2), var(--accent-3));
    }
    @media (max-width: 1050px) {
      .grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .badge { white-space: normal; }
      #ringCanvas { height: 330px; }
    }
    @media (max-width: 680px) {
      .shell { width: min(100vw - 20px, 1280px); padding-top: 14px; }
      h1 { font-size: 24px; }
      .controls { grid-template-columns: 1fr 84px; }
      .controls select:first-child { grid-column: 1 / -1; }
      .controls input[type="range"] { grid-column: 1 / -1; }
      .stats { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>1D Periodic Pedestrian Flow Demo</h1>
        <div class="subtitle">Replay of single-file pedestrian experiments under periodic boundary conditions</div>
      </div>
      <div class="badge" id="caseBadge"></div>
    </header>

    <section class="grid">
      <div>
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">1D Periodic Domain</div>
              <div class="panel-note">A 1D line is wrapped into a loop only for visualization; color tracks instantaneous velocity.</div>
            </div>
            <div class="legend"><span class="key"><span class="swatch"></span> slow to fast</span></div>
          </div>
          <div class="canvas-wrap"><canvas id="ringCanvas"></canvas></div>
          <div class="controls">
            <select id="caseSelect" aria-label="Simulation case"></select>
            <button id="playButton" type="button">Pause</button>
            <input id="frameSlider" type="range" min="0" max="0" value="0" aria-label="Frame">
            <select id="speedSelect" aria-label="Playback speed">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2x</option>
            </select>
          </div>
          <div class="case-copy" id="caseCopy"></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">Fundamental Diagram</div>
              <div class="panel-note">Mean velocity versus density; the selected replay point is highlighted.</div>
            </div>
          </div>
          <div class="canvas-wrap"><canvas id="fdCanvas"></canvas></div>
        </div>
      </div>

      <aside class="side">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Run State</div>
            <div class="panel-note" id="timeLabel"></div>
          </div>
          <div class="stats">
            <div class="stat"><label>density rho [1/m]</label><strong id="densityStat"></strong></div>
            <div class="stat"><label>N pedestrians</label><strong id="nStat"></strong></div>
            <div class="stat"><label>L [m]</label><strong id="lStat"></strong></div>
            <div class="stat"><label>mean velocity [m/s]</label><strong id="velocityStat"></strong></div>
            <div class="stat"><label>stopped pedestrians</label><strong id="stoppedStat"></strong></div>
            <div class="stat"><label>minimum gap [m]</label><strong id="gapStat"></strong></div>
            <div class="stat"><label>a [m]</label><strong id="aStat"></strong></div>
            <div class="stat"><label>b [s]</label><strong id="bStat"></strong></div>
            <div class="stat"><label>tau [s]</label><strong id="tauStat"></strong></div>
            <div class="stat"><label>model</label><strong id="modelStat" class="small-value"></strong></div>
            <div class="stat"><label>e</label><strong id="eStat"></strong></div>
            <div class="stat"><label>f</label><strong id="fStat"></strong></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Mean Velocity</div>
            <div class="panel-note">m/s over replay time</div>
          </div>
          <div class="canvas-wrap"><canvas id="speedCanvas"></canvas></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Space-Time Trace</div>
            <div class="panel-note">position x across recent frames</div>
          </div>
          <div class="canvas-wrap"><canvas id="spaceCanvas"></canvas></div>
        </div>
      </aside>
    </section>
  </main>

  <script>
    const PAYLOAD = __PAYLOAD__;
    const CASES = PAYLOAD.cases;
    const ringCanvas = document.getElementById("ringCanvas");
    const speedCanvas = document.getElementById("speedCanvas");
    const spaceCanvas = document.getElementById("spaceCanvas");
    const fdCanvas = document.getElementById("fdCanvas");
    const playButton = document.getElementById("playButton");
    const frameSlider = document.getElementById("frameSlider");
    const speedSelect = document.getElementById("speedSelect");
    const caseSelect = document.getElementById("caseSelect");
    const caseBadge = document.getElementById("caseBadge");
    const caseCopy = document.getElementById("caseCopy");
    const timeLabel = document.getElementById("timeLabel");
    const densityStat = document.getElementById("densityStat");
    const nStat = document.getElementById("nStat");
    const lStat = document.getElementById("lStat");
    const velocityStat = document.getElementById("velocityStat");
    const stoppedStat = document.getElementById("stoppedStat");
    const gapStat = document.getElementById("gapStat");
    const aStat = document.getElementById("aStat");
    const bStat = document.getElementById("bStat");
    const tauStat = document.getElementById("tauStat");
    const modelStat = document.getElementById("modelStat");
    const eStat = document.getElementById("eStat");
    const fStat = document.getElementById("fStat");

    let DATA = CASES[PAYLOAD.defaultCase];
    let frame = 0;
    let playing = true;
    let playback = 1;
    let lastTick = 0;
    let accumulator = 0;
    const canvasViews = new Map();
    let globalVmax = 1.25;

    for (const key of PAYLOAD.caseOrder) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = CASES[key].label;
      caseSelect.appendChild(option);
    }
    caseSelect.value = DATA.key;

    for (const item of Object.values(CASES)) {
      for (const velocities of item.velocities) {
        for (const velocity of velocities) {
          if (velocity > globalVmax) globalVmax = velocity;
        }
      }
    }

    function resizeCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      canvasViews.set(canvas, { ctx, width: rect.width, height: rect.height });
    }

    function resizeAllCanvases() {
      resizeCanvas(ringCanvas);
      resizeCanvas(fdCanvas);
      resizeCanvas(speedCanvas);
      resizeCanvas(spaceCanvas);
    }

    function setupCanvas(canvas) {
      if (!canvasViews.has(canvas)) resizeCanvas(canvas);
      return canvasViews.get(canvas);
    }

    function colorForVelocity(v) {
      const t = Math.max(0, Math.min(1, v / globalVmax));
      const stops = [
        [37, 99, 235],
        [15, 118, 110],
        [217, 119, 6],
        [190, 18, 60]
      ];
      const scaled = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(scaled));
      const f = scaled - i;
      const a = stops[i], b = stops[i + 1];
      const rgb = a.map((value, idx) => Math.round(value + (b[idx] - value) * f));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }

    function pointOnRing(x, width, height) {
      const cx = width / 2;
      const cy = height / 2 + 4;
      const rx = Math.max(80, width * 0.39);
      const ry = Math.max(58, height * 0.30);
      const theta = -Math.PI / 2 + 2 * Math.PI * (x / DATA.L);
      return [cx + rx * Math.cos(theta), cy + ry * Math.sin(theta)];
    }

    function drawRing() {
      const { ctx, width, height } = setupCanvas(ringCanvas);
      ctx.clearRect(0, 0, width, height);
      const cx = width / 2;
      const cy = height / 2 + 4;
      const rx = Math.max(80, width * 0.39);
      const ry = Math.max(58, height * 0.30);

      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = "#e7eaee";
      ctx.lineWidth = 34;
      ctx.stroke();

      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = "#bcc5ce";
      ctx.lineWidth = 2;
      ctx.stroke();

      for (let tick = 0; tick < 12; tick++) {
        const x = (tick / 12) * DATA.L;
        const [px, py] = pointOnRing(x, width, height);
        const [qx, qy] = pointOnRing(x + 0.001, width, height);
        const dx = qx - px, dy = qy - py;
        const len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len, ny = dx / len;
        ctx.beginPath();
        ctx.moveTo(px - nx * 11, py - ny * 11);
        ctx.lineTo(px + nx * 11, py + ny * 11);
        ctx.strokeStyle = "#9aa5b1";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      const positions = DATA.positions[frame];
      const velocities = DATA.velocities[frame];
      for (let i = 0; i < positions.length; i++) {
        const [px, py] = pointOnRing(positions[i], width, height);
        ctx.beginPath();
        ctx.arc(px, py, 8.5, 0, 2 * Math.PI);
        ctx.fillStyle = colorForVelocity(velocities[i]);
        ctx.fill();
        ctx.strokeStyle = "#17202a";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    function drawSpeed() {
      const { ctx, width, height } = setupCanvas(speedCanvas);
      ctx.clearRect(0, 0, width, height);
      const padL = 42, padR = 14, padT = 18, padB = 34;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const vmax = Math.max(1.0, ...DATA.meanVelocity) * 1.15;

      ctx.strokeStyle = "#d7dce2";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const y = padT + plotH * g / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
      }

      ctx.beginPath();
      for (let i = 0; i <= frame; i++) {
        const x = padL + plotW * i / (DATA.meanVelocity.length - 1);
        const y = padT + plotH * (1 - DATA.meanVelocity[i] / vmax);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2.5;
      ctx.stroke();

      const mx = padL + plotW * frame / (DATA.meanVelocity.length - 1);
      const my = padT + plotH * (1 - DATA.meanVelocity[frame] / vmax);
      ctx.beginPath();
      ctx.arc(mx, my, 4, 0, 2 * Math.PI);
      ctx.fillStyle = "#be123c";
      ctx.fill();

      ctx.fillStyle = "#64707d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("0", 8, padT + plotH + 4);
      ctx.fillText(vmax.toFixed(1), 8, padT + 4);
    }

    function drawSpaceTime() {
      const { ctx, width, height } = setupCanvas(spaceCanvas);
      ctx.clearRect(0, 0, width, height);
      const padL = 30, padR = 12, padT = 12, padB = 24;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const start = Math.max(0, frame - 95);
      const rows = Math.max(1, frame - start);

      ctx.strokeStyle = "#edf0f2";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const x = padL + plotW * g / 4;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }

      for (let f = start; f <= frame; f++) {
        const y = padT + plotH * (f - start) / rows;
        for (let i = 0; i < DATA.positions[f].length; i++) {
          const x = padL + plotW * DATA.positions[f][i] / DATA.L;
          ctx.fillStyle = colorForVelocity(DATA.velocities[f][i]);
          ctx.fillRect(x, y, 1.8, 1.8);
        }
      }

      ctx.fillStyle = "#64707d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("x = 0", padL, height - 7);
      ctx.fillText(`x = ${DATA.L.toFixed(1)} m`, width - padR - 72, height - 7);
    }

    function drawFundamentalDiagram() {
      const { ctx, width, height } = setupCanvas(fdCanvas);
      ctx.clearRect(0, 0, width, height);
      const padL = 46, padR = 18, padT = 82, padB = 38;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const maxRho = 3.0;
      const maxV = 1.4;
      const xScale = (rho) => padL + plotW * rho / maxRho;
      const yScale = (v) => padT + plotH * (1 - v / maxV);

      ctx.strokeStyle = "#d7dce2";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 5; g++) {
        const x = padL + plotW * g / 5;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }
      for (let g = 0; g <= 4; g++) {
        const y = padT + plotH * g / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(31, 41, 51, 0.35)";
      for (const point of PAYLOAD.fundamentalDiagram.empirical) {
        ctx.beginPath();
        ctx.arc(xScale(point.density), yScale(point.velocity), 2.2, 0, 2 * Math.PI);
        ctx.fill();
      }

      for (const curve of PAYLOAD.fundamentalDiagram.curves) {
        ctx.beginPath();
        curve.points.forEach((point, index) => {
          const x = xScale(point.density);
          const y = yScale(point.velocity);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = curve.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        for (const point of curve.points) {
          ctx.beginPath();
          ctx.arc(xScale(point.density), yScale(point.velocity), 3, 0, 2 * Math.PI);
          ctx.fillStyle = curve.color;
          ctx.fill();
        }
      }

      const selectedX = xScale(DATA.density);
      const selectedY = yScale(DATA.meanOverall);
      ctx.beginPath();
      ctx.arc(selectedX, selectedY, 7, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.fillStyle = "#64707d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("rho [1/m]", padL + plotW - 58, height - 9);
      ctx.save();
      ctx.translate(13, padT + 82);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("mean velocity [m/s]", 0, 0);
      ctx.restore();

      const legendX = padL;
      let legendY = 18;
      ctx.font = "12px system-ui, sans-serif";
      const legendItems = [
        { label: "empirical points", color: "rgba(31, 41, 51, 0.35)", empirical: true },
        ...PAYLOAD.fundamentalDiagram.curves
      ];
      let legendColumn = 0;
      for (let i = 0; i < legendItems.length; i++) {
        const item = legendItems[i];
        const x = legendX + legendColumn * 250;
        const y = legendY;
        if (x + 230 > width - padR && legendColumn > 0) {
          legendColumn = 0;
          legendY += 20;
        }
        const itemX = legendX + legendColumn * 250;
        const itemY = legendY;
        if (item.empirical) {
          ctx.fillStyle = item.color;
          ctx.beginPath();
          ctx.arc(itemX + 8, itemY, 3, 0, 2 * Math.PI);
          ctx.fill();
        } else {
          ctx.strokeStyle = item.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(itemX, itemY);
          ctx.lineTo(itemX + 16, itemY);
          ctx.stroke();
        }
        ctx.fillStyle = "#1f2933";
        ctx.fillText(item.label, itemX + 22, itemY + 4);
        legendColumn += 1;
      }

      ctx.strokeStyle = "#edf0f2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT - 15);
      ctx.lineTo(width - padR, padT - 15);
      ctx.stroke();
    }

    function stoppedCount(frameIndex) {
      return DATA.velocities[frameIndex].filter((v) => v < 0.05).length;
    }

    function minGap(frameIndex) {
      const pos = [...DATA.positions[frameIndex]].sort((a, b) => a - b);
      let gap = DATA.L;
      for (let i = 0; i < pos.length; i++) {
        const next = pos[(i + 1) % pos.length];
        const candidate = i === pos.length - 1 ? next + DATA.L - pos[i] : next - pos[i];
        gap = Math.min(gap, candidate);
      }
      return gap;
    }

    function setCase(key) {
      DATA = CASES[key];
      frame = 0;
      accumulator = 0;
      frameSlider.max = DATA.positions.length - 1;
      frameSlider.value = 0;
      caseBadge.textContent = `${DATA.modelLabel} | b=${DATA.b.toFixed(2)} | N=${DATA.n} | rho=${DATA.density.toFixed(2)}`;
      caseCopy.textContent = `${DATA.group}: ${DATA.purpose}`;
      draw();
    }

    function updateStats() {
      frameSlider.value = frame;
      timeLabel.textContent = `t = ${DATA.time[frame].toFixed(2)} s`;
      densityStat.textContent = DATA.density.toFixed(2);
      nStat.textContent = DATA.n;
      lStat.textContent = DATA.L.toFixed(1);
      velocityStat.textContent = DATA.meanVelocity[frame].toFixed(2);
      stoppedStat.textContent = stoppedCount(frame);
      gapStat.textContent = minGap(frame).toFixed(2);
      aStat.textContent = DATA.a.toFixed(2);
      bStat.textContent = DATA.b.toFixed(2);
      tauStat.textContent = DATA.tau.toFixed(2);
      modelStat.textContent = DATA.modelLabel;
      eStat.textContent = DATA.e === null ? "-" : DATA.e.toFixed(2);
      fStat.textContent = DATA.f === null ? "-" : DATA.f.toFixed(1);
    }

    function draw() {
      drawRing();
      drawFundamentalDiagram();
      drawSpeed();
      drawSpaceTime();
      updateStats();
    }

    function animate(timestamp) {
      if (!lastTick) lastTick = timestamp;
      const delta = timestamp - lastTick;
      lastTick = timestamp;
      if (playing) {
        accumulator += delta * playback;
        while (accumulator >= DATA.intervalMs) {
          frame = (frame + 1) % DATA.positions.length;
          accumulator -= DATA.intervalMs;
        }
      }
      draw();
      requestAnimationFrame(animate);
    }

    playButton.addEventListener("click", () => {
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    });
    frameSlider.addEventListener("input", () => {
      frame = Number(frameSlider.value);
      playing = false;
      playButton.textContent = "Play";
      draw();
    });
    speedSelect.addEventListener("change", () => {
      playback = Number(speedSelect.value);
    });
    caseSelect.addEventListener("change", () => {
      setCase(caseSelect.value);
    });
    window.addEventListener("resize", () => {
      resizeAllCanvases();
      draw();
    });
    resizeAllCanvases();
    setCase(DATA.key);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD__", data_json)


def write_html_animation(args) -> Path:
    payload = build_payload(args)
    html = render_html(payload)

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved animated demo to {out}")

    if args.show:
        webbrowser.open(out.resolve().as_uri())
    return out


def print_table_demo() -> None:
    densities = [0.6, 1.0, 1.4, 1.8, 2.2]
    cases = [
        ("Hard bodies without remote action, b=0.56", HardBodyModel, ModelParameters(a=0.36, b=0.56)),
        ("Remote action, b=0", RemoteActionModel, ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)),
    ]
    for name, model_cls, params in cases:
        results = fundamental_diagram(
            model_cls,
            params,
            L=17.3,
            densities=densities,
            relax_steps=5_000,
            measure_steps=5_000,
            seed=7,
            progress=False,
        )
        rho = np.array([r.density for r in results])
        velocity = np.array([r.mean_velocity for r in results])
        reference = empirical_mean_velocity_near_density(rho, half_width=0.075)

        print(f"\n{name}")
        print("rho [1/m]   simulation v [m/s]   nearby empirical mean [m/s]")
        for r, v, ref in zip(rho, velocity, reference):
            ref_text = "n/a" if np.isnan(ref) else f"{ref:.3f}"
            print(f"{r:8.3f}   {v:18.3f}   {ref_text:>25}")
        print(f"RMSE vs nearby empirical points: {rmse_against_empirical(rho, velocity):.3f} m/s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relax-steps", type=int, default=8_000)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--steps-per-frame", type=int, default=25)
    parser.add_argument("--fd-relax-steps", type=int, default=3_000)
    parser.add_argument("--fd-measure-steps", type=int, default=3_000)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--show", action="store_true", help="open the generated HTML in a browser")
    parser.add_argument("--table", action="store_true", help="print the old numeric smoke-test table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.table:
        print_table_demo()
        return
    write_html_animation(args)


if __name__ == "__main__":
    main()
