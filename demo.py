#!/usr/bin/env python3
"""Animated browser demo for the pedestrian-flow model.

Run:

    python3 demo.py

By default this writes ``figures/demo_simulation.html``: a self-contained
browser animation of the Fig. 3 case, namely the remote-action model (Eq. 6)
with ``b = 0`` near the velocity gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import webbrowser

import numpy as np

from pedestrian import (
    ModelParameters,
    HardBodyModel,
    RemoteActionModel,
    empirical_mean_velocity_near_density,
    fundamental_diagram,
    rmse_against_empirical,
)


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
DEFAULT_HTML = FIG_DIR / "demo_simulation.html"


def model_from_name(name: str):
    if name == "hardbody":
        return HardBodyModel
    if name == "remote":
        return RemoteActionModel
    raise ValueError(f"unknown model {name!r}")


def collect_frames(model, frame_count: int, steps_per_frame: int):
    positions = np.empty((frame_count, model.n), dtype=float)
    velocities = np.empty((frame_count, model.n), dtype=float)
    for frame in range(frame_count):
        for _ in range(steps_per_frame):
            model.step()
        positions[frame] = model.x
        velocities[frame] = model.v
    return positions, velocities


def simulate_demo(args):
    model_cls = model_from_name(args.model)
    n = max(1, int(round(args.density * args.L)))
    actual_density = n / args.L
    params = ModelParameters(a=0.36, b=args.b, e=0.07, f=2.0)
    model = model_cls(n=n, L=args.L, params=params, seed=args.seed)

    for _ in range(args.relax_steps):
        model.step()

    positions, velocities = collect_frames(model, args.frames, args.steps_per_frame)
    sim_time = np.arange(args.frames) * args.steps_per_frame * model.dt
    return n, actual_density, positions, velocities, sim_time


def build_payload(args, n, actual_density, positions, velocities, sim_time):
    return {
        "model": args.model,
        "b": args.b,
        "L": args.L,
        "n": n,
        "density": actual_density,
        "intervalMs": args.interval_ms,
        "stepsPerFrame": args.steps_per_frame,
        "positions": np.round(positions, 4).tolist(),
        "velocities": np.round(velocities, 4).tolist(),
        "meanVelocity": np.round(velocities.mean(axis=1), 4).tolist(),
        "time": np.round(sim_time, 4).tolist(),
    }


def render_html(payload) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pedestrian Flow Simulation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f2;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64707d;
      --line: #d7dce2;
      --track: #e7eaee;
      --track-edge: #bcc5ce;
      --accent: #0f766e;
      --accent-2: #d97706;
      --accent-3: #be123c;
      --blue: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .shell {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-end;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 750;
    }}
    .subtitle {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }}
    .badge {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.95fr);
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.08);
      overflow: hidden;
    }}
    .panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
    }}
    .panel-title {{
      font-weight: 700;
      font-size: 15px;
    }}
    .panel-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    .canvas-wrap {{ padding: 10px 12px 14px; }}
    canvas {{
      display: block;
      width: 100%;
      background: #fbfcfa;
      border: 1px solid #edf0f2;
      border-radius: 6px;
    }}
    #ringCanvas {{ height: 390px; }}
    #speedCanvas {{ height: 190px; }}
    #spaceCanvas {{ height: 270px; }}
    .side {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      padding: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
      background: #fcfdfb;
    }}
    .stat label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .stat strong {{
      font-size: 24px;
      line-height: 1;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      padding: 14px;
      align-items: center;
      border-top: 1px solid var(--line);
    }}
    button, select {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      height: 38px;
      padding: 0 12px;
      font: inherit;
      cursor: pointer;
    }}
    button {{
      min-width: 88px;
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
      font-weight: 650;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .legend {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 18px;
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--blue), var(--accent), var(--accent-2), var(--accent-3));
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .badge {{ white-space: normal; }}
      #ringCanvas {{ height: 320px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Pedestrian Flow Simulation</h1>
        <div class="subtitle">Remote-action single-file movement near the Fig. 2 velocity gap</div>
      </div>
      <div class="badge" id="caseBadge"></div>
    </header>

    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">Ring Movement</div>
            <div class="panel-note">Each marker is one pedestrian; color tracks instantaneous velocity.</div>
          </div>
          <div class="legend"><span class="key"><span class="swatch"></span> slow to fast</span></div>
        </div>
        <div class="canvas-wrap"><canvas id="ringCanvas"></canvas></div>
        <div class="controls">
          <button id="playButton" type="button">Pause</button>
          <input id="frameSlider" type="range" min="0" max="0" value="0">
          <select id="speedSelect" aria-label="Playback speed">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
          </select>
        </div>
      </div>

      <aside class="side">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Run State</div>
            <div class="panel-note" id="timeLabel"></div>
          </div>
          <div class="stats">
            <div class="stat"><label>density rho</label><strong id="densityStat"></strong></div>
            <div class="stat"><label>pedestrians</label><strong id="nStat"></strong></div>
            <div class="stat"><label>mean velocity</label><strong id="velocityStat"></strong></div>
            <div class="stat"><label>model b</label><strong id="bStat"></strong></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">Mean Velocity</div>
            <div class="panel-note">m/s over animation time</div>
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
    const DATA = {data_json};
    const ringCanvas = document.getElementById("ringCanvas");
    const speedCanvas = document.getElementById("speedCanvas");
    const spaceCanvas = document.getElementById("spaceCanvas");
    const playButton = document.getElementById("playButton");
    const frameSlider = document.getElementById("frameSlider");
    const speedSelect = document.getElementById("speedSelect");
    const caseBadge = document.getElementById("caseBadge");
    const timeLabel = document.getElementById("timeLabel");
    const densityStat = document.getElementById("densityStat");
    const nStat = document.getElementById("nStat");
    const velocityStat = document.getElementById("velocityStat");
    const bStat = document.getElementById("bStat");

    let frame = 0;
    let playing = true;
    let playback = 1;
    let lastTick = 0;
    let accumulator = 0;

    frameSlider.max = DATA.positions.length - 1;
    caseBadge.textContent = `${{DATA.model}} model | b=${{DATA.b}} | L=${{DATA.L}} m`;
    densityStat.textContent = DATA.density.toFixed(2);
    nStat.textContent = DATA.n;
    bStat.textContent = DATA.b.toFixed(2);

    function setupCanvas(canvas) {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, width: rect.width, height: rect.height }};
    }}

    function colorForVelocity(v) {{
      const vmax = Math.max(1.25, ...DATA.velocities.flat());
      const t = Math.max(0, Math.min(1, v / vmax));
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
      return `rgb(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}})`;
    }}

    function pointOnRing(x, width, height) {{
      const cx = width / 2;
      const cy = height / 2 + 4;
      const rx = Math.max(80, width * 0.39);
      const ry = Math.max(58, height * 0.30);
      const theta = -Math.PI / 2 + 2 * Math.PI * (x / DATA.L);
      return [cx + rx * Math.cos(theta), cy + ry * Math.sin(theta)];
    }}

    function drawRing() {{
      const {{ ctx, width, height }} = setupCanvas(ringCanvas);
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

      for (let tick = 0; tick < 12; tick++) {{
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
      }}

      const positions = DATA.positions[frame];
      const velocities = DATA.velocities[frame];
      for (let i = 0; i < positions.length; i++) {{
        const [px, py] = pointOnRing(positions[i], width, height);
        ctx.beginPath();
        ctx.arc(px, py, 8.5, 0, 2 * Math.PI);
        ctx.fillStyle = colorForVelocity(velocities[i]);
        ctx.fill();
        ctx.strokeStyle = "#17202a";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }}
    }}

    function drawSpeed() {{
      const {{ ctx, width, height }} = setupCanvas(speedCanvas);
      ctx.clearRect(0, 0, width, height);
      const padL = 42, padR = 14, padT = 18, padB = 34;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const vmax = Math.max(1.0, ...DATA.meanVelocity) * 1.15;

      ctx.strokeStyle = "#d7dce2";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {{
        const y = padT + plotH * g / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
      }}

      ctx.beginPath();
      for (let i = 0; i <= frame; i++) {{
        const x = padL + plotW * i / (DATA.meanVelocity.length - 1);
        const y = padT + plotH * (1 - DATA.meanVelocity[i] / vmax);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
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
    }}

    function drawSpaceTime() {{
      const {{ ctx, width, height }} = setupCanvas(spaceCanvas);
      ctx.clearRect(0, 0, width, height);
      const padL = 30, padR = 12, padT = 12, padB = 24;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const start = Math.max(0, frame - 95);
      const rows = Math.max(1, frame - start);

      ctx.strokeStyle = "#edf0f2";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {{
        const x = padL + plotW * g / 4;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }}

      for (let f = start; f <= frame; f++) {{
        const y = padT + plotH * (f - start) / rows;
        for (let i = 0; i < DATA.positions[f].length; i++) {{
          const x = padL + plotW * DATA.positions[f][i] / DATA.L;
          ctx.fillStyle = colorForVelocity(DATA.velocities[f][i]);
          ctx.fillRect(x, y, 1.8, 1.8);
        }}
      }}

      ctx.fillStyle = "#64707d";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("x = 0", padL, height - 7);
      ctx.fillText(`x = ${{DATA.L.toFixed(1)}} m`, width - padR - 72, height - 7);
    }}

    function updateStats() {{
      frameSlider.value = frame;
      timeLabel.textContent = `t = ${{DATA.time[frame].toFixed(2)}} s`;
      velocityStat.textContent = DATA.meanVelocity[frame].toFixed(2);
    }}

    function draw() {{
      drawRing();
      drawSpeed();
      drawSpaceTime();
      updateStats();
    }}

    function animate(timestamp) {{
      if (!lastTick) lastTick = timestamp;
      const delta = timestamp - lastTick;
      lastTick = timestamp;
      if (playing) {{
        accumulator += delta * playback;
        while (accumulator >= DATA.intervalMs) {{
          frame = (frame + 1) % DATA.positions.length;
          accumulator -= DATA.intervalMs;
        }}
      }}
      draw();
      requestAnimationFrame(animate);
    }}

    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    frameSlider.addEventListener("input", () => {{
      frame = Number(frameSlider.value);
      playing = false;
      playButton.textContent = "Play";
      draw();
    }});
    speedSelect.addEventListener("change", () => {{
      playback = Number(speedSelect.value);
    }});
    window.addEventListener("resize", draw);
    draw();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def write_html_animation(args) -> Path:
    n, actual_density, positions, velocities, sim_time = simulate_demo(args)
    payload = build_payload(args, n, actual_density, positions, velocities, sim_time)
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
    parser.add_argument("--model", choices=["remote", "hardbody"], default="remote")
    parser.add_argument("--density", type=float, default=1.21, help="target density [1/m]")
    parser.add_argument("--b", type=float, default=0.0, help="required-length velocity parameter [s]")
    parser.add_argument("--L", type=float, default=17.3, help="ring length [m]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relax-steps", type=int, default=8_000)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--steps-per-frame", type=int, default=25)
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
