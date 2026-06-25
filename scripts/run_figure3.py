#!/usr/bin/env python3
"""
Reproduce Figure 3 of the pedestrian-flow paper.

Figure 3 is a space-time plot of pedestrian positions x(t), not a
velocity-density diagram.

Paper-style choices:
- remote-action model, Eq. 6
- velocity-independent required length: b = 0
- densities near the velocity gap: rho ~= 1.16 and rho ~= 1.21
- open circular markers for all pedestrians
- one pedestrian trajectory highlighted by filled black circular markers
- time runs downward
- only a 20-second simulation window is displayed
- displayed vertical axis is rescaled to 0..16, like the paper-style figure
- caption is not embedded; keep caption in LaTeX/report

Run:

    python scripts/run_figure3.py --preset paper

Optional:

    python scripts/run_figure3.py --preset paper --layout vertical
    python scripts/run_figure3.py --preset paper --highlight-ped 7
    python scripts/run_figure3.py --preset paper --x-label L
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _common import add_common_args, resolve_steps, FIG_DIR
from pedestrian import ModelParameters, RemoteActionModel


DT = 0.001  # seconds, paper uses Delta t = 0.001 s

TARGET_DENSITIES = [
    (1.16, "below gap"),
    (1.21, "above gap"),
]


def unwrap_periodic_trajectory(x: np.ndarray, L: float) -> np.ndarray:
    """
    Unwrap one periodic trajectory so velocity/slope changes are computed correctly.

    This is only for scoring the highlighted pedestrian.
    The plotted data still use the original periodic positions.
    """
    xu = x.astype(float).copy()

    for k in range(1, len(xu)):
        dx = xu[k] - xu[k - 1]

        if dx < -0.5 * L:
            xu[k:] += L
        elif dx > 0.5 * L:
            xu[k:] -= L

    return xu


def choose_most_zigzag_ped(traj_by_id: np.ndarray, dt_sample: float, L: float) -> int:
    """
    Choose the pedestrian with the clearest stop-and-go / kinked trajectory.

    traj_by_id shape: (frames, N)
    Each column is the position history of one fixed pedestrian ID.
    """
    best_id = 0
    best_score = -np.inf

    for ped_id in range(traj_by_id.shape[1]):
        x = unwrap_periodic_trajectory(traj_by_id[:, ped_id], L)

        if len(x) < 5:
            continue

        v = np.diff(x) / dt_sample

        if len(v) < 4:
            continue

        # Strong changes in slope produce the kinked / stop-and-go look.
        score = np.sum(np.abs(np.diff(v))) + 0.25 * np.std(v)

        if score > best_score:
            best_score = score
            best_id = ped_id

    return best_id


def run_window_simulation(
    n: int,
    L: float,
    params: ModelParameters,
    relax_steps: int,
    window_steps: int,
    stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run relaxation, then record a short display window.

    Returns:
        traj:
            positions in current model order, shape (frames, N)
        traj_by_id:
            positions indexed by fixed pedestrian ID, shape (frames, N)
    """
    model = RemoteActionModel(n=n, L=L, params=params, seed=seed)

    for _ in range(relax_steps):
        model.step()

    traj = []
    traj_by_id = []

    for k in range(window_steps + 1):
        if k % stride == 0:
            traj.append(model.x.copy())

            frame_by_id = np.empty(n, dtype=float)
            frame_by_id[model.ids] = model.x
            traj_by_id.append(frame_by_id)

        model.step()

    return np.array(traj), np.array(traj_by_id)


def make_axes(layout: str):
    if layout == "vertical":
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(4.6, 8.2),
            sharex=True,
            sharey=True,
        )
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 4.8),
            sharex=True,
            sharey=True,
        )

    return fig, np.atleast_1d(axes).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument(
        "--highlight-ped",
        type=int,
        default=-1,
        help="pedestrian ID to highlight; -1 means auto-select the most kinked trajectory",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=20.0,
        help="real simulation time window to display, in seconds",
    )
    parser.add_argument(
        "--display-frames",
        type=int,
        default=150,
        help="number of sampled time rows to show in the 20-second window",
    )
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="horizontal for presentation; vertical is closer to the original TeX stacking",
    )
    parser.add_argument(
        "--x-label",
        default="x",
        help='x-axis label; paper uses "x", use --x-label L for your slide style',
    )

    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    # Fig. 3 uses the remote-action model with b = 0.
    params = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)

    time_window = float(args.time_window)
    window_steps = int(round(time_window / DT))

    # Respect small non-paper presets if measure is shorter than the requested window.
    window_steps = min(window_steps, measure)

    # Choose stride so the displayed window has about display_frames rows.
    display_frames = max(2, int(args.display_frames))
    stride = max(1, int(round(window_steps / (display_frames - 1))))

    dt_sample = stride * DT

    fig, axes = make_axes(args.layout)

    for ax, (rho, _panel_note) in zip(axes, TARGET_DENSITIES):
        n = int(round(rho * args.L))
        actual_rho = n / args.L

        print(f"rho={actual_rho:.3f} 1/m  (N={n})")

        traj, traj_by_id = run_window_simulation(
            n=n,
            L=args.L,
            params=params,
            relax_steps=relax,
            window_steps=window_steps,
            stride=stride,
            seed=args.seed,
        )

        frames = traj_by_id.shape[0]

        # Display coordinate only. This is not seconds.
        # It reproduces the 0..16 paper-style visual scale.
        t_display = np.linspace(0, 16, frames)

        if args.highlight_ped < 0:
            highlight = choose_most_zigzag_ped(traj_by_id, dt_sample, args.L)
        else:
            highlight = args.highlight_ped % n

        # Open circles for all non-highlighted pedestrians.
        for ped in range(n):
            if ped == highlight:
                continue

            ax.plot(
                traj_by_id[:, ped],
                t_display,
                linestyle="None",
                marker="o",
                markersize=3.0,
                markerfacecolor="none",
                markeredgecolor="0.20",
                markeredgewidth=0.55,
                alpha=0.90,
            )

        # Filled black circles for one highlighted trajectory.
        ax.plot(
            traj_by_id[:, highlight],
            t_display,
            linestyle="None",
            marker="o",
            markersize=4.4,
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.50,
            alpha=1.0,
        )

        ax.set_title(rf"$\rho = {actual_rho:.2f}$ [1/m]", fontsize=16)
        ax.set_xlabel(args.x_label, fontsize=14)
        ax.set_xlim(0, args.L)

        # Time runs downward. The numeric scale is display-coordinate 0..16,
        # not physical seconds.
        ax.set_ylim(16, 0)
        ax.set_yticks(np.arange(0, 17, 2))
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("t", fontsize=14)

    fig.tight_layout(w_pad=1.0, h_pad=1.0)

    out_dir = Path(FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_out = out_dir / "figure3_density_waves.png"
    pdf_out = out_dir / "figure3_density_waves.pdf"

    fig.savefig(png_out, dpi=250, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")

    print(f"saved {png_out}")
    print(f"saved {pdf_out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()