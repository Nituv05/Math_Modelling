#!/usr/bin/env python3
"""
Figure 3 of the paper: density waves near the velocity gap.

This is not another velocity-density diagram and it has no empirical overlay.
It is a space-time plot of pedestrian positions ``x(t)`` for the Eq. 6
remote-action model with velocity-independent required length (``b = 0``).
The two panels reproduce the paper's comparison just below and just above
the velocity gap from Fig. 2:

    rho ~= 1.16 1/m   below the gap, no strong density wave
    rho ~= 1.21 1/m   above the gap, density waves become visible

Time is plotted downward, matching the paper.  One pedestrian trajectory is
highlighted in black to make the individual motion easier to follow.

Caption:

    FIG. 3: Time-development of the positions for densities near the
    velocity-gap, see Figure 2. For rho > 1.2 m^-1 density waves are
    observable. Some individuals leave much larger than average gaps in front.

Run::

    python scripts/run_figure3.py
    python scripts/run_figure3.py --preset paper
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from _common import add_common_args, resolve_steps, FIG_DIR
from pedestrian import ModelParameters, RemoteActionModel


TARGET_DENSITIES = [
    (1.16, "below gap"),
    (1.21, "above gap: density waves"),
]


def break_at_periodic_wrap(x: np.ndarray, L: float) -> np.ndarray:
    """Insert NaNs where a trajectory wraps around the periodic boundary."""
    x_plot = x.astype(float).copy()
    wrapped = np.abs(np.diff(x_plot)) > (0.5 * L)
    x_plot[np.where(wrapped)[0] + 1] = np.nan
    return x_plot


def run_with_highlight(n, L, params, relax_steps, measure_steps, seed, stride, highlight_id):
    model = RemoteActionModel(n=n, L=L, params=params, seed=seed)
    highlight_id = highlight_id % n

    for _ in range(relax_steps):
        model.step()

    traj = []
    highlighted = []
    for k in range(measure_steps):
        model.step()
        if k % stride == 0:
            traj.append(model.x.copy())
            highlight_idx = int(np.flatnonzero(model.ids == highlight_id)[0])
            highlighted.append(model.x[highlight_idx])
    return np.array(traj), np.array(highlighted), highlight_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--highlight-ped", type=int, default=0,
                        help="pedestrian index to highlight in black (default: 0)")
    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    # Record a coarse-grained trajectory; we don't need every 1ms frame.
    stride = max(1, measure // 1500)
    params = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)

    fig, axes = plt.subplots(1, len(TARGET_DENSITIES), figsize=(11, 5.4),
                             sharex=True, sharey=True)

    for ax, (rho, panel_note) in zip(np.atleast_1d(axes), TARGET_DENSITIES):
        n = int(round(rho * args.L))
        actual_rho = n / args.L
        print(f"rho={actual_rho:.3f} 1/m  (N={n})")
        traj, highlighted, highlight = run_with_highlight(
            n=n,
            L=args.L,
            params=params,
            relax_steps=relax,
            measure_steps=measure,
            seed=args.seed,
            stride=stride,
            highlight_id=args.highlight_ped,
        )
        # traj: shape (frames, N).  Plot position vs time, with time downward.
        frames = traj.shape[0]
        t = np.arange(frames) * stride * 0.001  # seconds
        for ped in range(traj.shape[1]):
            ax.plot(traj[:, ped], t, ",", color="0.45", alpha=0.45)
        ax.plot(
            break_at_periodic_wrap(highlighted, args.L),
            t,
            "-",
            color="black",
            linewidth=1.0,
            alpha=1.0,
            label=f"highlighted pedestrian {highlight}",
        )
        ax.plot(highlighted, t, ".", color="black", markersize=1.8)
        ax.set_title(rf"$\rho$ = {actual_rho:.2f} [1/m] ({panel_note})")
        ax.set_xlabel("x  [m]")
        ax.set_xlim(0, args.L)
        ax.invert_yaxis()

    np.atleast_1d(axes)[0].set_ylabel("t  [s]  (downward)")
    caption = (
        "FIG. 3: Time-development of the positions for densities near\n"
        "the velocity-gap, see Figure 2. For $\\rho > 1.2\\,m^{-1}$ density waves\n"
        "are observable. Some individuals leave much larger than average gaps in front."
    )
    fig.text(0.07, 0.025, caption, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    out = f"{FIG_DIR}/figure3_density_waves.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
