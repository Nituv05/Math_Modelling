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

Time is plotted downward, matching the paper.  All pedestrians are shown as
filled black dots.  For rho > 1.2 m^-1 the nearly vertical stacks of dots
are the density waves discussed in the caption.

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


def run_space_time(n, L, params, relax_steps, measure_steps, seed, stride):
    model = RemoteActionModel(n=n, L=L, params=params, seed=seed)

    for _ in range(relax_steps):
        model.step()

    traj = []
    for k in range(measure_steps):
        model.step()
        if k % stride == 0:
            traj.append(model.x.copy())
    return np.array(traj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.set_defaults(seed=12)
    parser.add_argument("--plot-seconds", type=float, default=6.0,
                        help="time window shown after relaxation, in seconds (default: %(default)s)")
    parser.add_argument("--plot-frames", type=int, default=45,
                        help="number of sampled time rows in the plot (default: %(default)s)")
    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    # Fig. 3 in the paper shows a short space-time window after relaxation,
    # not the full 3e5 measurement interval used for the fundamental diagram.
    # The paper notes that density waves near the gap depend on the
    # distribution of individual velocities; seed=12 gives the same qualitative
    # contrast as the published figure: rho=1.16 remains almost homogeneous,
    # while rho=1.21 develops stopped density-wave stacks.
    plot_steps = min(measure, max(1, int(round(args.plot_seconds / 0.001))))
    stride = max(1, plot_steps // args.plot_frames)
    params = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)

    fig, axes = plt.subplots(len(TARGET_DENSITIES), 1, figsize=(3.1, 5.85),
                             sharex=True, sharey=True)

    for ax, (rho, panel_note) in zip(np.atleast_1d(axes), TARGET_DENSITIES):
        n = int(round(rho * args.L))
        actual_rho = n / args.L
        print(f"rho={actual_rho:.3f} 1/m  (N={n})")
        traj = run_space_time(
            n=n,
            L=args.L,
            params=params,
            relax_steps=relax,
            measure_steps=plot_steps,
            seed=args.seed,
            stride=stride,
        )
        # traj: shape (frames, N).  Fig. 3 plots all positions as identical
        # black dots; the visible vertical stacks are stopped density waves.
        frames = traj.shape[0]
        t = np.arange(frames) * stride * 0.001  # seconds
        ax.plot(
            traj.ravel(),
            np.repeat(t, traj.shape[1]),
            ".",
            color="black",
            markersize=2.0,
            linestyle="None",
        )
        ax.set_title(rf"$\rho$={actual_rho:.2f} [1/m]", fontsize=8)
        ax.set_xlabel("x", fontsize=8)
        ax.set_xlim(0, args.L)
        ax.set_xticks(np.arange(0, args.L, 2.0))
        ax.set_yticks([])
        ax.set_ylabel(r"$\leftarrow t$", rotation=90, fontsize=8)
        ax.tick_params(labelsize=7, length=2.5)
        ax.set_box_aspect(1.0)
        ax.invert_yaxis()

    caption = (
        "FIG. 3: Time-development of the positions for densities near\n"
        "the velocity-gap, see Figure 2. For $\\rho > 1.2\\,m^{-1}$ density waves\n"
        "are observable. Some individuals leave much larger than average gaps in front."
    )
    fig.text(0.04, 0.025, caption, ha="left", va="bottom", fontsize=6.8)
    fig.tight_layout(rect=(0, 0.14, 1, 1), h_pad=1.4)

    out = f"{FIG_DIR}/figure3_density_waves.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
