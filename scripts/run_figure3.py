#!/usr/bin/env python3
"""
Figure 3 of the paper.

Space-time development of pedestrian positions for the remote-action model
(Eq. 6) with ``b = 0`` at densities near the velocity gap.  For
``rho > 1.2 1/m`` distinct density waves ("stop-and-go") become visible:
some individuals leave much larger gaps in front than average.

Two densities are shown, as in the paper (rho ~ 1.16 and ~ 1.21 1/m).

Run::

    python scripts/run_figure3.py
    python scripts/run_figure3.py --preset paper
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from _common import add_common_args, resolve_steps, FIG_DIR
from pedestrian import ModelParameters, RemoteActionModel, run_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    # Record a coarse-grained trajectory; we don't need every 1ms frame.
    stride = max(1, measure // 1500)
    params = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)

    target_densities = [1.16, 1.21]
    fig, axes = plt.subplots(1, len(target_densities), figsize=(11, 5),
                             sharex=True, sharey=True)

    for ax, rho in zip(np.atleast_1d(axes), target_densities):
        n = int(round(rho * args.L))
        actual_rho = n / args.L
        print(f"rho={actual_rho:.3f} 1/m  (N={n})")
        _, traj = run_single(
            RemoteActionModel, n=n, L=args.L, params=params,
            relax_steps=relax, measure_steps=measure, seed=args.seed,
            record_trajectory=True, trajectory_stride=stride,
        )
        # traj: shape (frames, N).  Plot position vs time as a scatter, like
        # the paper's space-time diagram (time runs downward).
        frames = traj.shape[0]
        t = np.arange(frames) * stride * 0.001  # seconds
        for ped in range(traj.shape[1]):
            ax.plot(traj[:, ped], t, ",", color="black", alpha=0.5)
        ax.set_title(rf"$\rho$ = {actual_rho:.2f} 1/m")
        ax.set_xlabel("x  [m]")
        ax.invert_yaxis()

    np.atleast_1d(axes)[0].set_ylabel("t  [s]  (downward)")
    fig.suptitle("Fig. 3 — Space-time positions (remote action, b=0): density waves")
    fig.tight_layout()

    out = f"{FIG_DIR}/figure3_density_waves.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
