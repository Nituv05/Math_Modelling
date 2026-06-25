#!/usr/bin/env python3
"""
Figure 2 of the paper.

Velocity-density relation for *hard bodies with remote action* (Eq. 6),
compared to the hard bodies without remote action.  Remote-force
parameters ``e = 0.07 N`` and ``f = 2``.  Curves:

    without remote action, b = 0.56 s   (filled circles in the paper)
    with    remote action, b = 0.56 s
    with    remote action, b = 0.00 s   (shows the velocity gap / density waves)

Run::

    python scripts/run_figure2.py
    python scripts/run_figure2.py --preset paper
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from _common import add_common_args, resolve_steps, FIG_DIR
from pedestrian import (
    ModelParameters,
    HardBodyModel,
    RemoteActionModel,
    fundamental_diagram,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    densities = np.linspace(0.2, 2.9, 25)
    fig, ax = plt.subplots(figsize=(7, 5))
    cases = [
        ("without remote action, b=0.56", HardBodyModel,
         ModelParameters(a=0.36, b=0.56), "."),
        ("with remote action, b=0.56", RemoteActionModel,
         ModelParameters(a=0.36, b=0.56, e=0.07, f=2.0), "s"),
        ("with remote action, b=0.00", RemoteActionModel,
         ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0), "o"),
    ]

    for label, model_cls, params, marker in cases:
        print(label)
        results = fundamental_diagram(
            model_cls, params, L=args.L, densities=densities,
            relax_steps=relax, measure_steps=measure, seed=args.seed,
        )
        rho = [r.density for r in results]
        v = [r.mean_velocity for r in results]
        ax.plot(rho, v, marker, label=label, markersize=5,
                markerfacecolor="none" if marker != "." else None)

    ax.set_xlabel(r"$\rho$  [1/m]")
    ax.set_ylabel(r"$v$  [m/s]")
    ax.set_title("Fig. 2 — Remote action (Eq. 6) vs. hard bodies")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.4)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = f"{FIG_DIR}/figure2_remote.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
