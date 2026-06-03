#!/usr/bin/env python3
"""
Figure 1 of the paper.

Velocity-density relation for *hard bodies without remote action* (Eq. 5),
with ``a = 0.36 m`` and three values of the velocity dependence of the
required length:

    b = 0.00 s   (simple hard bodies, density-independent personal space)
    b = 0.56 s   (good agreement with empirical fundamental diagram)
    b = 1.06 s   (value reported empirically in Ref. [25])

Run::

    python scripts/run_figure1.py                 # default preset
    python scripts/run_figure1.py --preset paper  # full 3e5+3e5 steps
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from _common import add_common_args, resolve_steps, FIG_DIR
from pedestrian import ModelParameters, HardBodyModel, fundamental_diagram


B_VALUES = [0.0, 0.56, 1.06]
MARKERS = {0.0: "s", 0.56: "o", 1.06: "^"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    relax, measure = resolve_steps(args)

    densities = np.linspace(0.2, 2.9, 25)

    fig, ax = plt.subplots(figsize=(7, 5))
    for b in B_VALUES:
        print(f"hard-body model, b = {b} s")
        params = ModelParameters(a=0.36, b=b)
        results = fundamental_diagram(
            HardBodyModel, params, L=args.L, densities=densities,
            relax_steps=relax, measure_steps=measure, seed=args.seed,
        )
        rho = [r.density for r in results]
        v = [r.mean_velocity for r in results]
        ax.plot(rho, v, MARKERS[b], label=f"b = {b:.2f} s", markersize=5)

    ax.set_xlabel(r"$\rho$  [1/m]")
    ax.set_ylabel(r"$v$  [m/s]")
    ax.set_title("Fig. 1 — Hard bodies without remote action (Eq. 5)")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.4)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = f"{FIG_DIR}/figure1_hardbody.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
