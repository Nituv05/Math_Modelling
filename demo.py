#!/usr/bin/env python3
"""Small no-plot demo for the pedestrian-flow model.

Run:

    python3 demo.py

It uses short runs so the result is a smoke demonstration, not a paper-grade
reproduction.  Use the scripts in ``scripts/`` with ``--preset paper`` for
the full settings.
"""

from __future__ import annotations

import numpy as np

from pedestrian import (
    ModelParameters,
    HardBodyModel,
    RemoteActionModel,
    fundamental_diagram,
    empirical_mean_velocity_near_density,
    rmse_against_empirical,
)


def print_curve(name, model_cls, params, densities):
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


def main() -> None:
    densities = [0.6, 1.0, 1.4, 1.8, 2.2]
    print_curve(
        "Hard bodies without remote action, b=0.56",
        HardBodyModel,
        ModelParameters(a=0.36, b=0.56),
        densities,
    )
    print_curve(
        "Remote action, b=0",
        RemoteActionModel,
        ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0),
        densities,
    )


if __name__ == "__main__":
    main()
