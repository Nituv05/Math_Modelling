#!/usr/bin/env python3
"""Validation checks for the Seyfried-Steffen-Lippert reproduction.

This script is intentionally diagnostic: it reports quantitative checks that
were missing from the minimal reproduction instead of hard-failing on one
paper-specific number.  Use ``--preset paper`` for the expensive settings
closest to the publication.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import add_common_args, resolve_steps
from pedestrian import (
    ModelParameters,
    HardBodyModel,
    RemoteActionModel,
    fundamental_diagram,
    empirical_velocity_from_required_length,
    rmse_against_empirical,
    run_single,
)


def _results_to_arrays(results):
    rho = np.array([r.density for r in results], dtype=float)
    velocity = np.array([r.mean_velocity for r in results], dtype=float)
    return rho, velocity


def empirical_overlay_check(relax_steps, measure_steps, seed, L):
    densities = np.array([0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])
    params = ModelParameters(a=0.36, b=0.56)
    results = fundamental_diagram(
        HardBodyModel,
        params,
        L=L,
        densities=densities,
        relax_steps=relax_steps,
        measure_steps=measure_steps,
        seed=seed,
        progress=False,
    )
    rho, velocity = _results_to_arrays(results)
    reference = empirical_velocity_from_required_length(rho)
    return {
        "name": "empirical_reference_rmse",
        "model": "HardBodyModel(b=0.56)",
        "L": L,
        "rmse_m_per_s": rmse_against_empirical(rho, velocity),
        "points": [
            {
                "density": float(r),
                "simulation_velocity": float(v),
                "empirical_reference_velocity": float(ref),
                "delta": float(v - ref),
            }
            for r, v, ref in zip(rho, velocity, reference)
        ],
    }


def finite_size_check(relax_steps, measure_steps, seed):
    densities = np.array([0.8, 1.4, 2.0])
    lengths = [17.3, 20.0, 50.0]
    params = ModelParameters(a=0.36, b=0.56)
    rows = []
    for rho in densities:
        velocities = []
        for i, L in enumerate(lengths):
            n = max(1, int(round(rho * L)))
            res = run_single(
                HardBodyModel,
                n=n,
                L=L,
                params=params,
                relax_steps=relax_steps,
                measure_steps=measure_steps,
                seed=seed + i,
            )
            velocities.append(res.mean_velocity)
            rows.append(
                {
                    "target_density": float(rho),
                    "L": float(L),
                    "n": int(n),
                    "actual_density": float(res.density),
                    "mean_velocity": float(res.mean_velocity),
                }
            )
        rows.append(
            {
                "target_density": float(rho),
                "summary": "velocity_span_across_lengths",
                "span_m_per_s": float(np.max(velocities) - np.min(velocities)),
            }
        )
    return {"name": "finite_size", "model": "HardBodyModel(b=0.56)", "rows": rows}


def remote_action_gap_check(relax_steps, measure_steps, seed, L):
    params = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)
    target_densities = [1.16, 1.21]
    points = []
    for i, rho in enumerate(target_densities):
        n = max(1, int(round(rho * L)))
        res = run_single(
            RemoteActionModel,
            n=n,
            L=L,
            params=params,
            relax_steps=relax_steps,
            measure_steps=measure_steps,
            seed=seed + i,
        )
        points.append(
            {
                "target_density": float(rho),
                "actual_density": float(res.density),
                "n": int(n),
                "mean_velocity": float(res.mean_velocity),
                "std_velocity": float(res.std_velocity),
            }
        )
    return {
        "name": "remote_action_velocity_gap",
        "model": "RemoteActionModel(b=0)",
        "points": points,
        "velocity_drop_m_per_s": float(points[0]["mean_velocity"] - points[1]["mean_velocity"]),
    }


def regularization_sensitivity_check(relax_steps, measure_steps, seed, L):
    floors = [1e-5, 1e-6, 1e-7]
    rows = []
    rho = 1.21
    n = int(round(rho * L))
    for floor in floors:
        params = ModelParameters(
            a=0.36,
            b=0.0,
            e=0.07,
            f=2.0,
            clear_distance_floor=floor,
        )
        res = run_single(
            RemoteActionModel,
            n=n,
            L=L,
            params=params,
            relax_steps=relax_steps,
            measure_steps=measure_steps,
            seed=seed,
        )
        rows.append(
            {
                "clear_distance_floor": floor,
                "density": float(res.density),
                "mean_velocity": float(res.mean_velocity),
            }
        )
    velocities = [row["mean_velocity"] for row in rows]
    return {
        "name": "remote_force_regularization_sensitivity",
        "model": "RemoteActionModel(b=0)",
        "rows": rows,
        "span_m_per_s": float(np.max(velocities) - np.min(velocities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    relax_steps, measure_steps = resolve_steps(args)

    report = {
        "settings": {
            "relax_steps": relax_steps,
            "measure_steps": measure_steps,
            "L": args.L,
            "seed": args.seed,
        },
        "checks": [
            empirical_overlay_check(relax_steps, measure_steps, args.seed, args.L),
            finite_size_check(relax_steps, measure_steps, args.seed),
            remote_action_gap_check(relax_steps, measure_steps, args.seed, args.L),
            regularization_sensitivity_check(relax_steps, measure_steps, args.seed, args.L),
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("Validation report")
    print(json.dumps(report["settings"], indent=2))
    for check in report["checks"]:
        print(f"\n[{check['name']}]")
        print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
