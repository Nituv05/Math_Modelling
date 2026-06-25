#!/usr/bin/env python3
"""Run appendix sensitivity experiments from the defense context.

The experiments follow the "baseline reproduction + one-factor-at-a-time"
logic:

* Group A: vary b at fixed a, tau, L.
* Group B: vary a at fixed b, tau, L.
* Group C: vary tau and record both final diagram points and transient traces.
* Group D: sample N values to show that rho = N/L is the control variable.
* Group E: vary L while keeping target density approximately fixed.

Results are written to a separate output directory as CSV, JSON, and PNG files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the package importable when the script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import add_common_args, resolve_steps
from pedestrian import (
    HardBodyModel,
    ModelParameters,
    empirical_mean_velocity_near_density,
    fundamental_diagram,
    load_empirical_points,
    rmse_against_empirical,
    run_single,
)


BASELINE_A = 0.36
BASELINE_B = 0.56
BASELINE_TAU = 0.61
BASELINE_L = 17.3
COUNTS_BY_REGIME = [5, 10, 15, 20, 21, 25, 30, 35, 40, 45, 48]
B_VALUES = [0.0, 0.25, 0.56, 0.8, 1.06, 1.3]
A_VALUES = [0.30, 0.36, 0.45]
TAU_VALUES = [0.3, 0.61, 1.0]
FINITE_SIZE_LENGTHS = [17.3, 20.0, 50.0]
FINITE_SIZE_DENSITIES = [0.6, 1.2, 2.0]


def feasible_counts(counts: list[int], L: float, a: float) -> list[int]:
    """Keep only counts that can be initialized with minimum distance a."""
    n_max = int(L / a)
    return [n for n in counts if n <= n_max]


def result_row(group: str, varied_parameter: str, varied_value: float, result, params):
    return {
        "group": group,
        "model": "HardBodyModel",
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        "a": params.a,
        "b": params.b,
        "tau": params.tau,
        "L": result.L,
        "N": result.n,
        "density": result.density,
        "mean_velocity": result.mean_velocity,
        "std_velocity": result.std_velocity,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_rmse(rows: list[dict]) -> float | None:
    rho = np.array([row["density"] for row in rows], dtype=float)
    v = np.array([row["mean_velocity"] for row in rows], dtype=float)
    try:
        return rmse_against_empirical(rho, v)
    except ValueError:
        return None


def add_empirical_overlay(ax) -> None:
    empirical = load_empirical_points()
    ax.scatter(
        [p.density for p in empirical],
        [p.velocity for p in empirical],
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=18,
        linewidths=0.8,
        label="empirical data",
    )


def plot_group_lines(path: Path, rows: list[dict], parameter: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    add_empirical_overlay(ax)
    for value in sorted({row["varied_value"] for row in rows}):
        subset = [row for row in rows if row["varied_value"] == value]
        subset.sort(key=lambda row: row["density"])
        ax.plot(
            [row["density"] for row in subset],
            [row["mean_velocity"] for row in subset],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"{parameter} = {value:g}",
        )
    ax.set_xlabel(r"$\rho$ [1/m]")
    ax.set_ylabel(r"$\bar v$ [m/s]")
    ax.set_title(title)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.4)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transient(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for tau in sorted({row["tau"] for row in rows}):
        subset = [row for row in rows if row["tau"] == tau]
        subset.sort(key=lambda row: row["time_s"])
        ax.plot(
            [row["time_s"] for row in subset],
            [row["mean_velocity"] for row in subset],
            linewidth=1.2,
            label=rf"$\tau$ = {tau:g} s",
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\bar v(t)$ [m/s]")
    ax.set_title("Group C - transient mean velocity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_finite_size(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for L in sorted({row["L"] for row in rows}):
        subset = [row for row in rows if row["L"] == L]
        subset.sort(key=lambda row: row["target_density"])
        ax.plot(
            [row["actual_density"] for row in subset],
            [row["mean_velocity"] for row in subset],
            marker="o",
            markersize=5,
            linewidth=1.2,
            label=f"L = {L:g} m",
        )
    ax.set_xlabel(r"actual $\rho=N/L$ [1/m]")
    ax.set_ylabel(r"$\bar v$ [m/s]")
    ax.set_title("Group E - finite-size check at fixed density")
    ax.set_xlim(0, 2.25)
    ax.set_ylim(0, 1.4)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_diagram_group(
    group: str,
    varied_parameter: str,
    values: list[float],
    L: float,
    counts: list[int],
    relax_steps: int,
    measure_steps: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    for value_index, value in enumerate(values):
        params = ModelParameters(a=BASELINE_A, b=BASELINE_B, tau=BASELINE_TAU)
        setattr(params, varied_parameter, value)
        group_counts = feasible_counts(counts, L=L, a=params.a)
        print(f"{group}: {varied_parameter}={value:g}, counts={group_counts}")
        results = fundamental_diagram(
            HardBodyModel,
            params,
            L=L,
            counts=group_counts,
            relax_steps=relax_steps,
            measure_steps=measure_steps,
            seed=seed + 1000 * value_index,
            progress=True,
        )
        value_rows = [
            result_row(group, varied_parameter, value, result, params)
            for result in results
        ]
        rows.extend(value_rows)
        rmse = safe_rmse(value_rows)
        summaries.append(
            {
                "group": group,
                "varied_parameter": varied_parameter,
                "varied_value": value,
                "runs": len(value_rows),
                "rmse_against_nearby_empirical_m_per_s": rmse,
                "min_velocity": min(row["mean_velocity"] for row in value_rows),
                "max_velocity": max(row["mean_velocity"] for row in value_rows),
                "max_density": max(row["density"] for row in value_rows),
            }
        )
    return rows, summaries


def run_tau_trace(
    tau_values: list[float],
    n: int,
    L: float,
    total_steps: int,
    dt: float,
    stride: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for tau_index, tau in enumerate(tau_values):
        params = ModelParameters(a=BASELINE_A, b=BASELINE_B, tau=tau)
        model = HardBodyModel(n=n, L=L, params=params, dt=dt, seed=seed + tau_index)
        print(f"Group C trace: tau={tau:g}, N={n}, steps={total_steps}")
        for step in range(total_steps + 1):
            if step % stride == 0:
                rows.append(
                    {
                        "group": "C_tau_transient",
                        "model": "HardBodyModel",
                        "a": params.a,
                        "b": params.b,
                        "tau": params.tau,
                        "L": L,
                        "N": n,
                        "density": n / L,
                        "step": step,
                        "time_s": step * dt,
                        "mean_velocity": float(model.v.mean()),
                    }
                )
            if step < total_steps:
                model.step()
    return rows


def run_density_count_group(
    L: float,
    counts: list[int],
    relax_steps: int,
    measure_steps: int,
    seed: int,
) -> list[dict]:
    params = ModelParameters(a=BASELINE_A, b=BASELINE_B, tau=BASELINE_TAU)
    rows: list[dict] = []
    for i, n in enumerate(feasible_counts(counts, L=L, a=params.a)):
        print(f"Group D: N={n}, rho={n / L:.3f}")
        result = run_single(
            HardBodyModel,
            n=n,
            L=L,
            params=params,
            relax_steps=relax_steps,
            measure_steps=measure_steps,
            seed=seed + i,
        )
        row = result_row("D_N_rho", "N", float(n), result, params)
        nearby = empirical_mean_velocity_near_density([result.density])[0]
        row["nearby_empirical_mean_velocity"] = None if np.isnan(nearby) else float(nearby)
        rows.append(row)
    return rows


def run_finite_size_group(
    target_densities: list[float],
    lengths: list[float],
    relax_steps: int,
    measure_steps: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    params = ModelParameters(a=BASELINE_A, b=BASELINE_B, tau=BASELINE_TAU)
    rows: list[dict] = []
    summaries: list[dict] = []
    for rho_index, target_density in enumerate(target_densities):
        velocities = []
        for length_index, L in enumerate(lengths):
            n = int(round(target_density * L))
            if n * params.a > L:
                continue
            print(f"Group E: target rho={target_density:g}, L={L:g}, N={n}")
            result = run_single(
                HardBodyModel,
                n=n,
                L=L,
                params=params,
                relax_steps=relax_steps,
                measure_steps=measure_steps,
                seed=seed + rho_index * 100 + length_index,
            )
            rows.append(
                {
                    "group": "E_finite_size",
                    "model": "HardBodyModel",
                    "a": params.a,
                    "b": params.b,
                    "tau": params.tau,
                    "target_density": target_density,
                    "L": result.L,
                    "N": result.n,
                    "actual_density": result.density,
                    "mean_velocity": result.mean_velocity,
                    "std_velocity": result.std_velocity,
                }
            )
            velocities.append(result.mean_velocity)
        summaries.append(
            {
                "target_density": target_density,
                "velocity_span_across_lengths_m_per_s": (
                    float(np.max(velocities) - np.min(velocities)) if velocities else None
                ),
            }
        )
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("appendix_experiment_results"),
        help="directory for CSV/JSON/PNG outputs",
    )
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=500,
        help="record every N steps for tau transient traces",
    )
    args = parser.parse_args()

    relax_steps, measure_steps = resolve_steps(args)
    total_trace_steps = relax_steps + measure_steps
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    settings = {
        "preset": args.preset,
        "relax_steps": relax_steps,
        "measure_steps": measure_steps,
        "dt": 0.001,
        "L": args.L,
        "seed": args.seed,
        "counts_by_regime": COUNTS_BY_REGIME,
        "output_dir": str(out_dir),
    }

    group_a_rows, group_a_summary = run_diagram_group(
        "A_b_sensitivity",
        "b",
        B_VALUES,
        args.L,
        COUNTS_BY_REGIME,
        relax_steps,
        measure_steps,
        args.seed,
    )
    write_csv(out_dir / "group_a_b_sensitivity.csv", group_a_rows)
    plot_group_lines(
        out_dir / "group_a_b_sensitivity.png",
        group_a_rows,
        "b",
        "Group A - effect of velocity-dependent required space",
    )

    group_b_rows, group_b_summary = run_diagram_group(
        "B_a_sensitivity",
        "a",
        A_VALUES,
        args.L,
        COUNTS_BY_REGIME,
        relax_steps,
        measure_steps,
        args.seed + 10_000,
    )
    write_csv(out_dir / "group_b_a_sensitivity.csv", group_b_rows)
    plot_group_lines(
        out_dir / "group_b_a_sensitivity.png",
        group_b_rows,
        "a",
        "Group B - effect of minimum body depth",
    )

    group_c_rows, group_c_summary = run_diagram_group(
        "C_tau_sensitivity",
        "tau",
        TAU_VALUES,
        args.L,
        [10, 20, 30, 40],
        relax_steps,
        measure_steps,
        args.seed + 20_000,
    )
    write_csv(out_dir / "group_c_tau_sensitivity.csv", group_c_rows)
    plot_group_lines(
        out_dir / "group_c_tau_sensitivity.png",
        group_c_rows,
        "tau",
        "Group C - effect of relaxation time",
    )

    trace_rows = run_tau_trace(
        TAU_VALUES,
        n=20,
        L=args.L,
        total_steps=total_trace_steps,
        dt=0.001,
        stride=args.trace_stride,
        seed=args.seed + 30_000,
    )
    write_csv(out_dir / "group_c_tau_transient.csv", trace_rows)
    plot_transient(out_dir / "group_c_tau_transient.png", trace_rows)

    group_d_rows = run_density_count_group(
        args.L,
        COUNTS_BY_REGIME,
        relax_steps,
        measure_steps,
        args.seed + 40_000,
    )
    write_csv(out_dir / "group_d_n_rho.csv", group_d_rows)

    group_e_rows, group_e_summary = run_finite_size_group(
        FINITE_SIZE_DENSITIES,
        FINITE_SIZE_LENGTHS,
        relax_steps,
        measure_steps,
        args.seed + 50_000,
    )
    write_csv(out_dir / "group_e_finite_size.csv", group_e_rows)
    plot_finite_size(out_dir / "group_e_finite_size.png", group_e_rows)

    all_rows = group_a_rows + group_b_rows + group_c_rows + group_d_rows
    write_csv(out_dir / "all_fundamental_diagram_runs.csv", all_rows)

    summary = {
        "settings": settings,
        "elapsed_seconds": time.time() - started,
        "groups": {
            "A_b_sensitivity": group_a_summary,
            "B_a_sensitivity": group_b_summary,
            "C_tau_sensitivity": group_c_summary,
            "E_finite_size": group_e_summary,
        },
        "files": sorted(path.name for path in out_dir.iterdir()),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"saved appendix experiment results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
