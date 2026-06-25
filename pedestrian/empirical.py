"""Empirical-data helpers for the single-file fundamental diagram.

The modelling paper overlays measured single-file data points in Fig. 1.
Those points are not a fitted curve.  The bundled CSV stores the marker
locations digitized from the paper's Fig. 1 EPS source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMPIRICAL_POINTS_CSV = DATA_DIR / "empirical_single_file_points.csv"
EMPIRICAL_SOURCE = "Seyfried, Steffen, Lippert 2005 arXiv:physics/0506189 Fig. 1"
EMPIRICAL_NOTES = (
    "Digitized from Fig. 1 EPS markers; the paper shows data points, "
    "not an empirical fitted curve."
)


@dataclass(frozen=True)
class EmpiricalPoint:
    density: float
    velocity: float

    @property
    def inverse_density(self) -> float:
        return 1.0 / self.density


def load_empirical_points(path: Path = EMPIRICAL_POINTS_CSV) -> list[EmpiricalPoint]:
    """Load the bundled empirical data points."""
    points: list[EmpiricalPoint] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                EmpiricalPoint(
                    density=float(row["density_1_per_m"]),
                    velocity=float(row["velocity_m_per_s"]),
                )
            )
    return points

def empirical_mean_velocity_near_density(
    densities,
    half_width: float = 0.075,
    points: list[EmpiricalPoint] | None = None,
):
    """Mean empirical velocity in a density window around each density.

    Returns ``nan`` where the digitized data contain no point in the window.
    """
    import numpy as np

    rho = np.asarray(densities, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")

    empirical = points if points is not None else load_empirical_points()
    empirical_rho = np.array([p.density for p in empirical], dtype=float)
    empirical_v = np.array([p.velocity for p in empirical], dtype=float)

    means = np.full(rho.shape, np.nan, dtype=float)
    for idx, density in np.ndenumerate(rho):
        mask = np.abs(empirical_rho - density) <= half_width
        if mask.any():
            means[idx] = float(np.mean(empirical_v[mask]))
    return means


def rmse_against_empirical(densities, velocities, half_width: float = 0.075) -> float:
    """Root-mean-square error against nearby empirical data points."""
    import numpy as np

    rho = np.asarray(densities, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if rho.shape != v.shape:
        raise ValueError("densities and velocities must have the same shape")
    ref = empirical_mean_velocity_near_density(rho, half_width=half_width)
    valid = ~np.isnan(ref)
    if not valid.any():
        raise ValueError("no empirical data points found near the supplied densities")
    return float(np.sqrt(np.mean((v[valid] - ref[valid]) ** 2)))
