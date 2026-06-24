"""Empirical-reference helpers for the single-file fundamental diagram.

The modelling paper compares its curves against the single-file experiments
reported in Seyfried et al., "The Fundamental Diagram of Pedestrian Movement
Revisited" (arXiv:physics/0506170).  That paper reports the linear relation

    1 / rho = 0.36 m + 1.06 s * v

between inverse density and velocity.  The bundled CSV stores the curve
generated from that regression, not the raw experimental observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


EMPIRICAL_A = 0.36
EMPIRICAL_B = 1.06
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMPIRICAL_REFERENCE_CSV = DATA_DIR / "empirical_single_file_regression.csv"


@dataclass(frozen=True)
class EmpiricalPoint:
    density: float
    velocity: float
    inverse_density: float
    source: str
    notes: str


def empirical_velocity_from_required_length(
    density: np.ndarray | float,
    a: float = EMPIRICAL_A,
    b: float = EMPIRICAL_B,
) -> np.ndarray:
    """Return the regression reference velocity at density ``rho``.

    Rearranges ``1 / rho = a + b * v`` to ``v = (1 / rho - a) / b`` and
    clips negative velocities to zero.
    """
    rho = np.asarray(density, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    return np.maximum((1.0 / rho - a) / b, 0.0)


def load_empirical_reference(path: Path = EMPIRICAL_REFERENCE_CSV) -> list[EmpiricalPoint]:
    """Load the bundled regression reference points."""
    points: list[EmpiricalPoint] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                EmpiricalPoint(
                    density=float(row["density_1_per_m"]),
                    velocity=float(row["velocity_m_per_s"]),
                    inverse_density=float(row["inverse_density_m"]),
                    source=row["source"],
                    notes=row["notes"],
                )
            )
    return points


def rmse_against_empirical(densities, velocities) -> float:
    """Root-mean-square error against the regression reference curve."""
    rho = np.asarray(densities, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if rho.shape != v.shape:
        raise ValueError("densities and velocities must have the same shape")
    ref = empirical_velocity_from_required_length(rho)
    return float(np.sqrt(np.mean((v - ref) ** 2)))
