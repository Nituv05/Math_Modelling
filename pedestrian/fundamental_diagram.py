"""
Computing the velocity-density relation (fundamental diagram).

For a fixed ring length ``L`` the density ``rho = N / L`` is varied by
changing the number of pedestrians ``N``.  Each density yields one
time-averaged mean velocity, and the collection of ``(rho, v)`` points is
the fundamental diagram (Figs. 1 and 2 of the paper).
"""

from __future__ import annotations

import numpy as np

from .model import ModelParameters
from .simulation import run_single, RunResult, PAPER_RELAX_STEPS, PAPER_MEASURE_STEPS


def densities_to_counts(densities, L: float, a: float = 0.36) -> list[int]:
    """Convert a list of target densities [1/m] to integer pedestrian counts.

    Counts that cannot physically fit on the ring (``n * a > L``, i.e. above
    the jamming density ``1/a``) are dropped.
    """
    n_max = int(L / a)
    counts = sorted({max(1, int(round(rho * L))) for rho in densities})
    return [n for n in counts if n <= n_max]


def fundamental_diagram(model_cls, params: ModelParameters, L: float = 17.3,
                        densities=None, counts=None,
                        relax_steps: int = PAPER_RELAX_STEPS,
                        measure_steps: int = PAPER_MEASURE_STEPS,
                        dt: float = 0.001, seed: int | None = 0,
                        progress: bool = True) -> list[RunResult]:
    """Sweep density and return one :class:`RunResult` per pedestrian count.

    Either ``densities`` (target [1/m] values, converted to integer counts)
    or ``counts`` (explicit pedestrian numbers) must be given.
    """
    if counts is None:
        if densities is None:
            densities = np.linspace(0.1, 2.8, 28)
        counts = densities_to_counts(densities, L, a=params.a)

    results: list[RunResult] = []
    for i, n in enumerate(counts):
        run_seed = None if seed is None else seed + i
        res = run_single(model_cls, n=n, L=L, params=params,
                         relax_steps=relax_steps, measure_steps=measure_steps,
                         dt=dt, seed=run_seed)
        results.append(res)
        if progress:
            print(f"  [{i + 1:>2}/{len(counts)}] N={n:>3}  "
                  f"rho={res.density:5.3f} 1/m  v={res.mean_velocity:5.3f} m/s")
    return results
