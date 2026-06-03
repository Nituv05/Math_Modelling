"""
Driving a single simulation run and measuring the mean velocity.

Section III of the paper:

    "For every run we set at t = 0 all velocities to zero and distribute the
     persons randomly with a minimal distance of a in the system. After
     3e5 relaxation-steps we perform 3e5 measurement-steps. At every step
     we determine the mean value of the velocity over all particles and
     calculate the mean value over time."
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import HardBodyModel, RemoteActionModel, ModelParameters


# Step counts used in the paper.
PAPER_RELAX_STEPS = 300_000
PAPER_MEASURE_STEPS = 300_000


@dataclass
class RunResult:
    n: int
    L: float
    density: float          # rho = N / L  [1/m]
    mean_velocity: float    # time- and ensemble-averaged speed [m/s]
    std_velocity: float     # std of the per-step mean velocity over time


def run_single(model_cls, n: int, L: float, params: ModelParameters,
               relax_steps: int = PAPER_RELAX_STEPS,
               measure_steps: int = PAPER_MEASURE_STEPS,
               dt: float = 0.001, seed: int | None = None,
               record_trajectory: bool = False,
               trajectory_stride: int = 1):
    """Run one simulation and return a :class:`RunResult`.

    Parameters
    ----------
    model_cls
        :class:`~pedestrian.model.HardBodyModel` or
        :class:`~pedestrian.model.RemoteActionModel`.
    n, L
        Number of pedestrians and ring length; density is ``n / L``.
    relax_steps, measure_steps
        Number of integration steps for equilibration and for measurement.
    record_trajectory
        If true, also return the recorded positions (for Fig. 3 style
        space-time plots).
    """
    model = model_cls(n=n, L=L, params=params, dt=dt, seed=seed)

    for _ in range(relax_steps):
        model.step()

    mean_per_step = np.empty(measure_steps)
    traj = [] if record_trajectory else None
    for k in range(measure_steps):
        model.step()
        mean_per_step[k] = model.v.mean()
        if record_trajectory and (k % trajectory_stride == 0):
            traj.append(model.x.copy())

    result = RunResult(
        n=n,
        L=L,
        density=n / L,
        mean_velocity=float(mean_per_step.mean()),
        std_velocity=float(mean_per_step.std()),
    )
    if record_trajectory:
        return result, np.array(traj)
    return result
