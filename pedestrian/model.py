"""
Core 1D social-force pedestrian models.

Reproduces the two interaction approaches from

    A. Seyfried, B. Steffen, T. Lippert,
    "Basics of Modelling the Pedestrian Flow",
    Physica A 368 (2006) 232-238 (arXiv:physics/0506189)

Both models describe single-file pedestrian motion on a ring (periodic
boundary conditions) of length ``L``.  Pedestrians are ordered along the
ring, ``x_1 < x_2 < ... < x_N``, and each pedestrian ``i`` only interacts
with the pedestrian ``i+1`` directly in front (Eq. 5 / Eq. 6 of the paper).

The two models are:

* :class:`HardBodyModel`  -- "hard bodies without remote action" (Eq. 5).
  The pedestrian accelerates towards its desired speed unless the gap to
  the person in front drops to the *required length* ``d = a + b v``; in
  that case the velocity is set to zero and the move is rejected (a hard
  collision).  This requires the special quasi-parallel update of
  Section II C and is implemented by iterative relaxation.

* :class:`RemoteActionModel` -- "hard bodies with remote action" (Eq. 6).
  A smooth, diverging repulsive force replaces the hard collision and the
  system is integrated with an explicit Euler scheme.

Units: SI (metres, seconds, m/s).  Masses are set to ``m_i = 1`` as in the
paper, so forces and accelerations coincide.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class ModelParameters:
    """Parameters shared by both interaction models.

    Defaults follow the paper:

    * ``a = 0.36 m``                  required length offset (Eq. 4)
    * ``tau = 0.61 s``                acceleration time constant ("reliable value")
    * ``v0_mean = 1.24 m/s``         mean intended speed (normal distribution)
    * ``v0_sigma = 0.05 m/s``        spread of intended speed
    * ``e = 0.07 N``, ``f = 2``      strength / range of the remote force (Eq. 6)
    """

    a: float = 0.36          # [m]  required-length offset, Eq. (4)
    b: float = 0.56          # [s]  velocity dependence of required length, Eq. (4)
    tau: float = 0.61        # [s]  relaxation/acceleration time, Eq. (2)
    v0_mean: float = 1.24    # [m/s] mean intended speed
    v0_sigma: float = 0.05   # [m/s] std of intended speed
    e: float = 0.07          # [N]  remote-force strength, Eq. (6)
    f: float = 2.0           # [-]  remote-force range exponent, Eq. (6)
    clear_distance_floor: float = 1e-6  # [m] numerical guard for Eq. (6)

    def required_length(self, v: np.ndarray) -> np.ndarray:
        """Required length ``d_i = a + b * v_i`` (Eq. 4 / 5 / 6).

        Velocities are clamped at 0 so that a (numerically) negative
        velocity never shrinks the personal space below ``a``.
        """
        return self.a + self.b * np.maximum(v, 0.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def front_gaps(x: np.ndarray, L: float) -> np.ndarray:
    """Gap from each pedestrian to the one directly in front, on a ring.

    Pedestrians are assumed sorted: ``x[0] < x[1] < ... < x[N-1]``.
    The pedestrian in front of the last one is the first one, wrapped by
    ``+L``.  Returned gaps are strictly positive.
    """
    x_front = np.empty_like(x)
    x_front[:-1] = x[1:]
    x_front[-1] = x[0] + L
    return x_front - x


def initialise(n: int, L: float, params: ModelParameters,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial condition (Section III of the paper).

    At ``t = 0`` all velocities are zero and the persons are distributed
    randomly with a minimal mutual distance of ``a``.  Intended speeds are
    drawn from ``N(v0_mean, v0_sigma)``.

    The persons are placed on the ring by handing out the available slack
    ``L - n * a`` randomly among the ``n`` gaps, which guarantees the
    minimum distance ``a`` while using the full periodic length.
    """
    if n * params.a > L:
        raise ValueError(
            f"Cannot place {n} pedestrians with min distance {params.a} m "
            f"on a ring of length {L} m (need >= {n * params.a:.2f} m)."
        )

    slack = L - n * params.a
    # Random partition of the slack into n non-negative pieces (Dirichlet).
    extra = rng.dirichlet(np.ones(n)) * slack
    gaps = params.a + extra                      # each gap >= a
    x = np.cumsum(gaps) - gaps[0]                # start at 0, strictly sorted
    x = np.mod(x, L)
    x.sort()

    v = np.zeros(n)
    v0 = rng.normal(params.v0_mean, params.v0_sigma, size=n)
    v0 = np.maximum(v0, 0.0)                      # intended speed > 0 (paper)
    return x, v, v0


# --------------------------------------------------------------------------- #
# Model: hard bodies WITHOUT remote action  (Eq. 5)
# --------------------------------------------------------------------------- #
class HardBodyModel:
    """Hard bodies without remote action -- Eq. (5).

    Update rule per step (Section II C):

    1. Advance every pedestrian one explicit-Euler step with the driving
       force only,  ``a_i = (v0_i - v_i) / tau``.
    2. If, after the step, the gap to the person in front is smaller than
       the required length ``d_i = a + b v_i``, the velocity is set to zero
       and the position reset to the old value (rejected move).  Because a
       reset shifts the gap of the person *behind*, the check is repeated
       until no violation remains (iterative approximation of the exact
       parallel update).
    """

    def __init__(self, n: int, L: float, params: ModelParameters,
                 dt: float = 0.001, seed: int | None = None):
        self.n = n
        self.L = L
        self.params = params
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.x, self.v, self.v0 = initialise(n, L, params, self.rng)

    def step(self) -> None:
        p, dt, L = self.params, self.dt, self.L
        # Invariant on entry: self.x is sorted ascending in [0, L).
        x_old, v_old = self.x.copy(), self.v.copy()

        # (1) tentative driving-only Euler step.  We keep positions *unwrapped*
        # during the relaxation: per step everyone moves forward by less than
        # dt*v0 << a, so the ordering is preserved and front_gaps stays valid.
        acc = (self.v0 - v_old) / p.tau
        v_new = np.maximum(v_old + dt * acc, 0.0)  # velocities restricted to [0, v0]
        x_new = x_old + dt * v_old

        # (2) reject moves that violate the required length, relax to fixpoint.
        for _ in range(self.n + 1):
            d = p.required_length(v_new)
            gaps = front_gaps(x_new, L)
            violated = gaps <= d
            if not violated.any():
                break
            x_new[violated] = x_old[violated]
            v_new[violated] = 0.0

        # (3) now wrap onto the ring and restore the sorted invariant; the
        # front-runner that crossed L becomes the new first pedestrian.
        x_new = np.mod(x_new, L)
        order = np.argsort(x_new)
        self.x = x_new[order]
        self.v = v_new[order]
        self.v0 = self.v0[order]


# --------------------------------------------------------------------------- #
# Model: hard bodies WITH remote action  (Eq. 6)
# --------------------------------------------------------------------------- #
class RemoteActionModel:
    """Hard bodies with remote action -- Eq. (6).

    The force on pedestrian ``i`` is

        G_i = (v0_i - v_i) / tau - e * (1 / (gap_i - d_i)) ** f
        F_i = G_i              if v_i > 0
        F_i = max(0, G_i)      if v_i <= 0

    with ``d_i = a + b v_i``.  Integrated with an explicit Euler scheme of
    step ``dt = 0.001 s`` (Section II C).  Velocities are clamped at zero so
    that no backward motion against the intended direction occurs.
    """

    def __init__(self, n: int, L: float, params: ModelParameters,
                 dt: float = 0.001, seed: int | None = None):
        self.n = n
        self.L = L
        self.params = params
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.x, self.v, self.v0 = initialise(n, L, params, self.rng)

    def _force(self) -> np.ndarray:
        p = self.params
        d = p.required_length(self.v)
        gaps = front_gaps(self.x, self.L)
        # Effective clear distance; floored to a tiny epsilon so the strong
        # repulsion stays finite if a pair gets numerically too close.
        clear = np.maximum(gaps - d, p.clear_distance_floor)
        drive = (self.v0 - self.v) / p.tau
        repel = p.e * (1.0 / clear) ** p.f
        G = drive - repel
        # v <= 0  ->  force may only be non-negative (Eq. 6)
        return np.where(self.v > 0.0, G, np.maximum(0.0, G))

    def step(self) -> None:
        dt, L = self.dt, self.L
        x_old = self.x.copy()
        v_old = self.v.copy()
        F = self._force()
        self.v = np.maximum(self.v + dt * F, 0.0)   # no backward velocity
        self.x = np.mod(x_old + dt * v_old, L)
        # Maintain sorted order (ordering can in principle change if a fast
        # pedestrian overtakes, which the model should prevent, but we keep
        # the invariant defensively for the gap computation).
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.v = self.v[order]
        self.v0 = self.v0[order]
