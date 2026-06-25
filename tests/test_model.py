"""
Sanity tests for the pedestrian-flow models.

These check physical invariants and qualitative behaviour rather than exact
numbers (the paper's figures require ~3e5 steps to reproduce precisely).
Run with::

    pytest -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pedestrian import (ModelParameters, HardBodyModel, RemoteActionModel,
                        front_gaps, initialise, run_single,
                        empirical_mean_velocity_near_density,
                        load_empirical_points, rmse_against_empirical)


def test_required_length():
    p = ModelParameters(a=0.36, b=0.56)
    assert np.isclose(p.required_length(np.array([0.0]))[0], 0.36)
    assert np.isclose(p.required_length(np.array([1.0]))[0], 0.36 + 0.56)
    # negative velocity clamped -> never below a
    assert np.isclose(p.required_length(np.array([-5.0]))[0], 0.36)


def test_empirical_points():
    points = load_empirical_points()
    assert len(points) == 170
    assert min(p.density for p in points) < 0.6
    assert max(p.density for p in points) > 2.0
    assert min(p.velocity for p in points) < 0.12
    assert max(p.velocity for p in points) > 0.95

    rho = np.array([0.9, 1.4])
    expected = empirical_mean_velocity_near_density(rho)
    assert not np.isnan(expected).any()
    assert rmse_against_empirical(rho, expected) == 0.0


def test_front_gaps_sum_to_L():
    L = 17.3
    x = np.array([0.0, 3.0, 9.0, 15.0])
    gaps = front_gaps(x, L)
    assert np.all(gaps > 0)
    assert np.isclose(gaps.sum(), L)


def test_initialisation_respects_min_distance():
    p = ModelParameters(a=0.36)
    rng = np.random.default_rng(1)
    x, v, v0 = initialise(30, 17.3, p, rng)
    assert np.all(v == 0.0)
    assert np.all(v0 > 0.0)
    gaps = front_gaps(np.sort(x), 17.3)
    assert gaps.min() >= p.a - 1e-9


def test_initialisation_too_many_raises():
    p = ModelParameters(a=0.36)
    rng = np.random.default_rng(0)
    try:
        initialise(100, 17.3, p, rng)   # 100*0.36 = 36 m > 17.3 m
    except ValueError:
        return
    raise AssertionError("expected ValueError for overcrowded ring")


def test_hardbody_no_overlap():
    """Gaps must never drop below the required length (hard-body constraint)."""
    p = ModelParameters(a=0.36, b=0.56)
    m = HardBodyModel(n=20, L=17.3, params=p, seed=3)
    for _ in range(2000):
        m.step()
        gaps = front_gaps(m.x, m.L)
        d = p.required_length(m.v)
        # allow a tiny numerical tolerance
        assert np.all(gaps >= d - 1e-6)
        assert np.all(m.v >= -1e-9)


def test_remote_velocity_nonnegative_and_bounded():
    p = ModelParameters(a=0.36, b=0.56, e=0.07, f=2.0)
    m = RemoteActionModel(n=20, L=17.3, params=p, seed=4)
    for _ in range(2000):
        m.step()
        assert np.all(m.v >= 0.0)
        # speed should not exceed intended speed by much
        assert m.v.max() <= m.v0.max() + 0.1


def test_remote_action_step_uses_explicit_euler_position_update():
    """If old velocity is zero, one explicit Euler step must not move x."""
    p = ModelParameters(a=0.36, b=0.56, e=0.07, f=2.0)
    m = RemoteActionModel(n=2, L=50.0, params=p, seed=4)
    m.x = np.array([0.0, 20.0])
    m.v = np.array([0.0, 0.0])
    m.v0 = np.array([1.2, 1.2])
    x_old = m.x.copy()
    m.step()
    assert np.allclose(m.x, x_old)
    assert np.all(m.v > 0.0)


def test_low_density_reaches_free_speed():
    """At very low density the mean speed approaches the intended speed."""
    p = ModelParameters(a=0.36, b=0.56)
    res = run_single(HardBodyModel, n=2, L=50.0, params=p,
                     relax_steps=3000, measure_steps=3000, seed=0)
    assert res.mean_velocity > 1.0   # close to v0_mean = 1.24


def test_velocity_decreases_with_density():
    """Mean velocity must be monotonically lower at high density."""
    p = ModelParameters(a=0.36, b=0.56)
    low = run_single(HardBodyModel, n=3, L=17.3, params=p,
                     relax_steps=3000, measure_steps=3000, seed=0)
    high = run_single(HardBodyModel, n=40, L=17.3, params=p,
                      relax_steps=3000, measure_steps=3000, seed=0)
    assert high.mean_velocity < low.mean_velocity


def test_remote_action_b0_velocity_decreases_as_n_increases():
    """For remote action with b=0, increasing N raises density and lowers speed."""
    p = ModelParameters(a=0.36, b=0.0, e=0.07, f=2.0)
    low = run_single(RemoteActionModel, n=20, L=17.3, params=p,
                     relax_steps=2000, measure_steps=2000, seed=0)
    high = run_single(RemoteActionModel, n=40, L=17.3, params=p,
                      relax_steps=2000, measure_steps=2000, seed=0)

    assert high.density > low.density
    assert high.mean_velocity < low.mean_velocity
