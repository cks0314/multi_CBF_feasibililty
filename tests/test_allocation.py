"""
Tests for the allocation, and for the model it operates on.

The allocation has one guarantee worth protecting with a test: it never does
worse than the uniform split.  That is what Theorem 20 promises, and it is
exactly the kind of thing a refactor can silently break, because a
misformulated linear program usually still returns *something*.
"""
import numpy as np
import pytest

from mrcbf import (pairs, barrier_rows, random_scenario, actuation_bounds,
                   step, nominal_input, reserve, margins_under, share_of,
                   alloc_uniform, alloc_capability, alloc_certificate)
from mrcbf.model import AGENT_RADIUS


def conflicted_state(n, seed, squeeze=0.55):
    edges = pairs(n)
    q, v, heading, _ = random_scenario(n, seed)
    q = q * squeeze
    u_bar = actuation_bounds(n, seed)
    G, b, active_mask = barrier_rows(q, v, heading, edges)
    return edges, G, b, list(np.where(active_mask)[0]), u_bar


def worst_margin(G, b, edges, active, n, u_bar, theta):
    m = margins_under(G, b, edges, active, n, u_bar, theta)
    return np.nanmin(m) if np.any(np.isfinite(m)) else np.inf


# ------------------------------------------------------------- allocation
@pytest.mark.parametrize("seed", range(15))
def test_allocation_never_worse_than_uniform(seed):
    """The point of solving the linear program, stated as a test."""
    n = 4 + seed % 6
    edges, G, b, active, u_bar = conflicted_state(n, seed)
    if len(active) < 2:
        pytest.skip("no constraints active at this state")

    uniform = worst_margin(G, b, edges, active, n, u_bar,
                           alloc_uniform(G, b, edges, active, n, u_bar))
    theta, _, _ = alloc_certificate(G, b, edges, active, n, u_bar)
    optimal = worst_margin(G, b, edges, active, n, u_bar, theta)

    assert optimal >= uniform - 1e-7


@pytest.mark.parametrize("seed", range(10))
def test_reported_margin_matches_the_shares(seed):
    """The linear program's own value of t should equal the worst margin it
    achieves.  A mismatch means the objective and the constraints have drifted
    apart, which is easy to do and hard to notice."""
    n = 6
    edges, G, b, active, u_bar = conflicted_state(n, seed)
    if len(active) < 2:
        pytest.skip("no constraints active at this state")

    theta, t_star, _ = alloc_certificate(G, b, edges, active, n, u_bar)
    if not np.isfinite(t_star):
        pytest.skip("allocation program did not solve at this state")

    achieved = worst_margin(G, b, edges, active, n, u_bar, theta)
    assert abs(achieved - t_star) < 1e-5


@pytest.mark.parametrize("alloc", [alloc_uniform, alloc_capability])
def test_shares_are_a_valid_division(alloc):
    """Whatever the method, the two endpoints' shares must sum to one.

    This is what makes safety independent of the allocation: summing the local
    constraints recovers the original joint constraint exactly.
    """
    n = 8
    edges, G, b, active, u_bar = conflicted_state(n, 2)
    theta = alloc(G, b, edges, active, n, u_bar)
    for k in active:
        i, j = edges[k]
        assert abs(share_of(theta, k, i, edges)
                   + share_of(theta, k, j, edges) - 1.0) < 1e-12
        assert -1e-12 <= theta[k] <= 1.0 + 1e-12


def test_certificate_shares_are_a_valid_division():
    n = 8
    edges, G, b, active, u_bar = conflicted_state(n, 4)
    theta, _, _ = alloc_certificate(G, b, edges, active, n, u_bar)
    for k in active:
        i, j = edges[k]
        assert abs(share_of(theta, k, i, edges)
                   + share_of(theta, k, j, edges) - 1.0) < 1e-12


def test_allocation_does_not_change_the_certificate():
    """The joint reserve and the per robot leverage are properties of the
    state, not of how the constraints happen to be divided."""
    n = 8
    edges, G, b, active, u_bar = conflicted_state(n, 0)
    M_ref, _, lev_ref = reserve(G[active], b[active], u_bar)

    for alloc in (alloc_uniform, alloc_capability, alloc_certificate):
        out = alloc(G, b, edges, active, n, u_bar)
        _ = out[0] if isinstance(out, tuple) else out      # theta is unused
        M, _, lev = reserve(G[active], b[active], u_bar)
        assert abs(M - M_ref) < 1e-12
        assert np.allclose(lev, lev_ref, atol=1e-12)


# ------------------------------------------------------------------ model
def test_barrier_rows_only_touch_the_pair():
    n = 6
    edges, G, b, active, _ = conflicted_state(n, 5)
    for k in active:
        i, j = edges[k]
        others = [a for a in range(n) if a not in (i, j)]
        assert np.allclose(G[k, others], 0.0)


def test_distant_pairs_are_not_enforced():
    """Pairs beyond the sensing radius get a large offset and no input
    coefficients, so they drop out of every program."""
    n = 4
    edges = pairs(n)
    q = np.array([[0., 0.], [50., 0.], [0., 50.], [50., 50.]])
    v = np.zeros((n, 2))
    heading = np.zeros(n)
    G, b, active = barrier_rows(q, v, heading, edges)
    assert not active.any()
    assert np.allclose(G, 0.0)


def test_step_respects_the_heading_slew_limit():
    """One step cannot rotate further than the rate limit allows."""
    from mrcbf.model import OMEGA_MAX, DT
    n = 2
    q = np.zeros((n, 2))
    v = np.zeros((n, 2))
    heading = np.zeros(n)
    desired = np.full(n, np.pi)          # ask for a full reversal at once
    _, _, new_heading = step(q, v, heading, np.zeros(n), desired)
    assert np.all(np.abs(new_heading - heading) <= OMEGA_MAX * DT + 1e-12)


def test_agents_start_separated():
    """The scenario generator should not place robots in collision."""
    for seed in range(5):
        q, _, _, _ = random_scenario(8, seed)
        for i in range(8):
            for j in range(i + 1, 8):
                assert np.linalg.norm(q[i] - q[j]) > 2 * AGENT_RADIUS


def test_nominal_input_respects_the_bound():
    n = 6
    q, v, heading, goal = random_scenario(n, 0)
    u_bar = actuation_bounds(n, 0)
    u_nom, _ = nominal_input(q, v, heading, goal, u_bar)
    assert np.all(np.abs(u_nom) <= u_bar + 1e-12)
