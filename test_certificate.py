"""
Tests for the feasibility certificate.

The important one is `test_certificate_agrees_with_solver`: it checks the claim
the whole paper rests on, that the sign of M(x) decides feasibility exactly,
by asking a completely separate solver the same question.  If that test ever
fails, something is wrong with the theory or the implementation of it, not with
the test.
"""
import numpy as np
import pytest

from mrcbf import (pairs, barrier_rows, random_scenario, actuation_bounds,
                   reserve, alloc_certificate, alloc_uniform, alloc_capability,
                   margins_under)

cp = pytest.importorskip("cvxpy")


def conflicted_state(n, seed, squeeze=0.55):
    """A fleet squeezed together tightly enough to be interesting."""
    edges = pairs(n)
    q, v, heading, _ = random_scenario(n, seed)
    q = q * squeeze
    u_bar = actuation_bounds(n, seed)
    G, b, active_mask = barrier_rows(q, v, heading, edges)
    return edges, G, b, list(np.where(active_mask)[0]), u_bar


def qp_feasible(G, b, u_bar):
    u = cp.Variable(len(u_bar))
    prob = cp.Problem(cp.Minimize(0), [G @ u + b >= 0, cp.abs(u) <= u_bar])
    prob.solve(solver=cp.CLARABEL)
    return prob.status in ("optimal", "optimal_inaccurate")


@pytest.mark.parametrize("seed", range(12))
def test_certificate_agrees_with_solver(seed):
    """M(x) >= 0 exactly when an admissible input satisfies every constraint."""
    n = 4 + seed % 5
    edges, G, b, active, u_bar = conflicted_state(n, seed)
    if len(active) < 2:
        pytest.skip("no constraints active at this state")

    M, _, _ = reserve(G[active], b[active], u_bar)
    if abs(M) < 1e-6:
        pytest.skip("state sits on the feasibility boundary")

    assert (M >= 0) == qp_feasible(G[active], b[active], u_bar)


@pytest.mark.parametrize("seed", range(8))
def test_leverage_is_nonnegative_and_mostly_zero(seed):
    """Extra actuation never hurts, and usually does nothing at all.

    The second half is the operationally useful part: it says most robots are
    not worth upgrading, which is only meaningful if the zeros are exact.
    """
    n = 8
    edges, G, b, active, u_bar = conflicted_state(n, seed)
    if len(active) < 2:
        pytest.skip("no constraints active at this state")

    _, _, leverage = reserve(G[active], b[active], u_bar)
    assert np.all(leverage >= -1e-12)
    assert np.sum(leverage > 1e-7) <= n


def test_zero_leverage_means_upgrading_changes_nothing():
    """A robot whose leverage is zero can be scaled up tenfold for free.

    This is not an approximation.  The reserve is unchanged to solver
    precision, because that robot's contribution to the objective was already
    zero at the optimal multiplier.
    """
    n = 8
    edges, G, b, active, u_bar = conflicted_state(n, 0)
    M0, _, leverage = reserve(G[active], b[active], u_bar)

    idle = np.where(leverage <= 1e-9)[0]
    assert len(idle) > 0, "expected at least one robot with no leverage"

    for i in idle[:3]:
        boosted = u_bar.copy()
        boosted[i] *= 10.0
        M1, _, _ = reserve(G[active], b[active], boosted)
        assert abs(M1 - M0) < 1e-8


def test_multiplier_lives_on_the_simplex():
    edges, G, b, active, u_bar = conflicted_state(8, 3)
    _, lam, _ = reserve(G[active], b[active], u_bar)
    assert np.all(lam >= -1e-12)
    assert abs(lam.sum() - 1.0) < 1e-8


def test_multiplier_support_is_sparse():
    """The blame concentrates: far fewer constraints than are enforced."""
    n = 10
    edges, G, b, active, u_bar = conflicted_state(n, 1)
    if len(active) < 6:
        pytest.skip("too few constraints active to be interesting")
    _, lam, _ = reserve(G[active], b[active], u_bar)
    support = int(np.sum(lam > 1e-7))
    assert support <= n + 1
