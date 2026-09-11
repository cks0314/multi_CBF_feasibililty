"""
The feasibility certificate, and the three ways of dividing a shared
constraint.

Everything here operates on the affine system  b + G u >= 0  produced by
`model.barrier_rows`.  Three quantities do all the work:

  M(x)          the reserve.  Its sign decides, exactly, whether any admissible
                input satisfies every constraint at once.  Negative means the
                joint program is infeasible, and no division of the constraints
                can rescue it.

  dM/drho_i     how much the reserve would move if agent i's actuation were
                scaled up.  It is exactly zero for most agents, which is the
                useful part: it says upgrading them would achieve nothing.

  m_i^loc       the local margin of agent i under a given division theta.  It
                is nonnegative exactly when agent i's own program has a
                solution, which is what the decentralised filter needs.

The allocation problem is to pick theta so that the worst local margin is as
large as possible, and the point of the paper is that this is a single linear
program rather than a search.
"""
import numpy as np
from scipy.optimize import linprog


# ------------------------------------------------------------- certificate
def reserve(G, b, u_bar):
    """Evaluate the feasibility certificate M(x) and its multiplier.

    M(x) = min over the simplex of  lambda^T b + sum_i sigma_i(nu_i(lambda))

    where nu_i is the aggregate constraint normal that agent i carries and
    sigma_i is the support function of its input set.  For scalar inputs the
    support function is just u_bar_i |nu_i|, so the whole thing is a linear
    program once we introduce one epigraph variable per agent.

    Returns (M, lambda, leverage).  A negative M certifies infeasibility; the
    support of lambda names the constraints responsible, and `leverage` is the
    per-agent derivative dM/drho_i.
    """
    K, N = G.shape
    if K == 0:
        return np.inf, np.zeros(0), np.zeros(N)

    # variables: lambda (K, on the simplex), t (N, with t_i >= |nu_i|)
    nv = K + N
    A_ub, b_ub = [], []
    for i in range(N):
        for sign in (1.0, -1.0):
            row = np.zeros(nv)
            row[:K] = sign * G[:, i]
            row[K + i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    cost = np.concatenate([b, u_bar])
    simplex = np.concatenate([np.ones(K), np.zeros(N)])[None, :]
    res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=simplex, b_eq=[1.0],
                  bounds=[(0, None)] * K + [(None, None)] * N,
                  method='highs')
    if not res.success:
        return np.nan, np.zeros(K), np.zeros(N)

    lam = res.x[:K]
    nu = G.T @ lam
    return float(res.fun), lam, u_bar * np.abs(nu)


# ------------------------------------------------------------ local margin
def local_margin(G, b, incident, i, share, u_bar_i):
    """m_i^loc = max over |u_i| <= u_bar_i of the worst assigned constraint.

    Each constraint incident to agent i becomes an affine function of that
    agent's single input.  Their lower envelope is concave and piecewise
    linear, so its maximum over an interval is attained either at an endpoint
    or where two of the lines cross.  We solve it as a two variable linear
    program, which is exact and fast enough to call once per agent per step.

    `share` maps a constraint index to the fraction of b_k assigned to agent i.
    """
    if not incident:
        return np.inf
    offset = np.array([share[k] * b[k] for k in incident])
    slope = np.array([G[k, i] for k in incident])

    # maximise t subject to t <= offset_k + slope_k u for every k
    A = np.zeros((len(incident), 2))
    A[:, 0] = 1.0
    A[:, 1] = -slope
    res = linprog(np.array([-1.0, 0.0]), A_ub=A, b_ub=offset,
                  bounds=[(None, None), (-u_bar_i, u_bar_i)], method='highs')
    return -res.fun if res.success else -np.inf


def share_of(theta, k, i, edges):
    """The fraction of constraint k that falls on agent i.

    We store one number per constraint: the share of the lower indexed
    endpoint.  The other endpoint gets the remainder, which is what makes the
    division automatically conservative regardless of how theta is chosen.
    """
    return theta[k] if edges[k][0] == i else 1.0 - theta[k]


# ------------------------------------------------------------- allocations
def alloc_uniform(G, b, edges, active, n, u_bar):
    """Split every shared constraint down the middle.

    This is what a decentralised CBF filter does by default, and it is the
    baseline of Wang, Ames and Egerstedt.
    """
    return np.full(len(edges), 0.5)


def alloc_capability(G, b, edges, active, n, u_bar):
    """Weight each agent's share by how much it can push along the constraint.

    The intuition is reasonable: an agent with more authority in the relevant
    direction should carry more of the burden.  In our experiments this is
    indistinguishable from the uniform split, which is the negative result that
    motivates solving for the division exactly.
    """
    theta = np.full(len(edges), 0.5)
    for k in active:
        i, j = edges[k]
        si = u_bar[i] * abs(G[k, i])
        sj = u_bar[j] * abs(G[k, j])
        total = si + sj
        theta[k] = si / total if total > 1e-12 else 0.5
    return theta


def alloc_certificate(G, b, edges, active, n, u_bar):
    """Maximise the worst local margin.  This is the method of the paper.

    Substituting the local margin into the max-min objective and dualising the
    inner minimisation turns the problem into one linear program over

        t          the margin being maximised,
        theta_k    the division of each constraint,
        u_i        a witness input for each agent.

    The witness inputs are discarded when the program is feasible: they
    maximise the margin rather than track the task, so each agent recovers its
    own input locally instead.  When t < 0 no division works, and then u is
    exactly the input that maximises the worst margin across the fleet, which
    makes it a principled fallback.

    Returns (theta, t, u).
    """
    K = len(edges)
    if K == 0 or len(active) == 0:
        return np.full(K, 0.5), np.inf, np.zeros(n)

    nv = 1 + K + n                      # t, theta, u
    A_ub, b_ub = [], []
    for k in active:
        i, j = edges[k]
        # t <= theta_k b_k + G_ki u_i
        row = np.zeros(nv)
        row[0] = 1.0
        row[1 + k] = -b[k]
        row[1 + K + i] = -G[k, i]
        A_ub.append(row); b_ub.append(0.0)
        # t <= (1 - theta_k) b_k + G_kj u_j
        row = np.zeros(nv)
        row[0] = 1.0
        row[1 + k] = b[k]
        row[1 + K + j] = -G[k, j]
        A_ub.append(row); b_ub.append(b[k])

    cost = np.zeros(nv)
    cost[0] = -1.0                      # linprog minimises, so negate
    bounds = ([(None, None)] + [(0.0, 1.0)] * K
              + [(-u_bar[i], u_bar[i]) for i in range(n)])
    res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method='highs')
    if not res.success:
        return np.full(K, 0.5), -np.inf, np.zeros(n)

    theta = np.clip(res.x[1:1 + K], 0.0, 1.0)
    return theta, float(res.x[0]), res.x[1 + K:]


ALLOCATIONS = {
    'uniform': alloc_uniform,
    'capability': alloc_capability,
    'certificate': alloc_certificate,
}


def margins_under(G, b, edges, active, n, u_bar, theta):
    """Local margin of every agent under a given division.

    Convenience wrapper used by the experiments and figures.  An agent with no
    active constraints is unconstrained, and we report an infinite margin for
    it rather than a number that would distort the minimum.
    """
    out = np.full(n, np.nan)
    for i in range(n):
        incident = [k for k in active if i in edges[k]]
        if not incident:
            continue
        share = {k: share_of(theta, k, i, edges) for k in incident}
        out[i] = local_margin(G, b, incident, i, share, u_bar[i])
    return out
