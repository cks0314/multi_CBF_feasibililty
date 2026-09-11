"""
The safety filters themselves, and a closed loop rollout.

Two filters are implemented:

  LocalFilter    each agent solves for its own scalar input, enforcing only the
                 share of each constraint it has been assigned.  This is the
                 decentralised setting the paper is about, and the only thing
                 that distinguishes the three methods is how those shares are
                 chosen.

  JointFilter    one program over every input at once.  It is not deployable in
                 the decentralised setting, but it bounds what any division
                 could achieve, so we report it as a reference.

Both carry a slack variable, heavily penalised.  That matters more than it
looks: when a local program is infeasible the slack lets the agent ignore its
share and move anyway, which is how the heuristic baselines make progress
while violating the safety distance.  The comparison in the paper is partly a
comparison of how often each method has to rely on that escape hatch.
"""
import numpy as np
import cvxpy as cp

from .model import (barrier_rows, nominal_input, step, pairs,
                    AGENT_RADIUS, DT)
from .certificate import ALLOCATIONS, share_of

SLACK_PENALTY = 1e6
INFEASIBLE_TOL = 1e-4      # slack above this counts as an infeasible program


class LocalFilter:
    """One agent's program, compiled once and re-solved with new parameters.

    cvxpy problems are expensive to build and cheap to re-solve, so we allocate
    for the largest possible number of incident constraints and mask the unused
    rows.  The mask is what the `active` parameter does.
    """

    def __init__(self, max_incident):
        self.n = max_incident
        self.u = cp.Variable(1)
        self.slack = cp.Variable(max_incident, nonneg=True)
        self.slope = cp.Parameter(max_incident)
        self.offset = cp.Parameter(max_incident)
        self.mask = cp.Parameter(max_incident, nonneg=True)
        self.u_nom = cp.Parameter(1)
        self.u_bar = cp.Parameter(nonneg=True)
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.u - self.u_nom)
                        + SLACK_PENALTY * cp.sum_squares(self.slack)),
            [cp.multiply(self.mask,
                         self.slope * self.u + self.offset + self.slack) >= 0,
             cp.abs(self.u) <= self.u_bar])

    def __call__(self, slope, offset, u_nom, u_bar):
        """Returns (input, slack).  Slack above the tolerance means the
        agent had no admissible input for its assigned share."""
        k = len(slope)
        s = np.zeros(self.n); o = np.zeros(self.n); m = np.zeros(self.n)
        s[:k], o[:k], m[:k] = slope, offset, 1.0
        self.slope.value, self.offset.value, self.mask.value = s, o, m
        self.u_nom.value, self.u_bar.value = np.array([u_nom]), max(u_bar, 1e-6)
        try:
            self.problem.solve(solver=cp.CLARABEL)
            if self.u.value is None:
                return float(np.clip(u_nom, -u_bar, u_bar)), 1.0
            return float(self.u.value[0]), float(np.max(self.slack.value[:k]))
        except cp.error.SolverError:
            return float(np.clip(u_nom, -u_bar, u_bar)), 1.0


class JointFilter:
    """The centralised reference: every input in one program."""

    def __init__(self, n, n_constraints):
        self.u = cp.Variable(n)
        self.slack = cp.Variable(n_constraints, nonneg=True)
        self.G = cp.Parameter((n_constraints, n))
        self.b = cp.Parameter(n_constraints)
        self.u_nom = cp.Parameter(n)
        self.u_bar = cp.Parameter(n, nonneg=True)
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.u - self.u_nom)
                        + SLACK_PENALTY * cp.sum_squares(self.slack)),
            [self.G @ self.u + self.b + self.slack >= 0,
             cp.abs(self.u) <= self.u_bar])

    def __call__(self, G, b, u_nom, u_bar):
        self.G.value, self.b.value = G, b
        self.u_nom.value, self.u_bar.value = u_nom, u_bar
        try:
            self.problem.solve(solver=cp.CLARABEL)
            if self.u.value is None:
                return np.clip(u_nom, -u_bar, u_bar), 1.0
            return self.u.value, float(np.max(self.slack.value))
        except cp.error.SolverError:
            return np.clip(u_nom, -u_bar, u_bar), 1.0


def rollout(q, v, heading, goal, u_bar, mode, duration=4.0,
            local=None, joint=None, fallback=False, log=False):
    """Run one closed loop episode and report what happened.

    `mode` is 'centralized' for the joint reference, or one of the allocation
    names for the decentralised filter.  With `fallback` the certificate
    method applies the linear program's witness input when no division works,
    instead of letting each agent relax its own share independently.

    The infeasibility statistic counts *control steps at which some program had
    no solution*, not individual programs.  That distinction matters: the
    decentralised methods solve N programs per step and the centralised one
    solves a single program, so a per program count would divide the
    decentralised rates by N and make them look artificially good.
    """
    n = len(q)
    edges = pairs(n)
    local = local or LocalFilter(n - 1)
    joint = joint or JointFilter(n, len(edges))

    steps = infeasible_steps = fallback_steps = 0
    h_min = np.inf
    trace = []

    for _ in range(int(duration / DT)):
        u_nom, heading_des = nominal_input(q, v, heading, goal, u_bar)
        G, b, active_mask = barrier_rows(q, v, heading, edges)
        active = list(np.where(active_mask)[0])

        u = np.zeros(n)
        any_infeasible = False

        if mode == 'centralized':
            if active:
                u, slack = joint(G, b, u_nom, u_bar)
                any_infeasible = slack > INFEASIBLE_TOL
                steps += 1
                infeasible_steps += int(any_infeasible)
            else:
                u = u_nom
        else:
            out = ALLOCATIONS[mode](G, b, edges, active, n, u_bar)
            theta = out[0] if isinstance(out, tuple) else out
            t_star = out[1] if isinstance(out, tuple) else np.inf

            if fallback and np.isfinite(t_star) and t_star < 0:
                # No division makes every local program feasible.  Apply the
                # witness input, which maximises the worst margin over the
                # fleet, rather than letting each agent relax independently.
                u = np.clip(out[2], -u_bar, u_bar)
                any_infeasible = True
                fallback_steps += 1
                if active:
                    steps += 1; infeasible_steps += 1
            else:
                for i in range(n):
                    incident = [k for k in active if i in edges[k]]
                    if not incident:
                        u[i] = u_nom[i]
                        continue
                    slope = np.array([G[k, i] for k in incident])
                    offset = np.array([share_of(theta, k, i, edges) * b[k]
                                       for k in incident])
                    u[i], slack = local(slope, offset, u_nom[i], u_bar[i])
                    any_infeasible = any_infeasible or slack > INFEASIBLE_TOL
                if active:
                    steps += 1
                    infeasible_steps += int(any_infeasible)

        if log:
            trace.append(dict(q=q.copy(), heading=heading.copy(),
                              theta=theta.copy() if mode != 'centralized' else None))

        q, v, heading = step(q, v, heading, u, heading_des)
        closest = min(np.linalg.norm(q[i] - q[j]) for i, j in edges)
        h_min = min(h_min, closest ** 2 - (2 * AGENT_RADIUS) ** 2)

    reached = np.linalg.norm(q - goal, axis=1) < 0.45
    result = dict(
        infeasible_fraction=infeasible_steps / max(1, steps),
        fallback_fraction=fallback_steps / max(1, int(duration / DT)),
        min_h=float(h_min),
        reached=float(reached.mean()),
        mean_goal_distance=float(np.linalg.norm(q - goal, axis=1).mean()),
    )
    if log:
        result['trace'] = trace
    return result
