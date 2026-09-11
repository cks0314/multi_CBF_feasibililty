"""
Agent model and the high-order barrier chain.

The fleet consists of planar vehicles that can accelerate only along their own
heading.  The heading is a state, not an input, and it slews at a bounded rate
towards whatever direction the nominal controller asks for.  Each agent
therefore has a single scalar input (m_i = 1), which is what makes the
underactuation in the paper meaningful: an agent can have plenty of thrust and
still be unable to push in the direction a constraint needs.

    q_i     position           (2,)
    v_i     velocity           (2,)
    phi_i   heading            scalar, a state
    u_i     thrust along phi_i scalar, |u_i| <= u_bar_i

Safety between a pair (i, j) is h_k = |q_i - q_j|^2 - (2r)^2, which has
relative degree two, so we enforce it through the usual two-stage HOCBF chain

    psi_0 = h_k
    psi_1 = psi_0_dot + alpha_1 psi_0
    psi_2 = psi_1_dot + alpha_2 psi_1 >= 0.

Expanding psi_2 for this model puts every constraint in the affine form

    b_k(x) + sum_i G_{k,i}(x) u_i >= 0,

and the whole paper is about what can and cannot be done with the rows G and
the offsets b.
"""
import numpy as np

# ----------------------------------------------------------------- constants
# These are the values used for every experiment reported in the paper.
AGENT_RADIUS = 0.30        # m, so the pair constraint keeps centres 0.6 m apart
ALPHA1 = 3.0               # first class-K gain of the chain
ALPHA2 = 3.0               # second class-K gain
TAU = 0.12                 # s, heading tracking time constant
OMEGA_MAX = 6.0            # rad/s, heading slew limit
DT = 0.02                  # s, integration step (50 Hz)
KP, KD = 3.0, 3.6          # nominal PD gains, critically damped
SENSE_RADIUS = 2.4         # m, pairs closer than this are enforced


def pairs(n):
    """Every unordered pair of agents, in a fixed order.

    The index k of a constraint is its position in this list, and that ordering
    is what lets us talk about "constraint k" consistently across the
    certificate, the allocation and the local programs.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def barrier_rows(q, v, heading, edges, sense=SENSE_RADIUS):
    """Build the affine system  b + G u >= 0  at the current state.

    Returns
    -------
    G : (K, N) array
        Row k holds the input coefficients of constraint k.  Only the two
        agents in the pair have nonzero entries, and each is the projection of
        the separation vector onto that agent's heading.  This is the quantity
        Theorem 8 shows no class-K function can change.
    b : (K,) array
        The offset of constraint k, which collects the drift terms and the
        class-K contributions.  Unlike G, this *does* depend on the gains.
    active : (K,) bool array
        Which pairs are within sensing range.  Inactive pairs are given a large
        positive offset so they are trivially satisfied and can be ignored.
    """
    n, K = len(q), len(edges)
    e = np.stack([np.cos(heading), np.sin(heading)], axis=1)   # body axes
    G = np.zeros((K, n))
    b = np.zeros(K)
    active = np.zeros(K, dtype=bool)

    for k, (i, j) in enumerate(edges):
        p = q[i] - q[j]                       # separation
        if np.linalg.norm(p) > sense:
            b[k] = 1e3                        # far apart: not enforced
            continue
        active[k] = True
        w = v[i] - v[j]                       # relative velocity

        h = p @ p - (2 * AGENT_RADIUS) ** 2
        h_dot = 2.0 * (p @ w)

        # d/dt of h_dot contributes 2|w|^2 from the velocity term plus the
        # input terms below; the class-K gains only ever touch the offset.
        G[k, i] = 2.0 * (p @ e[i])
        G[k, j] = -2.0 * (p @ e[j])
        b[k] = 2.0 * (w @ w) + ALPHA1 * h_dot + ALPHA2 * (h_dot + ALPHA1 * h)

    return G, b, active


def nominal_input(q, v, heading, goal, u_bar):
    """Proportional-derivative controller, projected onto the body axis.

    The controller asks for an acceleration towards the goal.  Because the
    agent can only push along its heading, the achievable part is the
    projection, saturated to the actuation bound.  The desired heading is
    returned separately so the caller can slew towards it.
    """
    a_des = -KP * (q - goal) - KD * v
    heading_des = np.arctan2(a_des[:, 1], a_des[:, 0])
    e = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    u_nom = np.clip(np.einsum('im,im->i', a_des, e), -u_bar, u_bar)
    return u_nom, heading_des


def step(q, v, heading, u, heading_des, dt=DT):
    """Integrate one control step of the fleet.

    The heading tracks its command through a first-order lag with a slew
    limit, and the thrust acts along whatever heading results.  Ordering
    matters here: we rotate first, then accelerate along the new heading.
    """
    err = (heading_des - heading + np.pi) % (2 * np.pi) - np.pi
    omega = np.clip(err / TAU, -OMEGA_MAX, OMEGA_MAX)
    heading = heading + dt * omega
    e = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    v = v + dt * (u[:, None] * e)
    q = q + dt * v
    return q, v, heading


def random_scenario(n, seed, spread=2.6):
    """A random start and goal assignment inside a square.

    Starts are rejection sampled to keep a minimum separation, and goals are
    drawn independently of starts.  That independence is deliberate: it makes
    trajectories cross, which is what drives the fleet into the dense conflicts
    the paper is about.  A structured formation would be far easier.
    """
    rng = np.random.RandomState(seed)
    q = rng.uniform(-spread, spread, (n, 2))
    for i in range(n):
        for _ in range(400):
            if all(np.linalg.norm(q[i] - q[j]) > 0.75 for j in range(i)):
                break
            q[i] = rng.uniform(-spread, spread, 2)
    goal = rng.uniform(-spread, spread, (n, 2))
    d = goal - q
    heading = np.arctan2(d[:, 1], d[:, 0])
    return q, np.zeros_like(q), heading, goal


def actuation_bounds(n, seed, heterogeneous=True):
    """Per-agent thrust limits.

    The heterogeneous case spreads the bounds over a factor of four, which is
    the setting a capability-weighted split is designed for.  The homogeneous
    case gives everyone the same bound.
    """
    if not heterogeneous:
        return np.ones(n)
    rng = np.random.RandomState(1000 + seed)
    return rng.uniform(0.5, 2.0, n)
