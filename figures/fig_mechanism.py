"""
Fig.  What the allocation actually does.

(a) One conflict.  The same shared constraints, split two ways: each agent's
    local margin m_i^loc under the uniform split and under the optimiser of
    the LP.  Uniform leaves several agents with a negative margin -- their
    local programs have no solution -- while the LP lifts the worst of them.
(b) The geometry behind it (Lemma 28).  For one agent, each incident
    constraint is an affine function of that agent's own input; m_i^loc is
    the highest the lowest of them can be pushed within the input set.
    Re-allocating shifts the lines vertically by theta_{k,i} b_k.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as pubstyle
from style import BLUE, RED, GREY, GREEN, INK, ORANGE, COL, DCOL, save

from mrcbf import (pairs, rows, asym, alloc_uniform, alloc_capability,
                   alloc_exact, m_local, share, U0, RS)

# ---- find a conflict where the uniform split leaves someone infeasible ---
best = None
for seed in range(120):
    for N in (8, 10):
      for sq in (0.70, 0.65, 0.60, 0.55, 0.50):
        E = pairs(N)
        q, v, ang, goal = asym(N, seed)
        q = q * sq
        G, b, actm = rows(q, v, ang, E, RS)
        act = list(np.where(actm)[0])
        if len(act) < 4:
            continue
        Ei = {i: [k for k in act if i in E[k]] for i in range(N)}
        thu, _ = alloc_uniform(G, b, E, act, N, U0)
        thc, _ = alloc_capability(G, b, E, act, N, U0)
        the, _ = alloc_exact(G, b, E, act, N, U0)
        mu = np.array([m_local(G, b, Ei[i], i,
                               {k: share(thu, k, i, E) for k in Ei[i]}, U0)
                       for i in range(N)])
        mc = np.array([m_local(G, b, Ei[i], i,
                               {k: share(thc, k, i, E) for k in Ei[i]}, U0)
                       for i in range(N)])
        me = np.array([m_local(G, b, Ei[i], i,
                               {k: share(the, k, i, E) for k in Ei[i]}, U0)
                       for i in range(N)])
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(me))):
            continue
        # the case that matters: uniform leaves agents infeasible, the LP
        # does not.  Score by how many agents are rescued.
        # the case the algorithm is for: uniform leaves agents infeasible,
        # the optimal split leaves none
        rescued = int(((mu < 0) & (me >= 0)).sum())
        if not (mu.min() < 0 <= me.min()):
            continue
        score = (rescued, -mu.min())
        if best is None or score > best[0]:
            best = (score, N, E, act, Ei, G, b, thu, thc, the, mu, mc, me)
_, N, E, act, Ei, G, b, thu, thc, the, mu, mc, me = best

# capability leverage at this state: independent of the allocation
Ga, ba = G[act], b[act]
K, n = Ga.shape
cvec = np.concatenate([ba, U0 * np.ones(n)])
Amat = np.vstack([np.hstack([Ga.T, -np.eye(n)]),
                  np.hstack([-Ga.T, -np.eye(n)])])
rr = linprog(cvec, A_ub=Amat, b_ub=np.zeros(2 * n),
             A_eq=np.concatenate([np.ones(K), np.zeros(n)])[None, :],
             b_eq=[1.0], bounds=[(0, None)] * K + [(None, None)] * n,
             method='highs')
lev = U0 * np.abs(Ga.T @ rr.x[:K])
print(f'  leverage dM/drho_i = ' + ' '.join(f'{x:.3f}' for x in lev)
      + f'   ({int((lev <= 1e-8).sum())} of {N} zero)')
print(f'N={N}  {len(act)} active pairs')
print(f'  uniform     min_i m_i = {mu.min():+.3f}   infeasible agents: '
      f'{int((mu < 0).sum())}/{N}')
print(f'  capability  min_i m_i = {mc.min():+.3f}   infeasible agents: '
      f'{int((mc < 0).sum())}/{N}')
print()
print(f'{"agent":>6}{"uniform":>11}{"capability":>13}{"certificate":>13}')
for _i in np.argsort(mu):
    print(f'{_i:>6}{mu[_i]:>11.4f}{mc[_i]:>13.4f}{me[_i]:>13.4f}')
print(f'  agents within 1e-3 of the certificate worst: '
      f'{int((np.abs(me - me.min()) < 1e-3).sum())} of {N}')
print(f'  certificate min_i m_i = {me.min():+.3f}   infeasible agents: '
      f'{int((me < 0).sum())}/{N}')

# ============================================================== figure
fig, ax = plt.subplots(1, 3, figsize=(DCOL, 1.95),
                       gridspec_kw={'width_ratios': [1.32, 0.86, 1.02]})
fig.subplots_adjust(wspace=0.38)
a = ax[0]

x = np.arange(N)
w = 0.38
a.axhline(0, color=INK, lw=0.8)
w = 0.27
a.bar(x - w, mu, width=w, color=GREY, lw=0, label='uniform split')
a.bar(x, mc, width=w, color=ORANGE, lw=0, label='capability-weighted')
a.bar(x + w, me, width=w, color=RED, lw=0, label='certificate-optimal (LP)')
lo = min(mu.min(), mc.min(), me.min())
a.axhspan(lo * 1.35, 0, color='#f6dcd8', alpha=0.45, lw=0, zorder=0)
a.text(N - 0.4, lo * 0.75, 'local program\ninfeasible', fontsize=6.3,
       color=RED, ha='right', va='center')
order = np.argsort(mu)
mu, mc, me = mu[order], mc[order], me[order]
a.cla()
a.axhline(0, color=INK, lw=0.8)
w = 0.27
a.bar(x - w, mu, width=w, color=GREY, lw=0, label='uniform split')
a.bar(x, mc, width=w, color=ORANGE, lw=0, label='capability-weighted')
a.bar(x + w, me, width=w, color=RED, lw=0, label='certificate-optimal (LP)')
lo = min(mu.min(), mc.min(), me.min())
a.axhspan(lo * 1.35, 0, color='#f6dcd8', alpha=0.45, lw=0, zorder=0)
a.text(0.4, lo * 0.72, 'local program infeasible', fontsize=6.3,
       color=RED, ha='left', va='center')
a.set_xticks(x)
a.set_xticklabels([str(int(o)) for o in order], fontsize=6.6)
a.set_xlabel('agent  $i$   (ordered by uniform margin)')
a.set_ylabel(r'local margin  $m_i^{\mathrm{loc}}$')
hi = 1.9
nout = int(((mu > hi) | (mc > hi) | (me > hi)).sum())
a.set_ylim(lo * 1.45, hi)
if nout:
    a.text(N - 0.55, hi * 0.30, '%d beyond\naxis' % nout, fontsize=5.9,
           color=GREY, ha='right', va='top')
a.axhline(mu.min(), color=GREY, lw=0.7, ls=(0, (2, 2)))
a.axhline(me.min(), color=RED, lw=0.7, ls=(0, (2, 2)))
a.annotate('worst margin\n$%+.3f \\rightarrow %+.3f$' % (mu.min(), me.min()),
           xy=(2.6, me.min()), xytext=(2.2, 0.95), fontsize=6.3, color=INK,
           arrowprops=dict(arrowstyle='-', lw=0.5, color=INK))
pubstyle.legend(a, loc='upper left', fontsize=6.4)
a.set_title('what the allocation changes', loc='left',
            fontweight='bold', fontsize=7.4, pad=4)

a = ax[1]
lo2 = order
a.bar(range(N), lev[lo2], width=0.66, lw=0,
      color=[RED if lev[i] > 1e-8 else '#c8c8c8' for i in lo2])
a.set_xticks(range(N))
a.set_xticklabels([str(int(o)) for o in lo2], fontsize=6.2)
a.set_xlabel('agent  $i$   (same order)')
a.set_ylabel(r'$\partial M/\partial\rho_i$')
a.set_ylim(0, max(lev.max() * 1.42, 1e-3))
a.text(N - 0.4, lev.max() * 1.30,
       'identical under all three\nallocations; %d of %d zero'
       % (int((lev <= 1e-8).sum()), N),
       fontsize=6.0, color=GREY, ha='right', va='top')
a.set_title('what it cannot change', loc='left', fontweight='bold',
            fontsize=7.4, pad=4)

# ---- (c) does the leverage predict the effect of an upgrade? -----------
def reserve_at(Uvec):
    cv = np.concatenate([ba, Uvec])
    r = linprog(cv, A_ub=Amat, b_ub=np.zeros(2 * n),
                A_eq=np.concatenate([np.ones(K), np.zeros(n)])[None, :],
                b_eq=[1.0], bounds=[(0, None)] * K + [(None, None)] * n,
                method='highs')
    return r.fun


top = int(np.argmax(lev)); zero = int(np.argmin(lev))
M0 = reserve_at(U0 * np.ones(n))
ds = np.linspace(0, 0.5, 26)
dtop, dzero = [], []
for d in ds:
    Ua = U0 * np.ones(n); Ua[top] *= (1 + d)
    Ub = U0 * np.ones(n); Ub[zero] *= (1 + d)
    dtop.append(reserve_at(Ua) - M0); dzero.append(reserve_at(Ub) - M0)

a = ax[2]
a.plot(100 * ds, ds * lev[top], color=GREY, lw=1.0, ls=(0, (3, 2)),
       label=r'predicted, $\delta\,\partial M/\partial\rho_i$')
a.plot(100 * ds, dtop, color=RED, lw=1.5,
       label='agent %d  (largest leverage)' % top)
a.plot(100 * ds, dzero, color=GREY, lw=1.5,
       label='agent %d  (zero leverage)' % zero)
a.set_xlabel('actuation increase  (%)')
a.set_ylabel(r'$\Delta M$', labelpad=1)
a.set_xlim(0, 50)
pubstyle.legend(a, loc='upper left', fontsize=5.9)
a.set_title('what does change it', loc='left', fontweight='bold',
            fontsize=7.4, pad=4)
a.text(49, max(dtop) * 0.34, 'measured and\npredicted coincide', fontsize=5.9,
       color=GREY, ha='right', va='center')

save(fig, 'fig10_allocmech')
