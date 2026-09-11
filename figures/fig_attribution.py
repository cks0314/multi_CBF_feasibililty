"""
Fig. 8  Localising a conflict, across configurations.

(a) One conflict instant, drawn in full: the pair named by the dual support,
    and the agents shaded by their capability leverage.
(b) Aggregate over every conflict instant harvested from closed-loop runs
    with N = 6..16: how often relaxing each candidate pair restores
    feasibility.
(c) How concentrated the blame is: the number of agents carrying nonzero
    leverage, and the size of the dual support, against fleet size.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as pubstyle
from style import BLUE, RED, GREY, GREEN, INK, ORANGE, DCOL, save

from mrcbf.model import Filter, pairs, rows_and_b, DT, KP, KD, TAU, WMAX, r_ag

U0, RS = 1.0, 2.4


def reserve(G, b, U, keep=None, want=False):
    idx = np.arange(len(b)) if keep is None else np.asarray(keep)
    if len(idx) < 2:
        return (np.nan, None) if want else np.nan
    Gk, bk = G[idx], b[idx]
    K, N = Gk.shape
    c = np.concatenate([bk, U * np.ones(N)])
    A = np.vstack([np.hstack([Gk.T, -np.eye(N)]),
                   np.hstack([-Gk.T, -np.eye(N)])])
    r = linprog(c, A_ub=A, b_ub=np.zeros(2 * N),
                A_eq=np.concatenate([np.ones(K), np.zeros(N)])[None, :],
                b_eq=[1.0], bounds=[(0, None)] * K + [(None, None)] * N,
                method='highs')
    if not r.success:
        return (np.nan, None) if want else np.nan
    return (r.fun, r.x[:K]) if want else r.fun


def asym(N, seed):
    rng = np.random.RandomState(seed)
    q = rng.uniform(-2.6, 2.6, (N, 2))
    for i in range(N):
        for _ in range(400):
            if all(np.linalg.norm(q[i] - q[j]) > 0.75 for j in range(i)):
                break
            q[i] = rng.uniform(-2.6, 2.6, 2)
    goal = rng.uniform(-2.6, 2.6, (N, 2))
    d = goal - q
    return q, np.zeros_like(q), np.arctan2(d[:, 1], d[:, 0]), goal


def admissible(q, v, th, E, act):
    for k in act:
        i, j = E[k]
        p, w = q[i] - q[j], v[i] - v[j]
        h = p @ p - (2 * r_ag) ** 2
        if h <= 0 or 2.0 * (p @ w) + endtoend.ABAR * h < 0:
            return False
    return True


# ------------------------------------------------------- harvest conflicts
print('harvesting conflict instants ...')
CON = []
for N in (6, 8, 10, 12, 14, 16):
    E = pairs(N)
    f = Filter(N, len(E))
    for seed in range(7):
        q, v, th, goal = asym(N, seed)
        taken, last = 0, -99
        for step in range(int(5.0 / DT)):
            ades = -KP * (q - goal) - KD * v
            thd = np.arctan2(ades[:, 1], ades[:, 0])
            e = np.stack([np.cos(th), np.sin(th)], 1)
            un = np.clip(np.einsum('im,im->i', ades, e), -U0, U0)
            G, b, actm = rows_and_b(q, v, th, 0.0, E, RS, thd)
            act = np.where(actm)[0]
            if (len(act) >= 3 and taken < 6 and step - last > 10
                    and admissible(q, v, th, E, act)):
                M, lam = reserve(G[act], b[act], U0, want=True)
                if np.isfinite(M) and M < 0:
                    CON.append(dict(N=N, E=E, act=act, q=q.copy(), v=v.copy(),
                                    th=th.copy(), G=G.copy(), b=b.copy(),
                                    M=M, lam=lam))
                    taken += 1; last = step
            u, _ = f(G, b, un, U0)
            thdot = np.clip(((thd - th + np.pi) % (2 * np.pi) - np.pi) / TAU,
                            -WMAX, WMAX)
            th = th + DT * thdot
            v = v + DT * (u[:, None] * np.stack([np.cos(th), np.sin(th)], 1))
            q = q + DT * v
    print(f'  N={N}: {len(CON)} conflicts so far')

print(f'\nanalysing {len(CON)} conflicts ...')
rng = np.random.RandomState(0)
STRAT = ['certificate', 'closest', r'smallest $h$', r'smallest $\psi$',
         'random']
win = {s: 0 for s in STRAT}
percon = []                      # per-conflict outcome, for stratifying by N
nlev, nsupp, Ns, Ks = [], [], [], []
for n, C in enumerate(CON):
    E, act, lam = C['E'], C['act'], C['lam']
    Ga, ba = C['G'][act], C['b'][act]
    q, v = C['q'], C['v']
    d_, h_, p_ = [], [], []
    for k in act:
        i, j = E[k]
        p, w = q[i] - q[j], v[i] - v[j]
        h = p @ p - (2 * r_ag) ** 2
        d_.append(np.linalg.norm(p)); h_.append(h)
        p_.append(2.0 * p @ w + endtoend.ABAR * h)
    supp = np.where(lam > 1e-7)[0]
    cand = {'certificate': int(supp[np.argmax(lam[supp])]),
            'closest': int(np.argmin(d_)),
            r'smallest $h$': int(np.argmin(h_)),
            r'smallest $\psi$': int(np.argmin(p_))}
    row = {}
    for s, t in cand.items():
        keep = [u_ for u_ in range(len(act)) if u_ != t]
        Mk = reserve(Ga, ba, U0, keep)
        row[s] = int(np.isfinite(Mk) and Mk >= 0)
        win[s] += row[s]
    hits = 0
    for _ in range(12):
        t = rng.randint(len(act))
        keep = [u_ for u_ in range(len(act)) if u_ != t]
        Mk = reserve(Ga, ba, U0, keep)
        hits += int(np.isfinite(Mk) and Mk >= 0)
    win['random'] += hits / 12.0
    row['random'] = hits / 12.0
    percon.append([C['N']] + [row[s] for s in STRAT])
    lev = U0 * np.abs(Ga.T @ lam)
    nlev.append(int(np.sum(lev > 1e-8)))
    nsupp.append(len(supp)); Ns.append(C['N']); Ks.append(len(act))
    if (n + 1) % 40 == 0:
        print(f'  {n+1}/{len(CON)}')

n = len(CON)
nlev, nsupp, Ns, Ks = map(np.array, (nlev, nsupp, Ns, Ks))
print('\nrestores feasibility (fraction of %d conflicts):' % n)
for s in STRAT:
    print(f'  {s:<18} {win[s]/n:6.3f}')
print(f'\ndual support: mean {nsupp.mean():.2f}, max {nsupp.max()}')
print(f'agents with leverage: mean {nlev.mean():.2f}, max {nlev.max()}, '
      f'always <= 2*|supp|: {bool(np.all(nlev <= 2*nsupp))}')
print(f'enforced pairs: {Ks.min()}-{Ks.max()}')

# ------------------------------------------------------------- the snapshot
S = CON[int(np.argmin([c['M'] for c in CON]))]
E, act, lam = S['E'], S['act'], S['lam']
Ga, ba = S['G'][act], S['b'][act]
q, v, th, N = S['q'], S['v'], S['th'], S['N']
supp = np.where(lam > 1e-7)[0]
kstar = int(supp[np.argmax(lam[supp])])
lev = U0 * np.abs(Ga.T @ lam)
Mrel = reserve(Ga, ba, U0, [u_ for u_ in range(len(act)) if u_ != kstar])
print(f'\nsnapshot: N={N}, {len(act)} pairs, M={S["M"]:+.2f}, '
      f'|supp|={len(supp)}, relaxing the named pair -> {Mrel:+.2f}')

# ==================================================================== figure
fig = plt.figure(figsize=(DCOL, 2.65))
gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 0.82, 0.95, 0.90],
                      wspace=0.52)

a = fig.add_subplot(gs[0]); a.set_aspect('equal'); a.axis('off')
for t, k in enumerate(act):
    i, j = E[k]
    on = (t == kstar)
    a.plot(*zip(q[i], q[j]), color=(RED if on else '#d5d5d5'),
           lw=(2.6 if on else 0.8), zorder=3 if on else 1,
           solid_capstyle='round')
lmax = max(lev.max(), 1e-12)
for i in range(N):
    a.add_patch(Circle(q[i], r_ag, fc=BLUE, alpha=0.10 + 0.72 * lev[i] / lmax,
                       lw=0, zorder=3))
    a.add_patch(Circle(q[i], r_ag, fc='none', ec=INK,
                       lw=(1.5 if lev[i] > 1e-8 else 0.8), zorder=4))
    a.text(q[i][0], q[i][1] + r_ag * 1.75, str(i), fontsize=5.4,
           ha='center', va='center', zorder=7,
           fontweight=('bold' if lev[i] > 1e-8 else 'normal'),
           color=(RED if lev[i] > 1e-8 else GREY),
           bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.85))
    a.add_patch(FancyArrowPatch(q[i], q[i] + 0.55 * v[i], arrowstyle='-|>',
                                mutation_scale=6.5, lw=1.1, color=BLUE,
                                zorder=5, shrinkA=0, shrinkB=0))
    e = np.array([np.cos(th[i]), np.sin(th[i])])
    a.add_patch(FancyArrowPatch(q[i], q[i] + 0.44 * e, arrowstyle='-|>',
                                mutation_scale=6.5, lw=1.1, color=ORANGE,
                                zorder=5, shrinkA=0, shrinkB=0))
lim = np.abs(q).max() + 0.75
a.set_xlim(-lim, lim); a.set_ylim(-lim * 1.30, lim * 1.06)
a.text(-lim * 0.96, lim * 0.98,
       '$N=%d$,  %d enforced pairs\n$M=%+.2f$   ·   $|\\mathrm{supp}\\,\\lambda^{\\star}|=%d$'
       % (N, len(act), S['M'], len(supp)), fontsize=6.6, va='top', color=INK)
a.plot([], [], color=BLUE, lw=1.1, label='velocity')
a.plot([], [], color=ORANGE, lw=1.1, label='body axis')
a.plot([], [], color=RED, lw=2.4, label=r'pair named by $\lambda^{\star}$')
a.scatter([], [], s=24, fc=BLUE, ec=INK, lw=1.4,
          label=r'leverage $\partial M/\partial\rho_i>0$')
a.legend(loc='lower center', ncol=2, handlelength=1.3, handletextpad=0.4,
         borderpad=0.25, columnspacing=0.9, fontsize=5.9)
a.set_title('(a)  one conflict', loc='left', fontweight='bold',
            fontsize=7.0, pad=3)

a = fig.add_subplot(gs[1])
order = np.argsort(-lev)
a.bar(range(N), lev[order], width=0.66, lw=0,
      color=[RED if lev[i] > 1e-8 else '#c8c8c8' for i in order])
a.set_xticks(range(N))
a.set_xticklabels([str(int(o)) for o in order], fontsize=6.0)
a.set_xlabel('agent')
a.set_ylabel(r'$\partial M/\partial\rho_i$')
a.set_ylim(0, lev.max() * 1.32)
a.text(N - 0.4, lev.max() * 1.18, '%d of %d\nexactly zero'
       % (int((lev <= 1e-8).sum()), N), fontsize=6.0, color=GREY,
       ha='right', va='top')
a.set_title('(b)  per-agent leverage', loc='left',
            fontweight='bold', fontsize=7.0, pad=3)

a = fig.add_subplot(gs[2])
vals = [win[s] / n for s in STRAT]
cols = [GREEN, RED, RED, RED, GREY]
a.bar(range(len(STRAT)), vals, color=cols, width=0.66, lw=0)
for x, y in zip(range(len(STRAT)), vals):
    a.text(x, y + 0.018, '%.2f' % y, ha='center', va='bottom', fontsize=6.6)
a.set_xticks(range(len(STRAT)))
a.set_xticklabels(STRAT, fontsize=6.4, rotation=26, ha='right')
a.set_ylabel('conflicts where relaxing\nthat pair restores feasibility')
a.set_ylim(0, 1.12)
a.set_title('(c)  %d conflicts' % n, loc='left',
            fontweight='bold', fontsize=7.0, pad=3)

a = fig.add_subplot(gs[3])
jx = (rng.rand(len(Ns)) - 0.5) * 0.9
jy = (rng.rand(len(Ns)) - 0.5) * 0.22
uu = np.unique(Ns)
a.plot(uu, uu, color=GREY, lw=0.9, ls=(0, (2, 2)),
       label='all $N$ agents (for scale)')
a.scatter(Ns + jx, nlev + jy, s=13, c=BLUE, alpha=0.55, lw=0, zorder=3)
a.scatter(Ns + jx, nsupp + jy, s=22, c=RED, marker='x', lw=0.8, alpha=0.75,
          zorder=4)
a.plot(uu, [np.median(nlev[Ns == m]) for m in uu], color=BLUE, lw=1.4,
       zorder=5)
a.plot(uu, [np.median(nsupp[Ns == m]) for m in uu], color=RED, lw=1.4,
       ls=(0, (3, 2)), zorder=5)
a.set_ylim(0, max(uu) * 1.06)
a.set_xlabel('fleet size  $N$')
a.set_ylabel('count')
a.set_xticks(uu)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=GREY, lw=0.9, ls=(0, (2, 2)),
                  label='all $N$ agents (scale)'),
           Line2D([], [], color=BLUE, lw=1.4, marker='o', ms=3.2,
                  markerfacecolor=BLUE, markeredgecolor='none',
                  label='agents with leverage'),
           Line2D([], [], color=RED, lw=1.4, ls=(0, (3, 2)), marker='x',
                  ms=3.6, mew=0.9,
                  label=r'$|\mathrm{supp}\,\lambda^{\star}|$')]
lg = a.legend(handles=handles, loc='upper left', fontsize=6.1,
              handlelength=1.9, handletextpad=0.5, labelspacing=0.3,
              borderpad=0.3)
lg.get_frame().set_linewidth(0.5)
a.text(uu[-1] + 0.4, max(uu) * 0.34,
       'points: individual\nconflicts\nlines: median',
       fontsize=5.7, color=GREY, ha='right', va='top')
a.set_title('(d)  vs fleet size', loc='left', fontweight='bold',
            fontsize=7.0, pad=3)

save(fig, 'fig8_attribution')

# ------------------------------------------------------------- teaser (1 col)
from style import COL
figt, at = plt.subplots(figsize=(COL, COL * 0.86))
at.set_aspect('equal'); at.axis('off')
for t, k in enumerate(act):
    i, j = E[k]
    on = (t == kstar)
    at.plot(*zip(q[i], q[j]), color=(RED if on else '#d0d0d0'),
            lw=(2.8 if on else 0.9), zorder=3 if on else 1,
            solid_capstyle='round')
for i in range(N):
    at.add_patch(Circle(q[i], r_ag, fc=BLUE,
                        alpha=0.10 + 0.72 * lev[i] / lmax, lw=0, zorder=3))
    at.add_patch(Circle(q[i], r_ag, fc='none', ec=INK,
                        lw=(1.6 if lev[i] > 1e-8 else 0.8), zorder=4))
    at.add_patch(FancyArrowPatch(q[i], q[i] + 0.58 * v[i], arrowstyle='-|>',
                                 mutation_scale=7, lw=1.2, color=BLUE,
                                 zorder=5, shrinkA=0, shrinkB=0))
    e = np.array([np.cos(th[i]), np.sin(th[i])])
    at.add_patch(FancyArrowPatch(q[i], q[i] + 0.46 * e, arrowstyle='-|>',
                                 mutation_scale=7, lw=1.2, color=ORANGE,
                                 zorder=5, shrinkA=0, shrinkB=0))
at.set_xlim(-lim, lim); at.set_ylim(-lim * 1.34, lim * 1.10)
at.text(0, lim * 1.02, 'Infeasible safety filter', ha='center',
        fontsize=8.2, fontweight='bold', color=INK)
at.text(0, lim * 0.80,
        '%d agents,  %d enforced pairwise constraints'
        % (N, len(act)), ha='center', fontsize=6.8, color=GREY)
at.plot([], [], color=RED, lw=2.6,
        label='constraint identified by $\\lambda^{\\star}$')
at.scatter([], [], s=26, fc=BLUE, ec=INK, lw=1.5,
           label='agents with nonzero $\\partial M/\\partial\\rho_i$')
at.plot([], [], color=BLUE, lw=1.2, label='velocity')
at.plot([], [], color=ORANGE, lw=1.2, label='thrust axis')
at.legend(loc='lower center', ncol=2, handlelength=1.3, handletextpad=0.4,
          borderpad=0.3, columnspacing=0.8, fontsize=5.8)
save(figt, 'fig0_teaser')
np.save('attrib_stats.npy', np.c_[Ns, Ks, nsupp, nlev])
P = np.array(percon, float)
np.save('attrib_percon.npy', P)
print('\nrestoration rate stratified by fleet size:')
print(f"{'N':>4}{'n':>5}" + ''.join(f'{s.replace(chr(36),str()):>14}'
                                    for s in STRAT))
for m in np.unique(P[:, 0]):
    sel = P[:, 0] == m
    print(f'{int(m):>4}{int(sel.sum()):>5}' +
          ''.join(f'{P[sel, 1 + k].mean():>14.2f}' for k in range(len(STRAT))))
