"""Fig. 9  Certificate-optimal responsibility allocation."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as pubstyle
from style import BLUE, RED, GREY, GREEN, INK, ORANGE, DCOL, save

HET = [json.loads(l) for l in open('het_rows.jsonl')]
_s = {m: {(r['N'], r['seed']) for r in HET if r['mode'] == m}
      for m in ['centralized', 'uniform', 'capability', 'certificate']}
_common = set.intersection(*_s.values())
R = [r for r in HET if (r['N'], r['seed']) in _common]
MODES = [('centralized', BLUE, '^', 'centralized (reference)'),
         ('uniform', GREY, 'o', 'uniform split'),
         ('capability', ORANGE, 's', 'capability-weighted'),
         ('certificate', RED, 'D', 'certificate-optimal (proposed)')]
KEY = 'infstep'
Ns = sorted({int(r['N']) for r in R if
             len([x for x in R if x['N'] == r['N']
                  and x['mode'] == 'certificate']) >= 3})

def agg(N, m, key):
    v = np.array([r[key] for r in R if r['N'] == N and r['mode'] == m])
    return v.mean(), v.std(), len(v)


Ns = sorted({int(r['N']) for r in R
             if len([x for x in R if x['N'] == r['N']
                     and x['mode'] == 'certificate']) >= 3})

fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.35))
fig.subplots_adjust(wspace=0.46)

Ns = sorted({int(r['N']) for r in R})
W = 0.19


def grouped_box(a, key, scale=1.0):
    """one box per (fleet size, allocation); distributions are skewed, so
    median and IQR rather than mean and standard deviation"""
    for t, (m, col, mk, lab) in enumerate(MODES):
        data = [np.array([r[key] for r in R if r['N'] == N
                          and r['mode'] == m]) * scale for N in Ns]
        pos = np.arange(len(Ns)) + (t - 1.5) * W
        bp = a.boxplot(data, positions=pos, widths=W * 0.86, patch_artist=True,
                       showfliers=True, manage_ticks=False,
                       flierprops=dict(marker='.', ms=1.4, mfc=col, mec='none',
                                       alpha=1.0),
                       medianprops=dict(color='#111111', lw=0.9),
                       whiskerprops=dict(lw=0.6, color=col),
                       capprops=dict(lw=0.6, color=col),
                       boxprops=dict(lw=0.5, edgecolor=col))
        for patch in bp['boxes']:
            patch.set_facecolor(col); patch.set_alpha(1.0)
        a.plot([], [], color=col, lw=4, alpha=1.0, label=lab)
    a.set_xticks(np.arange(len(Ns)))
    a.set_xticklabels([str(N) for N in Ns])
    a.set_xlim(-0.5, len(Ns) - 0.5)
    a.set_xlabel('fleet size  $N$')


a = ax[0]
grouped_box(a, KEY, 100.0)
a.set_ylabel('steps with an infeasible\nprogram  (%)')
a.set_ylim(-3, None)
pubstyle.legend(a, loc='upper left', fontsize=5.9)
pubstyle.panel(a, 'a', 'local feasibility')

a = ax[1]
grouped_box(a, 'hmin')
a.axhline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
a.set_ylabel(r'closest approach  $\min_t\min_k h_k$', labelpad=1)
ylo2, yhi2 = -0.22, 0.13
a.set_ylim(ylo2, yhi2)
a.axhspan(ylo2, 0, color='#f6dcd8', alpha=1.0, lw=0, zorder=0)
a.text(-0.42, yhi2 * 0.80, 'safe', fontsize=6.4, color=GREEN, ha='left')
a.text(-0.42, ylo2 * 0.80, 'safety distance violated', fontsize=6.4,
       color=RED, ha='left')
pubstyle.panel(a, 'b', 'safety margin')

a = ax[2]


def wilson(k, n, z=1.96):
    """Wilson score interval: valid near 0 and 1, unlike the normal approx."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(c - h, 0), 100 * min(c + h, 1)


frac, lo, hi, cols = [], [], [], []
for m, col, mk, lab in MODES:
    hm = np.array([r['hmin'] for r in R if r['mode'] == m])
    p, l, u = wilson(int((hm < 0).sum()), len(hm))
    frac.append(p); lo.append(p - l); hi.append(u - p); cols.append(col)
a.bar(range(len(frac)), frac, color=cols, width=0.62, lw=0,
      yerr=[lo, hi], error_kw=dict(lw=0.8, capsize=2.0, ecolor='#444444'))
for x, y, u in zip(range(len(frac)), frac, hi):
    a.text(x, y + u + 2.0, '%.0f%%' % y, ha='center', va='bottom', fontsize=6.8)
a.set_xticks(range(len(frac)))
a.set_xticklabels(['central', 'uniform', 'capab.', 'certif.'], fontsize=6.2,
                  rotation=18, ha='right')
a.set_ylabel('runs with a safety violation  (%)')
a.set_ylim(0, 100)
pubstyle.panel(a, 'c', 'pooled over %d runs each'
               % len([r for r in R if r['mode'] == 'uniform']))

save(fig, 'fig9_allocation')

print('pooled:')
for m, _, _, _ in MODES:
    v = np.array([r[KEY] for r in R if r['mode'] == m])
    h = np.array([r['hmin'] for r in R if r['mode'] == m])
    t = np.array([r['ms'] for r in R if r['mode'] == m])
    print(f'  {m:<12} infeas {v.mean():.4f}+-{v.std():.4f}  '
          f'minh {h.mean():+.4f}  unsafe {int((h<0).sum())}/{len(h)}  '
          f'ms {t.mean():.2f}  n={len(v)}')
