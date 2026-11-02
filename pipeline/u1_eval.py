"""Shared U1-dev evaluation helpers (v12): cut rule + 5-fold CV inside dev + metrics. Rule (PLAN-v11 Amendments 2-5): 2-cut = highest recall of >=2 s.t. tricky-calm false alarms <= .15
(ties -> highest cut); then 1-/3-cuts by quadratic kappa."""
import numpy as np
from sklearn.model_selection import KFold
GRID = np.round(np.arange(0.5, 2.9001, 0.025), 4)
def qwk(a, b, k=4):
    a, b = np.asarray(a, int), np.asarray(b, int); O = np.zeros((k, k)); np.add.at(O, (a, b), 1); W = (np.arange(k)[:, None] - np.arange(k)[None]) ** 2 / (k - 1) ** 2
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum(); return float(1 - (W * O).sum() / (W * Ex).sum())
def tune(f, y, tricky):
    hi = y >= 2; rec = lambda c: float((f[hi] >= c).mean()); fa = lambda c: float((f[tricky] >= c).mean())
    ok = [c for c in GRID if fa(c) <= .15] or [GRID[-1]]; best = max(rec(c) for c in ok); c2 = max(c for c in ok if rec(c) == best)
    bk, b = -2, None
    for c1 in GRID[GRID < c2]:
        for c3 in GRID[GRID > c2]:
            k = qwk(np.digitize(f, (c1, c2, c3)), y)
            if k > bk: bk, b = k, (float(c1), float(c2), float(c3))
    return b
def metrics(l, y, tricky):
    hi = y >= 2; p = l >= 2; d = dict(kappa=round(qwk(l, y), 3), exact=round(float((l == y).mean()), 3))
    for k in range(4): d[f"exact_{k}"] = round(float((l[y == k] == k).mean()), 3)
    d.update(rec2=round(float((p & hi).sum() / hi.sum()), 3), prec2=round(float((p & hi).sum() / max(1, p.sum())), 3), rec3=round(float(((l == 3) & (y == 3)).sum() / max(1, (y == 3).sum())), 3),
             prec3=round(float(((l == 3) & (y == 3)).sum() / max(1, (l == 3).sum())), 3), fa=round(float((l[tricky] >= 2).mean()), 3), n=int(len(y)), n_tricky=int(tricky.sum()))
    return d
def cv_eval(f, y, tricky, seed=0):
    """5-fold CV inside dev: cuts tuned on 4 folds, applied to the 5th. Returns (metrics, fold cuts, final cuts on all dev, in-sample metrics)."""
    lv = np.zeros(len(y), int); fc = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(f):
        c = tune(f[tr], y[tr], tricky[tr]); fc.append(c); lv[te] = np.digitize(f[te], c)
    cf = tune(f, y, tricky); return metrics(lv, y, tricky), fc, cf, metrics(np.digitize(f, cf), y, tricky)
