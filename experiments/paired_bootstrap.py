"""Paired moving-block bootstrap of pooled F1 differences between methods (test segments, tau = 1 s).

The same 30-s block start times are used for all methods in each resample, so the
interval of the difference accounts for the correlation between methods evaluated on the
same vehicles. Writes results/paired_differences.csv
"""
import json
import numpy as np
import pandas as pd
from core import E, TESTS, split

rng = np.random.default_rng(20261120)
items = json.load(open(E + 'results/items_tau1.0.json'))
METHODS = ['proposed_global', 'proposed_moving', 'proposed_causal', 'framediff', 'mog2', 'yolo']
L, B = 30.0, 2000
arr = {}
for n in TESTS:
    for m in METHODS:
        it = items[f'{n}|{m}']
        arr[(n, m)] = (np.array([x[0] for x in it], float), np.array([x[1] for x in it]))
segs = {n: split(n) for n in TESTS}
F = {m: [] for m in METHODS}
for _ in range(B):
    starts = {n: rng.uniform(a, max(a, b - L), int(np.ceil((b - a) / L))) for n, (a, b) in segs.items()}
    for m in METHODS:
        tp = fp = fn = 0
        for n in TESTS:
            t, k = arr[(n, m)]
            for s in starts[n]:
                sel = (t >= s) & (t < s + L)
                tp += (k[sel] == 'TP').sum()
                fp += (k[sel] == 'FP').sum()
                fn += (k[sel] == 'FN').sum()
        F[m].append(2 * tp / (2 * tp + fp + fn))
F = {m: np.array(v) for m, v in F.items()}
rows = []
for ref in ('proposed_global',):
    for m in METHODS:
        if m == ref:
            continue
        d = F[m] - F[ref]
        rows.append(dict(method=m, reference=ref, diff_median=np.median(d), lo=np.percentile(d, 2.5),
                         hi=np.percentile(d, 97.5), frac_positive=(d > 0).mean()))
out = pd.DataFrame(rows)
out.to_csv(E + 'results/paired_differences.csv', index=False)
print(out.round(3).to_string())
