import os
"""Dusk recording of the Olaria camera (20 Nov 2026, 17:53-18:53; sunset 18:13).

extract : ROI mean gray level and frame difference at 1280x720 (INTER_AREA), with the decoded
          timestamp of every frame (this stream has short gaps, so times come from the timestamps).
evaluate: global, centred moving and causal median baselines of the proposed counter and frame
          differencing on the two annotated intervals (before and after sunset),
          calibration on the first 40 %, test on the remaining 60 % (same protocol as the paper).
Ground truth: gt/olaria_transicao[2]_A.csv and _B.csv (two annotators; frame indices are exact)
          and gt/olaria_transicao2_adjudication.csv.
Writes dusk/olaria_transicao.npz and results/dusk.csv.
Usage: python dusk.py [extract] [evaluate]   (the recording is not redistributed)"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from common import E
from core import causal_median, match, H_P, D_S, H_FD

VIDEO = E + 'videos/olaria_transicao_20261120_1753.ts'
NPZ = E + 'dusk/olaria_transicao.npz'
# annotated intervals (frame indices; 150-750 s and 1050-1650 s in the annotation tool)
WINDOWS = {'before_sunset': (3992, 19962, 'olaria_transicao'),
           'after_sunset': (27947, 43916, 'olaria_transicao2')}


def extract(max_frames=44500):
    import cv2
    poly = np.array(json.load(open(E + 'rois/olaria.json'))['polygon_normalized'], float)
    W, H = 1280, 720
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly * [W, H]).astype(np.int32)], 1)
    m = m.astype(bool)
    cap = cv2.VideoCapture(VIDEO); ts, I, FD = [], [], []; prev = None; t0 = None
    while cap.grab():
        ms = cap.get(cv2.CAP_PROP_POS_MSEC); t0 = ms if t0 is None else t0; t = (ms - t0) / 1000
        if len(ts) >= max_frames:
            break
        ok, fr = cap.retrieve()
        g = cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        r = g[m].astype(np.float32)
        ts.append(t); I.append(r.mean()); FD.append(np.abs(r - prev).mean() if prev is not None else 0.0); prev = r
    os.makedirs(E + 'dusk', exist_ok=True)
    np.savez(NPZ, t=np.array(ts), I=np.array(I), FD=np.array(FD), roi_px=m.sum())


def consensus_frames(stem):
    """Frames of the consensus events: mean frame of each matched pair (tau = 1 s) plus
    the unmatched marks accepted at adjudication (gt/<stem>_adjudication.csv, column keep)."""
    from scipy.optimize import linear_sum_assignment
    a = pd.read_csv(E + f'gt/{stem}_A.csv'); b = pd.read_csv(E + f'gt/{stem}_B.csv')
    C = np.abs(a.t_s.values[:, None] - b.t_s.values[None, :]); C = np.where(C <= 1.0, C, 1e6)
    r, c = linear_sum_assignment(C); ok = C[r, c] < 1e6
    fr = list(np.round((a.frame.values[r[ok]] + b.frame.values[c[ok]]) / 2).astype(int))
    adj = E + f'gt/{stem}_adjudication.csv'
    if os.path.exists(adj):
        j = pd.read_csv(adj); fr += list(j.frame[j.keep == 1].astype(int))
    return np.sort(np.array(fr))


def evaluate_window(name, f0, f1, stem, z):
    t = z['t']; G = t[consensus_frames(stem)]
    s = slice(f0, f1 + 1); ts = t[s]; I = z['I'][s].astype(float); FDs = z['FD'][s].astype(float)
    a, b = t[f0], t[f1]; fps = 1 / np.median(np.diff(ts)); tc = a + 0.4 * (b - a)

    def met(det, lo, hi, tau=1.0):
        g = G[(G >= lo) & (G < hi)]; d = np.asarray(det); d = d[(d >= lo) & (d < hi)]
        mg, md = match(g, d, tau); TP = int(mg.sum()); FP = int((~md).sum()); FN = int((~mg).sum())
        return dict(N_gt=len(g), N_det=len(d), TP=TP, FP=FP, FN=FN, P=TP / max(1, TP + FP),
                    R=TP / max(1, TP + FN), F1=2 * TP / max(1, 2 * TP + FP + FN), bias=len(d) - len(g))
    w = int(round(30 * fps))
    B = {'global': np.full_like(I, np.median(I)), 'moving': median_filter(I, size=w | 1, mode='nearest'),
         'causal': causal_median(I, w)}
    ours = lambda kind: (lambda h, d: ts[find_peaks(np.abs(I - B[kind]), height=h, distance=max(1, int(round(d * fps))))[0]])
    fdf = lambda h, d: ts[find_peaks(FDs, height=h, distance=max(1, int(round(d * fps))))[0]]
    rows = []
    for meth, f, grid in [('proposed_global', ours('global'), list(itertools.product(H_P, D_S))),
                          ('proposed_moving', ours('moving'), list(itertools.product(H_P, D_S))),
                          ('proposed_causal', ours('causal'), list(itertools.product(H_P, D_S))),
                          ('framediff', fdf, list(itertools.product(H_FD, D_S)))]:
        sc = {k: met(f(*k), a, tc)['F1'] for k in grid}
        mx = max(sc.values()); pl = sorted(k for k, v in sc.items() if v >= mx - 1e-9); k = pl[len(pl) // 2]
        rows.append(dict(window=name, method=meth, param=k, cal_F1=round(mx, 3), **met(f(*k), tc, b)))
    lvl = (np.median(I[ts < a + 60]), np.median(I[ts > b - 60]))
    print(f'{name}: {a:.1f}-{b:.1f} s, {len(G[(G >= a) & (G < b)])} vehicles, ROI level first/last minute {lvl[0]:.1f} / {lvl[1]:.1f}')
    return rows


def evaluate():
    z = np.load(NPZ); rows = []
    for name, (f0, f1, stem) in WINDOWS.items():
        if f1 < len(z['t']) and os.path.exists(E + f'gt/{stem}_A.csv'):
            rows += evaluate_window(name, f0, f1, stem, z)
    df = pd.DataFrame(rows).round(3)
    os.makedirs(E + 'results', exist_ok=True); df.to_csv(E + 'results/dusk.csv', index=False)
    print(df.to_string())


if __name__ == '__main__':
    steps = sys.argv[1:] or ['extract', 'evaluate']
    if 'extract' in steps:
        extract()
    if 'evaluate' in steps:
        evaluate()
