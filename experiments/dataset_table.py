"""Data-set summary (Table I) and reference-count uncertainty.

For every recording: frame rate, frame size, evaluation interval, calibration/test split,
consensus count (whole interval and test segment), the counts of each annotator in the test
segment, the flow, the ROI size at full processing resolution, and
    u(N_gt) = max(|N_A - N_gt|, |N_B - N_gt|) / sqrt(3)
i.e. a rectangular distribution whose half-width is the largest difference between the
count of either annotator and the consensus count in the test segment.
Writes results/table1_data.csv
"""
import numpy as np
import pandas as pd
from core import E, FPS, TESTS, Z, T_end, split, gt
from common import V
from evaluation import load_events

rows = []
for n in ['A'] + TESTS:
    z = Z(n)
    win = V[n][3]
    T = T_end(n)
    a, b = (0.0, T) if n == 'A' else split(n)
    g = gt(n)
    def cnt(x):
        x = np.asarray(x, float)
        return int(((x >= a) & (x < b)).sum())
    NA = cnt(load_events(E + f'gt/{n}_A.csv'))
    NB = cnt(load_events(E + f'gt/{n}_B.csv'))
    Ng = cnt(g)
    sizes = z['sizes']
    w_full, h_full = (int(sizes[0][0]), int(sizes[0][1]))
    roi_px = int(np.asarray(z['roi_px'])[0])
    W, H = (848, 478) if n == 'B' else (1920, 1080)
    rows.append(dict(video=n, fps=FPS[n], W=W, H=H, frames_eval=int(round(T * FPS[n])), T_eval=round(T, 1),
                     cal_s=round(a, 1), test_s=round(b - a, 1), N_gt=len(g), N_gt_test=Ng, N_A_test=NA, N_B_test=NB,
                     flow=int(round(3600 * len(g) / T)), roi_px=roi_px, roi_pct=100 * roi_px / (w_full * h_full),
                     full=f'{w_full}x{h_full}', u_Ngt=max(abs(NA - Ng), abs(NB - Ng)) / np.sqrt(3)))
d = pd.DataFrame(rows)
d.to_csv(E + 'results/table1_data.csv', index=False)
print(d.to_string())
