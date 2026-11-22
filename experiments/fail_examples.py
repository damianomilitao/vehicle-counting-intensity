"""Supplementary figure: one representative error of the proposed method per cause.

For each example the frame at the error time is cropped around the ROI and pixelated
(about 70 blocks across the crop, at least 3 pixels of the original frame per block) so
that license plates and faces cannot be read; the ROI outline is drawn afterwards. The right panel shows D(t) (global median,
full resolution) over +-4 s with the threshold h, the reference events and the detections.
Usage: python fail_examples.py   (paths as in common.py / core.py)
"""
import json
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import E, FPS, sig_full, gt
from common import paths

EX = [  # (cause code, recording, error type, time in s) chosen from fail/errors_classified.csv
    ('M', 'S1', 'FN', 134.258),
    ('S', 'B', 'FP', 86.590),
    ('Dp', 'raul_tarde_chuva', 'FN', 132.631),
    ('St', 'raul_tarde_chuva', 'FP', 140.303),
    ('LC', 'olaria_dia', 'FN', 191.213),
    ('Adj', 'S2', 'FP', 202.840),
    ('HL', 'olaria_noite', 'FP', 172.744),
    ('EX', 'olaria_noite', 'FP', 156.520),
    ('RN', 'S2', 'FP', 115.021)]
CAUSE = {'M': 'Merged events (FN)', 'S': 'One vehicle, two peaks (FP)',
         'Dp': 'Peak displaced by more than $\\tau$ (FN + FP)', 'St': 'Vehicle stopped on the ROI (FP)',
         'LC': 'Low contrast (FN)', 'Adj': 'Vehicle in the adjacent lane (FP)',
         'HL': 'Headlight reflection on wet pavement (FP)', 'EX': 'Automatic exposure change (FP)',
         'RN': 'Rain / spray, no vehicle (FP)'}
LETTER = {'B': 'B', 'olaria_dia': 'C', 'olaria_tarde_chuva': 'D', 'olaria_noite': 'E',
          'raul_tarde_chuva': 'F', 'raul_noite': 'G', 'S1': 'H', 'S2': 'I'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_SIG, C_REF, C_DET = '#2a78d6', '#008300', '#0b0b0b'
# on-screen camera label of the surveillance recordings (1920x1080 coordinates): removed by
# inpainting, as in the main-text figure of the camera views
OSD = {'olaria': (800, 420, 1340, 478), 'raul': (410, 235, 955, 295)}


def remove_label(im, n):
    site = 'olaria' if n.startswith('olaria') else 'raul' if n.startswith('raul') else None
    if site is None:
        return im
    x0, y0, x1, y1 = OSD[site]
    hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 185) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
    full = np.zeros(im.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = m
    return cv2.inpaint(im, full, 7, cv2.INPAINT_TELEA)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'TeX Gyre Termes', 'Liberation Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'savefig.dpi': 300, 'pdf.fonttype': 42})

errs = pd.read_csv(E + 'fail/errors_classified.csv')


def frame_at(video, k):
    cap = cv2.VideoCapture(video)
    i = 0
    fr = None
    while i <= k:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
    cap.release()
    return fr


def crop_pixelated(n, t):
    vid, roi, st, win, role = paths(n)
    fps = FPS[n]
    k = int(round(t * fps))
    fr = remove_label(frame_at(vid, k), n)
    H, W = fr.shape[:2]
    poly = np.array(roi.polygon, float)
    if st:
        A = np.load(st)
        A = np.linalg.inv(A[min(k, len(A) - 1)])
        q = np.c_[poly, np.ones(len(poly))] @ A.T
        poly = q[:, :2] / q[:, 2:]
    P = poly * [W, H]
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(1.3 * max(x1 - x0, y1 - y0), 120)
    xa, xb = int(max(0, cx - 1.35 * half)), int(min(W, cx + 1.35 * half))
    ya, yb = int(max(0, cy - half)), int(min(H, cy + half))
    c = fr[ya:yb, xa:xb]
    h, w = c.shape[:2]
    block = max(3, int(round(w / 70)))   # about 70 blocks across the crop
    small = cv2.resize(c, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    c = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    Pc = np.round(P - [xa, ya]).astype(np.int32)
    lw = max(2, int(round(w / 250)))
    cv2.polylines(c, [Pc], True, (0, 230, 255), lw, lineType=cv2.LINE_AA)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


fig, axs = plt.subplots(len(EX), 2, figsize=(6.6, 1.25 * len(EX)),
                        gridspec_kw={'width_ratios': [1, 2.1]})
for r, (code, n, kind, t) in enumerate(EX):
    fps = FPS[n]
    row = errs[(errs.video == n) & (np.abs(errs.t - t) < 0.05)].iloc[0]
    h = row.h
    dets = np.array(json.loads(row.dets))
    img = crop_pixelated(n, t)
    a = axs[r, 0]
    a.imshow(img)
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title(f'Video {LETTER[n]}, t = {t:.1f} s', fontsize=7, color=INK, loc='left', pad=2)
    I = sig_full(n)
    D = np.abs(I - np.median(I))
    t0, t1 = t - 4, t + 4
    kk = np.arange(max(0, int(t0 * fps)), min(len(D), int(t1 * fps)))
    a = axs[r, 1]
    a.plot(kk / fps, D[kk], color=C_SIG, lw=0.9)
    a.axhline(h, color=INK2, lw=0.6, ls='--')
    g = gt(n)
    g = g[(g > t0) & (g < t1)]
    for x in g:
        a.axvline(x, color=C_REF, lw=0.8, ls=':')
    dd = dets[(dets > t0) & (dets < t1)]
    a.plot(dd, np.interp(dd, kk / fps, D[kk]), 'o', color=C_DET, ms=3)
    a.axvspan(t - 0.12, t + 0.12, color='#eda100', alpha=0.35, lw=0)
    a.set_xlim(t0, t1)
    a.set_title(CAUSE[code], fontsize=7, color=INK, loc='left', pad=2)
    a.tick_params(labelsize=6.5, length=2)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.set_ylabel('$D$ (gray levels)', fontsize=6.5)
    a.grid(True, color=GRID, lw=0.4)
    a.set_axisbelow(True)
axs[-1, 1].set_xlabel('Time (s)')
fig.tight_layout(h_pad=0.6, w_pad=0.8)
fig.savefig(E + 'figures/figS_failures.png')
fig.savefig(E + 'figures/figS_failures.pdf')
print('ok')
