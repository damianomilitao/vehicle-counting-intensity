"""Figures of the revised manuscript and Supplementary Material.

Reads the result files written by the analysis scripts (results/*.csv) and writes
  fig_resolution.(png|pdf)      main text, F1 versus processing width
  figS_sweep_videos.(png|pdf)   Supplementary Material, F1 versus width per recording
  figS_sensitivity.(png|pdf)    Supplementary Material, pooled F1 over (h, dt) at four widths
Usage: python make_figures.py RESULTS_DIR OUT_DIR
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'results')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'figures')
os.makedirs(OUT, exist_ok=True)

# colours: three categorical slots (validated for colour-vision deficiency, all pairs)
C_AREA, C_LIN, C_FD = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'TeX Gyre Termes', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'lines.linewidth': 1.3,
    'savefig.dpi': 400, 'pdf.fonttype': 42})

NAMES = {'B': 'B', 'olaria_dia': 'C', 'olaria_tarde_chuva': 'D', 'olaria_noite': 'E',
         'raul_tarde_chuva': 'F', 'raul_noite': 'G', 'S1': 'H', 'S2': 'I'}
DESC = {'B': 'smartphone, day', 'olaria_dia': 'camera, day', 'olaria_tarde_chuva': 'camera, rain',
        'olaria_noite': 'camera, night', 'raul_tarde_chuva': 'camera, rain', 'raul_noite': 'camera, night',
        'S1': 'smartphone, rain, glass', 'S2': 'smartphone, rain'}

sw = pd.read_csv(os.path.join(RES, 'sweep_raw.csv'))
sw = sw[sw.kind == 'p']
HD = [v for v in NAMES if v != 'B']          # the seven 1920x1080 recordings


def pooled(df, variant, mode):
    g = df[df.variant == variant].groupby('w')[[f'TP_{mode}', f'FP_{mode}', f'FN_{mode}']].sum()
    tp, fp, fn = (g[f'{k}_{mode}'].values for k in ('TP', 'FP', 'FN'))
    return g.index.values, 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def wstar(w, f1, tol=0.02):
    """smallest width such that F1 >= F1(max width) - tol for every larger tested width"""
    order = np.argsort(w)[::-1]
    w, f1 = w[order], f1[order]
    ref = f1[0]
    ok = f1 >= ref - tol
    k = len(w) if ok.all() else int(np.argmin(ok))
    return w[k - 1]


def style(ax):
    ax.set_xscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.5)
    ax.grid(True, which='minor', axis='x', color=GRID, lw=0.3)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1280])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1280'])
    ax.minorticks_off()


SERIES = [('intensity_area', 'fixed', C_AREA, '-', 'Proposed, area, fixed'),
          ('intensity_area', 'retuned', C_AREA, (0, (3, 1.5)), 'Proposed, area, re-tuned'),
          ('intensity_linear', 'fixed', C_LIN, '-', 'Proposed, bilinear, fixed'),
          ('framediff_area', 'fixed', C_FD, '-', 'Frame diff., area, fixed'),
          ('framediff_area', 'retuned', C_FD, (0, (3, 1.5)), 'Frame diff., area, re-tuned')]

# ---------------------------------------------------------------- main-text figure
fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
hd = sw[sw.video.isin(HD)]
summary = []
for var, mode, col, ls, lab in SERIES:
    w, f1 = pooled(hd, var, mode)
    axs[0].plot(w, f1, color=col, ls=ls, label=lab)
    summary.append(('pooled', var, mode, wstar(w, f1), f1[-1]))
for var, mode, col, ls, lab in SERIES:
    w, f1 = pooled(sw[sw.video == 'B'], var, mode)
    axs[1].plot(w, f1, color=col, ls=ls, label=lab)
    summary.append(('B', var, mode, wstar(w, f1), f1[-1]))
for ax in axs:
    style(ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('$F_1$ (test segments)')
axs[1].set_xlabel('Processing width $w$ (pixels; aspect ratio preserved)')
axs[0].set_title('(a) Seven 1920$\\times$1080 recordings (C$-$I), pooled', loc='left', color=INK)
axs[1].set_title('(b) Video B (848$\\times$478 file)', loc='left', color=INK)
axs[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0.04), frameon=False, handlelength=2.6,
              borderaxespad=0.2)
# w* of the proposed method (area, fixed), pooled set only (one event changes F1 of a single
# recording by about 0.03, more than the 0.02 tolerance)
ws_ = [s[3] for s in summary if s[0] == 'pooled' and s[1] == 'intensity_area' and s[2] == 'fixed'][0]
axs[0].axvline(ws_, color=INK2, lw=0.6, ls=':')
axs[0].text(ws_ * 0.96, 0.04, f'$w^*$ = {ws_} px', color=INK2, fontsize=6.5, va='bottom', ha='right')
fig.tight_layout(h_pad=0.8)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_resolution.{ext}'))
plt.close(fig)
pd.DataFrame(summary, columns=['set', 'variant', 'mode', 'w_star', 'F1_full']).to_csv(
    os.path.join(OUT, 'w_star.csv'), index=False)

# ---------------------------------------------------------------- SI: per recording
fig, axs = plt.subplots(4, 2, figsize=(6.8, 7.6), sharex=True, sharey=True)
for ax, v in zip(axs.ravel(), NAMES):
    d = sw[sw.video == v]
    for var, mode, col, ls, lab in SERIES:
        dd = d[d.variant == var].sort_values('w')
        ax.plot(dd.w, dd[f'F1_{mode}'], color=col, ls=ls, label=lab, lw=1.1)
    style(ax)
    ax.set_ylim(0, 1.0)
    roi = int(d[(d.variant == 'intensity_area')].sort_values('w').roi_px.iloc[-1])
    ax.set_title(f'Video {NAMES[v]} ({DESC[v]}; ROI {roi} px at full res.)', loc='left', color=INK)
for ax in axs[-1]:
    ax.set_xlabel('Processing width $w$ (pixels)')
for ax in axs[:, 0]:
    ax.set_ylabel('$F_1$')
axs[0, 0].legend(loc='lower right', frameon=False, handlelength=2.6, fontsize=6.5)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'figS_sweep_videos.{ext}'))
plt.close(fig)

# ---------------------------------------------------------------- SI: sensitivity heat maps
sg = pd.read_csv(os.path.join(RES, 'sens_grid.csv'))
cmap = LinearSegmentedColormap.from_list('seq', SEQ)
widths = list(dict.fromkeys(sg.width))
fig, axs = plt.subplots(1, len(widths), figsize=(6.9, 2.6), sharey=True)
for ax, wd in zip(axs, widths):
    d = sg[sg.width == wd].pivot(index='h', columns='dt', values='F1')
    im = ax.imshow(d.values, origin='lower', aspect='auto', cmap=cmap, vmin=0.2, vmax=0.75)
    ax.contour(d.values, levels=[np.nanmax(d.values) - 0.02], colors=[INK], linewidths=0.7)
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels([f'{x:g}' for x in d.columns], rotation=90)
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels([f'{x:g}' for x in d.index])
    i, j = np.unravel_index(np.nanargmax(d.values), d.shape)
    ax.plot(j, i, marker='o', ms=4.5, mfc='none', mec='#e34948', mew=1.2)
    ax.set_title(('full resolution' if str(wd) == 'full' else f'w = {wd} px') +
                 f'\nmax $F_1$ = {d.values[i, j]:.2f}', color=INK)
    ax.set_xlabel('$\\Delta t_{\\min}$ (s)')
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_ylabel('$h$ (gray levels)')
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cb.set_label('pooled $F_1$ (test segments)')
cb.outline.set_visible(False)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'figS_sensitivity.{ext}'), bbox_inches='tight')
plt.close(fig)
print(pd.read_csv(os.path.join(OUT, 'w_star.csv')).to_string())

# ---------------------------------------------------------------- main text: D(t) of Video B
import json
from scipy.optimize import linear_sum_assignment
SIG = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, 'full')
GT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(BASE, 'gt')


def match(g, d, tau):
    g, d = np.asarray(g, float), np.asarray(d, float)
    mg, md = np.zeros(len(g), bool), np.zeros(len(d), bool)
    if len(g) and len(d):
        C = np.abs(g[:, None] - d[None, :])
        C = np.where(C <= tau, C, 1e6)
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 1e6
        mg[r[ok]] = True
        md[c[ok]] = True
    return mg, md


fps_B, h_B, dt_B, t0, t1 = 60.00672656380572, 8, 1.0, 80.0, 110.0
I = np.load(os.path.join(SIG, 'B.npz'))['I_area'][0].astype(float)
D = np.abs(I - np.median(I))
g = pd.read_csv(os.path.join(GT, 'B_consensus.csv')).t_s.values
g = g[(g >= 0) & (g < 125)]
det = np.array(json.load(open(os.path.join(RES, 'dets_tau1.0.json')))['B|proposed_global'])
mg, md = match(g, det, 1.0)
k = np.arange(int(t0 * fps_B), int(t1 * fps_B))
fig, ax = plt.subplots(figsize=(3.45, 2.15))
ax.plot(k / fps_B, D[k], color='#256abf', lw=0.7, zorder=2)
ax.axhline(h_B, color=INK2, lw=0.6, ls='--', zorder=1)
sel = (g > t0) & (g < t1)
for x, ok in zip(g[sel], mg[sel]):
    ax.axvline(x, color=INK2 if ok else '#e34948', lw=0.6 if ok else 0.9, ls=':' if ok else '--', zorder=1)
s = (det > t0) & (det < t1)
yy = np.interp(det, k / fps_B, D[k])
ax.plot(det[s & md], yy[s & md], 'o', ms=3.2, color='#0b0b0b', zorder=3)
ax.plot(det[s & ~md], yy[s & ~md], 'x', ms=4.5, mew=1.2, color='#e34948', zorder=3)
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=INK2, lw=0.6, ls=':'), Line2D([], [], color='#e34948', lw=0.9, ls='--'),
      Line2D([], [], color='#0b0b0b', marker='o', ms=3.2, ls=''),
      Line2D([], [], color='#e34948', marker='x', ms=4.5, mew=1.2, ls=''),
      Line2D([], [], color=INK2, lw=0.6, ls='--')]
ax.legend(hs, ['reference event, detected', 'reference event, missed (FN)', 'detection, matched (TP)',
               'detection, unmatched (FP)', f'threshold $h$ = {h_B} gray levels'], ncol=2, fontsize=6.3, frameon=False, loc='lower left',
          bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1, handlelength=1.8, columnspacing=1.0)
ax.set_xlim(t0, t1)
ax.set_ylim(0, max(D[k]) * 1.05)
ax.set_xlabel('Time (s)')
ax.set_ylabel('$D(t)$ (gray levels)')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis='y', color=GRID, lw=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_signal.{ext}'))
plt.close(fig)
print('TP', int((s & md).sum()), 'FP', int((s & ~md).sum()), 'FN', int((sel & ~mg).sum()))
