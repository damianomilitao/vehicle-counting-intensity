import os
import json, numpy as np, pandas as pd, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from core import *
R=E+'results/'; T=E+'tables/'; os.makedirs(T,exist_ok=True)
LAB={'A':'A','B':'B','olaria_dia':'C','olaria_tarde_chuva':'D','olaria_noite':'E','raul_tarde_chuva':'F','raul_noite':'G','S1':'H','S2':'I'}
DESC={'A':'site 1, smartphone','B':'site 2, smartphone','olaria_dia':'Olaria, camera','olaria_tarde_chuva':'Olaria, camera',
      'olaria_noite':'Olaria, camera','raul_tarde_chuva':'Raul Sert\\~a, camera','raul_noite':'Raul Sert\\~a, camera',
      'S1':'site 1, smartphone','S2':'site 1, smartphone'}
COND={'A':'day, dry','B':'day, dry','olaria_dia':'day, sunny','olaria_tarde_chuva':'day, rain, peak hour','olaria_noite':'night, street lights, wet',
      'raul_tarde_chuva':'day, rain, peak hour','raul_noite':'night, street lights, wet','S1':'day, rain, through window glass','S2':'day, rain'}
VIEW={'A':'window, $\\approx$10\\,m','B':'window, $\\approx$8\\,m, oblique','olaria_dia':'pole-mounted','olaria_tarde_chuva':'pole-mounted','olaria_noite':'pole-mounted',
      'raul_tarde_chuva':'pole-mounted','raul_noite':'pole-mounted','S1':'window, $\\approx$11\\,m','S2':'window, $\\approx$11\\,m'}
CODEC={'A':'H.264/MP4','B':'H.264/MP4','S1':'H.264/MP4','S2':'H.264/MP4'}
def f2(x): return f'{x:.2f}'
# ---- Table I
d=pd.read_csv(R+'table1_data.csv').set_index('video')
L=[]
for n in ['A']+TESTS:
    r=d.loc[n]; role='transfer calibration' if n=='A' else f"cal.\\ {r.cal_s:.0f}\\,s / test {r.test_s:.0f}\\,s"
    L.append(f"{LAB[n]} & {DESC[n]} & {VIEW[n]} & {COND[n]} & {r.fps:.1f} & {r.W}$\\times${r.H} & {r.T_eval:.1f} & {CODEC.get(n,'H.264/MPEG-TS')} & {int(r.roi_px)} ({r.roi_pct:.2f}) & {int(r.N_gt)} ({int(r.N_gt_test) if n!='A' else '--'}) & {int(r.flow)} & {role}\\\\")
open(T+'tab_data_rows.tex','w').write('\n'.join(L))
# ---- Table III agreement
a=pd.read_csv(R+'agreement.csv').set_index('video')
L=[]
for n in ['A']+TESTS:
    r=a.loc[n]
    L.append(f"{LAB[n]} & {int(r.N_A)} & {int(r.N_B)} & {int(r.matched)} & {int(r.only_A)} & {int(r.only_B)} & {f2(r.F1)} & {r.class_agree*100:.0f} & {r.off_mean:+.2f} & {r.off_sd:.2f} & {int(r.adjudicated)} & {int(r.N_gt)}\\\\")
open(T+'tab_agree_rows.tex','w').write('\n'.join(L))
# ---- Table IV main F1 per video x method
m=pd.read_csv(R+'main_tau1.0.csv'); p=pd.read_csv(R+'pooled_tau1.0.csv').set_index('method')
ORDER=['proposed_global','proposed_causal','framediff','mog2','yolo']
L=[]
for n in TESTS:
    s=m[m.video==n].set_index('method'); best=s.loc[ORDER].F1.max()
    cells=[]
    for k in ORDER:
        r=s.loc[k]; txt=f"{r.F1:.2f} [{r.F1_lo:.2f}, {r.F1_hi:.2f}]"
        cells.append(('\\textbf{'+txt+'}') if abs(r.F1-best)<1e-9 else txt)
    L.append(f"{LAB[n]} & {int(s.loc['proposed_global'].N_gt)} & "+' & '.join(cells)+'\\\\')
L.append('\\midrule')
cells=[]
bestp=max(p.loc[k].F1 for k in ORDER)
for k in ORDER:
    r=p.loc[k]; txt=f"{r.F1:.2f} [{r.F1_lo:.2f}, {r.F1_hi:.2f}]"; cells.append(('\\textbf{'+txt+'}') if abs(r.F1-bestp)<1e-9 else txt)
L.append(f"Pooled & {int(p.loc['proposed_global'].N_gt)} & "+' & '.join(cells)+'\\\\')
open(T+'tab_main_rows.tex','w').write('\n'.join(L))
# ---- Table V pooled metrics + cost
tim=json.load(open(R+'timing_pc.json'))['results']
try: tim+=json.load(open(R+'timing_pc_yolo.json'))['results']
except FileNotFoundError: pass
def tget(v,mth,w=None):
    for r in tim:
        if r['video']==v and r['method']==mth and (w is None or r['proc_w']==w): return r
    return None
NAME={'proposed_global':'Proposed, global median','proposed_moving':'Proposed, centred moving median','proposed_causal':'Proposed, causal median','framediff':'Frame differencing','mog2':'MOG2 virtual loop','yolo':'YOLOv8n + ByteTrack$^\\dagger$'}
TM={'proposed_global':'proposed_global','proposed_moving':None,'proposed_causal':'proposed_causal','framediff':'framediff','mog2':'mog2','yolo':'yolo'}
L=[]
for k in ['proposed_global','proposed_moving','proposed_causal','framediff','mog2','yolo']:
    r=p.loc[k]; t1=tget('olaria_noite',TM[k]) if TM[k] else None; t2=tget('B',TM[k]) if TM[k] else None
    if k.startswith('proposed') or k=='framediff':
        t1=tget('olaria_noite',TM[k],1280) if TM[k] else None; t2=tget('B',TM[k],848) if TM[k] else None
    fmt=lambda t: (f"{t['realtime_factor']*t['fps_video']:.0f}" if t else '--')   # video frames per second
    mem=lambda t: (f"{t['peak_working_set_mib']:.0f}" if t else '--')
    L.append(f"{NAME[k]} & {int(r.TP)} & {int(r.FP)} & {int(r.FN)} & {r.P:.2f} & {r.R:.2f} & {r.F1:.2f} [{r.F1_lo:.2f}, {r.F1_hi:.2f}] & {int(r.bias):+d} & {r.F1_day:.2f} / {r.F1_night:.2f} & {fmt(t1)} / {fmt(t2)} & {mem(t1)} / {mem(t2)}\\\\")
open(T+'tab_pooled_rows.tex','w').write('\n'.join(L))
# ---- transfer table (two panels, 7 columns)
tr=pd.read_csv(R+'transfer_from_A.csv'); cc=pd.read_csv(R+'cross_condition.csv')
PNAME={'proposed_global':'$h$, $\\Delta t_{\\min}$','proposed_causal':'$h$, $\\Delta t_{\\min}$','framediff':'$h$, $\\Delta t_{\\min}$','mog2':'on, $\\Delta t_{\\min}$'}
L=[]
for k in ['proposed_global','proposed_causal','framediff','mog2']:
    s_=tr[tr.method==k]; TP,FP,FN=s_.TP.sum(),s_.FP.sum(),s_.FN.sum(); own=p.loc[k]
    L.append(f"{NAME[k]} & \\multicolumn{{2}}{{c}}{{{s_.param.iloc[0].replace('(','').replace(')','')}}} & {2*TP/(2*TP+FP+FN):.2f} & {int(FP)} & {int(FN)} & {own.F1:.2f}\\\\")
open(T+'tab_transferA_rows.tex','w').write('\n'.join(L))
L=[]
for _,r in cc.iterrows():
    if r.method not in ('proposed_global','framediff','mog2'): continue
    L.append(f"{LAB[r.source]}$\\rightarrow${LAB[r.target]} & \\multicolumn{{2}}{{l}}{{{NAME[r.method].replace('Proposed, global median','Proposed (global)').replace(' virtual loop','')}}} & {r.F1_transfer:.2f} & \\multicolumn{{2}}{{c}}{{{int(r.bias_transfer):+d}}} & {r.F1_own:.2f}\\\\")
open(T+'tab_cross_rows.tex','w').write('\n'.join(L))
# ---- failure table
e=pd.read_csv(E+'fail/errors_classified.csv')
order=['M','S','Dp','St','LC','Adj','HL','EX','RN']
TXT={'M':'Merged events (short headway, side by side, occlusion)','S':'One vehicle producing two peaks (large, long or slow vehicle)',
 'Dp':'Peak displaced by more than $\\tau$ (slow or long vehicle)$^\\ast$','St':'Vehicle stopped on the ROI','LC':'Low contrast (dark or white vehicle on wet pavement, motorcycle, night)',
 'Adj':'Vehicle in the adjacent lane or partly over the ROI','HL':'Headlight reflections on the pavement (night, rain)','EX':'Automatic camera exposure change or slow illumination change','RN':'Rain, spray or wet-surface fluctuation without a vehicle'}
L=[]
for c in order:
    s=e[e.cause==c]; fp=(s.kind=='FP').sum(); fn=(s.kind=='FN').sum()
    vids=','.join(sorted({LAB[v] for v in s.video}))
    L.append(f"{TXT[c]} & {fp if fp else '--'} & {fn if fn else '--'} & {vids}\\\\")
L.append('\\midrule'); L.append(f"Total & {(e.kind=='FP').sum()} & {(e.kind=='FN').sum()} & \\\\")
open(T+'tab_fail_rows.tex','w').write('\n'.join(L))
print('ok')
# =====================================================================================
# additional main-text tables (resolution, Monte Carlo) and Supplementary tables
# =====================================================================================
T2 = os.path.join(os.path.dirname(T.rstrip('/')), 'si_tables') + '/'
os.makedirs(T2, exist_ok=True)
sw = pd.read_csv(R + 'sweep_raw.csv')
HD = [n for n in TESTS if n != 'B']


def pooled_row(df):
    out = {}
    for var in ('intensity_area', 'intensity_linear', 'framediff_area'):
        s = df[df.variant == var]
        for mode in ('fixed', 'retuned'):
            tp, fp, fn = s[f'TP_{mode}'].sum(), s[f'FP_{mode}'].sum(), s[f'FN_{mode}'].sum()
            out[(var, mode)] = (2 * tp / (2 * tp + fp + fn), int(s[f'N_det_{mode}'].sum() - s.N_gt.sum()))
    return out


def nearest(ws, target):
    ws = np.array(sorted(ws))
    return int(ws[np.argmin(np.abs(ws - target))])


for label, vids, full in (('hd', HD, 1280), ('b', ['B'], 848)):
    L = []
    d = sw[sw.video.isin(vids)]
    ws = sorted(d[d.kind == 'p'].w.unique())
    targets = [full, 640, 320, 160, 80, 48, 32, 16]
    rows = []
    for tw in targets:
        w = nearest(ws, tw)
        if rows and rows[-1][0] == w:
            continue
        rows.append((w, d[(d.kind == 'p') & (d.w == w)]))
    for wh in ((640, 480), (320, 240), (160, 120)):
        rows.append((wh, d[(d.kind == 's') & (d.w == wh[0]) & (d.h == wh[1])]))
    for w, s in rows:
        if s.empty:
            continue
        hh = int(s.h.iloc[0])
        ww = w if isinstance(w, int) else w[0]
        tag = '' if isinstance(w, int) else '$^\\dagger$'
        roi = int(np.median(s[s.variant == 'intensity_area'].roi_px))
        pr = pooled_row(s)
        L.append(f"{ww}$\\times${hh}{tag} & {roi} & {pr[('intensity_area','fixed')][0]:.2f} ({pr[('intensity_area','fixed')][1]:+d}) & "
                 f"{pr[('intensity_area','retuned')][0]:.2f} & {pr[('intensity_linear','fixed')][0]:.2f} & "
                 f"{pr[('framediff_area','fixed')][0]:.2f} & {pr[('framediff_area','retuned')][0]:.2f}\\\\")
    open(T + f'tab_res_{label}_rows.tex', 'w').write('\n'.join(L))

# ---- Monte Carlo (sigma_ROI = 4 px) with reference-count uncertainty
mc = pd.read_csv(R + 'montecarlo.csv')
L = []
for n in TESTS:
    r = mc[(mc.video == n) & (mc.sigma_px == 4)].iloc[0]
    u_gt = d1.loc[n].u_Ngt if 'd1' in dir() else pd.read_csv(R + 'table1_data.csv').set_index('video').loc[n].u_Ngt
    L.append(f"{LAB[n]} & {int(r.N_gt)} & {u_gt:.1f} & {int(r.N_det)} & {r.N_mean:.1f} & {r.u_N:.1f} & "
             f"[{r.N_lo:.0f}, {r.N_hi:.0f}] & {r.F1:.2f} & {r.u_F1:.2f}\\\\")
open(T + 'tab_mc_rows.tex', 'w').write('\n'.join(L))

# ---- S2: every resolution, per recording
SVAR = [('intensity_area', 'fixed'), ('intensity_area', 'retuned'), ('intensity_linear', 'fixed'),
        ('framediff_area', 'fixed'), ('framediff_area', 'retuned')]
for n in TESTS:
    d = sw[sw.video == n]
    a = d[d.variant == 'intensity_area'].set_index(['w', 'h', 'kind'])
    li = d[d.variant == 'intensity_linear'].set_index(['w', 'h', 'kind'])
    fd = d[d.variant == 'framediff_area'].set_index(['w', 'h', 'kind'])
    keys = sorted(a.index, key=lambda k: (k[2] == 's', -k[0]))
    L = ['\\begin{longtable}{rrrrrrrrrrrr}',
         f'\\caption{{Video {LAB[n]}: every processing size (test segment, $N_{{\\mathrm{{gt}}}} = {int(a.N_gt.iloc[0])}$). '
         'Area interpolation unless stated; fixed = parameters calibrated at full resolution; re-tuned = re-selected on the '
         'calibration segment at each size; $^\\dagger$ stretched 4:3.}\\\\',
         '\\toprule',
         '$w$ & $h$ & ROI px & $N_{\\mathrm{det}}$ & TP & FP & FN & $F_1$ & $F_1$ re-tuned & $F_1$ bilinear & $F_1$ FD & $F_1$ FD re-t.\\\\',
         '\\midrule\\endfirsthead',
         '\\toprule $w$ & $h$ & ROI px & $N_{\\mathrm{det}}$ & TP & FP & FN & $F_1$ & $F_1$ re-tuned & $F_1$ bilinear & $F_1$ FD & $F_1$ FD re-t.\\\\ \\midrule\\endhead',
         '\\bottomrule\\endfoot']
    for k in keys:
        r = a.loc[k]
        tag = '$^\\dagger$' if k[2] == 's' else ''
        L.append(f"{k[0]}{tag} & {k[1]} & {int(r.roi_px)} & {int(r.N_det_fixed)} & {int(r.TP_fixed)} & {int(r.FP_fixed)} & "
                 f"{int(r.FN_fixed)} & {r.F1_fixed:.2f} & {r.F1_retuned:.2f} & {li.loc[k].F1_fixed:.2f} & "
                 f"{fd.loc[k].F1_fixed:.2f} & {fd.loc[k].F1_retuned:.2f}\\\\")
    L.append('\\end{longtable}')
    open(T2 + f'tableS2_sweep_{LAB[n]}.tex', 'w').write('\n'.join(L))

# ---- S3: ROI variants and baseline variants
sr = pd.read_csv(R + 'sens_roi.csv')
VARS = ['dy-0.02', 'dy+0.02', 'dx-0.02', 'dx+0.02', 'dy-0.05', 'dy+0.05', 'dx-0.05', 'dx+0.05', 'scale0.6', 'scale0.8', 'scale1.2']
VLAB = {'dy-0.02': '$y-2\\%$', 'dy+0.02': '$y+2\\%$', 'dx-0.02': '$x-2\\%$', 'dx+0.02': '$x+2\\%$', 'dy-0.05': '$y-5\\%$',
        'dy+0.05': '$y+5\\%$', 'dx-0.05': '$x-5\\%$', 'dx+0.05': '$x+5\\%$', 'scale0.6': '$\\times 0.6$', 'scale0.8': '$\\times 0.8$',
        'scale1.2': '$\\times 1.2$'}
L = ['\\begin{tabular}{l' + 'c' * (len(VARS) + 1) + '}', '\\toprule',
     'Video & as drawn & ' + ' & '.join(VLAB[v] for v in VARS) + '\\\\', '\\midrule']
for n in TESTS:
    s = sr[sr.video == n].set_index('variant')
    L.append(f"{LAB[n]} & {s.F1_base.iloc[0]:.2f} & " + ' & '.join(f"{s.loc[v].F1_fixed:.2f}" for v in VARS) + '\\\\')
L.append('\\midrule')
cells = []
for v in VARS:
    s = sr[sr.variant == v]
    tp, fp, fn = s.TP_f.sum(), s.FP_f.sum(), s.FN_f.sum()
    cells.append(f"{2 * tp / (2 * tp + fp + fn):.2f}")
cells2 = []
for v in VARS:
    s = sr[sr.variant == v]
    tp, fp, fn = s.TP_r.sum(), s.FP_r.sum(), s.FN_r.sum()
    cells2.append(f"{2 * tp / (2 * tp + fp + fn):.2f}")
L.append(f"Pooled, fixed & {p.loc['proposed_global'].F1:.2f} & " + ' & '.join(cells) + '\\\\')
L.append(f"Pooled, re-tuned & {p.loc['proposed_global'].F1:.2f} & " + ' & '.join(cells2) + '\\\\')
L += ['\\bottomrule', '\\end{tabular}']
open(T2 + 'tableS3_roi.tex', 'w').write('\n'.join(L))

L = ['\\begin{tabular}{lccc}', '\\toprule', 'Video & global median & centred moving median & causal median\\\\', '\\midrule']
for n in TESTS:
    s = m[m.video == n].set_index('method')
    L.append(f"{LAB[n]} & " + ' & '.join(f"{s.loc[k].F1:.2f} ({s.loc[k].param.replace('(', '').replace(')', '')})"
                                         for k in ('proposed_global', 'proposed_moving', 'proposed_causal')) + '\\\\')
L.append('\\midrule')
L.append('Pooled & ' + ' & '.join(f"{p.loc[k].F1:.2f} [{p.loc[k].F1_lo:.2f}, {p.loc[k].F1_hi:.2f}]"
                                  for k in ('proposed_global', 'proposed_moving', 'proposed_causal')) + '\\\\')
L += ['\\bottomrule', '\\end{tabular}']
open(T2 + 'tableS3_baseline.tex', 'w').write('\n'.join(L))

# ---- S4: timing
hw = json.load(open(R + 'timing_pc.json'))['hardware']
MN = {'decode': 'Decoding only', 'proposed_global': 'Proposed, global median', 'proposed_causal': 'Proposed, causal median',
      'stab_estimate': 'Camera-motion estimation', 'framediff': 'Frame differencing', 'mog2': 'MOG2 virtual loop',
      'yolo': 'YOLOv8n + ByteTrack (10 Hz)'}
L = ['\\begin{tabular}{llrrrrrrr}', '\\toprule',
     'Clip & Configuration & Size & Runs & Median (s) & IQR (s) & Throughput (fps) & Real-time factor & Peak memory (MiB)\\\\',
     '\\midrule']
for vid, lab in (('olaria_noite', 'E (1920$\\times$1080, 29.9 fps)'), ('B', 'B (848$\\times$478, 60.0 fps)')):
    rows = [r for r in tim if r['video'] == vid]
    first = True
    for r in rows:
        size = f"{r['proc_w']}$\\times${r['proc_h']}" if r['method'] not in ('decode', 'stab_estimate', 'yolo') else \
            ('640 px wide' if r['method'] == 'yolo' else ('640 px wide' if r['method'] == 'stab_estimate' else 'native'))
        thr = r['realtime_factor'] * r['fps_video']
        L.append(f"{lab if first else ''} & {MN[r['method']]} & {size} & {r['R']} & {r['time_median_s']:.2f} & "
                 f"{r['time_iqr_s']:.2f} & {thr:.0f} & {r['realtime_factor']:.2f} & {r['peak_working_set_mib']:.0f} ({r['tracemalloc_peak_mib']:.1f})\\\\")
        first = False
    L.append('\\midrule')
L[-1] = '\\bottomrule'
L.append('\\end{tabular}')
open(T2 + 'tableS4_timing.tex', 'w').write('\n'.join(L))

# ---- S5: Monte Carlo, three ROI standard deviations
L = ['\\begin{tabular}{lcccccccc}', '\\toprule',
     'Video & $N_{\\mathrm{gt}}$ & $N_{\\mathrm{det}}$ & $\\sigma_{\\mathrm{ROI}}$ (px) & $h$ range & $\\Delta t_{\\min}$ range (s) & '
     '$\\overline{N}$ & $u(N)$ [95\\,\\% interval] & $\\overline{F_1}$ ($u$)\\\\', '\\midrule']
for n in TESTS:
    s = mc[mc.video == n]
    for i, (_, r) in enumerate(s.iterrows()):
        L.append(f"{LAB[n] if i == 0 else ''} & {int(r.N_gt) if i == 0 else ''} & {int(r.N_det) if i == 0 else ''} & {int(r.sigma_px)} & "
                 f"{r.h_range.replace('(', '').replace(')', '').replace(', ', '--') if i == 0 else ''} & "
                 f"{r.dt_range.replace('(', '').replace(')', '').replace(', ', '--') if i == 0 else ''} & {r.N_mean:.1f} & "
                 f"{r.u_N:.1f} [{r.N_lo:.0f}, {r.N_hi:.0f}] & {r.F1_mean:.2f} ({r.u_F1:.2f})\\\\")
L += ['\\bottomrule', '\\end{tabular}']
open(T2 + 'tableS5_mc.tex', 'w').write('\n'.join(L))

# ---- S6: complete event-level results
MN2 = {'proposed_global': 'Proposed, global', 'proposed_moving': 'Proposed, moving', 'proposed_causal': 'Proposed, causal',
       'framediff': 'Frame differencing', 'mog2': 'MOG2', 'yolo': 'YOLOv8n + ByteTrack'}
PN = {'proposed_global': '$h$, $\\Delta t_{\\min}$', 'proposed_moving': '$h$, $\\Delta t_{\\min}$', 'proposed_causal': '$h$, $\\Delta t_{\\min}$',
      'framediff': '$h$, $\\Delta t_{\\min}$', 'mog2': 'on, $\\Delta t_{\\min}$', 'yolo': 'conf, cover'}
L = ['\\begin{longtable}{llllrrrrrrrr}',
     '\\caption{Complete event-level results on the test segments ($\\tau = 1.0$\\,s). Parameters: calibrated on the first 40\\,\\% of each '
     'recording (plateau = number of grid points with the maximum calibration $F_1$; the median of the plateau was selected).}\\\\',
     '\\toprule Video & Method & Parameters & Plateau & $N_{\\mathrm{det}}$ & TP & FP & FN & $P$ & $R$ & $F_1$ & $b$\\\\ \\midrule\\endfirsthead',
     '\\toprule Video & Method & Parameters & Plateau & $N_{\\mathrm{det}}$ & TP & FP & FN & $P$ & $R$ & $F_1$ & $b$\\\\ \\midrule\\endhead',
     '\\bottomrule\\endfoot']
for n in TESTS:
    s = m[m.video == n]
    for i, (_, r) in enumerate(s.iterrows()):
        L.append(f"{LAB[n] + ' (' + str(int(r.N_gt)) + ')' if i == 0 else ''} & {MN2[r.method]} & {PN[r.method]} = "
                 f"{r.param.replace('(', '').replace(')', '')} & {int(r.plateau)} & {int(r.N_det)} & {int(r.TP)} & {int(r.FP)} & "
                 f"{int(r.FN)} & {r.P:.2f} & {r.R:.2f} & {r.F1:.2f} & {int(r.bias):+d}\\\\")
    L.append('\\midrule')
for tau in ('1.0', '1.5', '2.0'):
    mt = pd.read_csv(R + f'main_tau{tau}.csv')
    g = mt.groupby('method')[['TP', 'FP', 'FN', 'N_gt', 'N_det']].sum()
    for i, k in enumerate(['proposed_global', 'proposed_moving', 'proposed_causal', 'framediff', 'mog2', 'yolo']):
        r = g.loc[k]
        P_ = r.TP / (r.TP + r.FP)
        R_ = r.TP / (r.TP + r.FN)
        L.append(f"{'Pooled, $\\tau$ = ' + tau + ' s' if i == 0 else ''} & {MN2[k]} & & & {int(r.N_det)} & {int(r.TP)} & {int(r.FP)} & "
                 f"{int(r.FN)} & {P_:.2f} & {R_:.2f} & {2 * r.TP / (2 * r.TP + r.FP + r.FN):.2f} & {int(r.N_det - r.N_gt):+d}\\\\")
    if tau != '2.0':
        L.append('\\midrule')
L.append('\\end{longtable}')
open(T2 + 'tableS6_full.tex', 'w').write('\n'.join(L))

# ---- S7: detector agreement with each annotator
av = pd.read_csv(R + 'annotator_vs_detector.csv')
L = ['\\begin{tabular}{lcccc}', '\\toprule',
     'Video & $N$ (F.~S.~R.) & $F_1$ vs.\\ F.~S.~R. & $N$ (second) & $F_1$ vs.\\ second annotator\\\\', '\\midrule']
for n in TESTS:
    s = av[av.video == n].set_index('annotator')
    L.append(f"{LAB[n]} & {int(s.loc['A'].N)} & {s.loc['A'].F1:.2f} & {int(s.loc['B'].N)} & {s.loc['B'].F1:.2f}\\\\")
L += ['\\bottomrule', '\\end{tabular}']
open(T2 + 'tableS7_annotators.tex', 'w').write('\n'.join(L))
print('ok extra')

# ---- the main file adds the row terminator before \\bottomrule: drop the last one of each rows file
import glob
for f in glob.glob(T + 'tab_*_rows.tex'):
    txt = open(f).read().rstrip()
    if txt.endswith('\\\\'):
        txt = txt[:-2]
    open(f, 'w').write(txt + '\n')
