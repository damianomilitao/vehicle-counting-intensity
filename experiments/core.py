import os
import numpy as np, json, itertools, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import V, E
from evaluation import load_events
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
import baselines as BL
from tvic import ROI

FPS={'A':30.0,'B':60.00672656380572,'olaria_dia':24.24,'olaria_tarde_chuva':30.002006823198872,'olaria_noite':29.894021195760846,
     'raul_tarde_chuva':29.97794265089232,'raul_noite':29.89103269019294,'S1':59.71708301084363,'S2':57.902642741926584}
TESTS=['B','olaria_dia','olaria_tarde_chuva','olaria_noite','raul_tarde_chuva','raul_noite','S1','S2']
LABEL={'A':'A','B':'B','olaria_dia':'C','olaria_tarde_chuva':'D','olaria_noite':'E',
       'raul_tarde_chuva':'F','raul_noite':'G','S1':'H','S2':'I'}   # labels used in the paper
H_P=[1,2,3,4,5,6,8,10,12,15,18,22,26,30,35,40]; D_S=[0.3,0.5,0.7,1.0,1.5,2.0,3.0]
H_FD=[0.5,1,1.5,2,3,4,5,6,8,10,12,15]; ON_MOG=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
CONF_Y=[0.1,0.25,0.4,0.55]; COVER_Y=[0.1,0.3,0.5,0.7]
TAU=1.0
_Z={}
def Z(n):
    if n not in _Z: _Z[n]=np.load(E+f'full/{n}.npz')
    return _Z[n]
def T_end(n):
    w=V[n][3]
    return w[1] if w else Z(n)['frames']/FPS[n]
def split(n): T=T_end(n); return 0.4*T, T
_G={}
def gt(n):
    if n not in _G:
        g=load_events(E+f'gt/{n}_consensus.csv'); w=V[n][3]
        _G[n]=g if w is None else g[(g>=w[0])&(g<w[1])]
    return _G[n]
def match(g,d,tau):
    g=np.asarray(g,float); d=np.asarray(d,float)
    if len(g)==0 or len(d)==0: return np.zeros(len(g),bool),np.zeros(len(d),bool)
    C=np.abs(g[:,None]-d[None,:]); Cm=np.where(C<=tau,C,1e6); r,c=linear_sum_assignment(Cm); ok=Cm[r,c]<1e6
    mg=np.zeros(len(g),bool); md=np.zeros(len(d),bool); mg[r[ok]]=True; md[c[ok]]=True
    return mg,md
def metrics_seg(n,det,a,b,tau=None):
    tau=TAU if tau is None else tau
    g=gt(n); g=g[(g>=a)&(g<b)]; d=np.asarray(det,float); d=d[(d>=a)&(d<b)]
    mg,md=match(g,d,tau); TP=int(mg.sum()); FP=int((~md).sum()); FN=int((~mg).sum())
    P=TP/(TP+FP) if TP+FP else 0.0; R=TP/(TP+FN) if TP+FN else 0.0; F=2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) else 1.0
    return dict(N_gt=len(g),N_det=len(d),TP=TP,FP=FP,FN=FN,P=P,R=R,F1=F,bias=len(d)-len(g),
                Er=100*abs(len(d)-len(g))/len(g) if len(g) else np.nan,g=g,d=d,mg=mg,md=md)
# ---------------- detectors ----------------
def causal_median(I,win):
    out=np.empty_like(I); buf=[]
    import bisect
    for i,x in enumerate(I):
        bisect.insort(buf,x)
        if len(buf)>win:
            old=I[i-win]; j=bisect.bisect_left(buf,old); buf.pop(j)
        m=len(buf); out[i]=buf[m//2] if m%2 else 0.5*(buf[m//2-1]+buf[m//2])
    return out
_B={}
def baseline_of(I,fps,kind,key=None):
    if key is not None and (key,kind) in _B: return _B[(key,kind)]
    if kind=='global': B=np.full_like(I,np.median(I))
    elif kind=='moving': B=median_filter(I,size=max(3,int(round(30*fps))|1),mode='nearest')
    elif kind=='causal': B=causal_median(I,max(3,int(round(30*fps))))
    if key is not None: _B[(key,kind)]=B
    return B
def det_intensity(I,fps,h,dsec,kind='global',key=None):
    D=np.abs(I-baseline_of(I,fps,kind,key))
    return find_peaks(D,height=h,distance=max(1,int(round(dsec*fps))))[0]/fps
def det_fd(s,fps,h,dsec):
    return find_peaks(s,height=h,distance=max(1,int(round(dsec*fps))))[0]/fps
def det_mog(occ,fps,on,dsec):
    return BL.mog2_events(occ,fps,on,dsec)
def pick(scores):
    mx=max(scores.values()); pl=sorted(k for k,v in scores.items() if v>=mx-1e-12)
    return pl[len(pl)//2], mx, pl
def calibrate(n,fn,grid,a,b):
    sc={k:metrics_seg(n,fn(*k),a,b)['F1'] for k in grid}
    return pick(sc)
# signals
def sig_full(n,kind='I_area'):  return Z(n)[kind][0].astype(float)
def mog_occ(n): return np.load(E+f'bsig/{n}_mog.npy')
_Y={}
def yolo_tracks(n):
    if n not in _Y:
        tr=[]
        for off,f in ((10**6,f'yolo/{n}_cal.json'),(0,f'yolo/{n}.json')):
            try: tr+=[(t,tid+off,c,x1,y1,x2,y2) for t,tid,c,x1,y1,x2,y2 in json.load(open(E+f))['tracks']]
            except FileNotFoundError: pass
        d=json.load(open(E+f'yolo/{n}.json')); _Y[n]=(tr,(d['W'],d['H']))
    return _Y[n]
def det_yolo(n,conf,cover):
    tr,sz=yolo_tracks(n); roi=ROI.load(E+'rois/'+V[n][1])
    # separate track ids of the two runs (cal and test runs restart ids): offset by run
    return BL.yolo_events(tr,roi,sz,conf,cover)
