import os
import sys, json, numpy as np, cv2, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import core
from core import *
from common import paths
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ev=pd.read_csv(E+'fail/errors.csv')
for n in ev.video.unique():
    e=ev[ev.video==n].reset_index(drop=True); fps=FPS[n]
    vid,roi,st,win,role=paths(n); As=np.load(st) if st else None
    offs=[-0.4,0.0,0.4]
    need={}
    for i,r in e.iterrows():
        for j,o in enumerate(offs): need.setdefault(int(round((r.t+o)*fps)),[]).append((i,j))
    frames={}
    cap=cv2.VideoCapture(vid); k=0; mx=max(need)
    while k<=mx:
        ok,fr=cap.read()
        if not ok: break
        if k in need:
            H,W=fr.shape[:2]; poly=np.array(roi.polygon)
            if As is not None:
                A=np.linalg.inv(As[min(k,len(As)-1)]); q=np.c_[poly,np.ones(len(poly))]@A.T; poly=q[:,:2]/q[:,2:]
            P=(poly*[W,H]).astype(np.int32); x0,y0=P.min(0); x1,y1=P.max(0); cx,cy=(x0+x1)//2,(y0+y1)//2
            half=max(int(1.6*max(x1-x0,y1-y0)),110)
            img=fr.copy(); cv2.polylines(img,[P],True,(0,255,255),2)
            crop=img[max(0,cy-half):cy+half, max(0,cx-int(half*1.4)):cx+int(half*1.4)]
            for (i,j) in need[k]: frames[(i,j)]=cv2.resize(crop,(300,int(300*crop.shape[0]/max(1,crop.shape[1]))))
        k+=1
    cap.release()
    I=sig_full(n); D=np.abs(I-np.median(I)); h=e.h.iloc[0]
    per=6
    for p0 in range(0,len(e),per):
        sub=e.iloc[p0:p0+per]
        fig,axs=plt.subplots(len(sub),4,figsize=(13,2.3*len(sub)),gridspec_kw={'width_ratios':[1,1,1,1.3]})
        axs=np.atleast_2d(axs)
        for ri,(i,r) in enumerate(sub.iterrows()):
            for j in range(3):
                a=axs[ri,j]; im=frames.get((i,j))
                if im is not None: a.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                a.axis('off'); a.set_title(f"{offs[j]:+.1f} s",fontsize=7)
            a=axs[ri,3]; t0=r.t-4; t1=r.t+4; kk=np.arange(max(0,int(t0*fps)),min(len(D),int(t1*fps)))
            a.plot(kk/fps,D[kk],lw=0.8); a.axhline(h,color='r',lw=0.6,ls='--')
            g=gt(n); g=g[(g>t0)&(g<t1)]
            for x in g: a.axvline(x,color='g',lw=0.8,ls=':')
            dd=np.array(json.loads(r.dets)); dd=dd[(dd>t0)&(dd<t1)]
            a.plot(dd,np.interp(dd,kk/fps,D[kk]),'k.',ms=5)
            a.axvline(r.t,color='m',lw=1.2,alpha=0.5)
            a.set_title(f"#{i} {n} {r.kind} t={r.t:.2f}s {r.note}",fontsize=7); a.tick_params(labelsize=6)
        plt.tight_layout(); plt.savefig(E+f'fail/{n}_{p0//per:02d}.png',dpi=70); plt.close()
    print(n,len(e),flush=True)
