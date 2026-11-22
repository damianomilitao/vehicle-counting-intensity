import os
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import core
from core import *
rng=np.random.default_rng(2026)
OUT=E+'results/'; import os; os.makedirs(OUT,exist_ok=True)

def methods_for(n):
    fps=FPS[n]; I=sig_full(n); fd=Z(n)['FD_area'][0].astype(float)
    M={'proposed_global':(lambda h,d: det_intensity(I,fps,h,d,'global',(n,'full')), list(itertools.product(H_P,D_S))),
       'proposed_moving':(lambda h,d: det_intensity(I,fps,h,d,'moving',(n,'full')), list(itertools.product(H_P,D_S))),
       'proposed_causal':(lambda h,d: det_intensity(I,fps,h,d,'causal',(n,'full')), list(itertools.product(H_P,D_S))),
       'framediff':(lambda h,d: det_fd(fd,fps,h,d), list(itertools.product(H_FD,D_S)))}
    try:
        occ=mog_occ(n); M['mog2']=(lambda on,d: det_mog(occ,fps,on,d), list(itertools.product(ON_MOG,D_S)))
    except FileNotFoundError: pass
    if os.path.exists(E+f'yolo/{n}_cal.json') and n!='A' and not os.environ.get('NOYOLO'):
        M['yolo']=(lambda c,v: det_yolo(n,c,v), list(itertools.product(CONF_Y,COVER_Y)))
    return M

def boot_ci(items,a,b,L=30.0,B=2000):
    """items: list of (time, kind) kind in TP/FP/FN within [a,b). moving-block bootstrap of F1."""
    t=np.array([x[0] for x in items]); k=np.array([x[1] for x in items])
    nb=int(np.ceil((b-a)/L)); f=[]
    for _ in range(B):
        st=rng.uniform(a,max(a,b-L),nb); tp=fp=fn=0
        for s in st:
            m=(t>=s)&(t<s+L); tp+=(k[m]=='TP').sum(); fp+=(k[m]=='FP').sum(); fn+=(k[m]=='FN').sum()
        f.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan)
    return np.nanpercentile(f,[2.5,97.5])

def items_of(m):
    it=[(x,'TP') for x in m['g'][m['mg']]]+[(x,'FN') for x in m['g'][~m['mg']]]+[(x,'FP') for x in m['d'][~m['md']]]
    return it

def run_main(tau):
    core.TAU=tau; rows=[]; dets={}; items={}
    for n in TESTS:
        tc,T=split(n)
        for mname,(fn,grid) in methods_for(n).items():
            k,fcal,pl=calibrate(n,fn,grid,0,tc)
            det=fn(*k); m=metrics_seg(n,det,tc,T)
            dets[(n,mname)]=det; items[(n,mname)]=items_of(m)
            rows.append(dict(video=n,method=mname,param=k,plateau=len(pl),F1_cal=fcal,
                             **{x:m[x] for x in ('N_gt','N_det','TP','FP','FN','P','R','F1','bias','Er')}))
    df=pd.DataFrame(rows)
    return df,dets,items

if __name__=='__main__':
    res={}
    for tau in (1.0,1.5,2.0):
        df,dets,items=run_main(tau)
        if tau==1.0:
            # bootstrap CIs per video and pooled
            ci=[]
            for i,r in df.iterrows():
                tc,T=split(r.video); lo,hi=boot_ci(items[(r.video,r.method)],tc,T); ci.append((lo,hi))
            df['F1_lo']=[c[0] for c in ci]; df['F1_hi']=[c[1] for c in ci]
            pooled=[]
            for mname in df.method.unique():
                s=df[df.method==mname]; TP,FP,FN=s.TP.sum(),s.FP.sum(),s.FN.sum()
                # pooled bootstrap: resample blocks per video
                f=[]
                for _ in range(1000):
                    tp=fp=fn=0
                    for n in s.video:
                        tc,T=split(n); it=items[(n,mname)]; t=np.array([x[0] for x in it]); k=np.array([x[1] for x in it])
                        nb=int(np.ceil((T-tc)/30)); st=rng.uniform(tc,max(tc,T-30),nb)
                        for s0 in st:
                            m=(t>=s0)&(t<s0+30); tp+=(k[m]=='TP').sum(); fp+=(k[m]=='FP').sum(); fn+=(k[m]=='FN').sum()
                    f.append(2*tp/(2*tp+fp+fn))
                lo,hi=np.percentile(f,[2.5,97.5])
                pooled.append(dict(method=mname,videos=len(s),N_gt=s.N_gt.sum(),N_det=s.N_det.sum(),TP=TP,FP=FP,FN=FN,
                    P=TP/(TP+FP),R=TP/(TP+FN),F1=2*TP/(2*TP+FP+FN),F1_lo=lo,F1_hi=hi,bias=s.bias.sum(),
                    F1_mean_videos=s.F1.mean(),F1_day=s[~s.video.str.contains('noite')].F1.mean(),F1_night=s[s.video.str.contains('noite')].F1.mean()))
            pd.DataFrame(pooled).to_csv(OUT+'pooled_tau1.0.csv',index=False)
            json.dump({f'{k[0]}|{k[1]}':list(map(float,v)) for k,v in dets.items()},open(OUT+'dets_tau1.0.json','w'))
            json.dump({f'{k[0]}|{k[1]}':[(float(a),b) for a,b in v] for k,v in items.items()},open(OUT+'items_tau1.0.json','w'))
        df.to_csv(OUT+f'main_tau{tau}.csv',index=False)
        print('tau',tau); print(df.pivot_table(index='video',columns='method',values='F1').round(2).to_string())
    print(pd.read_csv(OUT+'pooled_tau1.0.csv').round(3).to_string())
