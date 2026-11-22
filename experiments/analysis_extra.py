import os
import sys, json, itertools, numpy as np, pandas as pd, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import core
from core import *
from analysis_main import methods_for
OUT=E+'results/'
core.TAU=1.0
what=sys.argv[1:] or ['transfer','cross','stab','sweep','sens','mc','fail','influence','occ']

def full_metrics_whole(n,det):   # whole video (A)
    T=T_end(n); return metrics_seg(n,det,0,T)

if 'transfer' in what:
    rows=[]
    MA=methods_for('A'); TA=T_end('A')
    for mname in ('proposed_global','proposed_causal','framediff','mog2'):
        fnA,grid=MA[mname]
        k,fcal,pl=calibrate('A',fnA,grid,0,TA)
        for n in TESTS:
            fn,_=methods_for(n)[mname]; tc,T=split(n); m=metrics_seg(n,fn(*k),tc,T)
            rows.append(dict(method=mname,param=k,plateau_A=len(pl),video=n,**{x:m[x] for x in ('N_gt','N_det','TP','FP','FN','P','R','F1','bias')}))
    df=pd.DataFrame(rows); df.to_csv(OUT+'transfer_from_A.csv',index=False)
    print(df.pivot_table(index='video',columns='method',values='F1').round(2).to_string())
    print(df.groupby('method')[['TP','FP','FN']].sum().assign(F1=lambda x:2*x.TP/(2*x.TP+x.FP+x.FN)).round(3))
    print(df.drop_duplicates('method')[['method','param','plateau_A']])

if 'cross' in what:
    pairs=[('olaria_tarde_chuva','olaria_noite'),('olaria_noite','olaria_tarde_chuva'),('raul_tarde_chuva','raul_noite'),('raul_noite','raul_tarde_chuva')]
    rows=[]
    for src,dst in pairs:
        for mname in ('proposed_global','proposed_causal','framediff','mog2'):
            fs,grid=methods_for(src)[mname]; tcs,_=split(src); k,_,_=calibrate(src,fs,grid,0,tcs)
            fd,_=methods_for(dst)[mname]; tc,T=split(dst); m=metrics_seg(dst,fd(*k),tc,T)
            kd,_,_=calibrate(dst,fd,grid,0,tc); md=metrics_seg(dst,fd(*kd),tc,T)
            rows.append(dict(source=src,target=dst,method=mname,param_src=k,F1_transfer=m['F1'],bias_transfer=m['bias'],param_own=kd,F1_own=md['F1']))
    df=pd.DataFrame(rows); df.to_csv(OUT+'cross_condition.csv',index=False); print(df.round(3).to_string())

if 'stab' in what:
    rows=[]
    for n in ('B','S1','S2'):
        fps=FPS[n]; tc,T=split(n); w=1280 if n!='B' else 848
        for lab,I in (('with',sig_full(n)),('without',np.load(E+f'sig/{n}_w{w}_area_n.npy'))):
            fn=lambda h,d,I=I: det_intensity(I,fps,h,d,'global')
            k,_,_=calibrate(n,fn,list(itertools.product(H_P,D_S)),0,tc); m=metrics_seg(n,fn(*k),tc,T)
            rows.append(dict(video=n,stabilization=lab,param=k,**{x:m[x] for x in ('N_det','TP','FP','FN','F1','bias')}))
        z=Z(n); rows[-1]['outside_frames']=int(z['outside_frames']); rows[-1]['track_fail_frames']=int(z['track_fail']); rows[-1]['frames']=int(z['frames'])
    df=pd.DataFrame(rows); df.to_csv(OUT+'stabilization.csv',index=False); print(df.to_string())

if 'sweep' in what:
    rows=[]
    for n in TESTS:
        z=Z(n); fps=FPS[n]; tc,T=split(n); sizes=z['sizes']; kinds=z['kinds']
        # fixed params from full-res calibration (area, global) and framediff full
        fI=lambda I,h,d: det_intensity(I,fps,h,d,'global')
        I0=z['I_area'][0].astype(float); kI,_,_=calibrate(n,lambda h,d: fI(I0,h,d),list(itertools.product(H_P,D_S)),0,tc)
        F0=z['FD_area'][0].astype(float); kF,_,_=calibrate(n,lambda h,d: det_fd(F0,fps,h,d),list(itertools.product(H_FD,D_S)),0,tc)
        for si,((w,h),kd) in enumerate(zip(sizes,kinds)):
            for var,arr,kfix,gridf,detf in (('intensity_area',z['I_area'][si],kI,list(itertools.product(H_P,D_S)),fI),
                                           ('intensity_linear',z['I_linear'][si],kI,list(itertools.product(H_P,D_S)),fI),
                                           ('framediff_area',z['FD_area'][si],kF,list(itertools.product(H_FD,D_S)),lambda s,h,d: det_fd(s,fps,h,d))):
                s=arr.astype(float)
                if np.isnan(s).all(): continue
                s=np.where(np.isnan(s),np.nanmedian(s),s)
                mf=metrics_seg(n,detf(s,*kfix),tc,T)
                kr,_,_=calibrate(n,lambda a,b: detf(s,a,b),gridf,0,tc); mr=metrics_seg(n,detf(s,*kr),tc,T)
                if var.startswith('intensity'):
                    D=np.abs(s-np.median(s)); seg=D[int(tc*fps):int(T*fps)]
                    sig=1.4826*np.median(np.abs(seg-np.median(seg)))
                else:
                    sig=1.4826*np.median(np.abs(s[int(tc*fps):int(T*fps)]-np.median(s[int(tc*fps):int(T*fps)])))
                rows.append(dict(video=n,w=int(w),h=int(h),kind=kd,variant=var,roi_px=int(z['roi_px'][si]),
                    N_gt=mf['N_gt'],N_det_fixed=mf['N_det'],TP_fixed=mf['TP'],FP_fixed=mf['FP'],FN_fixed=mf['FN'],F1_fixed=mf['F1'],Er_fixed=mf['Er'],
                    N_det_retuned=mr['N_det'],TP_retuned=mr['TP'],FP_retuned=mr['FP'],FN_retuned=mr['FN'],F1_retuned=mr['F1'],param_retuned=kr,
                    sigma_D=sig,h_over_sigma=(kfix[0]/sig if sig>0 else np.nan)))
        print(n,'done',flush=True)
    pd.DataFrame(rows).to_csv(OUT+'sweep_raw.csv',index=False)

if 'sens' in what:
    # parameter grids at 4 widths (pooled over test videos), ROI variants, baseline variants
    targets=[None,640,320,160]; rows=[]
    for tw in targets:
        for (h,d) in itertools.product(H_P,D_S):
            TP=FP=FN=0
            for n in TESTS:
                z=Z(n); fps=FPS[n]; tc,T=split(n)
                si=0 if tw is None else int(np.argmin([abs(int(x[0])-tw) if k=='p' else 1e9 for x,k in zip(z['sizes'],z['kinds'])]))
                s=z['I_area'][si].astype(float); m=metrics_seg(n,det_intensity(s,fps,h,d,'global',(n,si)),tc,T)
                TP+=m['TP']; FP+=m['FP']; FN+=m['FN']
            rows.append(dict(width='full' if tw is None else tw,h=h,dt=d,F1=2*TP/(2*TP+FP+FN),bias=TP+FP-(TP+FN)))
    pd.DataFrame(rows).to_csv(OUT+'sens_grid.csv',index=False)
    rows=[]
    for n in TESTS:
        z=Z(n); fps=FPS[n]; tc,T=split(n)
        I0=z['I_area'][0].astype(float); grid=list(itertools.product(H_P,D_S))
        kI,_,_=calibrate(n,lambda h,d: det_intensity(I0,fps,h,d,'global'),grid,0,tc)
        base=metrics_seg(n,det_intensity(I0,fps,*kI,'global'),tc,T)['F1']
        for j,name in enumerate(z['sens_names']):
            s=z['SENS'][j].astype(float)
            if np.isnan(s).all(): continue
            s=np.where(np.isnan(s),np.nanmedian(s),s)
            mf=metrics_seg(n,det_intensity(s,fps,*kI,'global'),tc,T)
            kr,_,_=calibrate(n,lambda h,d: det_intensity(s,fps,h,d,'global'),grid,0,tc); mr=metrics_seg(n,det_intensity(s,fps,*kr,'global'),tc,T)
            rows.append(dict(video=n,variant=str(name),F1_base=base,F1_fixed=mf['F1'],F1_retuned=mr['F1'],TP_f=mf['TP'],FP_f=mf['FP'],FN_f=mf['FN'],TP_r=mr['TP'],FP_r=mr['FP'],FN_r=mr['FN']))
    pd.DataFrame(rows).to_csv(OUT+'sens_roi.csv',index=False)
    print(pd.read_csv(OUT+'sens_roi.csv').groupby('variant')[['F1_fixed','F1_retuned']].mean().round(3))

if 'mc' in what:
    rows=[]; Mtr=2000; rng=np.random.default_rng(7)
    for n in TESTS:
        z=Z(n); fps=FPS[n]; tc,T=split(n); I0=z['I_area'][0].astype(float); grid=list(itertools.product(H_P,D_S))
        sc={k:metrics_seg(n,det_intensity(I0,fps,*k,'global'),0,tc)['F1'] for k in grid}
        k0,mx,pl=pick(sc)
        hs=sorted({h for h,d in pl if d==k0[1]}); ds=sorted({d for h,d in pl if h==k0[0]})
        def rng_of(vals,grid_vals):
            lo,hi=min(vals),max(vals); i0=grid_vals.index(lo); i1=grid_vals.index(hi)
            a=(grid_vals[i0-1]+lo)/2 if i0>0 else lo-(grid_vals[1]-grid_vals[0])/2
            b=(grid_vals[i1+1]+hi)/2 if i1<len(grid_vals)-1 else hi+(hi-grid_vals[-2])/2
            return a,b
        hr=rng_of(hs,H_P); dr=rng_of(ds,D_S)
        sigmas=z['MC_sigma']
        for sg in (2,4,8):
            idx=np.flatnonzero(sigmas==sg); N=[]; F=[]
            for t in range(Mtr):
                h=rng.uniform(*hr); d=rng.uniform(*dr); j=rng.choice(idx); s=z['MC'][j].astype(float)
                s=np.where(np.isnan(s),np.nanmedian(s),s)
                m=metrics_seg(n,det_intensity(s,fps,h,d,'global'),tc,T); N.append(m['N_det']); F.append(m['F1'])
            N=np.array(N); F=np.array(F); m0=metrics_seg(n,det_intensity(I0,fps,*k0,'global'),tc,T)
            rows.append(dict(video=n,sigma_px=sg,h0=k0[0],dt0=k0[1],h_range=hr,dt_range=dr,N_gt=m0['N_gt'],N_det=m0['N_det'],
                N_mean=N.mean(),u_N=N.std(ddof=1),N_lo=np.percentile(N,2.5),N_hi=np.percentile(N,97.5),
                F1=m0['F1'],F1_mean=F.mean(),u_F1=F.std(ddof=1),F1_lo=np.percentile(F,2.5),F1_hi=np.percentile(F,97.5)))
        print(n,flush=True)
    pd.DataFrame(rows).to_csv(OUT+'montecarlo.csv',index=False); print(pd.DataFrame(rows).round(2).to_string())

if 'fail' in what:
    dets=json.load(open(OUT+'dets_tau1.0.json')); main=pd.read_csv(OUT+'main_tau1.0.csv')
    rows=[]
    for n in TESTS:
        tc,T=split(n); d=np.array(dets[f'{n}|proposed_global']); m=metrics_seg(n,d,tc,T)
        k=eval(main[(main.video==n)&(main.method=='proposed_global')].param.iloc[0])
        cls=pd.read_csv(E+f'gt/{n}_consensus.csv')
        for x in m['g'][~m['mg']]:
            nd=m['d'][~m['md']]; near=nd[np.abs(nd-x)<=2.5]
            c=cls.iloc[np.argmin(np.abs(cls.t_s.values-x))]['class']
            rows.append(dict(video=n,kind='FN',t=x,h=k[0],dt=k[1],cls=c,note=('timing' if len(near) else '')+f' [{c}]',dets=json.dumps(list(m['d'][(m['d']>x-4)&(m['d']<x+4)]))))
        for x in m['d'][~m['md']]:
            ng=m['g'][~m['mg']]; near=ng[np.abs(ng-x)<=2.5]
            tpg=m['g'][m['mg']]; prev=tpg[(tpg<x)&(tpg>x-3)]
            note='timing' if len(near) else ('after TP' if len(prev) else '')
            rows.append(dict(video=n,kind='FP',t=x,h=k[0],dt=k[1],cls='',note=note,dets=json.dumps(list(m['d'][(m['d']>x-4)&(m['d']<x+4)]))))
    pd.DataFrame(rows).to_csv(E+'fail/errors.csv',index=False); print(pd.DataFrame(rows).groupby(['video','kind']).size())

if 'influence' in what:
    dets=json.load(open(OUT+'dets_tau1.0.json')); rows=[]
    for n in TESTS:
        tc,T=split(n); d=np.array(dets[f'{n}|proposed_global'])
        for tag in ('A','B'):
            a=pd.read_csv(E+f'gt/{n}_{tag}.csv').t_s.values; a=a[(a>=tc)&(a<T)]; d2=d[(d>=tc)&(d<T)]
            mg,md=match(a,d2,1.0); TP=mg.sum(); rows.append(dict(video=n,annotator=tag,N=len(a),TP=int(TP),F1=2*TP/(len(a)+len(d2))))
    df=pd.DataFrame(rows); df.to_csv(OUT+'annotator_vs_detector.csv',index=False)
    print(df.pivot_table(index='video',columns='annotator',values='F1').round(3))

if 'occ' in what:
    main=pd.read_csv(OUT+'main_tau1.0.csv'); rows=[]
    for n in TESTS:
        k=eval(main[(main.video==n)&(main.method=='proposed_global')].param.iloc[0]); I=sig_full(n); D=np.abs(I-np.median(I))
        rows.append(dict(video=n,frac_D_above_h=(D>=k[0]).mean(),h=k[0]))
    print(pd.DataFrame(rows).round(3)); pd.DataFrame(rows).to_csv(OUT+'occupancy.csv',index=False)
