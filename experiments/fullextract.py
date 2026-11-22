import os
import sys, time, json, numpy as np, cv2
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import V, paths, E
from tvic import video_info, target_size
from resample import weights
cv2.setNumThreads(1)

def mapped(polys, A):
    """polys: (n,v,2) normalized in anchor frame -> current frame (inverse of A: current->anchor)."""
    if A is None: return polys
    Ai=np.linalg.inv(A); P=np.concatenate([polys,np.ones(polys.shape[:2]+(1,))],-1)@Ai.T
    return P[...,:2]/P[...,2:]

def build_rois(poly, full_wh, rng):
    P0=np.asarray(poly,float); fw,fh=full_wh
    mc={}
    for s in (2,4,8):
        mc[s]=np.stack([P0+rng.normal(0,1,P0.shape)*[s/fw,s/fh] for _ in range(30)])
    c=P0.mean(0); sens=[]; names=[]
    for d in (-0.05,-0.02,0.02,0.05): sens.append(P0+[0,d]); names.append(f'dy{d:+.2f}')
    for d in (-0.05,-0.02,0.02,0.05): sens.append(P0+[d,0]); names.append(f'dx{d:+.2f}')
    for k in (0.6,0.8,1.2): sens.append(c+(P0-c)*k); names.append(f'scale{k}')
    return P0, mc, np.stack(sens), names

def job(n):
    t_start=time.time()
    vid,roi,st,win,role=paths(n); info=video_info(vid); fps=info['fps']; W,H=info['width'],info['height']
    As=np.load(st) if st else None
    full_w=1280 if W>=1280 else W
    ws=sorted(set(np.round(np.geomspace(full_w,16,100)).astype(int).tolist()),reverse=True)
    sizes=[target_size((W,H),w) for w in ws]+[(640,480),(320,240),(160,120)]
    kinds=['p']*len(ws)+['s']*3
    full=sizes[0]
    rng=np.random.default_rng(12345)
    P0,mc,sens,sens_names=build_rois(roi.polygon,full,rng)
    allp=np.concatenate([P0[None],*mc.values(),sens])
    mf=int(win[1]*fps)+60 if win else 10**9
    nfr=min(mf,info['frames']) if info['frames'] else mf
    def bbox(polys):
        if As is not None:
            lo=np.array([1.,1.]); hi=np.array([0.,0.])
            for k in range(0,min(len(As),nfr),5):
                q=mapped(polys,As[k]).reshape(-1,2); lo=np.minimum(lo,q.min(0)); hi=np.maximum(hi,q.max(0))
        else:
            lo=polys.reshape(-1,2).min(0); hi=polys.reshape(-1,2).max(0)
        return np.clip(lo,0,1),np.clip(hi,0,1)
    lo_m,hi_m=bbox(P0[None]); lo_x,hi_x=bbox(allp[1:])
    def make(w,h,kd,lo,hi,ips):
        up=(w>W or h>H)
        c0=max(0,int(np.floor(lo[0]*w))-1); c1=min(w,int(np.ceil(hi[0]*w))+2)
        r0=max(0,int(np.floor(lo[1]*h))-1); r1=min(h,int(np.ceil(hi[1]*h))+2)
        ent={'w':w,'h':h,'kind':kd,'c0':c0,'c1':c1,'r0':r0,'r1':r1,'up':up,'Wx':{},'Wy':{}}
        if not up:
            for ip in ips:
                Wx=weights(W,w,ip)[c0:c1]; Wy=weights(H,h,ip)[r0:r1]
                nzx=np.flatnonzero(Wx.any(0)); nzy=np.flatnonzero(Wy.any(0))
                ent['xs']=(min(ent.get('xs',(W,0))[0],nzx[0]),max(ent.get('xs',(W,0))[1],nzx[-1]+1))
                ent['ys']=(min(ent.get('ys',(H,0))[0],nzy[0]),max(ent.get('ys',(H,0))[1],nzy[-1]+1))
                ent['Wx'][ip]=Wx; ent['Wy'][ip]=Wy
            for ip in ips:
                ent['Wx'][ip]=np.ascontiguousarray(ent['Wx'][ip][:,ent['xs'][0]:ent['xs'][1]].T.astype(np.float32))
                ent['Wy'][ip]=np.ascontiguousarray(ent['Wy'][ip][:,ent['ys'][0]:ent['ys'][1]].astype(np.float32))
        return ent
    plan=[make(w,h,kd,lo_m,hi_m,('area','linear')) for (w,h),kd in zip(sizes,kinds)]
    xplan=make(full[0],full[1],'x',lo_x,hi_x,('area',))
    xs0=min([e['xs'][0] for e in plan+[xplan] if 'xs' in e]); xs1=max([e['xs'][1] for e in plan+[xplan] if 'xs' in e])
    ys0=min([e['ys'][0] for e in plan+[xplan] if 'ys' in e]); ys1=max([e['ys'][1] for e in plan+[xplan] if 'ys' in e])
    def raster(polys,ent):
        w,h,c0,r0=ent['w'],ent['h'],ent['c0'],ent['r0']
        out=np.zeros((len(polys),ent['r1']-r0,ent['c1']-c0),np.uint8)
        for i,p in enumerate(polys):
            pts=np.array([[round(x*w)-c0,round(y*h)-r0] for x,y in p],np.int32)
            cv2.fillPoly(out[i],[pts],1)
        return out.astype(bool)
    nS=len(sizes)
    I={ip:[[] for _ in range(nS)] for ip in ('area','linear')}; FD=[[] for _ in range(nS)]
    extra=[]; prevO=[None]*nS; masks=[None]*nS; xmasks=None; A_prev=None
    roi_px=[None]*nS; outside=0; fail=0
    cap=cv2.VideoCapture(vid); k=0
    while k<mf:
        ok,fr=cap.read()
        if not ok: break
        A=As[min(k,len(As)-1)] if As is not None else None
        if A is not None and k>0 and k<len(As) and np.array_equal(As[k],As[k-1]): fail+=1
        changed = (k==0) or (A is not None and not np.allclose(A,A_prev,atol=1e-6))
        if changed:
            mp=mapped(allp,A); A_prev=A
            for si,ent in enumerate(plan):
                masks[si]=raster(mp[:1],ent)[0]
                if roi_px[si] is None: roi_px[si]=int(masks[si].sum())
            xmasks=raster(mp[1:],xplan).reshape(len(mp)-1,-1).astype(np.float32)
            cnt=xmasks.sum(1); cnt[cnt==0]=np.nan; xmasks_n=xmasks/cnt[:,None]
            if (mp[0]<0).any() or (mp[0]>1).any(): outside+=1
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        Gw=g[ys0:ys1,xs0:xs1].astype(np.float32)
        Xo=xplan['Wy']['area']@Gw[xplan['ys'][0]-ys0:xplan['ys'][1]-ys0, xplan['xs'][0]-xs0:xplan['xs'][1]-xs0]@xplan['Wx']['area']
        extra.append(xmasks_n@Xo.ravel())
        for si,ent in enumerate(plan):
            m=masks[si]
            for ip in ('area','linear'):
                if ent['up']:
                    rs=cv2.resize(fr,(ent['w'],ent['h']),interpolation=cv2.INTER_AREA if ip=='area' else cv2.INTER_LINEAR)
                    O=cv2.cvtColor(rs,cv2.COLOR_BGR2GRAY)[ent['r0']:ent['r1'],ent['c0']:ent['c1']].astype(np.float32)
                else:
                    G_=Gw[ent['ys'][0]-ys0:ent['ys'][1]-ys0, ent['xs'][0]-xs0:ent['xs'][1]-xs0]
                    O=ent['Wy'][ip]@G_@ent['Wx'][ip]
                I[ip][si].append(float(O[m].mean()) if m.any() else np.nan)
                if ip=='area':
                    Oq=np.clip(np.rint(O),0,255)
                    FD[si].append(0.0 if prevO[si] is None or not m.any() else float(np.abs(Oq-prevO[si])[m].mean()))
                    prevO[si]=Oq
        k+=1
    cap.release()
    nmc=sum(len(v) for v in mc.values())
    ex=np.array(extra,np.float32)
    out=dict(sizes=np.array(sizes),kinds=np.array(kinds),fps=fps,W=W,H=H,frames=k,
             I_area=np.array(I['area'],np.float32),I_linear=np.array(I['linear'],np.float32),
             FD_area=np.array(FD,np.float32),roi_px=np.array([r if r is not None else -1 for r in roi_px]),
             MC=ex[:,:nmc].T, MC_sigma=np.repeat([2,4,8],30), SENS=ex[:,nmc:].T, sens_names=np.array(sens_names),
             outside_frames=outside, track_fail=fail, window=(xs0,xs1,ys0,ys1), xwindow=(xplan['xs'][0],xplan['xs'][1],xplan['ys'][0],xplan['ys'][1]))
    np.savez_compressed(E+f'full/{n}.npz',**out)
    return n,k,round(time.time()-t_start),(xs1-xs0,ys1-ys0)

if __name__=='__main__':
    from multiprocessing import Pool
    names=sys.argv[1:] or ['S1','S2','olaria_noite','raul_tarde_chuva','olaria_tarde_chuva','raul_noite','olaria_dia','B','A']
    with Pool(2) as p:
        for r in p.imap_unordered(job,names): print(r,flush=True)
