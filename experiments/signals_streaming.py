"""ROI signal at full resolution with the streaming implementation (tvic.extract_signal), with and
without camera-motion compensation; writes sig/<name>_w<width>_area_<s|n>.npy. The signals without
compensation are used by analysis_extra.py (stab)."""
import sys, numpy as np, time
from common import *
def job(args):
    n,w,interp,stab=args
    vid,roi,st,win,role=paths(n); info=video_info(vid)
    size=target_size((info['width'],info['height']),w) if w else None
    mf=int(win[1]*info['fps'])+60 if win else None
    I,_,meta=extract_signal(vid,roi,size,interp,max_frames=mf,stabilize=(st if stab else False))
    tag=f"{n}_w{w}_{interp}_{'s' if stab else 'n'}"
    np.save(E+f'sig/{tag}.npy',I); return tag,info['fps'],len(I)
if __name__=='__main__':
    from multiprocessing import Pool
    names=list(V); jobs=[(n,full_w(n),'area',True) for n in names]+[(n,full_w(n),'area',False) for n in names if V[n][2]]
    t=time.time()
    with Pool(2) as p:
        for r in p.imap_unordered(job,jobs): print(r,round(time.time()-t),flush=True)
