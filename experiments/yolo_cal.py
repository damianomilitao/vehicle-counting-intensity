import os
import sys, json, time, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import V, paths, E
from tvic import video_info
import baselines as BL
for n in sys.argv[1:]:
    vid,roi,st,win,role=paths(n); info=video_info(vid); fps=info['fps']
    T=win[1] if win else info['frames']/fps; t0=0.4*T
    t=time.time()
    tracks,sz,k=BL.yolo_tracks(vid,roi,fps,os.environ.get('YOLO_WEIGHTS','yolov8n.pt'),640,10.0,st or None,None,0.0,t0)
    json.dump(dict(video=n,t_start=0.0,t_end=t0,W=sz[0],H=sz[1],tracks=tracks,time_s=time.time()-t),open(E+f'yolo/{n}_cal.json','w'))
    print(n,len(tracks),round(time.time()-t),flush=True)
