import os
import sys, numpy as np, time, cv2, json
from common import *
from tvic import Stabilizer, video_info
from ultralytics import YOLO
TARGET_HZ=10.0; W=640
def run(n):
    vid,roi,st,win,role=paths(n); info=video_info(vid); fps=info['fps']
    T=win[1] if win else info['frames']/fps; t0=0.4*T
    step=max(1,int(round(fps/TARGET_HZ)))
    As=np.load(st) if st else None
    model=YOLO(os.environ.get('YOLO_WEIGHTS','yolov8n.pt'))
    h=int(round(W*info['height']/info['width'])); poly=np.array([[x*W,y*h] for x,y in roi.polygon],np.float32)
    cap=cv2.VideoCapture(vid); k=0; first={}; conf={}; tracks=[]; t=time.time(); nproc=0
    while True:
        ok=cap.grab()
        if not ok: break
        tk=k/fps
        if tk>=T: break
        if tk>=t0 and (k%step==0):
            ok,fr=cap.retrieve()
            fr=cv2.resize(fr,(W,h),interpolation=cv2.INTER_AREA)
            if As is not None: fr=Stabilizer.warp_to_anchor(fr,As[min(k,len(As)-1)])
            r=model.track(fr,persist=True,conf=0.1,classes=[2,3,5,7],device='cpu',imgsz=640,verbose=False,tracker='bytetrack.yaml')[0]
            nproc+=1
            if r.boxes.id is not None:
                for (x1,y1,x2,y2),tid,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.id.cpu().numpy(),r.boxes.conf.cpu().numpy()):
                    tracks.append((round(tk,3),int(tid),float(c),float(x1),float(y1),float(x2),float(y2)))
        k+=1
    json.dump(dict(video=n,t0=t0,T=T,hz=fps/step,frames=nproc,time_s=time.time()-t,W=W,H=h,poly=poly.tolist(),tracks=tracks),open(E+f'yolo/{n}.json','w'))
    return n,nproc,round(time.time()-t),len(set(x[1] for x in tracks))
if __name__=='__main__':
    for n in sys.argv[1:]: print(run(n),flush=True)
