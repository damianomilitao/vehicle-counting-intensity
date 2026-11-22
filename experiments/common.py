import os
import sys, json, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from tvic import ROI, extract_signal, video_info, target_size, detect, Params
from evaluation import load_events, metrics
E=os.path.dirname(os.path.abspath(__file__))+os.sep
V={ # name: (video, roi, stab, window_s, role)
 'A':('A.mp4','A.json','A.npy',None,'cal'),
 'B':('B.mp4','B.json','B.npy',(0,125),'test'),
 'olaria_dia':('olaria_dia_20261114_1230.ts','olaria_dia.json',None,(0,300),'test'),
 'olaria_tarde_chuva':('olaria_tarde_chuva_20261116_1713.ts','olaria.json',None,(0,300),'test'),
 'olaria_noite':('olaria_noite_20261116_1846.ts','olaria.json',None,(0,300),'test'),
 'raul_tarde_chuva':('trevo_raulserta_tarde_chuva_20261116_1713.ts','raul.json',None,(0,300),'test'),
 'raul_noite':('trevo_raulserta_noite_20261116_1846.ts','raul.json',None,(0,300),'test'),
 'S1':('S1.mp4','S1.json','S1.npy',None,'test'),
 'S2':('S2.mp4','S2.json','S2.npy',None,'test')}
def paths(n):
    v,r,s,w,role=V[n]
    return E+'videos/'+v, ROI.load(E+'rois/'+r), (E+'stab/'+s if s else False), w, role
_FW={'A':1280,'B':848}
def full_w(n):
    return _FW.get(n,1280)
def gt(n):
    g=load_events(E+f'gt/{n}_consensus.csv'); w=V[n][3]
    return g if w is None else g[(g>=w[0])&(g<w[1])]
def clip(n,t):
    w=V[n][3]; t=np.asarray(t,float)
    return t if w is None else t[(t>=w[0])&(t<w[1])]
