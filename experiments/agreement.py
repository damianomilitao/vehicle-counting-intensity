import os
import numpy as np, pandas as pd, sys
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import V, E
rows=[]; offs_all=[]
for n in V:
    a=pd.read_csv(E+f'gt/{n}_A.csv'); b=pd.read_csv(E+f'gt/{n}_B.csv'); w=V[n][3]
    if w: a=a[(a.t_s>=w[0])&(a.t_s<w[1])]; b=b[(b.t_s>=w[0])&(b.t_s<w[1])]
    a=a.reset_index(drop=True); b=b.reset_index(drop=True)
    tol=1.5 if n=='olaria_dia' else 2.0
    C=np.abs(a.t_s.values[:,None]-b.t_s.values[None,:]); Cm=np.where(C<=tol,C,1e6)
    r,c=linear_sum_assignment(Cm); ok=Cm[r,c]<1e6; r,c=r[ok],c[ok]
    off=b.t_s.values[c]-a.t_s.values[r]; offs_all+=list(off)
    cls=(a['class'].values[r]==b['class'].values[c]).mean()
    g=pd.read_csv(E+f'gt/{n}_consensus.csv')
    if w: g=g[(g.t_s>=w[0])&(g.t_s<w[1])]
    P=len(r)/len(b); R=len(r)/len(a)
    rows.append(dict(video=n,N_A=len(a),N_B=len(b),matched=len(r),only_A=len(a)-len(r),only_B=len(b)-len(r),
        F1=2*len(r)/(len(a)+len(b)),class_agree=cls,off_mean=off.mean(),off_sd=off.std(ddof=1),N_gt=len(g),
        adjudicated=(g.origin!='agreed').sum()))
df=pd.DataFrame(rows); print(df.round(3).to_string())
offs=np.array(offs_all); print('pooled offsets: n',len(offs),'mean',offs.mean().round(3),'sd',offs.std(ddof=1).round(3),
  'MAD-sigma',(1.4826*np.median(np.abs(offs-np.median(offs)))).round(3))
o2=np.array([x for x in offs]); 
df.to_csv(E+'results/agreement.csv',index=False)
# without olaria_dia (time base converted)
