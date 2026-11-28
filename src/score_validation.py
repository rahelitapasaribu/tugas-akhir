import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
R='/kaggle/working/tugas-akhir'; P=f'{R}/data/processed'; A=f'{P}/label_audit'; D=f'{R}/results/metrics_v3_parent_rdrop'
L={'Negatif':0,'Netral':1,'Positif':2}; IL={v:k for k,v in L.items()}
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
sh=pd.read_csv(f'{A}/validation_blind_sheet_filled.csv'); key=pd.read_csv(f'{A}/validation_blind_key.csv'); d=sh.merge(key,on='blind_id')
d=d[d.label_anotator1.isin(L)].copy()
two=d.label_anotator2.isin(L)
if two.sum()>=30: print('Cohen kappa (n=%d): %.3f'%(two.sum(),cohen_kappa_score(d[two].label_anotator1,d[two].label_anotator2)))
d['gold']=np.where(two&(d.label_anotator1!=d.label_anotator2),d.label_adjudikasi,d.label_anotator1)
d=d[d.gold.isin(L)]
dev=pd.concat([pd.read_csv(f'{P}/splits/train_audited_v3.csv'),pd.read_csv(f'{P}/splits/validation_audited_v3.csv')],ignore_index=True)
te=pd.read_csv(f'{P}/splits/test.csv')
po=np.mean([softmax(np.load(f'{D}/oof_logits_seed{s}.npy')/1.15,1) for s in (42,2026)],0).argmax(1)
pt=np.mean([softmax(np.load(f'{D}/test_logits_seed{s}.npy')/1.15,1) for s in (42,2026)],0).argmax(1)
mp={**dict(zip(dev.comment_id,po)),**dict(zip(te.comment_id,pt))}
d['model']=d.comment_id.map(mp).map(IL)
rows=[]
for st,g in d.groupby('stratum'):
    n=len(g); r={'stratum':st,'n':n}
    for name,col in [('gold=label_sebelum','label_before'),('gold=label_sesudah','label_after'),('gold=prediksi_model','model')]:
        k=int((g.gold==g[col]).sum()); lo,hi=wilson(k,n); r[name]=f'{k/n:.2f} [{lo:.2f}-{hi:.2f}]'
    rows.append(r)
pd.set_option('display.width',250); print(pd.DataFrame(rows).to_string(index=False))
