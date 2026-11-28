import numpy as np, pandas as pd
from scipy.special import softmax
R='/kaggle/working/tugas-akhir'; P=f'{R}/data/processed'; D=f'{R}/results/metrics_v3_parent_rdrop'
L={'Negatif':0,'Netral':1,'Positif':2}; THR=0.80
dev=pd.concat([pd.read_csv(f'{P}/splits/train_audited_v3.csv'),pd.read_csv(f'{P}/splits/validation_audited_v3.csv')],ignore_index=True)
y=dev.label_final.map(L).values
pr=[softmax(np.load(f'{D}/oof_logits_seed{s}.npy')/1.15,1) for s in (42,2026)]
pm=np.mean(pr,0); a=pm.argmax(1)
f=(pr[0].argmax(1)==pr[1].argmax(1))&(pr[0].argmax(1)==a)&(a!=y)&(pm.max(1)>=THR)&(dev.label_source=='Qwen_dipertahankan').values
dev.loc[f,['comment_id','label_final']].to_csv(f'{P}/label_audit/train_drop_ids.csv',index=False)
print('baris yang dikeluarkan dari data LATIH:',f.sum())
