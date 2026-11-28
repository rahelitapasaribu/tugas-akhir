import numpy as np, pandas as pd
R='/kaggle/working/tugas-akhir'; P=f'{R}/data/processed'; A=f'{P}/label_audit'
N_TEST_RANDOM, N_TEST_KEPT, N_TEST_CHANGED, N_DEV_CHANGED, N_DEV_UNCHANGED, SEED = 50, 60, 40, 60, 30, 7

dev=pd.concat([pd.read_csv(f'{P}/splits/train_audited_v3.csv'),pd.read_csv(f'{P}/splits/validation_audited_v3.csv')],ignore_index=True)
te=pd.read_csv(f'{P}/splits/test.csv')
aud=pd.read_csv(f'{A}/test_audited_final.csv')[['comment_id','label_audited']].drop_duplicates('comment_id')
te=te.merge(aud,on='comment_id',how='left'); te['label_audited']=te['label_audited'].fillna(te['label_final'])
cf=pd.read_csv(f'{A}/test_conflicts_to_audit.csv')
te['changed']=te.label_final!=te.label_audited; dev['changed']=dev.label_final!=dev.label_before_audit
def pick(df,n): return df.sample(n=min(n,len(df)),random_state=SEED)

parts=[]
def add(df,stratum,split,before,after):
    d=df.copy(); d['stratum']=stratum; d['split']=split; d['label_before']=d[before]; d['label_after']=d[after]; parts.append(d)
add(pick(te,N_TEST_RANDOM),'test_acak','test','label_final','label_audited')
add(pick(te[te.comment_id.isin(cf.comment_id)&~te.changed],N_TEST_KEPT),'test_ditinjau_dipertahankan','test','label_final','label_audited')
add(pick(te[te.changed],N_TEST_CHANGED),'test_diubah','test','label_final','label_audited')
add(pick(dev[dev.changed],N_DEV_CHANGED),'dev_diubah','dev','label_before_audit','label_final')
add(pick(dev[~dev.changed],N_DEV_UNCHANGED),'dev_tidak_diubah','dev','label_before_audit','label_final')
s=pd.concat(parts,ignore_index=True).drop_duplicates('comment_id').sample(frac=1,random_state=SEED).reset_index(drop=True)
s['blind_id']=[f'B{i:04d}' for i in range(len(s))]

raw=pd.read_csv(f'{R}/data/raw/comments_raw.csv',usecols=['comment_id','comment']).drop_duplicates('comment_id'); pm=dict(zip(raw.comment_id,raw.comment))
isrep=s.is_reply.astype(str).str.lower().eq('true')
sheet=pd.DataFrame({'blind_id':s.blind_id,'video_title':s.video_title,
    'induk':[pm.get(p,'') if r and pd.notna(p) else '' for p,r in zip(s.parent_id,isrep)],
    'komentar':s.comment_original,'label_anotator1':'','label_anotator2':'','label_adjudikasi':''})
sheet.to_csv(f'{A}/validation_blind_sheet.csv',index=False)           # HANYA ini yang dianotasi
s[['blind_id','comment_id','stratum','split','label_before','label_after']].to_csv(f'{A}/validation_blind_key.csv',index=False)
print(len(sheet),'baris'); print(s.stratum.value_counts())
