from pathlib import Path
from datetime import datetime
import time, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'public_stack'; DATA=ROOT/'data'
def main():
 y=pd.read_csv(DATA/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(); te=pd.read_csv(DATA/'test.csv',usecols=['id'])
 seed=int(os.environ.get('CV_SEED','42'))
 cv=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
 logs=[]; best=-1
 print('CV seed',seed,flush=True)
 for suffix in ['','ext','ext2u','ext3u','ext4u','ext5u']:
  name='X_oof'+('_'+suffix if suffix else '')+'.npy'; xtn='X_test'+('_'+suffix if suffix else '')+'.npy'
  xp=OUT/name; tp=OUT/xtn
  if not xp.exists() or not tp.exists(): continue
  X=np.load(xp,mmap_mode='r'); XT=np.load(tp,mmap_mode='r')
  # Focus on compact models to limit noisy meta fits.
  tabp=OUT/('model_table'+('_'+suffix if suffix else '')+'.csv')
  nmp=OUT/('model_names'+('_'+suffix if suffix else '')+'.csv')
  names=pd.read_csv(nmp).model.tolist()
  if tabp.exists():
   top=pd.read_csv(tabp).model.head(32).tolist(); idx=[names.index(n) for n in top if n in names]
  else: idx=list(range(min(X.shape[1],32)))
  for nfeat in [8,16,32]:
   ix=idx[:nfeat]
   for C in [0.003,0.01,0.03,0.1]:
    oof=np.zeros(len(y)); testp=np.zeros(len(te)); t0=time.time()
    for tr,va in cv:
     m=LogisticRegression(C=C,solver='lbfgs',max_iter=250,tol=1e-4)
     m.fit(X[tr][:,ix],y[tr]); oof[va]=m.predict_proba(X[va][:,ix])[:,1]; testp+=m.predict_proba(XT[:,ix])[:,1]/5
    auc=roc_auc_score(y,oof); row={'matrix':suffix or 'base','features':nfeat,'C':C,'auc':auc,'seconds':round(time.time()-t0,1)}; logs.append(row)
    print(row,flush=True)
    if auc>best:
     best=auc; np.save(OUT/'iterate_best_oof.npy',oof); np.save(OUT/'iterate_best_test.npy',testp)
     pd.DataFrame({'id':te.id,'satisfaction':testp}).to_csv(OUT/'submission_iterate_models.csv',index=False)
     print('NEW BEST',row,flush=True)
  del X,XT
 pd.DataFrame(logs).to_csv(OUT/'iterate_model_results.csv',index=False)
 # Only replace active submission when this round beats the known ext_l2 OOF result.
 champion=np.load(OUT/'ext_l2_oof.npy'); ca=roc_auc_score(y,champion)
 if best>ca:
  sub=pd.read_csv(OUT/'submission_iterate_models.csv'); sub.to_csv(DATA/'submission.csv',index=False)
  print('ACTIVE SUBMISSION UPDATED',best,flush=True)
 print('ROUND COMPLETE',datetime.now().isoformat(timespec='seconds'),'best',best,'baseline',ca,flush=True)
if __name__=='__main__': main()
