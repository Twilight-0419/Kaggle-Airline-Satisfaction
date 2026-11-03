from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'public_stack'
DATA=ROOT/'data'
EXT=next(ROOT.glob('某个公开算法*'))/'extra_oof'
def rank(x): return rankdata(x,method='average')/len(x)
def main():
 tr=pd.read_csv(DATA/'train.csv'); te=pd.read_csv(DATA/'test.csv'); y=tr.satisfaction.to_numpy(); ids=te.id.to_numpy()
 d=EXT/'s6e10-oof'
 pairs=[]
 for name in ['ext_l2','ext_l1','ext2_l2','ext3_l2','ext4_l2','ext5_l2','stack','l1_c003']:
  op=OUT/f'{name}_oof.npy'; tp=OUT/f'{name}_test.npy'
  if op.exists() and tp.exists(): pairs.append((name,np.load(op),np.load(tp)))
 for name in ['05','06','04','07']:
  op=d/f'{name}_blend_oof.csv'; tp=d/f'{name}_submission.csv'
  if op.exists() and tp.exists():
   a=pd.read_csv(op).set_index('id').reindex(tr.id).satisfaction.to_numpy(); b=pd.read_csv(tp).set_index('id').reindex(te.id).satisfaction.to_numpy(); pairs.append(('najiama_'+name,a,b))
 # Id-aware isolated submissions as extra diversity.
 for p in (ROOT/'outputs'/'final_candidates').glob('submission*.csv'):
  z=pd.read_csv(p).set_index('id').reindex(te.id).satisfaction.to_numpy()
  # no OOF counterpart, so don't use these in CV blend search.
 matrix=np.column_stack([rank(x[1]) for x in pairs]); testmat=np.column_stack([rank(x[2]) for x in pairs])
 print('single AUC',[(p[0],round(roc_auc_score(y,p[1]),8)) for p in pairs],flush=True)
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(matrix,y))
 # Learn blend weights on four folds, score untouched fold. Small diverse shortlist avoids huge noisy search.
 aucs={}
 for j,p in enumerate(pairs): aucs[p[0]]=[]
 # Candidates convex mixtures of every strong single with ext_l2, plus top pairs and rank/equal blend.
 base=next((i for i,p in enumerate(pairs) if p[0]=='ext_l2'),0)
 for i,p in enumerate(pairs):
  for w in np.arange(0,1.001,.05):
   pred=w*matrix[:,base]+(1-w)*matrix[:,i]
   for k,(it,iv) in enumerate(folds): aucs.setdefault(f'{p[0]}_w{w:.2f}',[]).append((k,roc_auc_score(y[iv],pred[iv])))
 # direct candidates, CV average
 results=[]
 for name, vals in aucs.items():
  if vals and isinstance(vals[0],tuple) and len(vals)==5: results.append((np.mean([v for _,v in vals]),name))
 results.sort(reverse=True); print('CV top',results[:15],flush=True)
 # Pick via CV; refit selected blend using its identified two-model construction.
 best=results[0][1]; import re
 m=re.match(r'(.+)_w([0-9.]+)',best); idx=next(i for i,p in enumerate(pairs) if p[0]==m.group(1)); w=float(m.group(2))
 pred=w*matrix[:,base]+(1-w)*matrix[:,idx]; tp=w*testmat[:,base]+(1-w)*testmat[:,idx]
 # Compare full OOF to established champion, persist only if actual OOF improves.
 champion=np.load(OUT/'ext_l2_oof.npy'); ca=roc_auc_score(y,champion); ba=roc_auc_score(y,pred)
 if ba>ca:
  final=tp; label=f'{best}'; score=ba
 else:
  final=testmat[:,base]; label='ext_l2'; score=ca
 sub=pd.DataFrame({'id':ids,'satisfaction':final}); sub.to_csv(DATA/'submission.csv',index=False)
 sub.to_csv(OUT/'submission_iterative_best.csv',index=False)
 print('SAVED',label,'OOF',score,'CV',results[0],flush=True)
if __name__=='__main__': main()
