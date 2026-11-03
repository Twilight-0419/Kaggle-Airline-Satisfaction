from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs'/'public_stack'; D=ROOT/'data'; EXT=next(ROOT.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def r(x): return rankdata(x,method='average')/len(x)
def main():
 tr=pd.read_csv(D/'train.csv'); te=pd.read_csv(D/'test.csv'); y=tr.satisfaction.to_numpy(); ids=te.id.to_numpy(); pred=[]
 for n in ['ext_l2','ext_l1','ext2_l2','ext3_l2','ext4_l2','ext5_l2','stack','l1_c003']:
  if (O/f'{n}_oof.npy').exists(): pred.append((n,r(np.load(O/f'{n}_oof.npy')),r(np.load(O/f'{n}_test.npy'))))
 for n in ['04','05','06','07']:
  a=EXT/f'{n}_blend_oof.csv'; b=EXT/f'{n}_submission.csv'
  if a.exists() and b.exists(): pred.append(('najiama_'+n,r(pd.read_csv(a).set_index('id').reindex(tr.id).satisfaction.to_numpy()),r(pd.read_csv(b).set_index('id').reindex(te.id).satisfaction.to_numpy())))
 X=np.column_stack([p[1] for p in pred]); XT=np.column_stack([p[2] for p in pred]); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)); W=[]; scores=[]
 # Coordinate grid search learns weights on four folds then scores the fifth.
 for f,(it,iv) in enumerate(folds):
  w=np.zeros(X.shape[1]); w[np.argmax([roc_auc_score(y[it],X[it,j]) for j in range(X.shape[1])])]=1
  for _ in range(4):
   for j in range(X.shape[1]):
    best=(roc_auc_score(y[it],X[it]@w),w[j])
    for v in np.arange(0,1.001,.1):
     ww=w.copy(); ww[j]=v; s=roc_auc_score(y[it],X[it]@ww)
     if s>best[0]: best=(s,v)
    w[j]=best[1]
  W.append(w); scores.append(roc_auc_score(y[iv],X[iv]@w)); print('fold',f,'cvauc',scores[-1],'weights',[(pred[j][0],round(v,3)) for j,v in enumerate(w) if v>.01],flush=True)
 w=np.mean(W,axis=0); po=X@w; pt=XT@w; auc=roc_auc_score(y,po); print('mean weights',[(pred[j][0],round(v,4)) for j,v in enumerate(w) if v>.005],'OOF',auc,'foldmean',np.mean(scores),flush=True)
 if auc>roc_auc_score(y,np.load(O/'ext_l2_oof.npy')):
  pd.DataFrame({'id':ids,'satisfaction':pt}).to_csv(D/'submission.csv',index=False); pd.DataFrame({'id':ids,'satisfaction':pt}).to_csv(O/'submission_crossfit_blend.csv',index=False); print('UPDATED',flush=True)
if __name__=='__main__':main()
