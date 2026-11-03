from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1];O=R/'outputs'/'public_stack';D=R/'data';E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def rank(x):return rankdata(x,method='average')/len(x)
def main():
 tr=pd.read_csv(D/'train.csv');te=pd.read_csv(D/'test.csv');y=tr.satisfaction.to_numpy();pred=[]
 for n in ['ext_l2','ext_l1','stack','xgboost_meta_tuned','lightgbm_meta','catboost_external']:
  a=O/f'{n}_oof.npy';b=O/f'{n}_test.npy'
  if a.exists() and b.exists():pred.append((n,np.load(a),np.load(b)))
 for n in ['04','05','06','07']:
  a=E/f'{n}_blend_oof.csv';b=E/f'{n}_submission.csv'
  if a.exists() and b.exists():pred.append(('najiama_'+n,pd.read_csv(a).set_index('id').reindex(tr.id).satisfaction.to_numpy(),pd.read_csv(b).set_index('id').reindex(te.id).satisfaction.to_numpy()))
 X=np.column_stack([rank(x[1]) for x in pred]);T=np.column_stack([rank(x[2]) for x in pred]);cv=StratifiedKFold(5,shuffle=True,random_state=42);results=[]
 for C in [.003,.01,.03,.1,.3,1.0]:
  o=np.zeros(len(y));t=np.zeros(len(te))
  for k,(ix,iv) in enumerate(cv.split(X,y)):
   m=LogisticRegression(C=C,max_iter=300,tol=1e-5);m.fit(X[ix],y[ix]);o[iv]=m.predict_proba(X[iv])[:,1];t+=m.predict_proba(T)[:,1]/5
  auc=roc_auc_score(y,o);row={'C':C,'auc':auc,'fold_mean':np.mean([roc_auc_score(y[iv],o[iv]) for _,iv in cv.split(X,y)])};results.append(row);print(row,flush=True)
  if auc>roc_auc_score(y,np.load(O/'ext_l2_oof.npy')):
   pd.DataFrame({'id':te.id,'satisfaction':t}).to_csv(O/'submission_meta_lr_top.csv',index=False);np.save(O/'meta_lr_top_oof.npy',o);np.save(O/'meta_lr_top_test.npy',t)
 print('columns', [p[0] for p in pred],flush=True)
if __name__=='__main__':main()
