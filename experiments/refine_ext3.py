from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'public_stack';D=R/'data'
def rn(x):return rankdata(x,method='average')/len(x)
def main():
 y=pd.read_csv(D/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(); ids=pd.read_csv(D/'test.csv',usecols=['id']).id.to_numpy();X=np.load(O/'X_oof_ext3u.npy',mmap_mode='r');T=np.load(O/'X_test_ext3u.npy',mmap_mode='r'); names=pd.read_csv(O/'model_names_ext3u.csv').model.tolist(); tab=pd.read_csv(O/'model_table_ext3u.csv'); ix=[names.index(n) for n in tab.model.head(24) if n in names];best=0
 for seed in [7]:
  cv=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
  for k in [6,8,12,16]:
   for C in [.005,.01,.015,.02,.025,.03]:
    o=np.zeros(len(y));t=np.zeros(len(ids))
    for tr,va in cv:
     m=LogisticRegression(C=C,max_iter=300,tol=1e-4);m.fit(X[tr][:,:][:,ix[:k]],y[tr]);o[va]=m.predict_proba(X[va][:,ix[:k]])[:,1];t+=m.predict_proba(T[:,ix[:k]])[:,1]/5
    a=roc_auc_score(y,o);print(seed,k,C,round(a,9),flush=True)
    if a>best:
     best=a; np.save(O/'refine_ext3_oof.npy',o);np.save(O/'refine_ext3_test.npy',t);pd.DataFrame({'id':ids,'satisfaction':t}).to_csv(O/'submission_refine_ext3.csv',index=False);print('BEST',best,flush=True)
if __name__=='__main__':main()
