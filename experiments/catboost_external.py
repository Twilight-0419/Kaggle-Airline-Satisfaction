from pathlib import Path
import time
import numpy as np,pandas as pd,catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1]; D=R/'data'; O=R/'outputs'/'public_stack'
def main():
 comp=pd.read_csv(D/'train.csv'); test=pd.read_csv(D/'test.csv'); orig=pd.read_csv(next(R.glob('某个公开算法*'))/'original_data'/'train.csv'); y=comp.satisfaction.astype(int).to_numpy(); yo=orig.satisfaction.map({'neutral or dissatisfied':0,'satisfied':1}).astype(int).to_numpy()
 cats=['Gender','Customer Type','Type of Travel','Class']; drop=['id','satisfaction']; feat=[c for c in comp.columns if c not in drop]; X=comp[feat].copy(); XT=test[feat].copy(); XO=orig[[c for c in feat if c!='Inflight service']].copy();
 for c in cats:
  X[c]=X[c].fillna('missing').astype(str); XT[c]=XT[c].fillna('missing').astype(str); XO[c]=XO[c].fillna('missing').astype(str)
 # add missing feature omitted from external raw data with neutral sentinel
 for c in feat:
  if c not in XO: XO[c]=0
 XO=XO[feat]; means=X['Arrival Delay in Minutes'].mean();
 for z in [X,XT,XO]:
  z['Arrival Delay in Minutes']=z['Arrival Delay in Minutes'].fillna(means)
 cv=StratifiedKFold(5,shuffle=True,random_state=42); o=np.zeros(len(y)); t=np.zeros(len(test)); started=time.time()
 for f,(tr,va) in enumerate(cv.split(X,y),1):
  xx=pd.concat([X.iloc[tr],XO],ignore_index=True); yy=np.r_[y[tr],yo]; sw=np.r_[np.ones(len(tr)),np.full(len(XO),.5)]
  m=cb.CatBoostClassifier(iterations=800,depth=6,learning_rate=.05,l2_leaf_reg=8,loss_function='Logloss',eval_metric='AUC',thread_count=8,random_seed=700+f,verbose=False,allow_writing_files=False,od_type='Iter',od_wait=80,cat_features=cats)
  m.fit(xx,yy,sample_weight=sw,eval_set=(X.iloc[va],y[va]),verbose=False);o[va]=m.predict_proba(X.iloc[va])[:,1];t+=m.predict_proba(XT)[:,1]/5; print('fold',f,'auc',roc_auc_score(y[va],o[va]),'iter',m.best_iteration_,'elapsed',round(time.time()-started),flush=True)
 print('OOF',roc_auc_score(y,o),flush=True);np.save(O/'catboost_external_oof.npy',o);np.save(O/'catboost_external_test.npy',t);pd.DataFrame({'id':test.id,'satisfaction':t}).to_csv(O/'catboost_external_submission.csv',index=False)
if __name__=='__main__':main()
