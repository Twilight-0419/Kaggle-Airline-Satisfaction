from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];O=R/'outputs'/'public_stack';D=R/'data';E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def main():
 tr=pd.read_csv(D/'train.csv',usecols=['id','satisfaction']);te=pd.read_csv(D/'test.csv',usecols=['id']);y=tr.satisfaction.to_numpy()
 p0=np.load(O/'ext_l2_oof.npy');p1=np.load(O/'xgboost_meta_tuned_oof.npy');p2=pd.read_csv(E/'05_blend_oof.csv').set_index('id').reindex(tr.id).satisfaction.to_numpy()
 T0=np.load(O/'ext_l2_test.npy');T1=np.load(O/'xgboost_meta_tuned_test.npy');T2=pd.read_csv(E/'05_submission.csv').set_index('id').reindex(te.id).satisfaction.to_numpy()
 P=np.column_stack([p0,p1,p2]);T=np.column_stack([T0,T1,T2]); seeds=[42,7,2026,123,31415]; folds=[list(StratifiedKFold(5,shuffle=True,random_state=s).split(np.zeros(len(y)),y)) for s in seeds]
 rows=[]
 for w1 in np.arange(.06,.101,.01):
  for w2 in np.arange(0,.041,.01):
   if w1+w2>.18:continue
   w=np.array([1-w1-w2,w1,w2]);q=P@w;scores=[np.mean([roc_auc_score(y[v],q[v]) for _,v in fs]) for fs in folds]
   rows.append((np.mean(scores),np.std(scores),min(scores),roc_auc_score(y,q),w[0],w[1],w[2]))
 d=pd.DataFrame(rows,columns=['mean_cv','std_cv','min_cv','oof_auc','w_ext_l2','w_xgb_meta','w_najiama05']).sort_values(['mean_cv','min_cv'],ascending=False);d.to_csv(O/'fine_triple_cv.csv',index=False);print(d.to_string(index=False),flush=True)
 b=d.iloc[0];w=b[['w_ext_l2','w_xgb_meta','w_najiama05']].to_numpy(float);pd.DataFrame({'id':te.id,'satisfaction':T@w}).to_csv(O/'submission_triple_blend_fine.csv',index=False);print('BEST',b.to_dict(),flush=True)
if __name__=='__main__':main()
