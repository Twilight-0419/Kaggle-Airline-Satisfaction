from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit,expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];O=R/'outputs'/'public_stack';D=R/'data';E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def main():
    tr=pd.read_csv(D/'train.csv',usecols=['id','satisfaction']);te=pd.read_csv(D/'test.csv',usecols=['id']);y=tr.satisfaction.to_numpy()
    p0=np.load(O/'ext_l2_oof.npy');p1=np.load(O/'xgboost_meta_tuned_oof.npy')
    p2=pd.read_csv(E/'05_blend_oof.csv').set_index('id').reindex(tr.id).satisfaction.to_numpy()
    t0=np.load(O/'ext_l2_test.npy');t1=np.load(O/'xgboost_meta_tuned_test.npy');t2=pd.read_csv(E/'05_submission.csv').set_index('id').reindex(te.id).satisfaction.to_numpy()
    methods={'probability':(np.column_stack([p0,p1,p2]),np.column_stack([t0,t1,t2])),
             'logit':(np.column_stack([logit(np.clip(p0,1e-7,1-1e-7)),logit(np.clip(p1,1e-7,1-1e-7)),logit(np.clip(p2,1e-7,1-1e-7))]),np.column_stack([logit(np.clip(t0,1e-7,1-1e-7)),logit(np.clip(t1,1e-7,1-1e-7)),logit(np.clip(t2,1e-7,1-1e-7))]))}
    seeds=[42,7,2026,123,31415]; folds_by_seed=[list(StratifiedKFold(5,shuffle=True,random_state=s).split(np.zeros(len(y)),y)) for s in seeds]
    weights=[]
    for w1 in np.arange(0,.161,.02):
        for w2 in np.arange(0,.161,.02):
            if w1+w2<=.20:weights.append((1-w1-w2,w1,w2))
    rows=[]
    for method,(P,T) in methods.items():
        for w in weights:
            q=P@np.array(w); scores=[]
            for folds in folds_by_seed:scores.append(np.mean([roc_auc_score(y[v],q[v]) for _,v in folds]))
            rows.append((np.mean(scores),np.std(scores),min(scores),roc_auc_score(y,q),method,*w))
    df=pd.DataFrame(rows,columns=['mean_cv','std_cv','min_cv','oof_auc','method','w_ext_l2','w_xgb_meta','w_najiama05']).sort_values(['mean_cv','min_cv'],ascending=False)
    df.to_csv(O/'triple_blend_cv.csv',index=False);print(df.head(15).to_string(index=False),flush=True)
    b=df.iloc[0]; m=b['method'];ww=b[['w_ext_l2','w_xgb_meta','w_najiama05']].to_numpy(float); T=methods[m][1]; q=T@ww
    pd.DataFrame({'id':te.id,'satisfaction':expit(q) if m=='logit' else q}).to_csv(O/'submission_triple_blend.csv',index=False)
    print('BEST',b.to_dict(),flush=True)
if __name__=='__main__':main()
