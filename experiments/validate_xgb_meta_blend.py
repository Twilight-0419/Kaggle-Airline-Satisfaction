from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'public_stack'; D=R/'data'
def main():
    y=pd.read_csv(D/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy()
    a=np.load(O/'ext_l2_oof.npy'); b=np.load(O/'xgboost_meta_tuned_oof.npy')
    xforms={'probability':(a,b),'logit':(logit(np.clip(a,1e-7,1-1e-7)),logit(np.clip(b,1e-7,1-1e-7)))}
    rows=[]
    for seed in [7,2026,123,31415,20261029]:
        folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
        for method,(x,z) in xforms.items():
            for w in np.arange(.05,.151,.005):
                p=(1-w)*x+w*z; ss=[roc_auc_score(y[v],p[v]) for _,v in folds]
                rows.append((seed,method,w,np.mean(ss),np.std(ss),roc_auc_score(y,p)))
        print('seed done',seed,flush=True)
    df=pd.DataFrame(rows,columns=['seed','method','weight','fold_mean','fold_std','oof_auc'])
    df.to_csv(O/'xgb_meta_blend_multiseed.csv',index=False)
    agg=df.groupby(['method','weight']).agg(mean_cv=('fold_mean','mean'),std_cv=('fold_mean','std'),min_cv=('fold_mean','min'),mean_oof=('oof_auc','mean')).reset_index().sort_values(['mean_cv','min_cv'],ascending=False)
    print(agg.head(15).to_string(index=False),flush=True)
    best=agg.iloc[0]; print('ROBUST_BEST',best.to_dict(),flush=True)
    base=np.load(O/'ext_l2_test.npy'); pred=np.load(O/'xgboost_meta_tuned_test.npy'); w=float(best.weight)
    if best.method=='logit':
        from scipy.special import expit
        test=expit((1-w)*logit(np.clip(base,1e-7,1-1e-7))+w*logit(np.clip(pred,1e-7,1-1e-7)))
    else:test=(1-w)*base+w*pred
    te=pd.read_csv(D/'test.csv',usecols=['id'])
    pd.DataFrame({'id':te.id,'satisfaction':test}).to_csv(O/'submission_xgb_meta_robust.csv',index=False)
if __name__=='__main__':main()
