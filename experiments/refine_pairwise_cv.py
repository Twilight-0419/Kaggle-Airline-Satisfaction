from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'public_stack'; D=R/'data'
E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def rank(x): return rankdata(x,method='average')/len(x)
def main():
    tr=pd.read_csv(D/'train.csv',usecols=['id','satisfaction']); te=pd.read_csv(D/'test.csv',usecols=['id'])
    y=tr.satisfaction.to_numpy(); ids=te.id.to_numpy()
    base=np.load(O/'ext_l2_oof.npy'); baset=np.load(O/'ext_l2_test.npy')
    candidates=[]
    for n in ['ext_l1','stack','l1_c003','xgboost_meta_tuned','lightgbm_meta','catboost_external','external_lgb','refine_ext3','public24_stack','lgb_aux_route']:
        a=O/f'{n}_oof.npy'; b=O/f'{n}_test.npy'
        if a.exists() and b.exists(): candidates.append((n,np.load(a),np.load(b)))
    for n in ['04','05','06','07']:
        a=E/f'{n}_blend_oof.csv'; b=E/f'{n}_submission.csv'
        if a.exists() and b.exists():
            candidates.append(('najiama_'+n,pd.read_csv(a).set_index('id').reindex(tr.id).satisfaction.to_numpy(),pd.read_csv(b).set_index('id').reindex(te.id).satisfaction.to_numpy()))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    transforms={'rank':(rank(base),rank(baset)), 'probability':(base,baset), 'logit':(logit(np.clip(base,1e-7,1-1e-7)),logit(np.clip(baset,1e-7,1-1e-7)))}
    rows=[]
    for name,a,b in candidates:
        for trname, (x,xt) in transforms.items():
            z={'rank':rank(a),'probability':a,'logit':logit(np.clip(a,1e-7,1-1e-7))}[trname]
            zt={'rank':rank(b),'probability':b,'logit':logit(np.clip(b,1e-7,1-1e-7))}[trname]
            for w in np.arange(0,0.501,.025):
                p=(1-w)*x+w*z
                scores=[roc_auc_score(y[v],p[v]) for _,v in folds]
                rows.append((np.mean(scores),roc_auc_score(y,p),name,trname,w,np.std(scores)))
        print('checked',name,flush=True)
    res=pd.DataFrame(rows,columns=['fold_mean','oof_auc','candidate','transform','weight','fold_std']).sort_values(['fold_mean','oof_auc'],ascending=False)
    res.to_csv(O/'pairwise_cv_refine.csv',index=False)
    print(res.head(20).to_string(index=False),flush=True)
    best=res.iloc[0]; name=str(best['candidate']); trname=str(best['transform']); w=float(best['weight'])
    a,b=next((a,b) for n,a,b in candidates if n==name)
    x,xt=transforms[trname]
    z={'rank':rank(a),'probability':a,'logit':logit(np.clip(a,1e-7,1-1e-7))}[trname]
    zt={'rank':rank(b),'probability':b,'logit':logit(np.clip(b,1e-7,1-1e-7))}[trname]
    o=(1-w)*x+w*z; t=(1-w)*xt+w*zt
    print('BEST',name,trname,w,'OOF',roc_auc_score(y,o),'CV',best.fold_mean,flush=True)
    if roc_auc_score(y,o)>roc_auc_score(y,base):
        pd.DataFrame({'id':ids,'satisfaction':t}).to_csv(O/'submission_pairwise_cv.csv',index=False)
        print('SAVED candidate; active submission unchanged until review',flush=True)
if __name__=='__main__': main()
