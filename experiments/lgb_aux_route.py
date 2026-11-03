from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'/'public_stack'
PUBLIC=next(ROOT.glob('某个公开算法*'))/'extra_oof'/'s6e10-stack-oof-predictions'
TARGET='satisfaction'; ID='id'

def main():
    train=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv')
    y=train[TARGET].astype('int8').to_numpy(); n=len(train)
    allx=pd.concat([train.drop(columns=[TARGET]),test],ignore_index=True)
    aux=pd.read_parquet(PUBLIC/'aux_ev.parquet')
    if len(aux)!=len(allx): raise ValueError(f'aux rows {len(aux)} != data rows {len(allx)}')
    X=allx.drop(columns=[ID]).copy()
    # Cross-predicted expected ratings from the public auxiliary feature recipe.
    for c in aux.columns: X[c]=aux[c].to_numpy()
    # Route profile: distance-group means of other numeric variables, label-free.
    numeric=[c for c in allx.columns if c not in [ID,'Flight Distance'] and pd.api.types.is_numeric_dtype(allx[c])]
    route=allx.groupby('Flight Distance',dropna=False)[numeric].mean()
    for c in numeric:
        X[f'route_mean_{c}']=allx['Flight Distance'].map(route[c]).astype('float32')
    # Frequency features are calculated over train plus test, without labels.
    for c in ['Flight Distance','Age',*numeric[-13:]]:
        freq=allx[c].map(allx[c].value_counts(dropna=False))
        X[f'count_{c}']=np.log1p(freq).astype('float32')
    cats=[c for c in ['Gender','Customer Type','Type of Travel','Class'] if c in X]
    X=pd.get_dummies(X,columns=cats,dtype=np.int8)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(-1).astype('float32')
    X_train=X.iloc[:n].reset_index(drop=True); X_test=X.iloc[n:].reset_index(drop=True)
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    oof=np.zeros(n,dtype='float32'); testpred=np.zeros(len(test),dtype='float32'); started=time.time()
    for fold,(tr,va) in enumerate(cv.split(X_train,y),1):
        model=lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=2200,learning_rate=.025,
            num_leaves=63,min_child_samples=35,subsample=.85,subsample_freq=1,colsample_bytree=.8,
            reg_lambda=3.,n_jobs=8,random_state=420+fold,verbosity=-1)
        model.fit(X_train.iloc[tr],y[tr],eval_set=[(X_train.iloc[va],y[va])],eval_metric='auc',
            callbacks=[lgb.early_stopping(120,verbose=False)])
        oof[va]=model.predict_proba(X_train.iloc[va])[:,1]
        testpred+=model.predict_proba(X_test)[:,1]/5
        print(f'fold={fold} auc={roc_auc_score(y[va],oof[va]):.6f} iter={model.best_iteration_} elapsed={time.time()-started:.0f}s',flush=True)
    auc=roc_auc_score(y,oof); print('OOF',auc,'features',X_train.shape[1],flush=True)
    np.save(OUT/'lgb_aux_route_oof.npy',oof);np.save(OUT/'lgb_aux_route_test.npy',testpred)
    sub=pd.DataFrame({ID:test[ID].to_numpy(),TARGET:np.clip(testpred,0,1)})
    sub.to_csv(OUT/'submission_lgb_aux_route.csv',index=False)
    # Compare blend weights with the best established OOF stack.
    base=np.load(OUT/'ext_l2_oof.npy');base_t=np.load(OUT/'ext_l2_test.npy')
    from scipy.stats import rankdata
    rank=lambda v:rankdata(v,method='average')/len(v)
    best=(auc,'standalone',0.0)
    for w in np.arange(0,0.301,.025):
        p=(1-w)*rank(base)+w*rank(oof); score=roc_auc_score(y,p)
        if score>best[0]:best=(score,'rank_ext_l2_blend',float(w))
    print('BLEND_BEST',best,flush=True)
    if best[2]>0:
        w=best[2]; z=(1-w)*rank(base_t)+w*rank(testpred)
        pd.DataFrame({ID:test[ID].to_numpy(),TARGET:z}).to_csv(OUT/'submission_lgb_aux_route_blend.csv',index=False)

if __name__=='__main__': main()
