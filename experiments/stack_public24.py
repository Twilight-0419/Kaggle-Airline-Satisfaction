from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'/'public_stack'
PUB=next(ROOT.glob('某个公开算法*'))/'extra_oof'/'s6e10-stack-oof-predictions'

def as_logit(x):
    x=np.asarray(x,dtype=np.float64)
    return logit(np.clip(x,1e-6,1-1e-6)).astype(np.float32)

def main():
    train=pd.read_csv(DATA/'train.csv',usecols=['satisfaction']); y=train.satisfaction.to_numpy()
    X0=np.load(OUT/'X_oof_ext5u.npy',mmap_mode='r'); T0=np.load(OUT/'X_test_ext5u.npy',mmap_mode='r')
    n0=pd.read_csv(OUT/'model_names_ext5u.csv').model.astype(str).tolist()
    aucs=pd.read_csv(OUT/'model_table_ext5u.csv').set_index('model').oof_auc
    ix=[n0.index(n) for n in aucs.sort_values(ascending=False).index if n in n0][:32]
    cols_o=[np.asarray(X0[:,i]) for i in ix]; cols_t=[np.asarray(T0[:,i]) for i in ix]
    names=[n0[i] for i in ix]; seen={hashlib.sha1(np.ascontiguousarray(v).view(np.uint8)).hexdigest() for v in cols_o}
    added=[]
    for p in sorted(PUB.glob('*.npz')):
        z=np.load(p); o=as_logit(z['oof']); t=as_logit(z['test'])
        h=hashlib.sha1(np.ascontiguousarray(o).view(np.uint8)).hexdigest()
        if h in seen: continue
        seen.add(h); cols_o.append(o); cols_t.append(t); names.append('public24:'+p.stem); added.append(p.stem)
    X=np.column_stack(cols_o).astype(np.float32); XT=np.column_stack(cols_t).astype(np.float32)
    print('matrix',X.shape,'new public members',added,flush=True)
    cv=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    best=(-1,None,None,None)
    for c in [.003,.01,.03,.1,.3]:
        oof=np.zeros(len(y),dtype=np.float32); testp=np.zeros(XT.shape[0],dtype=np.float32); coefs=[]
        for fold,(tr,va) in enumerate(cv,1):
            m=LogisticRegression(C=c,solver='lbfgs',max_iter=300,tol=1e-5)
            m.fit(X[tr],y[tr]); oof[va]=m.predict_proba(X[va])[:,1]; testp+=m.predict_proba(XT)[:,1]/5; coefs.append(m.coef_[0])
            print(f'C={c} fold={fold} auc={roc_auc_score(y[va],oof[va]):.6f}',flush=True)
        auc=roc_auc_score(y,oof); foldmean=np.mean([roc_auc_score(y[va],oof[va]) for _,va in cv])
        print('RESULT',{'C':c,'oof_auc':auc,'fold_mean':foldmean},flush=True)
        if auc>best[0]: best=(auc,c,oof,testp)
    auc,c,oof,testp=best
    np.save(OUT/'public24_stack_oof.npy',oof);np.save(OUT/'public24_stack_test.npy',testp)
    base=np.load(OUT/'ext_l2_oof.npy'); baset=np.load(OUT/'ext_l2_test.npy')
    rows=[]
    from scipy.stats import rankdata
    r=lambda x:rankdata(x,method='average')/len(x)
    for w in np.arange(0,1.001,.025):
        p=(1-w)*r(base)+w*r(oof); fm=np.mean([roc_auc_score(y[v],p[v]) for _,v in cv]); rows.append((fm,roc_auc_score(y,p),w))
    print('BLENDS',sorted(rows,reverse=True)[:5],flush=True)
    pd.DataFrame(rows,columns=['fold_mean','oof_auc','weight_public24']).to_csv(OUT/'public24_stack_blends.csv',index=False)
    w=max(rows)[2]; pred=(1-w)*r(baset)+w*r(testp)
    pd.DataFrame({'id':pd.read_csv(DATA/'test.csv',usecols=['id']).id,'satisfaction':pred}).to_csv(OUT/'submission_public24_stack.csv',index=False)

if __name__=='__main__': main()
