from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];O=R/'outputs'/'public_stack';D=R/'data'
def main():
    y=pd.read_csv(D/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy();ids=pd.read_csv(D/'test.csv',usecols=['id']).id.to_numpy()
    X=np.load(O/'X_oof_ext5u.npy',mmap_mode='r');T=np.load(O/'X_test_ext5u.npy',mmap_mode='r')
    ix=pd.read_csv(O/'model_table_ext5u.csv').sort_values('oof_auc',ascending=False).head(32).model
    names=pd.read_csv(O/'model_names_ext5u.csv').model.tolist(); cols=[names.index(x) for x in ix]
    X=np.asarray(X[:,cols],dtype=np.float32);T=np.asarray(T[:,cols],dtype=np.float32)
    cv=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y));oof=np.zeros(len(y),np.float32);test=np.zeros(len(T),np.float32)
    for f,(tr,va) in enumerate(cv,1):
        m=CatBoostClassifier(iterations=700,depth=5,learning_rate=.04,loss_function='Logloss',eval_metric='AUC',l2_leaf_reg=5,random_seed=42,thread_count=8,verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=80,verbose=False)
        oof[va]=m.predict_proba(X[va])[:,1];test+=m.predict_proba(T)[:,1]/5
        np.save(O/'catboost_meta_public24_oof_partial.npy',oof);np.save(O/'catboost_meta_public24_test_partial.npy',test)
        print('fold',f,'auc',roc_auc_score(y[va],oof[va]),'bestiter',m.best_iteration_,flush=True)
    print('OOF',roc_auc_score(y,oof),flush=True)
    np.save(O/'catboost_meta_public24_oof.npy',oof);np.save(O/'catboost_meta_public24_test.npy',test)
    pd.DataFrame({'id':ids,'satisfaction':test}).to_csv(O/'submission_catboost_meta_public24.csv',index=False)
if __name__=='__main__':main()
