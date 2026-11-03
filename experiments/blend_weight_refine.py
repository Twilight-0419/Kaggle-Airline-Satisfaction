from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];O=R/'outputs'/'public_stack';D=R/'data';E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def main():
 tr=pd.read_csv(D/'train.csv'); y=tr.satisfaction.to_numpy(); a=np.load(O/'ext_l2_oof.npy');b=pd.read_csv(E/'05_blend_oof.csv').set_index('id').reindex(tr.id).satisfaction.to_numpy(); x=logit(np.clip(a,1e-7,1-1e-7));z=logit(np.clip(b,1e-7,1-1e-7));cv=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(np.zeros(len(y)),y)); rows=[]
 for w in np.arange(.08,.181,.005):
  p=expit((1-w)*x+w*z); fold=[roc_auc_score(y[v],p[v]) for _,v in cv]; rows.append({'weight_05':w,'oof_auc':roc_auc_score(y,p),'fold_mean':np.mean(fold),'fold_std':np.std(fold)});print(rows[-1],flush=True)
 pd.DataFrame(rows).to_csv(O/'blend_weight_refine.csv',index=False)
if __name__=='__main__':main()
