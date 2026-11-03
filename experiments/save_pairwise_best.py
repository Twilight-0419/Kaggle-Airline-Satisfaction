from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'public_stack'; D=R/'data'
def rank(x): return rankdata(x,method='average')/len(x)
def main():
    tr=pd.read_csv(D/'train.csv',usecols=['id','satisfaction']); te=pd.read_csv(D/'test.csv',usecols=['id'])
    tab=pd.read_csv(O/'pairwise_cv_refine.csv'); best=tab.iloc[0]; n=best['candidate']; trans=best['transform']; w=float(best['weight'])
    a=np.load(O/f'{n}_oof.npy'); b=np.load(O/f'{n}_test.npy')
    base=np.load(O/'ext_l2_oof.npy'); baset=np.load(O/'ext_l2_test.npy')
    conv={'rank':rank,'probability':lambda x:x,'logit':lambda x:logit(np.clip(x,1e-7,1-1e-7))}[trans]
    o=(1-w)*conv(base)+w*conv(a); t=(1-w)*conv(baset)+w*conv(b)
    print('selected',n,trans,w,'OOF',roc_auc_score(tr.satisfaction,o),'active baseline',roc_auc_score(tr.satisfaction,base),'CV mean',best['fold_mean'])
    pd.DataFrame({'id':te.id,'satisfaction':t}).to_csv(O/'submission_pairwise_cv.csv',index=False)
if __name__=='__main__':main()
