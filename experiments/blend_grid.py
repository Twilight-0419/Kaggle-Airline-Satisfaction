from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'public_stack'; D=R/'data'; E=next(R.glob('某个公开算法*'))/'extra_oof'/'s6e10-oof'
def rn(x):return rankdata(x,method='average')/len(x)
def main():
 tr=pd.read_csv(D/'train.csv'); te=pd.read_csv(D/'test.csv'); y=tr.satisfaction.to_numpy(); ids=te.id.to_numpy(); ps=[]
 for n in ['ext_l2','ext_l1','stack','l1_c003','external_lgb']:
  if (O/f'{n}_oof.npy').exists():ps.append((n,np.load(O/f'{n}_oof.npy'),np.load(O/f'{n}_test.npy')))
 for n in ['04','05','06','07']:
  a=E/f'{n}_blend_oof.csv';b=E/f'{n}_submission.csv'
  if a.exists() and b.exists():ps.append(('najiama_'+n,pd.read_csv(a).set_index('id').reindex(tr.id).satisfaction.to_numpy(),pd.read_csv(b).set_index('id').reindex(te.id).satisfaction.to_numpy()))
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)); results=[]
 for trans in ['rank','logit']:
  A=np.column_stack([rn(p[1]) if trans=='rank' else logit(np.clip(p[1],1e-7,1-1e-7)) for p in ps]); T=np.column_stack([rn(p[2]) if trans=='rank' else logit(np.clip(p[2],1e-7,1-1e-7)) for p in ps])
  for i in range(len(ps)):
   for j in range(i+1,len(ps)):
    vals=[]
    for w in np.arange(0,1.001,.025):
     z=w*A[:,i]+(1-w)*A[:,j]; sc=[roc_auc_score(y[v],z[v]) for _,v in folds]; results.append((np.mean(sc),trans,i,j,w,roc_auc_score(y,z)));

 results.sort(reverse=True);print('TOP',[(round(a,8),b,ps[c][0],ps[d][0],e,round(f,8)) for a,b,c,d,e,f in results[:20]],flush=True)
 # Candidate maximize fold-mean among pairwise fixed blends. Write only if full OOF exceeds current ext_l2.
 _,trans,i,j,w,auc=results[0]; A=np.column_stack([rn(p[1]) if trans=='rank' else logit(np.clip(p[1],1e-7,1-1e-7)) for p in ps]);T=np.column_stack([rn(p[2]) if trans=='rank' else logit(np.clip(p[2],1e-7,1-1e-7)) for p in ps]); o=w*A[:,i]+(1-w)*A[:,j];t=w*T[:,i]+(1-w)*T[:,j]
 print('BEST',ps[i][0],ps[j][0],trans,w,'OOF',roc_auc_score(y,o),'CV',auc,flush=True)
 if roc_auc_score(y,o)>roc_auc_score(y,np.load(O/'ext_l2_oof.npy')):
  pd.DataFrame({'id':ids,'satisfaction':t}).to_csv(D/'submission.csv',index=False);pd.DataFrame({'id':ids,'satisfaction':t}).to_csv(O/'submission_grid_blend.csv',index=False);print('UPDATED',flush=True)
if __name__=='__main__':main()
