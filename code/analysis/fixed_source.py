"""Fixed Day-1 source comparison extracted from the local v6 build script."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def build_fixed_source(raw, output):
 RAW=Path(raw)
 OUT=Path(output)
 features=['envelope','mel_10','wav2vecbase9']; labels=['Envelope','Mel-10','wav2vec 2.0']
 def holm(p):
  p=np.array(p); order=np.argsort(p); q=np.empty(len(p)); q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return q
 rows=[]; pairs=[]; hashes=[]
 for f,label in zip(features,labels):
  pa=RAW/'module_A_intra_domain'/f/'neurascan.csv';pb=RAW/'module_B_sda'/f/'neurascan_to_brk.csv'
  a=pd.read_csv(pa).set_index('subject')['r1.0'];b=pd.read_csv(pb).set_index('subject')['S1_r1.0']
  z=pd.concat([a.rename('day2_accuracy'),b.rename('day3_accuracy')],axis=1).dropna();d=z.day2_accuracy-z.day3_accuracy;n=len(d);lo,hi=stats.t.interval(.95,n-1,loc=d.mean(),scale=d.sem())
  rows.append(dict(feature=label,n=n,day2_mean=z.day2_accuracy.mean(),day3_mean=z.day3_accuracy.mean(),delta_mean=d.mean(),ci_low=lo,ci_high=hi,t=stats.ttest_1samp(d,0).statistic,p=stats.ttest_1samp(d,0).pvalue,cohen_dz=d.mean()/d.std(ddof=1),missing_subjects=';'.join(a.index.difference(z.index))))
  for sub,row in z.iterrows():pairs.append(dict(feature=label,subject=sub,**row.to_dict(),delta=row.day2_accuracy-row.day3_accuracy,day2_source=str(pa.relative_to(RAW)),day2_column='r1.0',day3_source=str(pb.relative_to(RAW)),day3_column='S1_r1.0'))
 for row,q in zip(rows,holm([r['p'] for r in rows])):row['p_holm']=q
 pd.DataFrame(rows).to_csv(OUT/'fixed_source_comparison.csv',index=False)
 pd.DataFrame(pairs).to_csv(OUT/'fixed_source_participant_pairs.csv',index=False)
