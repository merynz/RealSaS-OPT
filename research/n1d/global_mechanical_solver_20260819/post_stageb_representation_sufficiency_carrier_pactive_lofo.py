from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import testB_carrier_collision as tb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

roots=[
    Path('/mnt/data/sufficiency/currentH/09908'),
    Path('/mnt/data/sufficiency/currentH/11032'),
    Path('/mnt/data/sufficiency/currentH/12772'),
    Path('/mnt/data/sufficiency/currentH/13203'),
    Path('/mnt/data/sufficiency/currentH/14404'),
    Path('/mnt/data/sufficiency/currentH/14702'),
    Path('/mnt/data/sufficiency/currentH/14758'),
    Path('/mnt/data/sufficiency/currentH/15290'),
]

rows=[]
for r in roots:
    rows.extend(tb.family_features(r))

X=np.array([[r['num'][17]] for r in rows],float)
y=np.array([r['target_active'] for r in rows],int)
fam=np.array([r['family'] for r in rows])
p=np.full(len(rows),np.nan)

for f in sorted(set(fam)):
    tr=fam!=f
    te=~tr
    model=Pipeline([
        ('scale',StandardScaler()),
        ('clf',LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=0)),
    ])
    model.fit(X[tr],y[tr])
    p[te]=model.predict_proba(X[te])[:,1]

res={
    'schema':'RealSaS.N1D.PostStageB.RepresentationSufficiency.CarrierPActiveLOFO.v1',
    'feature':'log(eps + ||consensus(delta_point_map_srcA)||)',
    'families':{},
    'overall_auroc':float(roc_auc_score(y,p)),
    'overall_balanced_accuracy_0_5':float(balanced_accuracy_score(y,p>=.5)),
}
for f in sorted(set(fam)):
    q=fam==f
    yy=y[q]
    pp=p[q]
    res['families'][str(int(f))]={
        'n':int(q.sum()),
        'moving':int(yy.sum()),
        'nonmoving':int((1-yy).sum()),
        'auroc':float(roc_auc_score(yy,pp)) if len(set(yy))==2 else None,
        'balanced_accuracy_0_5':float(balanced_accuracy_score(yy,pp>=.5)),
    }
print(json.dumps(res,indent=2))
Path('/mnt/data/sufficiency/CARRIER_PACTIVE_LOFO.json').write_text(json.dumps(res,indent=2))
