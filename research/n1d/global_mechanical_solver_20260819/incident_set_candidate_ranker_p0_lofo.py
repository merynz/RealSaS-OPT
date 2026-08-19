from __future__ import annotations
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import multi_edge_incident_audit_v1 as a

ROOT=Path('/mnt/data/sufficiency');OUT=ROOT/'incident_set_candidate_ranker_p0_lofo';OUT.mkdir(exist_ok=True);FAMS=m.FAMILIES

def primary(n): return bool(n['reliable'] and n['active'] and n['full_cont2'] and n['M_cont2'])

def build_examples(f):
    nodes=f['nodes'];edges=m.graph(f);u_state=np.array([int(np.argmin(n['Uraw'])) for n in nodes],int)
    inc=[[] for _ in range(64)]
    for i,j,vs in edges:
        rz=a.robust_z(m.pair_raw(f,i,j,vs))
        inc[i].append(a.midrank_vec(rz[:,u_state[j]]));inc[j].append(a.midrank_vec(rz[u_state[i],:]))
    out=[]
    for i,n in enumerate(nodes):
        if not primary(n): continue
        if not inc[i]: raise RuntimeError(('isolated primary',i))
        P=np.stack(inc[i],0); u=a.midrank_vec(n['Uraw']);
        feats=np.stack([u,P.mean(0),np.median(P,0),P.min(0),P.max(0),P.std(0),(P<=.25).mean(0),(P<=.10).mean(0)],1)
        err=np.linalg.norm(n['M']-n['target'][None,:],axis=1); y=(err<=2*n['scale']).astype(np.int64)
        out.append({'i':i,'X':feats,'y':y,'err':err,'scale':float(n['scale']),'Uidx':int(np.argmin(n['Uraw']))})
    return out

def eval_choices(f,exs,choices):
    rec=[]
    for e,q in zip(exs,choices):
        er=float(e['err'][q]); sc=e['scale'];rec.append((er/sc,er<=sc,er<=2*sc))
    return {'n':len(rec),'contain1':float(np.mean([r[1] for r in rec])),'contain2':float(np.mean([r[2] for r in rec])),'errors':[float(r[0]) for r in rec]}

def main():
    t=time.time();fams={fid:m.build_family(fid) for fid in FAMS};pf=m.preflight(fams)
    parity=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-.975609756097561)<1e-12 and abs(pf['worst']-.9322033898305084)<1e-12 and pf['families90']==8)
    examples={fid:build_examples(fams[fid]) for fid in FAMS};n=sum(len(v) for v in examples.values())
    # U baseline parity
    ub={};uall=[]
    for fid in FAMS:
        q=eval_choices(fams[fid],examples[fid],[e['Uidx'] for e in examples[fid]]);ub[str(fid)]=q;uall += q['errors']
    uden=sum(q['n'] for q in ub.values()); uc2=sum(q['contain2']*q['n'] for q in ub.values())/uden; uc1=sum(q['contain1']*q['n'] for q in ub.values())/uden
    parity=bool(parity and n==261 and abs(uc2-.6781609195402298)<1e-12)
    if not parity: raise SystemExit(('PARITY_FAIL',n,uc2))
    folds={};allrec=[]
    for hold in FAMS:
        Xtr=[];ytr=[]
        for fid in FAMS:
            if fid==hold:continue
            for e in examples[fid]:Xtr.append(e['X']);ytr.append(e['y'])
        Xtr=np.concatenate(Xtr,0);ytr=np.concatenate(ytr,0)
        sc=StandardScaler().fit(Xtr); Xs=sc.transform(Xtr)
        clf=LogisticRegression(C=1.0,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=1000,random_state=0)
        clf.fit(Xs,ytr)
        choices=[];rec=[]
        for e in examples[hold]:
            prob=clf.predict_proba(sc.transform(e['X']))[:,1];q=int(np.argmax(prob));choices.append(q)
            er=float(e['err'][q]);s=e['scale'];rec.append({'i':e['i'],'q':q,'norm_err':er/s,'contain1':bool(er<=s),'contain2':bool(er<=2*s),'pmax':float(prob[q])})
        folds[str(hold)]={'n':len(rec),'contain1':float(np.mean([r['contain1'] for r in rec])),'contain2':float(np.mean([r['contain2'] for r in rec])),'median_norm_err':float(np.median([r['norm_err'] for r in rec])),'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),'train_rows':int(len(Xtr)),'train_positive_rate':float(np.mean(ytr)),'records':rec}
        allrec += [(hold,r) for r in rec]
        print('FOLD',hold,'n',len(rec),'c1',folds[str(hold)]['contain1'],'c2',folds[str(hold)]['contain2'],'med',folds[str(hold)]['median_norm_err'],flush=True)
    N=len(allrec);c1=float(np.mean([r['contain1'] for _,r in allrec]));c2=float(np.mean([r['contain2'] for _,r in allrec]));med=float(np.median([r['norm_err'] for _,r in allrec]));p90=float(np.quantile([r['norm_err'] for _,r in allrec],.9));famc2={str(fid):folds[str(fid)]['contain2'] for fid in FAMS};famc1={str(fid):folds[str(fid)]['contain1'] for fid in FAMS};worstc2=min(famc2.values());worstc1=min(famc1.values());gap=max(famc2.values())-min(famc2.values())
    hard=[11032,13203,15290];gains={str(fid):famc2[str(fid)]-ub[str(fid)]['contain2'] for fid in hard};worst_gain=worstc2-min(q['contain2'] for q in ub.values())
    abs_gate=bool(c2>=.75-1e-12 and worstc2>=.60-1e-12 and all(famc2[str(fid)]>=.60-1e-12 for fid in hard) and c1>=.50-1e-12 and worstc1>=.35-1e-12 and med<=1.0+1e-12 and gap<=.30+1e-12)
    causal_pool=bool(c2-uc2>=.08-1e-12);hardcond=bool(worst_gain>=.08-1e-12 or (sum(g>=.10-1e-12 for g in gains.values())>=2 and all(g>=-.05-1e-12 for g in gains.values())));causal=bool(causal_pool and hardcond)
    verdict='INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_PASS' if abs_gate and causal else 'INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_FAIL'
    result={'schema':'RealSaS.N1D.IncidentSetCandidateRankerP0LOFO.v1','date':'2026-08-19','prereg_commit':'37a6ba7b975176fa8d549223166578707dfa2329','parity':{'all':parity,'den':pf['den'],'m256_pooled2':pf['pooled2'],'m256_worst':pf['worst'],'primary_n':n,'u_only_contain1':uc1,'u_only_contain2':uc2},'folds':folds,'aggregate':{'n':N,'contain1':c1,'contain2':c2,'worst_family_contain1':worstc1,'worst_family_contain2':worstc2,'median_norm_err':med,'p90_norm_err':p90,'best_worst_gap':gap,'per_family_contain1':famc1,'per_family_contain2':famc2},'hardtail_gains_vs_u':gains,'gains':{'pooled_contain2':c2-uc2,'worst_family_contain2':worst_gain},'gates':{'absolute':abs_gate,'causal_pool':causal_pool,'hardtail_causal':hardcond,'causal':causal},'verdict':verdict,'seconds':time.time()-t}
    path=OUT/'RESULT.json';path.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='folds'},indent=2));print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
