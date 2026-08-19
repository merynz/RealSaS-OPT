from __future__ import annotations
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import multi_edge_incident_audit_v1 as p

ROOT=Path('/mnt/data/sufficiency'); OUT=ROOT/'edge_reliability_cross_consensus_audit_v1';OUT.mkdir(exist_ok=True)
FAMS=m.FAMILIES

def ols_flow_inconsistency(f,i,j,views):
    A=[]; b=[]; norms=[]
    for v in views:
        # project_np displacement: +255*R dot g in x, -255*U dot g in y
        A.append((m.IMAGE_NATIVE-1)*m.RNP[v]); A.append(-(m.IMAGE_NATIVE-1)*m.UNP[v])
        r=np.asarray(f['dis'][v,i]-f['dis'][v,j],np.float64)
        b.extend([float(r[0]),float(r[1])]); norms.append(float(np.linalg.norm(r)))
    A=np.asarray(A,np.float64); b=np.asarray(b,np.float64)
    g=np.linalg.lstsq(A,b,rcond=None)[0]
    res=[]
    for k,v in enumerate(views):
        pred=np.array([(m.IMAGE_NATIVE-1)*np.dot(m.RNP[v],g),-(m.IMAGE_NATIVE-1)*np.dot(m.UNP[v],g)])
        r=np.asarray(f['dis'][v,i]-f['dis'][v,j],np.float64)
        res.append(float(np.linalg.norm(pred-r)))
    return float(np.median(res)/(np.median(norms)+1e-9))

def score_metrics(rows,key):
    hits1=[];hits2=[];nr=[]; per={}; degstr={'le4':[],'eq5':[],'ge6':[]}
    for r in rows:
        scores=np.asarray(r[key],float); target=int(r['damaging_pos']); order=np.argsort(-scores,kind='stable'); pos=int(np.where(order==target)[0][0]); d=len(scores)
        hits1.append(pos==0);hits2.append(pos<min(2,d));nr.append(pos/max(d-1,1))
        per.setdefault(str(r['fid']),[]).append(pos/max(d-1,1))
        degstr['le4' if d<=4 else ('eq5' if d==5 else 'ge6')].append(pos/max(d-1,1))
    fammean={k:float(np.mean(v)) for k,v in per.items()}
    out={'n':len(rows),'top1':float(np.mean(hits1)),'top2':float(np.mean(hits2)),'median_norm_rank':float(np.median(nr)),'mean_norm_rank':float(np.mean(nr)),'families_mean_rank_le_050':int(sum(v<=.50 for v in fammean.values())),'per_family_mean_norm_rank':fammean,'degree_strata_mean_norm_rank':{k:(float(np.mean(v)) if v else None) for k,v in degstr.items()}}
    out['supported']=bool(out['top1']>=.35-1e-12 and out['top2']>=.60-1e-12 and out['median_norm_rank']<=.35+1e-12 and out['mean_norm_rank']<=.40+1e-12 and out['families_mean_rank_le_050']>=5)
    return out

def main():
    t=time.time();fams={fid:m.build_family(fid) for fid in FAMS};pf=m.preflight(fams)
    parity=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-.975609756097561)<1e-12 and abs(pf['worst']-.9322033898305084)<1e-12 and pf['families90']==8)
    if not parity: raise SystemExit('PREFLIGHT_PARITY_FAIL')
    rows=[]; parent_all=[]; perbase={}
    for fid in FAMS:
        f=fams[fid];nodes=f['nodes'];edges=m.graph(f);deg=np.zeros(64,int)
        for i,j,_ in edges:deg[i]+=1;deg[j]+=1
        oracle=np.array([p.oracle_idx(n) for n in nodes],int)
        eval_state=np.array([oracle[i] if (n['reliable'] and n['M_cont2']) else int(np.argmin(n['Uraw'])) for i,n in enumerate(nodes)],int)
        u_state=np.array([int(np.argmin(n['Uraw'])) for n in nodes],int)
        # incident entries contain both evaluator conditional contribution and observation-only conditional vectors/scores
        inc=[[] for _ in range(64)]
        for edge_pos,(i,j,vs) in enumerate(edges):
            raw=m.pair_raw(f,i,j,vs);rz=p.robust_z(raw);w=1/max(int(deg[i]),1)+1/max(int(deg[j]),1)
            s3=ols_flow_inconsistency(f,i,j,vs)
            inc[i].append({'edge':(i,j),'eval_c':w*rz[:,eval_state[j]],'u_c':w*rz[:,u_state[j]],'s3':s3})
            inc[j].append({'edge':(i,j),'eval_c':w*rz[eval_state[i],:],'u_c':w*rz[u_state[i],:],'s3':s3})
        fam_base=[]
        for i,n in enumerate(nodes):
            if not p.primary(n):continue
            ii=inc[i];oi=int(oracle[i]);C=np.sum(np.stack([e['eval_c'] for e in ii]),0);sr=float(p.midrank_vec(C)[oi]);q=int(np.argmin(C));err=float(np.linalg.norm(n['M'][q]-n['target'])/n['scale']);fam_base.append((err<=2,sr))
            parent_all.append({'fid':fid,'i':i,'contain2':bool(err<=2),'sum_rank':sr})
            if sr<=.25:continue
            # evaluator damaging-edge target from exact LOEO
            loeo=[float(p.midrank_vec(C-e['eval_c'])[oi]) for e in ii]
            damaging=int(np.argmin(np.asarray(loeo)))
            # observation-only edge conditional percentile vectors under U_ONLY neighbor state
            pv=np.stack([p.midrank_vec(e['u_c']) for e in ii],0); med=np.median(pv,0)
            s1=np.mean(np.abs(pv-med[None,:]),axis=1)
            # observation-only edge endpoint votes and geometric outlier
            votes=np.stack([n['M'][int(np.argmin(e['u_c']))] for e in ii],0);vmed=np.median(votes,axis=0);s2=np.linalg.norm(votes-vmed[None,:],axis=1)/max(float(n['scale']),1e-12)
            s3=np.array([e['s3'] for e in ii],float)
            rows.append({'fid':int(fid),'i':int(i),'degree':len(ii),'damaging_pos':damaging,'damaging_edge':list(ii[damaging]['edge']),'loeo_best_rank':float(min(loeo)),'sum_rank':sr,'S1':s1.tolist(),'S2':s2.tolist(),'S3':s3.tolist()})
        perbase[str(fid)]={'n':len(fam_base),'contain2':float(np.mean([x[0] for x in fam_base])),'median_rank':float(np.median([x[1] for x in fam_base]))}
    # parent guards
    pooled_c2=float(np.mean([x['contain2'] for x in parent_all]));medrank=float(np.median([x['sum_rank'] for x in parent_all]));worst_c2=min(x['contain2'] for x in perbase.values());worst_rank=max(x['median_rank'] for x in perbase.values())
    rescue=float(np.mean([r['loeo_best_rank']<=.25 for r in rows]))
    guards={'primary_n':len(parent_all)==261,'bad_n':len(rows)==46,'pair_sum_pooled_contain2':abs(pooled_c2-.8544061302681992)<1e-12,'pair_sum_worst_family_contain2':abs(worst_c2-.68)<1e-12,'pooled_median_oracle_rank':abs(medrank-.072265625)<1e-12,'worst_family_median_oracle_rank':abs(worst_rank-.201171875)<1e-12,'loeo_rescue_025':abs(rescue-.45652173913043476)<1e-12}; valid=all(guards.values())
    metrics={k:score_metrics(rows,k) for k in ['S1','S2','S3']}; passing=[k for k,v in metrics.items() if v['supported']]
    verdict='OBS_EDGE_RELIABILITY_SIGNAL_FOUND_V1' if passing else 'NO_SIMPLE_OBS_EDGE_RELIABILITY_SIGNAL_V1'
    if not valid:verdict='INVALID_PARENT_PARITY_FAIL'
    result={'schema':'RealSaS.N1D.EdgeReliabilityCrossConsensusAudit.v1','date':'2026-08-19','prereg_commit':'f08f788301eef559abf245da1e703bd52b50d2c4','parent_guards':{**guards,'all':valid},'parent_reproduced':{'pooled_c2':pooled_c2,'worst_c2':worst_c2,'median_rank':medrank,'worst_rank':worst_rank,'bad_n':len(rows),'loeo_rescue_025':rescue},'metrics':metrics,'passing_predictors':passing,'verdict':verdict,'rows':rows,'seconds':time.time()-t}
    path=OUT/'RESULT.json';path.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
