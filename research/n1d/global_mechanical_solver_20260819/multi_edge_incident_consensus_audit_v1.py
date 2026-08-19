from __future__ import annotations
import json, math, hashlib, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/data/sufficiency')
import v3_rank_runner as m

ROOT=Path('/mnt/data/sufficiency')
OUT=ROOT/'multi_edge_incident_audit_v1'
OUT.mkdir(exist_ok=True)
FAMS=m.FAMILIES

# Frozen V2 pair calibration: median/IQR, epsilon floor 1e-6.
def robust_z(a):
    a=np.asarray(a,dtype=np.float64)
    med=float(np.median(a))
    iqr=max(float(np.quantile(a,.75)-np.quantile(a,.25)),1e-6)
    return (a-med)/iqr

def midrank_vec(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    s=np.sort(a,kind='stable')
    left=np.searchsorted(s,a,side='left')
    right=np.searchsorted(s,a,side='right')
    return (left+0.5*(right-left))/len(a)

def oracle_idx(n):
    return int(np.argmin(np.linalg.norm(np.asarray(n['M'])-np.asarray(n['target'])[None,:],axis=1)))

def primary(n):
    return bool(n['reliable'] and n['M_cont2'] and n['active'] and n['full_cont2'])

def quality_for_family(f, choices, eligible_ids):
    rec=[]
    for i in eligible_ids:
        n=f['nodes'][i]
        q=int(choices[i])
        err=float(np.linalg.norm(n['M'][q]-n['target']))
        sc=float(n['scale'])
        rec.append({'i':i,'norm_err':err/max(sc,1e-12),'contain1':bool(err<=sc),'contain2':bool(err<=2*sc)})
    if not rec:
        return {'n':0,'contain1':None,'contain2':None,'median_norm_err':None,'records':[]}
    return {'n':len(rec),'contain1':float(np.mean([r['contain1'] for r in rec])),'contain2':float(np.mean([r['contain2'] for r in rec])),'median_norm_err':float(np.median([r['norm_err'] for r in rec])),'records':rec}

def pooled_quality(per_family, arm):
    rec=[]; fam_c2={}; fam_c1={}
    for fid in FAMS:
        q=per_family[str(fid)]['quality'][arm]
        fam_c2[str(fid)]=q['contain2']; fam_c1[str(fid)]=q['contain1']; rec += q['records']
    return {'n':len(rec),'contain1':float(np.mean([r['contain1'] for r in rec])),'contain2':float(np.mean([r['contain2'] for r in rec])),'median_norm_err':float(np.median([r['norm_err'] for r in rec])),'worst_family_contain1':float(min(fam_c1.values())),'worst_family_contain2':float(min(fam_c2.values())),'best_worst_contain2_gap':float(max(fam_c2.values())-min(fam_c2.values())),'per_family_contain2':fam_c2,'per_family_contain1':fam_c1}

def summarize_edge_stats(rows):
    if not rows: return {'n':0}
    keys=['degree','median_edge_oracle_rank','max_edge_oracle_rank','iqr_edge_oracle_rank','frac_edges_rank_le_025','frac_edges_rank_le_010','sum_oracle_rank','loeo_best_rank','loeo_rank_improvement']
    out={'n':len(rows)}
    for k in keys:
        vals=[r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        if vals: out[k]={'median':float(np.median(vals)),'mean':float(np.mean(vals)),'p90':float(np.quantile(vals,.9))}
    return out

def main():
    t=time.time(); fams={fid:m.build_family(fid) for fid in FAMS}; pf=m.preflight(fams)
    parity=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-0.975609756097561)<1e-12 and abs(pf['worst']-0.9322033898305084)<1e-12 and pf['families90']==8)
    if not parity: raise SystemExit('PREFLIGHT_PARITY_FAIL')
    per_family={}; all_rows=[]; all_bad=[]
    for fid in FAMS:
        f=fams[fid]; nodes=f['nodes']; edges=m.graph(f); deg=np.zeros(64,dtype=int)
        for i,j,_ in edges: deg[i]+=1; deg[j]+=1
        oracle=np.array([oracle_idx(n) for n in nodes],dtype=int)
        neighbor_state=np.array([oracle_idx(n) if (n['reliable'] and n['M_cont2']) else int(np.argmin(n['Uraw'])) for n in nodes],dtype=int)
        incident=[[] for _ in range(64)]
        for i,j,vs in edges:
            raw=m.pair_raw(f,i,j,vs); rz=robust_z(raw); w=1/max(int(deg[i]),1)+1/max(int(deg[j]),1)
            incident[i].append({'neighbor':j,'w':float(w),'c':rz[:,neighbor_state[j]],'edge':(i,j)})
            incident[j].append({'neighbor':i,'w':float(w),'c':rz[neighbor_state[i],:],'edge':(i,j)})
        eligible=[i for i,n in enumerate(nodes) if primary(n)]
        choices={'C_SUM':{},'TRIM_MAX1_Z':{},'MEDIAN_EDGE_RANK':{}}; rows=[]
        for i in eligible:
            n=nodes[i]; oi=int(oracle[i]); inc=incident[i]
            if not inc: continue
            edge_ranks=[]; contrib=[]; pvec=[]
            for e in inc:
                c=np.asarray(e['c'],dtype=np.float64); pv=midrank_vec(c)
                edge_ranks.append(float(pv[oi])); contrib.append(float(e['w'])*c); pvec.append(pv)
            A=np.stack(contrib,axis=0); Csum=np.sum(A,axis=0); sum_rank=float(midrank_vec(Csum)[oi])
            choices['C_SUM'][i]=int(np.argmin(Csum))
            Ctrim=Csum-np.max(A,axis=0) if len(inc)>1 else Csum.copy(); choices['TRIM_MAX1_Z'][i]=int(np.argmin(Ctrim))
            Cmed=np.median(np.stack(pvec,axis=0),axis=0); choices['MEDIAN_EDGE_RANK'][i]=int(np.argmin(Cmed))
            loeo_best=sum_rank
            if len(inc)>=2:
                loeo=[float(midrank_vec(Csum-A[k])[oi]) for k in range(len(inc))]; loeo_best=float(min(loeo))
            ranks=np.asarray(edge_ranks,dtype=float)
            row={'fid':int(fid),'i':int(i),'degree':int(len(inc)),'oracle_idx':oi,'median_edge_oracle_rank':float(np.median(ranks)),'max_edge_oracle_rank':float(np.max(ranks)),'iqr_edge_oracle_rank':float(np.quantile(ranks,.75)-np.quantile(ranks,.25)),'frac_edges_rank_le_025':float(np.mean(ranks<=.25)),'frac_edges_rank_le_010':float(np.mean(ranks<=.10)),'sum_oracle_rank':sum_rank,'loeo_best_rank':loeo_best,'loeo_rank_improvement':float(sum_rank-loeo_best),'sum_bad':bool(sum_rank>.25),'loeo_rescue_025':bool(sum_rank>.25 and loeo_best<=.25),'loeo_rescue_010':bool(sum_rank>.25 and loeo_best<=.10)}
            rows.append(row); all_rows.append(row)
            if row['sum_bad']: all_bad.append(row)
        qout={}
        for arm in ['C_SUM','TRIM_MAX1_Z','MEDIAN_EDGE_RANK']:
            xx=np.zeros(64,dtype=int)
            for i in eligible:
                if i not in choices[arm]: raise RuntimeError(('missing choice',fid,i,arm))
                xx[i]=choices[arm][i]
            qout[arm]=quality_for_family(f,xx,eligible)
        per_family[str(fid)]={'eligible_n':len(eligible),'edge_stats':summarize_edge_stats(rows),'bad_n':int(sum(r['sum_bad'] for r in rows)),'quality':qout,'rows':rows}
        print('FAMILY',fid,'n',len(eligible),'bad',per_family[str(fid)]['bad_n'],'C2',round(qout['C_SUM']['contain2'],6),'TRIM',round(qout['TRIM_MAX1_Z']['contain2'],6),'MED',round(qout['MEDIAN_EDGE_RANK']['contain2'],6),flush=True)
    pooled={arm:pooled_quality(per_family,arm) for arm in ['C_SUM','TRIM_MAX1_Z','MEDIAN_EDGE_RANK']}; base=pooled['C_SUM']
    consistency={'n_261':pooled['C_SUM']['n']==261,'pooled_contain2':abs(base['contain2']-0.8544061302681992)<1e-10,'worst_family_contain2':abs(base['worst_family_contain2']-0.68)<1e-10,'pooled_median_oracle_rank':abs(float(np.median([r['sum_oracle_rank'] for r in all_rows]))-0.072265625)<1e-12,'worst_family_median_oracle_rank':abs(max(float(np.median([r['sum_oracle_rank'] for r in per_family[str(fid)]['rows']])) for fid in FAMS)-0.201171875)<1e-12}
    consistency_all=bool(all(consistency.values())); bad_n=len(all_bad)
    rescue025=float(np.mean([r['loeo_rescue_025'] for r in all_bad])) if all_bad else 0.0; rescue010=float(np.mean([r['loeo_rescue_010'] for r in all_bad])) if all_bad else 0.0; bad_support_median=float(np.median([r['frac_edges_rank_le_025'] for r in all_bad])) if all_bad else None
    def robust_arm_support(arm):
        q=pooled[arm]
        return {'pooled_preservation':bool(q['contain2']>=base['contain2']-.03-1e-12),'worst_family_gain':float(q['worst_family_contain2']-base['worst_family_contain2']),'worst_family_gain_pass':bool(q['worst_family_contain2']>=base['worst_family_contain2']+.08-1e-12),'11032_gain':float(q['per_family_contain2']['11032']-base['per_family_contain2']['11032']),'11032_gain_pass':bool(q['per_family_contain2']['11032']>=base['per_family_contain2']['11032']+.08-1e-12),'all_required':bool(q['contain2']>=base['contain2']-.03-1e-12 and q['worst_family_contain2']>=base['worst_family_contain2']+.08-1e-12 and q['per_family_contain2']['11032']>=base['per_family_contain2']['11032']+.08-1e-12)}
    robust={arm:robust_arm_support(arm) for arm in ['TRIM_MAX1_Z','MEDIAN_EDGE_RANK']}; any_robust=any(x['all_required'] for x in robust.values())
    if bad_n>=10 and rescue025>=.50-1e-12 and any_robust: verdict='SINGLE_EDGE_OUTLIER_MATERIAL'
    elif bad_n>=10 and rescue025<.30 and bad_support_median<.50: verdict='COHERENT_MULTI_EDGE_WRONG'
    else: verdict='MIXED_MULTI_EDGE_FAILURE'
    result={'schema':'RealSaS.N1D.MultiEdgeIncidentConsensusAudit.v1','date':'2026-08-19','prereg_commit':'39c80ce3a636307ba9414d14a3f9099ed6d10ef8','preflight':{'parity':parity,'den':int(pf['den']),'pooled2':float(pf['pooled2']),'worst':float(pf['worst']),'families90':int(pf['families90']),'per2':{str(k):float(v) for k,v in pf['per2'].items()}},'behavioral_consistency':{**consistency,'all':consistency_all},'pooled_quality':pooled,'pooled_edge_stats':summarize_edge_stats(all_rows),'bad_nodes':{'n':bad_n,'loeo_rescue_to_025_fraction':rescue025,'loeo_rescue_to_010_fraction':rescue010,'median_edge_support_fraction_le_025':bad_support_median},'robust_arm_rule_checks':robust,'per_family':per_family,'verdict':verdict if consistency_all else 'INVALID_BASELINE_CONSISTENCY_FAIL','seconds':time.time()-t}
    p=OUT/'RESULT.json'; p.write_text(json.dumps(result,indent=2,allow_nan=False)); print('\nBASE CONSISTENCY',json.dumps(result['behavioral_consistency'],indent=2)); print('BAD',json.dumps(result['bad_nodes'],indent=2)); print('ROBUST',json.dumps(robust,indent=2)); print('POOLED',json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('per_family_contain1','per_family_contain2')} for k,v in pooled.items()},indent=2)); print('VERDICT',result['verdict']); print('RESULT_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=='__main__': main()
