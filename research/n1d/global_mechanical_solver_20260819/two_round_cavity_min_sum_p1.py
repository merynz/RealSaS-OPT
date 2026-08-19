from __future__ import annotations
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import multi_edge_incident_audit_v1 as a
import one_pass_min_sum_message_solver_p0 as p0

ROOT=Path('/mnt/data/sufficiency');OUT=ROOT/'two_round_cavity_min_sum_p1';OUT.mkdir(exist_ok=True);FAMS=m.FAMILIES

def evalset(f,x,primary):
    rec=[]
    for i,n in enumerate(f['nodes']):
        ok=(n['reliable'] and n['M_cont2'] and (n['active'] and n['full_cont2'] if primary else True))
        if not ok: continue
        er=float(np.linalg.norm(n['M'][x[i]]-n['target'])); s=float(n['scale'])
        rec.append((er/s,er<=s,er<=2*s))
    return {'n':len(rec),'contain1':float(np.mean([r[1] for r in rec])),'contain2':float(np.mean([r[2] for r in rec])),'errors':[float(r[0]) for r in rec]}

def normmsg(v):
    v=np.asarray(v,np.float64)
    return v-float(np.min(v))

def solve_two_round(f):
    nodes=f['nodes']; U=[a.robust_z(n['Uraw']) for n in nodes]; edges=m.graph(f); deg=np.zeros(64,int)
    for i,j,_ in edges: deg[i]+=1; deg[j]+=1
    # Factor store only once with canonical orientation axes i,j.
    factors={}; neigh=[[] for _ in range(64)]
    for i,j,vs in edges:
        R=a.robust_z(m.pair_raw(f,i,j,vs)); w=1/max(int(deg[i]),1)+1/max(int(deg[j]),1)
        factors[(i,j)]=(R,float(w)); neigh[i].append(j); neigh[j].append(i)
    # Round 1 directed messages.
    m1={}
    for (i,j),(R,w) in factors.items():
        # i -> j : result indexed by k_j
        mij=np.min(U[i][:,None] + w*R,axis=0)
        # j -> i : result indexed by k_i
        mji=np.min(U[j][None,:] + w*R,axis=1)
        m1[(i,j)]=normmsg(mij); m1[(j,i)]=normmsg(mji)
    B1=[]
    for i in range(64):
        b=U[i].copy()
        for j in neigh[i]: b += m1[(j,i)]
        B1.append(b)
    x1=np.array([int(np.argmin(b)) for b in B1],int)
    # Round 2 synchronous cavity messages: source receives all round1 except destination echo.
    m2={}
    for (i,j),(R,w) in factors.items():
        cav_i=U[i].copy()
        for q in neigh[i]:
            if q!=j: cav_i += m1[(q,i)]
        cav_j=U[j].copy()
        for q in neigh[j]:
            if q!=i: cav_j += m1[(q,j)]
        mij=np.min(cav_i[:,None] + w*R,axis=0)
        mji=np.min(cav_j[None,:] + w*R,axis=1)
        m2[(i,j)]=normmsg(mij); m2[(j,i)]=normmsg(mji)
    B2=[]
    for i in range(64):
        b=U[i].copy()
        for j in neigh[i]: b += m2[(j,i)]
        B2.append(b)
    x2=np.array([int(np.argmin(b)) for b in B2],int)
    return x1,x2

def aggregate(per,arm,key):
    N=sum(per[str(fid)][key][arm]['n'] for fid in FAMS)
    c1=sum(per[str(fid)][key][arm]['contain1']*per[str(fid)][key][arm]['n'] for fid in FAMS)/N
    c2=sum(per[str(fid)][key][arm]['contain2']*per[str(fid)][key][arm]['n'] for fid in FAMS)/N
    errs=[];fc1={};fc2={}
    for fid in FAMS:
        q=per[str(fid)][key][arm]; errs+=q['errors']; fc1[str(fid)]=q['contain1'];fc2[str(fid)]=q['contain2']
    return {'n':N,'contain1':c1,'contain2':c2,'worst_family_contain1':min(fc1.values()),'worst_family_contain2':min(fc2.values()),'best_worst_gap':max(fc2.values())-min(fc2.values()),'median_norm_err':float(np.median(errs)),'p90_norm_err':float(np.quantile(errs,.9)),'per_family_contain1':fc1,'per_family_contain2':fc2}

def main():
    t=time.time(); fams={fid:m.build_family(fid) for fid in FAMS}; pf=m.preflight(fams)
    parity=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-.975609756097561)<1e-12 and abs(pf['worst']-.9322033898305084)<1e-12 and pf['families90']==8)
    per={}; index_equal={}; total_equal=0
    for fid in FAMS:
        f=fams[fid]
        xu,_,xp0=p0.solve(f)
        x1,x2=solve_two_round(f)
        eq=int(np.sum(x1==xp0)); index_equal[str(fid)]={'equal':eq,'n':64,'all':bool(eq==64)};total_equal+=eq
        per[str(fid)]={'primary':{},'secondary':{}}
        for key,pri in [('primary',True),('secondary',False)]:
            per[str(fid)][key]['U_ONLY']=evalset(f,xu,pri)
            per[str(fid)][key]['ROUND1']=evalset(f,x1,pri)
            per[str(fid)][key]['ROUND2']=evalset(f,x2,pri)
        print('FAMILY',fid,'IDX',eq,'R1',per[str(fid)]['primary']['ROUND1']['contain2'],'R2',per[str(fid)]['primary']['ROUND2']['contain2'],flush=True)
    agg={key:{arm:aggregate(per,arm,key) for arm in ['U_ONLY','ROUND1','ROUND2']} for key in ['primary','secondary']}
    U=agg['primary']['U_ONLY'];R1=agg['primary']['ROUND1'];R2=agg['primary']['ROUND2'];S0=agg['secondary']['U_ONLY'];S1=agg['secondary']['ROUND1'];S2=agg['secondary']['ROUND2']
    round1_metrics=bool(R1['n']==261 and abs(R1['contain1']-.47126436781609193)<1e-12 and abs(R1['contain2']-.7777777777777778)<1e-12 and abs(R1['worst_family_contain2']-.4)<1e-12 and abs(R1['median_norm_err']-1.0715530716411676)<1e-12 and abs(S1['contain2']-.8666666666666667)<1e-12)
    parity=bool(parity and U['n']==261 and abs(U['contain2']-.6781609195402298)<1e-12 and total_equal==512 and round1_metrics)
    hard=[11032,13203,15290]
    gains_u={str(fid):R2['per_family_contain2'][str(fid)]-U['per_family_contain2'][str(fid)] for fid in hard}
    worstgain_u=R2['worst_family_contain2']-U['worst_family_contain2']
    absolute=bool(R2['contain2']>=.75-1e-12 and R2['worst_family_contain2']>=.60-1e-12 and all(R2['per_family_contain2'][str(fid)]>=.60-1e-12 for fid in hard) and R2['contain1']>=.50-1e-12 and R2['worst_family_contain1']>=.35-1e-12 and R2['median_norm_err']<=1.0+1e-12 and R2['best_worst_gap']<=.30+1e-12)
    causal_pool=bool(R2['contain2']-U['contain2']>=.08-1e-12)
    hard_causal=bool(worstgain_u>=.08-1e-12 or (sum(v>=.10-1e-12 for v in gains_u.values())>=2 and all(v>=-.05-1e-12 for v in gains_u.values())))
    causal=bool(causal_pool and hard_causal)
    safety=bool(S2['contain2']>=S0['contain2']-.03-1e-12)
    fam_delta_r1={str(fid):R2['per_family_contain2'][str(fid)]-R1['per_family_contain2'][str(fid)] for fid in FAMS}
    hard_delta_r1={str(fid):fam_delta_r1[str(fid)] for fid in hard}
    depth=bool(R2['contain2']-R1['contain2']>=.03-1e-12 and R2['worst_family_contain2']-R1['worst_family_contain2']>=.08-1e-12 and all(v>=-.05-1e-12 for v in fam_delta_r1.values()) and S2['contain2']>=S1['contain2']-.03-1e-12)
    amplifies=bool(R2['contain2']<R1['contain2']-.03-1e-12 or any(v<-.10-1e-12 for v in hard_delta_r1.values()))
    if not parity: verdict='INVALID_ROUND1_PARITY_FAIL'
    elif absolute and causal and safety: verdict='TWO_ROUND_CAVITY_MIN_SUM_P1_ABSOLUTE_PASS'
    elif depth: verdict='TWO_ROUND_CAVITY_MIN_SUM_DEPTH_SUPPORTED_BUT_ABSOLUTE_FAIL'
    elif amplifies: verdict='SECOND_ROUND_AMPLIFIES_FACTOR_BIAS'
    else: verdict='SECOND_ROUND_NOT_MATERIAL__FACTOR_OR_OBJECTIVE_INSUFFICIENCY_REMAINS'
    result={'schema':'RealSaS.N1D.TwoRoundCavityMinSumP1.v1','date':'2026-08-19','prereg_commit':'85267a10f0cc53451ee9101b1d21338366aa4260','parity':{'all':parity,'m256_den':int(pf['den']),'m256_pooled2':float(pf['pooled2']),'m256_worst':float(pf['worst']),'primary_n':U['n'],'u_contain2':U['contain2'],'round1_metric_parity':round1_metrics,'round1_index_equal_total':total_equal,'round1_index_equal_per_family':index_equal},'aggregate':agg,'gains_vs_u':{'pooled_contain2':R2['contain2']-U['contain2'],'worst_family_contain2':worstgain_u,'hardtails':gains_u,'secondary_contain2':S2['contain2']-S0['contain2']},'round2_vs_round1':{'pooled_contain2':R2['contain2']-R1['contain2'],'worst_family_contain2':R2['worst_family_contain2']-R1['worst_family_contain2'],'per_family_contain2':fam_delta_r1,'secondary_contain2':S2['contain2']-S1['contain2']},'gates':{'absolute':absolute,'causal_pool':causal_pool,'hardtail_causal':hard_causal,'causal':causal,'safety':safety,'depth_materiality':depth,'amplifies_bias':amplifies},'verdict':verdict,'seconds':time.time()-t}
    path=OUT/'RESULT.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('PARITY',json.dumps(result['parity'],indent=2));print('PRIMARY',json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('errors','per_family_contain1')} for k,v in agg['primary'].items()},indent=2));print('SECONDARY',json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('errors','per_family_contain1','per_family_contain2')} for k,v in agg['secondary'].items()},indent=2));print('R2_VS_R1',json.dumps(result['round2_vs_round1'],indent=2));print('GATES',json.dumps(result['gates'],indent=2));print('VERDICT',verdict);print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__': main()
