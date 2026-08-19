from __future__ import annotations
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/sufficiency')
import v3_rank_runner as m
import multi_edge_incident_audit_v1 as a
ROOT=Path('/mnt/data/sufficiency');OUT=ROOT/'one_pass_min_sum_message_solver_p0';OUT.mkdir(exist_ok=True);FAMS=m.FAMILIES

def evalset(f,x,primary):
 rec=[]
 for i,n in enumerate(f['nodes']):
  ok=(n['reliable'] and n['M_cont2'] and (n['active'] and n['full_cont2'] if primary else True))
  if not ok:continue
  er=float(np.linalg.norm(n['M'][x[i]]-n['target']));s=float(n['scale']);rec.append((er/s,er<=s,er<=2*s))
 return {'n':len(rec),'contain1':float(np.mean([r[1] for r in rec])),'contain2':float(np.mean([r[2] for r in rec])),'errors':[float(r[0]) for r in rec]}

def solve(f):
 nodes=f['nodes'];U=[a.robust_z(n['Uraw']) for n in nodes];xu=np.array([int(np.argmin(u)) for u in U],int);edges=m.graph(f);deg=np.zeros(64,int)
 for i,j,_ in edges:deg[i]+=1;deg[j]+=1
 hard=[u.copy() for u in U]; belief=[u.copy() for u in U]
 for i,j,vs in edges:
  R=a.robust_z(m.pair_raw(f,i,j,vs));w=1/max(int(deg[i]),1)+1/max(int(deg[j]),1)
  hard[i]+=w*R[:,xu[j]];hard[j]+=w*R[xu[i],:]
  # j -> i and i -> j one-pass min-sum, frozen unary incoming only
  mji=np.min(U[j][None,:]+w*R,axis=1);mji-=np.min(mji)
  mij=np.min(U[i][:,None]+w*R,axis=0);mij-=np.min(mij)
  belief[i]+=mji;belief[j]+=mij
 xh=np.array([int(np.argmin(v)) for v in hard],int);xm=np.array([int(np.argmin(v)) for v in belief],int)
 return xu,xh,xm

def aggregate(per,arm,key):
 N=sum(per[str(fid)][key][arm]['n'] for fid in FAMS);c1=sum(per[str(fid)][key][arm]['contain1']*per[str(fid)][key][arm]['n'] for fid in FAMS)/N;c2=sum(per[str(fid)][key][arm]['contain2']*per[str(fid)][key][arm]['n'] for fid in FAMS)/N;errs=[];fc1={};fc2={}
 for fid in FAMS:
  q=per[str(fid)][key][arm];errs+=q['errors'];fc1[str(fid)]=q['contain1'];fc2[str(fid)]=q['contain2']
 return {'n':N,'contain1':c1,'contain2':c2,'worst_family_contain1':min(fc1.values()),'worst_family_contain2':min(fc2.values()),'best_worst_gap':max(fc2.values())-min(fc2.values()),'median_norm_err':float(np.median(errs)),'p90_norm_err':float(np.quantile(errs,.9)),'per_family_contain1':fc1,'per_family_contain2':fc2}

def main():
 t=time.time();fams={fid:m.build_family(fid) for fid in FAMS};pf=m.preflight(fams);parity=bool(pf['parity'] and pf['den']==492 and abs(pf['pooled2']-.975609756097561)<1e-12 and abs(pf['worst']-.9322033898305084)<1e-12 and pf['families90']==8)
 per={}
 for fid in FAMS:
  xu,xh,xm=solve(fams[fid]);per[str(fid)]={'primary':{},'secondary':{}}
  for key,pri in [('primary',True),('secondary',False)]:
   per[str(fid)][key]['U_ONLY']=evalset(fams[fid],xu,pri);per[str(fid)][key]['HARD_U_LOCAL']=evalset(fams[fid],xh,pri);per[str(fid)][key]['MIN_SUM_P0']=evalset(fams[fid],xm,pri)
  print('FAMILY',fid,'U',per[str(fid)]['primary']['U_ONLY']['contain2'],'H',per[str(fid)]['primary']['HARD_U_LOCAL']['contain2'],'M',per[str(fid)]['primary']['MIN_SUM_P0']['contain2'],flush=True)
 agg={key:{arm:aggregate(per,arm,key) for arm in ['U_ONLY','HARD_U_LOCAL','MIN_SUM_P0']} for key in ['primary','secondary']};U=agg['primary']['U_ONLY'];G=agg['primary']['MIN_SUM_P0'];S0=agg['secondary']['U_ONLY'];S=agg['secondary']['MIN_SUM_P0']
 parity=bool(parity and U['n']==261 and abs(U['contain2']-.6781609195402298)<1e-12)
 hard=[11032,13203,15290];gains={str(fid):G['per_family_contain2'][str(fid)]-U['per_family_contain2'][str(fid)] for fid in hard};worstgain=G['worst_family_contain2']-U['worst_family_contain2']
 absolute=bool(G['contain2']>=.75-1e-12 and G['worst_family_contain2']>=.60-1e-12 and all(G['per_family_contain2'][str(fid)]>=.60-1e-12 for fid in hard) and G['contain1']>=.50-1e-12 and G['worst_family_contain1']>=.35-1e-12 and G['median_norm_err']<=1.0+1e-12 and G['best_worst_gap']<=.30+1e-12)
 causalpool=bool(G['contain2']-U['contain2']>=.08-1e-12);hardcausal=bool(worstgain>=.08-1e-12 or (sum(v>=.10-1e-12 for v in gains.values())>=2 and all(v>=-.05-1e-12 for v in gains.values())));causal=causalpool and hardcausal;safety=bool(S['contain2']>=S0['contain2']-.03-1e-12)
 verdict='ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_PASS' if parity and absolute and causal and safety else ('INVALID_PARITY_FAIL' if not parity else 'ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL')
 result={'schema':'RealSaS.N1D.OnePassMinSumMessageSolverP0.v1','date':'2026-08-19','prereg_commit':'a1169be16dea0b706a179c59c979770b0ae06347','parity':{'all':parity,'den':pf['den'],'m256_pooled2':pf['pooled2'],'m256_worst':pf['worst'],'primary_n':U['n'],'u_contain2':U['contain2']},'aggregate':agg,'hardtail_gains':gains,'gains':{'pooled_primary_contain2':G['contain2']-U['contain2'],'worst_family_contain2':worstgain,'secondary_contain2':S['contain2']-S0['contain2']},'gates':{'absolute':absolute,'causal_pool':causalpool,'hardtail_causal':hardcausal,'causal':causal,'safety':safety},'verdict':verdict,'seconds':time.time()-t}
 path=OUT/'RESULT.json';path.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='aggregate'},indent=2));print('PRIMARY',json.dumps(agg['primary'],indent=2));print('SECONDARY',json.dumps({k:{kk:vv for kk,vv in v.items() if not kk.startswith('per_family') and kk not in ('errors',)} for k,v in agg['secondary'].items()},indent=2));print('RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
