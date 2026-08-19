from __future__ import annotations
import sys,pickle,json,math,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/mnt/data/sufficiency')
import adaptive_h_v1 as a
import adaptive_fast_helpers as fh
import candidate_specific_local_motion_energy_v1 as prev
m=a.m; F=a.FAMILIES; DIR=a.DIRNAME; FAMS=pickle.load(open('/mnt/data/sufficiency/adaptive_fams.pkl','rb'))
EPS=1e-3; IQREPS=1e-6; DIRTHR=1.0
OUT=Path('/mnt/data/sufficiency/vector_motion_energy_v15');OUT.mkdir(exist_ok=True)

def H16(fam,i):
 C=[]
 for v in range(8):
  r=fam['views'][i][v];C.append(np.empty((0,2),np.float32) if r is None else np.asarray(r['c8'])[:16])
 return fh.make_H_fast(C)

def candidate_vectors(H,PA,usable):
 p0={v:fh.project_np(PA[None],v)[0] for v in usable}
 return {v:fh.project_np(H,v)-p0[v][None] for v in usable}

def vec_energy(vecs,usable,obs):
 vals=[]
 for v in usable:vals.append(np.linalg.norm(vecs[v]-obs[v][None],axis=1))
 return np.median(np.stack(vals,1),1) if len(vals)>=2 else np.full(len(next(iter(vecs.values()))),np.inf)

def polar_energy(vecs,usable,obs):
 vals=[]
 for v in usable:
  q=vecs[v];qm=np.linalg.norm(q,axis=1);o=np.asarray(obs[v],float);om=float(np.linalg.norm(o));ra=np.abs(np.log(EPS+qm)-math.log(EPS+om));rd=np.zeros(len(q))
  if om>=DIRTHR:
   mask=qm>=DIRTHR
   if mask.any():rd[mask]=1-np.clip((q[mask]@o)/(qm[mask]*om),-1,1)
  vals.append(ra+rd)
 return np.median(np.stack(vals,1),1) if len(vals)>=2 else np.full(len(next(iter(vecs.values()))),np.inf)

def fuse(Es):
 z=[]
 for E in Es:
  E=np.asarray(E,float);ok=np.isfinite(E)
  if ok.sum()<2:continue
  med=np.median(E[ok]);iq=max(np.quantile(E[ok],.75)-np.quantile(E[ok],.25),IQREPS);q=(E-med)/iq;q[~ok]=np.nan;z.append(q)
 if not z:return np.full_like(Es[0],np.inf)
 A=np.stack(z,1);r=np.nanmedian(A,1);r[~np.isfinite(r)]=np.inf;return r

def summarize(records,arm):
 rr=[r for r in records if r.get('eval')];ne=np.array([r[arm]['norm_err'] for r in rr]);c1=np.array([r[arm]['contain1'] for r in rr]);c2=np.array([r[arm]['contain2'] for r in rr]);w25=np.array([r[arm]['amp_within25'] for r in rr]);w50=np.array([r[arm]['amp_within50'] for r in rr])
 return {'n':len(rr),'contain1':float(c1.mean()),'contain2':float(c2.mean()),'median_norm_err':float(np.median(ne)),'p90_norm_err':float(np.quantile(ne,.90)),'amp_within25':float(w25.mean()),'amp_within50':float(w50.mean())}

def run_family(fid):
 fam=FAMS[fid];root=Path('/mnt/data/sufficiency/currentH')/DIR[fid];A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'))
 x=m.load_model_tensor(A,B)
 with torch.no_grad():out=m.MODEL(x)
 PA,VA,_,_,_,_=m.exact_problem_a_frontdoor(A,B,n=64,res=96);xy=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
 flows=m.forward_dis_flows(A,B);dis=np.stack([prev.bilinear_flow_samples(flows[v],xy[v]) for v in range(8)],0)
 tr=m.sample_field_np(out['transport_offset_srcA'][0],xy).astype(float)*(255/31)
 d3=m.sample_field_np(out['delta_point_map_srcA'][0],xy).astype(float);d2=np.empty((8,64,2),float)
 for v in range(8):d2[v]=fh.project_np(PA+d3[v],v)-fh.project_np(PA,v)
 z=np.load(root/'B'/'observation_sidecar.npz');center=np.asarray(z['camera_center'],float);half=float(z['camera_half_extent']);TA=(np.asarray(z['surface_points_A'],float)-center)/(2*half);TB=(np.asarray(z['surface_points_B'],float)-center)/(2*half);active=np.linalg.norm(TB[fam['j']]-TA[fam['j']],axis=1)>.005
 arms=['V_DIS','V_TRANSPORT','V_DELTA3D','V_FUSION','P_DIS','P_TRANSPORT','P_DELTA3D','P_FUSION'];rec=[]
 for i in range(64):
  H=H16(fam,i);target=fam['TB'][fam['j'][i]];sc=float(fam['sB'][fam['j'][i]]);herr=np.linalg.norm(H-target[None],axis=1) if len(H) else np.array([]);contained=bool(len(H) and herr.min()<=2*sc);ev=bool(fam['reliable'][i] and active[i] and contained);r={'i':i,'eval':ev,'reliable':bool(fam['reliable'][i]),'active':bool(active[i]),'contained':contained,'H_n':int(len(H))}
  if not ev:rec.append(r);continue
  hstar=H[int(herr.argmin())];astar=float(np.linalg.norm(hstar-PA[i]));amps=np.linalg.norm(H-PA[i][None],axis=1);usable=[v for v in range(8) if VA[v,i]];vecs=candidate_vectors(H,PA[i],usable)
  vd=vec_energy(vecs,usable,dis[:,i]);vt=vec_energy(vecs,usable,tr[:,i]);v3=vec_energy(vecs,usable,d2[:,i]);vf=fuse([vd,vt,v3]);pd=polar_energy(vecs,usable,dis[:,i]);pt=polar_energy(vecs,usable,tr[:,i]);p3=polar_energy(vecs,usable,d2[:,i]);pf=fuse([pd,pt,p3])
  em={'V_DIS':vd,'V_TRANSPORT':vt,'V_DELTA3D':v3,'V_FUSION':vf,'P_DIS':pd,'P_TRANSPORT':pt,'P_DELTA3D':p3,'P_FUSION':pf}
  for name,E in em.items():
   k=int(np.nanargmin(E));err=float(np.linalg.norm(H[k]-target));asel=float(amps[k]);ratio=max(asel,astar)/max(min(asel,astar),EPS);r[name]={'norm_err':err/max(sc,1e-12),'contain1':bool(err<=sc),'contain2':bool(err<=2*sc),'amp_within25':bool(ratio<=1.25),'amp_within50':bool(ratio<=1.5)}
  rec.append(r)
 return {'family':fid,'n_eval':sum(r['eval'] for r in rec),'arms':{x:summarize(rec,x) for x in arms},'records':rec}

def aggregate():
 fr={str(f):json.load(open(OUT/f'{f}.json')) for f in F};arms=['V_DIS','V_TRANSPORT','V_DELTA3D','V_FUSION','P_DIS','P_TRANSPORT','P_DELTA3D','P_FUSION'];out={'schema':'RealSaS.N1D.CandidateSpecificVectorMotionEnergy.v15','date':'2026-08-19','prereg_commit':'52da162d25c598ee793a0cabac807baf75198e32','families':fr,'arms':{}}
 for arm in arms:
  per={f:fr[str(f)]['arms'][arm] for f in F};n=sum(q['n'] for q in per.values());c1=sum(q['contain1']*q['n'] for q in per.values())/n;c2=sum(q['contain2']*q['n'] for q in per.values())/n;errs=[r[arm]['norm_err'] for q in fr.values() for r in q['records'] if r.get('eval')];vals=[q['contain2'] for q in per.values()]
  pool={'n':n,'contain1':float(c1),'contain2':float(c2),'worst_family_contain1':float(min(q['contain1'] for q in per.values())),'worst_family_contain2':float(min(vals)),'best_family_contain2':float(max(vals)),'best_worst_gap':float(max(vals)-min(vals)),'median_norm_err':float(np.median(errs)),'p90_norm_err':float(np.quantile(errs,.9)),'hardtail_13203_contain2':float(per[13203]['contain2']),'hardtail_15290_contain2':float(per[15290]['contain2']),'amp_within25':float(sum(q['amp_within25']*q['n'] for q in per.values())/n),'amp_within50':float(sum(q['amp_within50']*q['n'] for q in per.values())/n)}
  gate=bool(pool['contain2']>=.75 and pool['worst_family_contain2']>=.60 and pool['hardtail_13203_contain2']>=.60 and pool['hardtail_15290_contain2']>=.60 and pool['contain1']>=.55 and pool['worst_family_contain1']>=.40 and pool['median_norm_err']<=.90 and pool['best_worst_gap']<=.25)
  out['arms'][arm]={'per_family':per,'pooled':pool,'gate':gate}
 passes=[x for x in arms if out['arms'][x]['gate']]
 if not passes:dec='NO_VECTOR_ARM_PASSES__SINGLE_CARRIER_MOTION_VECTOR_INSUFFICIENT'
 else:
  singles=[x for x in passes if 'FUSION' not in x]
  if singles:
   best=max(singles,key=lambda x:out['arms'][x]['pooled']['worst_family_contain2']);dec=best;f='V_FUSION' if best.startswith('V_') else 'P_FUSION'
   if f in passes and out['arms'][f]['pooled']['worst_family_contain2']>=out['arms'][best]['pooled']['worst_family_contain2']+.05:dec=f
  else:dec=passes[0]
 out['decision']=dec;p=OUT/'AGGREGATE.json';p.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'decision':dec,'arms':{x:{'pooled':out['arms'][x]['pooled'],'gate':out['arms'][x]['gate'],'per_family_c2':{str(f):out['arms'][x]['per_family'][f]['contain2'] for f in F}} for x in arms}},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--family',type=int);ap.add_argument('--aggregate',action='store_true');z=ap.parse_args()
 if z.aggregate:aggregate()
 else:
  r=run_family(z.family);p=OUT/f'{z.family}.json';p.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({'family':z.family,'n_eval':r['n_eval'],'arms':r['arms']},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())
