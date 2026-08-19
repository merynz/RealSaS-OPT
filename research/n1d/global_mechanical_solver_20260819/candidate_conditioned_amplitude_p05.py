from __future__ import annotations
import sys,pickle,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
sys.path.insert(0,'/mnt/data/sufficiency')
import adaptive_h_v1 as a
import adaptive_fast_helpers as fh
m=a.m;F=a.FAMILIES;DIR=a.DIRNAME;EPS=1e-8
fams=pickle.load(open('/mnt/data/sufficiency/adaptive_fams.pkl','rb'))

def stats_visible(x,V):
 x=np.asarray(x,float);V=np.asarray(V,bool);out=[]
 for i in range(x.shape[1]):
  q=x[V[:,i],i];out.append(q if len(q) else x[:,i])
 return out
def scalar_stats(arrs):return np.array([[float(np.mean(q)),float(np.std(q)),float(np.max(q))] for q in arrs],float)
def pairacc(score,amp):
 q=[]
 for i in range(len(amp)):
  for j in range(i+1,len(amp)):
   if abs(amp[i]-amp[j])/max(amp[i],amp[j],EPS)<.10:continue
   q.append((score[i]-score[j])*(amp[i]-amp[j])>0)
 return float(np.mean(q)) if q else None

def extract(fid):
 root=Path('/mnt/data/sufficiency/currentH')/DIR[fid];A=sorted((root/'A').glob('*.png'));B=sorted((root/'B').glob('*.png'))
 x=m.load_model_tensor(A,B)
 with torch.no_grad():out=m.MODEL(x)
 PA,VA,_,_,_,_=m.exact_problem_a_frontdoor(A,B,n=64,res=96);xy=np.stack([m.project_points(PA,v) for v in range(8)],0).astype(np.float32)
 cur=m.sample_field_np(out['delta_point_map_srcA'][0],xy);dsig=m.sample_field_np(out['descriptor_log_sigma'][0,0],xy);lsig=m.sample_field_np(out['log_sigma_A'][0,0],xy);vlog=m.sample_field_np(out['visibility_logit_A'][0,0],xy);gate=m.sample_field_np(out['transport_gate_srcA'][0],xy);ent=m.sample_field_np(out['transport_entropy_srcA'][0],xy);peak=m.sample_field_np(out['transport_peak_srcA'][0],xy);edge=m.sample_field_np(out['transport_edge_mass_srcA'][0],xy)
 w=VA[...,None].astype(float);cons=(cur*w).sum(0)/np.maximum(w.sum(0),1);camp=np.linalg.norm(cons,axis=1);viewamp=np.linalg.norm(cur,axis=2);va=stats_visible(viewamp[...,None],VA);vast=np.array([[np.mean(q),np.std(q),np.max(q),np.median(q)] for q in va]);agreement=camp/np.maximum(vast[:,0],EPS);obs_scale=m.local_scale(PA)
 feats=[cons,camp[:,None],vast,agreement[:,None]]
 for q in [gate,ent,peak,edge,dsig,lsig]:feats.append(scalar_stats(stats_visible(q,VA)))
 visprob=1/(1+np.exp(-vlog));feats.append(scalar_stats(stats_visible(visprob,np.ones_like(VA))));feats.append(VA.sum(0)[:,None].astype(float));feats.append(obs_scale[:,None]);B1=np.concatenate(feats,1)
 z=np.load(root/'B'/'observation_sidecar.npz');center=np.asarray(z['camera_center'],float);half=float(z['camera_half_extent']);TA=(np.asarray(z['surface_points_A'],float)-center)/(2*half);TB=(np.asarray(z['surface_points_B'],float)-center)/(2*half);sA=m.local_scale(TA);sB=m.local_scale(TB);D=np.linalg.norm(PA[:,None]-TA[None],axis=2);j=D.argmin(1);reliable=D[np.arange(64),j]<=2*sA[j];tamp=np.linalg.norm(TB[j]-TA[j],axis=1);active=tamp>.005
 rows=[];fa=fams[fid]
 for i in range(64):
  C=[]
  for v in range(8):
   r=fa['views'][i][v];C.append(np.empty((0,2),np.float32) if r is None else np.asarray(r['c8'])[:16])
  H=fh.make_H_fast(C);target=TB[j[i]];herr=np.linalg.norm(H-target[None],axis=1) if len(H) else np.array([]);contained=bool(len(H) and herr.min()<=2*sB[j[i]])
  if len(H):
   hstar=H[int(herr.argmin())];amps=np.linalg.norm(H-PA[i][None],axis=1);astar=float(np.linalg.norm(hstar-PA[i]));la=np.log(EPS+amps);med=float(np.median(la));q25=float(np.quantile(la,.25));q75=float(np.quantile(la,.75));iqr=max(q75-q25,EPS);zh=(la-med)/iqr;zstar=float((math.log(EPS+astar)-med)/iqr);zcur=float((math.log(EPS+camp[i])-med)/iqr);qstar=float(np.mean(amps<=astar));hs=np.array([med,iqr,float(np.quantile(la,.10)),q25,q75,float(np.quantile(la,.90)),float(la.min()),float(la.max()),float(la.max()-la.min()),zcur])
  else:
   amps=np.array([]);zh=np.array([]);astar=np.nan;zstar=np.nan;zcur=np.nan;qstar=np.nan;hs=np.full(10,np.nan)
  rows.append({'fid':fid,'i':i,'reliable':bool(reliable[i]),'active':bool(active[i]),'contained':contained,'astar':astar,'zstar':zstar,'zcur':zcur,'qstar':qstar,'zh':zh,'amps':amps,'R0':np.array([zcur]),'R2':np.r_[B1[i],hs]})
 return rows

rows=[]
for f in F:
 print('[EXTRACT]',f,flush=True);rows.extend(extract(f))
R=[r for r in rows if r['reliable'] and r['active'] and r['contained'] and np.isfinite(r['zstar'])]
out={'schema':'RealSaS.N1D.CandidateConditionedAmplitude.P05.v1','date':'2026-08-19','prereg_commit':'de4ca727956a96db095181f49613db8ff9e7ee24','n_eval':len(R),'arms':{}}
for arm in ['R0','R1','R2']:
 pf={};allz=[];alls=[];allamp=[];allwithin=[]
 for held in F:
  tr=[r for r in R if r['fid']!=held];te=[r for r in R if r['fid']==held]
  if arm=='R0':pred=np.array([r['zcur'] for r in te])
  elif arm=='R1':
   Xtr=np.array([[r['zcur']] for r in tr]);ytr=np.array([r['zstar'] for r in tr]);Xte=np.array([[r['zcur']] for r in te]);rg=make_pipeline(StandardScaler(),Ridge(alpha=10));rg.fit(Xtr,ytr);pred=rg.predict(Xte)
  else:
   Xtr=np.stack([r['R2'] for r in tr]);ytr=np.array([r['zstar'] for r in tr]);Xte=np.stack([r['R2'] for r in te]);rg=make_pipeline(StandardScaler(),Ridge(alpha=10));rg.fit(Xtr,ytr);pred=rg.predict(Xte)
  zt=np.array([r['zstar'] for r in te]);amp=np.array([r['astar'] for r in te]);selected=[];within=[];logerr=[]
  for p,r in zip(pred,te):
   k=int(np.argmin(np.abs(r['zh']-p)));aa=float(r['amps'][k]);selected.append(aa);ratio=max(aa,r['astar'])/max(min(aa,r['astar']),EPS);within.append(ratio<=1.25);logerr.append(abs(math.log((aa+EPS)/(r['astar']+EPS))))
  rho=float(spearmanr(zt,pred).statistic) if len(te)>=3 else None;pa=pairacc(pred,amp)
  pf[str(held)]={'n':len(te),'spearman_z':rho,'pairwise_accuracy':pa,'median_abs_z_error':float(np.median(np.abs(pred-zt))),'within25':float(np.mean(within)),'median_abs_log_amp_ratio':float(np.median(logerr)),'oracle_q_median':float(np.median([r['qstar'] for r in te]))}
  allz.extend(zt);alls.extend(pred);allamp.extend(amp);allwithin.extend(within)
 rhos=[q['spearman_z'] for q in pf.values() if q['spearman_z'] is not None];pairs=[q['pairwise_accuracy'] for q in pf.values() if q['pairwise_accuracy'] is not None];withs=[q['within25'] for q in pf.values()]
 pool={'pooled_spearman':float(spearmanr(allz,alls).statistic),'median_family_spearman':float(np.median(rhos)),'worst_family_spearman':float(min(rhos)),'negative_family_count':int(sum(x<0 for x in rhos)),'mean_family_pairwise_accuracy':float(np.mean(pairs)),'worst_family_pairwise_accuracy':float(min(pairs)),'pooled_within25':float(np.mean(allwithin)),'worst_family_within25':float(min(withs))}
 gate=bool(pool['pooled_spearman']>=.75 and pool['median_family_spearman']>=.65 and pool['worst_family_spearman']>=.45 and pool['negative_family_count']==0 and pool['mean_family_pairwise_accuracy']>=.72 and pool['worst_family_pairwise_accuracy']>=.60 and pool['pooled_within25']>=.70 and pool['worst_family_within25']>=.55)
 out['arms'][arm]={'per_family':pf,'pooled':pool,'gate':gate}
if out['arms']['R1']['gate']:
 r1=out['arms']['R1']['pooled'];r2=out['arms']['R2']['pooled'];out['decision']='R2_TYPED_PLUS_H' if out['arms']['R2']['gate'] and r2['worst_family_spearman']>=r1['worst_family_spearman']+.05 and r2['worst_family_within25']>=r1['worst_family_within25']+.05 else 'R1_CALIBRATED_H_RELATIVE_CURRENT'
elif out['arms']['R2']['gate']:out['decision']='R2_TYPED_PLUS_H'
elif out['arms']['R0']['gate']:out['decision']='R0_FROZEN_CURRENT_H_RELATIVE__NO_RICHER_HEAD'
else:out['decision']='NO_ARM_PASSES__AMPLITUDE_EVIDENCE_CHANNEL_INSUFFICIENT'
p=Path('/mnt/data/sufficiency/CANDIDATE_CONDITIONED_AMPLITUDE_P05_RESULT.json');p.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'n_eval':len(R),'decision':out['decision'],'arms':{k:{'pool':v['pooled'],'gate':v['gate'],'per_family':{f:{'rho':q['spearman_z'],'pair':q['pairwise_accuracy'],'within25':q['within25']} for f,q in v['per_family'].items()}} for k,v in out['arms'].items()}},indent=2));print('sha',hashlib.sha256(p.read_bytes()).hexdigest())
