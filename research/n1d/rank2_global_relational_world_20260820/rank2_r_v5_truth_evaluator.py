from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

V3A_SELECTION_SHA='82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41'
V5_SELECTION_SHA='277946d94a3685222fb0cef4ad3b34d4c4656657309cccdc65204e01a2386887'
V5_SOURCE_SHA='ea7ab996f881a08538c3cc049ca799e98783919255ed2c5214c01feada3161d7'
RANK_TOL=1e-8
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)
FAMS=(9908,11032,12772,13203,14404,14702,14758,15290)
HIST={
 'overall': {'n':2288,'g_hit_n':490,'v3a_hit_n':679,'g_mean':3.640371613838409,'g_median':1.7272416852096728,'v3a_mean':2.5693494032099724,'v3a_median':1.4062604064852646,'abstain_n':4,'improve':522,'same':1443,'worsen':319},
 'hardtail_rank3': {'n':79,'g_hit_n':18,'v3a_hit_n':21,'g_mean':4.665084409688869,'g_median':1.5543076080365525,'v3a_mean':3.2236907342392276,'v3a_median':1.5204439867767832,'abstain_n':0,'improve':12,'same':52,'worsen':15},
 'carrier': {'improved':6,'same':3,'worsened':6},
}

def sha256_file(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,str(path));mod=importlib.util.module_from_spec(spec);assert spec.loader is not None;sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def camera_basis():
 R=[];U=[]
 for deg in YAW_DEG:
  t=math.radians(deg);radial=np.array([math.sin(t),-math.cos(t),0.],float);fw=-radial;up=np.array([0.,0.,1.]);right=np.cross(fw,up);right/=np.linalg.norm(right);R.append(right);U.append(up)
 return np.stack(R),np.stack(U)
CAM_R,CAM_U=camera_basis()
def projective_rank(vcol):
 views=np.where(np.asarray(vcol)>0)[0]
 if not len(views):return 0
 A=[]
 for v in views:A += [CAM_R[int(v)],-CAM_U[int(v)]]
 return int(np.linalg.matrix_rank(np.asarray(A),tol=RANK_TOL))

def logical_evidence_digest(z):
 h=hashlib.sha256()
 for name in ('coords','scores','valid','V_A','XY_A','output_idx'):
  a=np.ascontiguousarray(z[name]);h.update(name.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()

def fam_entry(sel,family):
 x=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
 if x is None:raise RuntimeError(('family missing',family))
 return x

def view_entry(fam,v):
 x=next((x for x in fam['views'] if int(x['view'])==int(v)),None)
 if x is None:raise RuntimeError(('view missing',fam.get('family'),v))
 return x

def eval_family(science_dir,state_path,evidence_path,v3sel,v5sel,sidecar,family):
 vf=fam_entry(v3sel,family); wf=fam_entry(v5sel,family)
 if sha256_file(state_path)!=vf['state_sha256']:raise RuntimeError(('state SHA mismatch',family))
 ez=np.load(evidence_path,allow_pickle=False)
 dg=logical_evidence_digest(ez)
 if dg!=vf['proposal_evidence_content_sha256'] or dg!=wf['proposal_evidence_content_sha256']:raise RuntimeError(('evidence digest mismatch',family))
 z=np.load(state_path,allow_pickle=False);meta=json.loads(state_path.with_suffix('.json').read_text())
 if meta.get('truth_access')!='NONE':raise RuntimeError(('state truth contract invalid',family))
 sys.path.insert(0,str(science_dir))
 ev=load_module(f'v5_eval_{family}',science_dir/'evaluation_phase.py')
 obs=load_module(f'v5_obs_{family}',science_dir/'observable_phase.py')
 PA=np.asarray(z['P_A'],np.float32);PB=np.asarray(z['P_B'],np.float32);H=np.asarray(z['H_xyz'],np.float32);off=np.asarray(z['H_offsets'],np.int64)
 t512=ev._truth(sidecar);ids,map_err=ev._map64(PA,t512);t=ev._subset_truth(t512,ids)
 sA=ev.local_scale(t512['P_A'],4)[ids];sB=ev.local_scale(t512['P_B'],4)[ids]
 reliable=map_err<=2*sA;active=np.linalg.norm(np.asarray(t['P_B'])-np.asarray(t['P_A']),axis=1)>.005
 base_err=np.linalg.norm(PB-np.asarray(t['P_B']),axis=1);contained2=np.zeros(64,bool)
 for i in range(64):
  a,b=int(off[i]),int(off[i+1]);er=np.linalg.norm(H[a:b]-np.asarray(t['P_B'])[i][None,:],axis=1)
  if len(er):contained2[i]=bool(np.min(er)<=2*sB[i])
 hardtail=set(int(i) for i in range(64) if reliable[i] and active[i] and contained2[i] and base_err[i]>2*sB[i])
 rank3=set(i for i in range(64) if projective_rank(z['V_A'][:,i])>=3 and projective_rank(z['V_B'][:,i])>=3)
 m=obs._route_module(meta['route'])
 truthB=np.asarray(t['P_B'],np.float32);truthVB=np.asarray(t['V_B'],np.uint8)
 truthXY=np.stack([m.project_points(truthB,v) for v in range(8)]).astype(np.float32)
 coords=np.asarray(ez['coords'],np.float32);valid=np.asarray(ez['valid'],np.uint8);VA=np.asarray(ez['V_A'],np.uint8)
 rows=[]
 for v in range(8):
  vv=view_entry(vf,v); ww=view_entry(wf,v)
  va={int(k):tuple(map(int,val)) for k,val in vv['assigned'].items()}; vab=set(map(int,vv['abstained']))
  wa={int(k):tuple(map(int,val)) for k,val in ww['assigned'].items()}; wab=set(map(int,ww['abstained']))
  gtop={int(k):tuple(map(int,val)) for k,val in vv['g_top1'].items()}
  for i0 in np.where(VA[v]>0)[0]:
   i=int(i0)
   if not reliable[i] or not bool(truthVB[v,i]):continue
   sites=[]
   for k in range(4):
    if valid[i,v,k]:
     q=tuple(map(int,np.rint(coords[i,v,k]).astype(int)))
     if q not in sites:sites.append(q)
   if not sites:continue
   target=np.asarray(truthXY[v,i],float);dist=np.asarray([np.linalg.norm(np.asarray(s,float)-target) for s in sites],float);best=float(np.min(dist));oracle={sites[k] for k in np.where(np.isclose(dist,best,rtol=0,atol=1e-9))[0]}
   gs=gtop[i]; gd=float(np.linalg.norm(np.asarray(gs,float)-target))
   vs=None if i in vab else va.get(i); vd=None if vs is None else float(np.linalg.norm(np.asarray(vs,float)-target))
   ws=None if i in wab else wa.get(i); wd=None if ws is None else float(np.linalg.norm(np.asarray(ws,float)-target))
   rows.append({'family':int(family),'view':v,'carrier':i,'rank3':bool(i in rank3),'hardtail':bool(i in hardtail),'hardtail_rank3':bool(i in hardtail and i in rank3),'oracle_best_sites':[list(x) for x in sorted(oracle)],'oracle_best_distance_px':best,'g_site':list(gs),'g_distance_px':gd,'g_hit':bool(gs in oracle),'g_regret_px':float(gd-best),'v3a_abstain':bool(vs is None),'v3a_site':None if vs is None else list(vs),'v3a_distance_px':vd,'v3a_hit':bool(False if vs is None else vs in oracle),'v3a_regret_px':None if vd is None else float(vd-best),'v5_abstain':bool(ws is None),'v5_site':None if ws is None else list(ws),'v5_distance_px':wd,'v5_hit':bool(False if ws is None else ws in oracle),'v5_regret_px':None if wd is None else float(wd-best)})
 return {'family':int(family),'mapping_reliable':int(reliable.sum()),'hardtail_carriers':sorted(hardtail),'rank3_hardtail_carriers':sorted(hardtail&rank3),'rows':rows,'sidecar_sha256':sha256_file(sidecar)}

def metrics(rows,prefix):
 n=len(rows); hits=sum(bool(r[f'{prefix}_hit']) for r in rows); reg=[r[f'{prefix}_regret_px'] for r in rows if r[f'{prefix}_regret_px'] is not None]
 return {'n':n,'hit_n':hits,'hit_rate':hits/n if n else None,'abstain_n':n-len(reg),'mean_regret_px_assigned':float(np.mean(reg)) if reg else None,'median_regret_px_assigned':float(np.median(reg)) if reg else None,'assigned_n':len(reg)}

def paired_direction(rows,prefix,eps=1e-12):
 imp=same=wor=0
 for r in rows:
  x=r[f'{prefix}_regret_px']
  if x is None:continue
  d=x-r['g_regret_px']
  if d < -eps:imp+=1
  elif d > eps:wor+=1
  else:same+=1
 return {'improve_n':imp,'same_n':same,'worsen_n':wor,'paired_n':imp+same+wor}

def carrier_direction(rows,prefix,eps=1e-12):
 g=defaultdict(list); m=defaultdict(list); abst=set()
 for r in rows:
  if not r['hardtail_rank3']:continue
  key=(r['family'],r['carrier']);g[key].append(r['g_regret_px'])
  x=r[f'{prefix}_regret_px']
  if x is None:abst.add(key)
  else:m[key].append(x)
 out={'improved_n':0,'same_n':0,'worsened_n':0,'abstained_carrier_n':0,'carriers':[]}
 for key in sorted(g):
  if key in abst or len(m[key])!=len(g[key]):out['abstained_carrier_n']+=1;direction='ABSTAINED';mg=float(np.mean(g[key]));mm=None
  else:
   mg=float(np.mean(g[key]));mm=float(np.mean(m[key]));d=mm-mg
   if d < -eps:out['improved_n']+=1;direction='IMPROVED'
   elif d > eps:out['worsened_n']+=1;direction='WORSENED'
   else:out['same_n']+=1;direction='SAME'
  out['carriers'].append({'family':key[0],'carrier':key[1],'g_mean_regret_px':mg,f'{prefix}_mean_regret_px':mm,'direction':direction})
 return out

def parity(result):
 o=result['overall'];h=result['hardtail_rank3'];c=result['rank3_hardtail_carrier_direction']['v3a']
 tests={'overall_n':o['g']['n']==HIST['overall']['n'],'overall_g_hit':o['g']['hit_n']==HIST['overall']['g_hit_n'],'overall_v3a_hit':o['v3a']['hit_n']==HIST['overall']['v3a_hit_n'],'overall_v3a_abstain':o['v3a']['abstain_n']==HIST['overall']['abstain_n'],'overall_g_mean':abs(o['g']['mean_regret_px_assigned']-HIST['overall']['g_mean'])<1e-12,'overall_g_median':abs(o['g']['median_regret_px_assigned']-HIST['overall']['g_median'])<1e-12,'overall_v3a_mean':abs(o['v3a']['mean_regret_px_assigned']-HIST['overall']['v3a_mean'])<1e-12,'overall_v3a_median':abs(o['v3a']['median_regret_px_assigned']-HIST['overall']['v3a_median'])<1e-12,'overall_direction':(o['v3a_vs_g']['improve_n'],o['v3a_vs_g']['same_n'],o['v3a_vs_g']['worsen_n'])==(HIST['overall']['improve'],HIST['overall']['same'],HIST['overall']['worsen']),'hard_n':h['g']['n']==HIST['hardtail_rank3']['n'],'hard_g_hit':h['g']['hit_n']==HIST['hardtail_rank3']['g_hit_n'],'hard_v3a_hit':h['v3a']['hit_n']==HIST['hardtail_rank3']['v3a_hit_n'],'hard_v3a_abstain':h['v3a']['abstain_n']==HIST['hardtail_rank3']['abstain_n'],'hard_g_mean':abs(h['g']['mean_regret_px_assigned']-HIST['hardtail_rank3']['g_mean'])<1e-12,'hard_g_median':abs(h['g']['median_regret_px_assigned']-HIST['hardtail_rank3']['g_median'])<1e-12,'hard_v3a_mean':abs(h['v3a']['mean_regret_px_assigned']-HIST['hardtail_rank3']['v3a_mean'])<1e-12,'hard_v3a_median':abs(h['v3a']['median_regret_px_assigned']-HIST['hardtail_rank3']['v3a_median'])<1e-12,'hard_direction':(h['v3a_vs_g']['improve_n'],h['v3a_vs_g']['same_n'],h['v3a_vs_g']['worsen_n'])==(HIST['hardtail_rank3']['improve'],HIST['hardtail_rank3']['same'],HIST['hardtail_rank3']['worsen']),'carrier_direction':(c['improved_n'],c['same_n'],c['worsened_n'])==(HIST['carrier']['improved'],HIST['carrier']['same'],HIST['carrier']['worsened'])}
 return {'pass':all(tests.values()),'tests':tests}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--science-dir',type=Path,required=True);ap.add_argument('--panel',type=Path,required=True);ap.add_argument('--evidence-dir',type=Path,required=True);ap.add_argument('--sidecar-dir',type=Path,required=True);ap.add_argument('--v3a-selection',type=Path,required=True);ap.add_argument('--v5-selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if sha256_file(a.v3a_selection)!=V3A_SELECTION_SHA:raise RuntimeError('V3A selection SHA mismatch')
 if sha256_file(a.v5_selection)!=V5_SELECTION_SHA:raise RuntimeError('V5 selection SHA mismatch')
 v3=json.loads(a.v3a_selection.read_text());v5=json.loads(a.v5_selection.read_text())
 if v3.get('truth_access')!='NONE' or v5.get('truth_access')!='NONE':raise RuntimeError('pretruth authority invalid')
 if v5.get('source_sha256')!=V5_SOURCE_SHA:raise RuntimeError('V5 source authority mismatch')
 families=[];rows=[]
 for fam in FAMS:
  pf='09908' if fam==9908 else str(fam);state=a.panel/pf/f'{fam}_e01_OBSERVABLE_STATE.npz';evidence=a.evidence_dir/f'{fam}_V3A_PROPOSAL_EVIDENCE.npz';side=a.sidecar_dir/str(fam)/'observation_sidecar.npz';fr=eval_family(a.science_dir,state,evidence,v3,v5,side,fam);families.append({k:v for k,v in fr.items() if k!='rows'});rows.extend(fr['rows'])
 hard=[r for r in rows if r['hardtail_rank3']]
 result={'schema':'RealSaS.N1D.Rank2R.V5.TruthEvaluation.v1','status':'COMPLETE_UNTUNED_TRUTH_EVALUATION','authorities':{'v3a_selection_sha256':V3A_SELECTION_SHA,'v5_selection_sha256':V5_SELECTION_SHA,'v5_source_sha256':V5_SOURCE_SHA},'population':{'applicable_rows':len(rows),'rank3_hardtail_view_rows':len(hard),'hardtail_n':sum(len(x['hardtail_carriers']) for x in families),'rank3_hardtail_carriers':sum(len(x['rank3_hardtail_carriers']) for x in families)},'overall':{'g':metrics(rows,'g'),'v3a':metrics(rows,'v3a'),'v5':metrics(rows,'v5'),'v3a_vs_g':paired_direction(rows,'v3a'),'v5_vs_g':paired_direction(rows,'v5')},'hardtail_rank3':{'g':metrics(hard,'g'),'v3a':metrics(hard,'v3a'),'v5':metrics(hard,'v5'),'v3a_vs_g':paired_direction(hard,'v3a'),'v5_vs_g':paired_direction(hard,'v5')},'rank3_hardtail_carrier_direction':{'v3a':carrier_direction(rows,'v3a'),'v5':carrier_direction(rows,'v5')},'families':families,'rows':rows}
 result['v3a_evaluator_parity']=parity(result)
 if not result['v3a_evaluator_parity']['pass']:raise RuntimeError(('V3A evaluator replay parity FAIL',result['v3a_evaluator_parity']))
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False));print(json.dumps({k:result[k] for k in ('status','population','overall','hardtail_rank3','v3a_evaluator_parity')},indent=2,sort_keys=True))
if __name__=='__main__':main()
