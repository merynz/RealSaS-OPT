from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V3C.TruthEvaluation.Recovery.v1'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
WORLD_SHA={
9908:'326c0a6425c633c1a5acaac9ab6838496d2d97f7f582fe126ae1a8e16269c6c5',
11032:'c291254f79ea49fb1bdb72dbbe3e5d96b85ba01f9ce4f4e830d732140e6be57a',
12772:'51d5e0c41b0b7c0b1354f7bba23dae76151f54fcea814cc7c7bd206f0846945f',
13203:'646c06bc296ca3df8e5218055e32ded18d575181faa59a636314f76a86433033',
14404:'be1a70c7c7ce76203170e7bce2dd68f06229fb693104d45e4e1255c1f3e420c4',
14702:'0c70e4874e08a912b205fc319d76aef89eca8d7fecc65cf1eeaeb3660c80956e',
14758:'1ebd219d8222bf5c7ff5d165620899ba11087d20d5b95193c7bf2a5f1996890a',
15290:'36915e7837c0738b57a152e07851f55e528fc8e2e29632f70c245b9310756625'}
EXPECTED_HARDTAIL={9908:[],11032:[0,42,53,58],12772:[63],13203:[],14404:[2,6,32,37,40,43,56],14702:[23,24,25],14758:[1,15,31,35,37,44,45,59],15290:[]}
RANK_TOL=1e-8; YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);assert s.loader;sys.modules[name]=m;s.loader.exec_module(m);return m
def material(b,n):
 if not(np.isfinite(b) and np.isfinite(n)):return float('nan'),False,False
 r=float((b-n)/(abs(b)+1e-8));return r,bool(r>=.20 and b-n>=.05),bool(r<=-.20 and n-b>=.05)
def basis():
 R=[];U=[]
 for d in YAW_DEG:
  t=math.radians(d);rad=np.array([math.sin(t),-math.cos(t),0.]);fw=-rad;up=np.array([0.,0.,1.]);right=np.cross(fw,up);right/=np.linalg.norm(right);R.append(right);U.append(up)
 return np.stack(R),np.stack(U)
CR,CU=basis()
def prank(v):
 ids=np.where(np.asarray(v)>0)[0]
 if not len(ids):return 0
 A=[]
 for q in ids:A += [CR[int(q)],-CU[int(q)]]
 return int(np.linalg.matrix_rank(np.asarray(A),tol=RANK_TOL))
def rank3(VA,VB):return np.array([prank(VA[:,i])>=3 and prank(VB[:,i])>=3 for i in range(64)],bool)
def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,(np.floating,float)):
  v=float(x);return v if np.isfinite(v) else None
 if isinstance(x,(np.integer,int)):return int(x)
 if isinstance(x,(np.bool_,bool)):return bool(x)
 return x

def eval_family(root,science,state,world,sidecar,family,episode):
 # Pre-truth guards: frozen world bytes + state metadata + baseline identity.
 if sha(world)!=WORLD_SHA[int(family)]:raise RuntimeError(('world SHA mismatch',family,sha(world),WORLD_SHA[int(family)]))
 z=np.load(state,allow_pickle=False);meta=json.load(open(Path(state).with_suffix('.json')))
 if meta.get('truth_access')!='NONE' or sha(state)!=meta['observable_state_sha256']:raise RuntimeError('state authority invalid')
 w=np.load(world,allow_pickle=False);PA=np.asarray(z['P_A'],np.float32);PB0=np.asarray(z['P_B'],np.float32);PB=np.asarray(w['P_B_V3C'],np.float32)
 if not np.array_equal(np.asarray(w['P_A'],np.float32),PA):raise RuntimeError('world PA mismatch')
 if not np.array_equal(np.asarray(w['P_B_baseline'],np.float32),PB0):raise RuntimeError('world baseline mismatch')
 sys.path.insert(0,str(science));ev=mod(f'v3c_ev_{family}',science/'evaluation_phase.py');mm=mod(f'v3c_mm_{family}',science/'mechanics_metrics.py');obs=mod(f'v3c_obs_{family}',science/'observable_phase.py')
 base={k:np.asarray(z[k]) for k in ('P_A','P_B','N_A','N_B','V_A','V_B')};gb=ev.gfdr.compute_gfdr_v2(**base,Z=None,U=None)
 NB,VB,XY=obs.decorate_pb_observable(root,int(family),episode,meta['route'],PA,PB); nw={'P_A':PA,'P_B':PB,'N_A':np.asarray(z['N_A']),'N_B':NB,'V_A':np.asarray(z['V_A']),'V_B':VB};gn=ev.gfdr.compute_gfdr_v2(**nw,Z=None,U=None)
 # Truth opens only after every guard above passed.
 T512=ev._truth(sidecar);ids,maperr=ev._map64(PA,T512);T=ev._subset_truth(T512,ids);sA=ev.local_scale(T512['P_A'],4)[ids];sB=ev.local_scale(T512['P_B'],4)[ids]
 reliable=maperr<=2*sA;active=np.linalg.norm(np.asarray(T['P_B'])-np.asarray(T['P_A']),axis=1)>.005;baseerr=np.linalg.norm(PB0-np.asarray(T['P_B']),axis=1);newerr=np.linalg.norm(PB-np.asarray(T['P_B']),axis=1);gt=ev.gfdr.compute_gfdr_v2(**T,Z=None,U=None)
 H=np.asarray(z['H_xyz'],np.float32);off=np.asarray(z['H_offsets'],np.int64);contained=np.zeros(64,bool)
 for i in range(64):
  a,b=int(off[i]),int(off[i+1]);er=np.linalg.norm(H[a:b]-np.asarray(T['P_B'])[i][None,:],axis=1);contained[i]=bool(len(er) and np.min(er)<=2*sB[i])
 witness=[i for i in range(64) if reliable[i] and active[i] and contained[i] and baseerr[i]>2*sB[i]]
 if witness!=EXPECTED_HARDTAIL[int(family)]:raise RuntimeError(('hardtail parity mismatch',family,witness,EXPECTED_HARDTAIL[int(family)]))
 r3=rank3(z['V_A'],z['V_B']);hr=[]
 for i in witness:
  bt=mm.truth_block_errors(gb,gt,PA,i);nt=mm.truth_block_errors(gn,gt,PA,i);bc=float(bt['composite_median_capped10']);nc=float(nt['composite_median_capped10']);rel,imp,deg=material(bc,nc);eff=mm.effect_blocks(gb,gn,PA,i)
  hr.append({'carrier':int(i),'is_rank3':bool(r3[i]),'baseline_error_over_scaleB':float(baseerr[i]/max(sB[i],1e-12)),'v3c_error_over_scaleB':float(newerr[i]/max(sB[i],1e-12)),'geometry_improved':bool(newerr[i]<baseerr[i]),'baseline_composite':bc,'v3c_composite':nc,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg,'mechanics_equivalent':bool(eff['equivalent'])})
 ar=[]
 for i in np.where(reliable&active)[0]:
  bt=mm.truth_block_errors(gb,gt,PA,int(i));nt=mm.truth_block_errors(gn,gt,PA,int(i));bc=float(bt['composite_median_capped10']);nc=float(nt['composite_median_capped10']);rel,imp,deg=material(bc,nc)
  ar.append({'carrier':int(i),'is_rank3':bool(r3[i]),'geometry_improved':bool(newerr[i]<baseerr[i]),'baseline_composite':bc,'v3c_composite':nc,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg})
 return {'schema':SCHEMA,'family':int(family),'world_sha256':sha(world),'sidecar_sha256':sha(sidecar),'hardtail':hr,'reliable_active':ar,'population':{'hardtail':len(witness),'reliable_active':len(ar),'rank3_hardtail':sum(bool(r3[i]) for i in witness)},'truth_access':'EVALUATOR_ONLY_AFTER_WORLD_FREEZE'}

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--science',type=Path,required=True);a.add_argument('--state',type=Path,required=True);a.add_argument('--world',type=Path,required=True);a.add_argument('--sidecar',type=Path,required=True);a.add_argument('--family',type=int,required=True);a.add_argument('--episode',default='e01');a.add_argument('--out',type=Path,required=True);q=a.parse_args();r=eval_family(q.root,q.science,q.state,q.world,q.sidecar,q.family,q.episode);q.out.parent.mkdir(parents=True,exist_ok=True);q.out.write_text(json.dumps(safe(r),indent=2,sort_keys=True)+'\n');print(json.dumps(r['population'],sort_keys=True))
if __name__=='__main__':main()
