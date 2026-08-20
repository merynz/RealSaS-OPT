from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V4.FrontierMechanicsEvaluation.v1'
SET_AUTH_SHA256='70a2ed96a81672feee95bb5f4d64e11ed80d6362cdb8bf92c1d8dab17f62aa15'
V4_SOURCE_SHA256='d1fbc210d7f5a85fcca774ccb483779207a7932e124fc7fe0393568d74d9a1f5'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
RANK_TOL=1e-8; YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def js(x):
    if isinstance(x,dict): return {k:js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [js(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,float)):
        v=float(x); return v if np.isfinite(v) else None
    if isinstance(x,(np.integer,int)): return int(x)
    if isinstance(x,(np.bool_,bool)): return bool(x)
    return x
def material_flags(b,n):
    if not(np.isfinite(b) and np.isfinite(n)): return float('nan'),False,False
    rel=float((b-n)/(abs(b)+1e-8)); return rel,bool(rel>=.20 and (b-n)>=.05),bool(rel<=-.20 and (n-b)>=.05)
def camera_basis():
    R=[];U=[]
    for deg in YAW_DEG:
        t=math.radians(deg); radial=np.array([math.sin(t),-math.cos(t),0.]); fw=-radial; up=np.array([0.,0.,1.]); right=np.cross(fw,up); right/=np.linalg.norm(right); R.append(right);U.append(up)
    return np.stack(R),np.stack(U)
CAM_R,CAM_U=camera_basis()
def prank(vcol):
    views=np.where(np.asarray(vcol)>0)[0]; A=[]
    for v in views:A += [CAM_R[int(v)],-CAM_U[int(v)]]
    return int(np.linalg.matrix_rank(np.asarray(A),tol=RANK_TOL)) if A else 0
def rank3mask(VA,VB): return np.array([(prank(VA[:,i])>=3 and prank(VB[:,i])>=3) for i in range(64)],bool)

def evaluate(root,science_dir,state_path,rep_path,auth_path,sidecar,family,episode):
    # representation authority before evaluator truth
    if sha256_file(auth_path)!=SET_AUTH_SHA256: raise RuntimeError('set authority SHA mismatch')
    auth=json.loads(auth_path.read_text()); frec=next((x for x in auth['families'] if int(x['family'])==int(family)),None)
    if auth.get('source_sha256')!=V4_SOURCE_SHA256 or auth.get('observable_panel_manifest_sha256')!=OBS_PANEL_SHA256 or frec is None: raise RuntimeError('set authority invalid')
    if sha256_file(rep_path)!=frec['representation_sha256']: raise RuntimeError('representation SHA mismatch')
    if sha256_file(state_path)!=frec['state_sha256']:
        raise RuntimeError('state SHA mismatch')
    z=np.load(state_path,allow_pickle=False); meta=json.loads(state_path.with_suffix('.json').read_text()); rep=np.load(rep_path,allow_pickle=False)
    if meta.get('truth_access')!='NONE': raise RuntimeError('state truth contract invalid')
    PA=np.asarray(z['P_A'],np.float32); PB=np.asarray(z['P_B'],np.float32)
    off=np.asarray(rep['offsets'],np.int64); XYZ=np.asarray(rep['xyz'],np.float32); pl=np.asarray(rep['pareto_layer'],np.int64); baseflag=np.asarray(rep['baseline_flag'],bool)
    gr=np.asarray(rep['G_rank'],float); rr=np.asarray(rep['R_rank'],float); srcidx=np.asarray(rep['source_global_index'],np.int64)
    sys.path.insert(0,str(science_dir)); ev=load_module('v4_eval',science_dir/'evaluation_phase.py'); mm=load_module('v4_mm',science_dir/'mechanics_metrics.py'); obs=load_module('v4_obs',science_dir/'observable_phase.py')
    base_world={k:np.asarray(z[k]) for k in ('P_A','P_B','N_A','N_B','V_A','V_B')}; base_g=ev.gfdr.compute_gfdr_v2(**base_world,Z=None,U=None)
    # truth opens here only after representation freeze checks
    t512=ev._truth(sidecar); ids,map_err=ev._map64(PA,t512); t=ev._subset_truth(t512,ids); true_g=ev.gfdr.compute_gfdr_v2(**t,Z=None,U=None)
    sA=ev.local_scale(t512['P_A'],4)[ids]; sB=ev.local_scale(t512['P_B'],4)[ids]; reliable=map_err<=2*sA; active=np.linalg.norm(np.asarray(t['P_B'])-np.asarray(t['P_A']),axis=1)>.005
    base_err=np.linalg.norm(PB-np.asarray(t['P_B']),axis=1); H=np.asarray(z['H_xyz'],np.float32); hoff=np.asarray(z['H_offsets'],np.int64); contained=np.zeros(64,bool)
    for i in range(64):
        a,b=int(hoff[i]),int(hoff[i+1]); er=np.linalg.norm(H[a:b]-np.asarray(t['P_B'])[i][None,:],axis=1); contained[i]=bool(len(er) and np.min(er)<=2*sB[i])
    witness=[i for i in range(64) if reliable[i] and active[i] and contained[i] and base_err[i]>2*sB[i]]; r3=rank3mask(z['V_A'],z['V_B'])
    outrows=[]
    for i in witness:
        a,b=int(off[i]),int(off[i+1]); inds=np.arange(a,b); front=inds[pl[a:b]==0]; bi=inds[baseflag[a:b]][0]
        bt=mm.truth_block_errors(base_g,true_g,PA,i); bc=float(bt['composite_median_capped10']); cand=[]
        for gi in front:
            p=XYZ[gi]; geom=float(np.linalg.norm(p-np.asarray(t['P_B'])[i])); isbase=bool(baseflag[gi])
            if isbase:
                nt=bt; nc=bc; effect={'equivalent':1,'passes':{}}; rel,imp,deg=0.0,False,False
            else:
                Pcf=PB.copy(); Pcf[i]=p; NB,VB,_=obs.decorate_pb_observable(root,int(family),episode,meta['route'],PA,Pcf)
                ng=ev.gfdr.compute_gfdr_v2(P_A=PA,P_B=Pcf,N_A=np.asarray(z['N_A']),N_B=NB,V_A=np.asarray(z['V_A']),V_B=VB,Z=None,U=None)
                nt=mm.truth_block_errors(ng,true_g,PA,i); nc=float(nt['composite_median_capped10']); effect=mm.effect_blocks(base_g,ng,PA,i); rel,imp,deg=material_flags(bc,nc)
            cand.append({'rep_global_index':int(gi),'source_global_index':int(srcidx[gi]),'is_baseline':isbase,'G_rank':float(gr[gi]),'R_rank':float(rr[gi]),'geometry_error_over_scaleB':geom/max(float(sB[i]),1e-12),'composite':nc,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg,'mechanically_equivalent_to_baseline':bool(effect.get('equivalent',0))})
        best=min(cand,key=lambda x:(x['composite'],x['G_rank'],x['R_rank'],x['rep_global_index']))
        outrows.append({'carrier':int(i),'is_rank3':bool(r3[i]),'frontier_n':len(cand),'baseline_on_frontier':bool(baseflag[bi] and pl[bi]==0),'baseline_composite':bc,'baseline_error_over_scaleB':float(base_err[i]/max(sB[i],1e-12)),'frontier_any_material_improve':bool(any(x['material_improve'] for x in cand)),'frontier_any_mechanical_equivalent':bool(any(x['mechanically_equivalent_to_baseline'] for x in cand)),'frontier_any_geometry_improve':bool(any(x['geometry_error_over_scaleB'] < base_err[i]/max(sB[i],1e-12) for x in cand)),'frontier_best':best,'candidates':cand})
    return {'schema':SCHEMA,'family':int(family),'episode':episode,'authorities':{'set_authority_sha256':SET_AUTH_SHA256,'representation_sha256':sha256_file(rep_path),'state_sha256':sha256_file(state_path),'sidecar_sha256':sha256_file(sidecar)},'population':{'hardtail_witnesses':len(witness),'hardtail_carriers':[int(x) for x in witness]},'hardtail_rows':outrows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True);ap.add_argument('--science-dir',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--rep',type=Path,required=True);ap.add_argument('--authority',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',default='e01');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise RuntimeError('refusing overwrite')
    r=evaluate(a.root,a.science_dir,a.state,a.rep,a.authority,a.sidecar,a.family,a.episode);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(js(r),indent=2,sort_keys=True,allow_nan=False));print(json.dumps({'family':r['family'],'population':r['population'],'frontier_sizes':[x['frontier_n'] for x in r['hardtail_rows']]},indent=2))
if __name__=='__main__':main()
