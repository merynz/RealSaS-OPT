from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V3B.TruthEvaluation.v1'
WORLD_AUTH_SHA256='d57e61b59c1126a9597759350ef2756218446a43e833018106e5afcdb546d3e0'
V3B_SOURCE_SHA256='5a6ca40e834822c5da8c1fef64ea8d71e104902695905eb60ac34642f4b85d24'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
RANK_TOL=1e-8
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def json_safe(x):
    if isinstance(x,dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,float)):
        v=float(x); return v if np.isfinite(v) else None
    if isinstance(x,(np.integer,int)): return int(x)
    if isinstance(x,(np.bool_,bool)): return bool(x)
    return x

def material_flags(base_comp,new_comp):
    if not (np.isfinite(base_comp) and np.isfinite(new_comp)): return float('nan'),False,False
    rel=float((base_comp-new_comp)/(abs(base_comp)+1e-8))
    return rel,bool(rel>=.20 and (base_comp-new_comp)>=.05),bool(rel<=-.20 and (new_comp-base_comp)>=.05)

def camera_basis():
    R=[]; U=[]
    for deg in YAW_DEG:
        t=math.radians(deg); radial=np.array([math.sin(t),-math.cos(t),0.],float); fw=-radial; up=np.array([0.,0.,1.]); right=np.cross(fw,up); right/=np.linalg.norm(right); R.append(right); U.append(up)
    return np.stack(R),np.stack(U)
CAM_R,CAM_U=camera_basis()

def projective_rank(vcol):
    views=np.where(np.asarray(vcol)>0)[0]
    if not len(views): return 0
    A=[]
    for v in views: A += [CAM_R[int(v)],-CAM_U[int(v)]]
    return int(np.linalg.matrix_rank(np.asarray(A),tol=RANK_TOL))

def rank3_mask(VA,VB):
    return np.array([(projective_rank(VA[:,i])>=3 and projective_rank(VB[:,i])>=3) for i in range(64)],bool)

def evaluate_family(root,science_dir,state_path,world_path,world_authority_path,sidecar,family,episode):
    # All source/world authority checks happen before truth opens.
    if sha256_file(world_authority_path)!=WORLD_AUTH_SHA256: raise RuntimeError('V3B world authority SHA mismatch')
    auth=json.loads(world_authority_path.read_text())
    if auth.get('truth_access')!='NONE' or auth.get('status')!='FROZEN_PRE_TRUTH': raise RuntimeError('V3B world authority not pre-truth frozen')
    if auth.get('source_sha256')!=V3B_SOURCE_SHA256 or auth.get('observable_panel_manifest_sha256')!=OBS_PANEL_SHA256: raise RuntimeError('V3B source/panel authority mismatch')
    famrec=next((x for x in auth['families'] if int(x['family'])==int(family)),None)
    if famrec is None: raise RuntimeError('family missing world authority')
    if sha256_file(state_path)!=famrec['state_sha256']: raise RuntimeError('state SHA mismatch')
    if sha256_file(world_path)!=famrec['world_sha256']: raise RuntimeError('world SHA mismatch')
    meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE' or sha256_file(state_path)!=meta['observable_state_sha256']: raise RuntimeError('observable state authority invalid')
    z=np.load(state_path,allow_pickle=False); wz=np.load(world_path,allow_pickle=False)
    PA=np.asarray(z['P_A'],np.float32); PB0=np.asarray(z['P_B'],np.float32); PBv=np.asarray(wz['P_B_V3B'],np.float32)
    if not np.array_equal(np.asarray(wz['P_A'],np.float32),PA): raise RuntimeError('world P_A mismatch')
    if not np.array_equal(np.asarray(wz['P_B_baseline'],np.float32),PB0): raise RuntimeError('world baseline mismatch')
    if PBv.shape!=(64,3): raise RuntimeError('world P_B shape invalid')

    sys.path.insert(0,str(science_dir))
    ev=load_module('v3b_r6_eval_exact',science_dir/'evaluation_phase.py')
    mm=load_module('v3b_r6_metrics_exact',science_dir/'mechanics_metrics.py')
    obs=load_module('v3b_r6_observable_exact',science_dir/'observable_phase.py')
    base_world={k:np.asarray(z[k]) for k in ('P_A','P_B','N_A','N_B','V_A','V_B')}
    base_g=ev.gfdr.compute_gfdr_v2(**base_world,Z=None,U=None)
    NBv,VBv,XYBv=obs.decorate_pb_observable(root,int(family),episode,meta['route'],PA,PBv)
    new_world={'P_A':PA,'P_B':PBv,'N_A':np.asarray(z['N_A']),'N_B':NBv,'V_A':np.asarray(z['V_A']),'V_B':VBv}
    new_g=ev.gfdr.compute_gfdr_v2(**new_world,Z=None,U=None)

    # Truth opens only here, after all V3-B world bytes are verified frozen.
    t512=ev._truth(sidecar); ids,map_err=ev._map64(PA,t512); t=ev._subset_truth(t512,ids)
    sA=ev.local_scale(t512['P_A'],4)[ids]; sB=ev.local_scale(t512['P_B'],4)[ids]
    reliable=map_err<=2*sA; active=np.linalg.norm(np.asarray(t['P_B'])-np.asarray(t['P_A']),axis=1)>.005
    base_err=np.linalg.norm(PB0-np.asarray(t['P_B']),axis=1); new_err=np.linalg.norm(PBv-np.asarray(t['P_B']),axis=1)
    true_g=ev.gfdr.compute_gfdr_v2(**t,Z=None,U=None)
    H=np.asarray(z['H_xyz'],np.float32); off=np.asarray(z['H_offsets'],np.int64)
    nearest_err=np.full(64,np.inf); contained2=np.zeros(64,bool)
    for i in range(64):
        a,b=int(off[i]),int(off[i+1]); HH=H[a:b]; er=np.linalg.norm(HH-np.asarray(t['P_B'])[i][None,:],axis=1)
        if len(er): nearest_err[i]=float(np.min(er)); contained2[i]=bool(np.min(er)<=2*sB[i])
    witness=[i for i in range(64) if reliable[i] and active[i] and contained2[i] and base_err[i]>2*sB[i]]
    r3=rank3_mask(z['V_A'],z['V_B'])
    rows=[]
    for i in witness:
        bt=mm.truth_block_errors(base_g,true_g,PA,i); nt=mm.truth_block_errors(new_g,true_g,PA,i)
        bc=float(bt['composite_median_capped10']); nc=float(nt['composite_median_capped10']); rel,imp,deg=material_flags(bc,nc)
        effect=mm.effect_blocks(base_g,new_g,PA,i)
        rows.append({'carrier':int(i),'is_rank3':bool(r3[i]),'truth_surface_id':int(ids[i]),'mapping_error_over_scaleA':float(map_err[i]/max(sA[i],1e-12)),'baseline_error_over_scaleB':float(base_err[i]/max(sB[i],1e-12)),'v3b_error_over_scaleB':float(new_err[i]/max(sB[i],1e-12)),'geometry_improved_vs_baseline':bool(new_err[i]<base_err[i]),'geometry_delta_over_scaleB':float((base_err[i]-new_err[i])/max(sB[i],1e-12)),'truth_relative':{'baseline':bt,'v3b':nt,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg},'baseline_vs_v3b_effect':effect})
    all_rows=[]
    for i in np.where(reliable&active)[0]:
        bt=mm.truth_block_errors(base_g,true_g,PA,int(i)); nt=mm.truth_block_errors(new_g,true_g,PA,int(i)); bc=float(bt['composite_median_capped10']); nc=float(nt['composite_median_capped10']); rel,imp,deg=material_flags(bc,nc)
        all_rows.append({'carrier':int(i),'is_rank3':bool(r3[i]),'baseline_composite':bc,'v3b_composite':nc,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg,'geometry_improved_vs_baseline':bool(new_err[i]<base_err[i])})
    changed=np.linalg.norm(PBv-PB0,axis=1)
    return {'schema':SCHEMA,'family':int(family),'episode':episode,'authorities':{'state_sha256':sha256_file(state_path),'world_sha256':sha256_file(world_path),'world_authority_sha256':WORLD_AUTH_SHA256,'sidecar_sha256':sha256_file(sidecar),'observable_panel_manifest_sha256':OBS_PANEL_SHA256,'v3b_source_sha256':V3B_SOURCE_SHA256},'population':{'mapping_reliable':int(reliable.sum()),'active':int(active.sum()),'reliable_active':int((reliable&active).sum()),'feasible_contained2':int(contained2.sum()),'hardtail_witnesses':len(witness),'hardtail_carriers':[int(x) for x in witness],'rank3_hardtail_n':int(sum(bool(r3[i]) for i in witness)),'rankdef_hardtail_n':int(sum(not bool(r3[i]) for i in witness))},'world':{'baseline_exact_n':int(np.sum(changed==0)),'changed_n_gt_1e_4':int(np.sum(changed>1e-4)),'mean_distance_from_baseline':float(np.mean(changed)),'max_distance_from_baseline':float(np.max(changed)),'observable_decorator':{'route':meta['route'],'N_B_shape':list(NBv.shape),'V_B_shape':list(VBv.shape),'XY_B_shape':list(XYBv.shape)}},'hardtail_rows':rows,'reliable_active_rows':all_rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--science-dir',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--world',type=Path,required=True); ap.add_argument('--world-authority',type=Path,required=True); ap.add_argument('--sidecar',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--episode',default='e01'); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError('refusing overwrite')
    r=evaluate_family(a.root,a.science_dir,a.state,a.world,a.world_authority,a.sidecar,a.family,a.episode); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(json_safe(r),indent=2,sort_keys=True,allow_nan=False)); print(json.dumps({'family':r['family'],'population':r['population'],'world':{k:r['world'][k] for k in ('baseline_exact_n','changed_n_gt_1e_4','mean_distance_from_baseline','max_distance_from_baseline')}},indent=2,sort_keys=True))
if __name__=='__main__': main()
