from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2GlobalRelationalWorld.V2.TruthEvaluation.v1'
V2_SOURCE_SHA256='1cf8d8fff72cd8596207a605e7a1b687096e15178dcc81f0f28b395c0b1fe5dd'
SELECTION_LOGICAL_SHA256='0a851b4ccd6fd29ed7de497e59290a453d2556c585c96fb986629e631320234a'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
RANK_TOL=1e-8
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def logical_selection_sha(sel):
    logical={
      'source_sha256':sel['source_sha256'],
      'observable_panel_manifest_sha256':sel['observable_panel_manifest_sha256'],
      'families':[{'family':f['family'],'state_sha256':f['state_sha256'],'variable_ids':f['variable_ids'],'selected_original_global_index':f['selected_original_global_index']} for f in sel['families']]
    }
    b=json.dumps(logical,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(b).hexdigest()
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
def variable_ids_from_observation(VA,VB):
    out=[]
    for i in range(64):
        if not (projective_rank(VA[:,i])>=3 and projective_rank(VB[:,i])>=3): out.append(i)
    return out

def evaluate_family(root,science_dir,state_path,selection_path,sidecar,family,episode):
    sys.path.insert(0,str(science_dir))
    ev=load_module('r6_eval_exact',science_dir/'evaluation_phase.py')
    mm=load_module('r6_metrics_exact',science_dir/'mechanics_metrics.py')
    obs=load_module('r6_observable_exact',science_dir/'observable_phase.py')
    sel=json.loads(selection_path.read_text())
    if sel.get('truth_access')!='NONE' or sel.get('source_sha256')!=V2_SOURCE_SHA256: raise RuntimeError('V2 selection authority invalid')
    if logical_selection_sha(sel)!=SELECTION_LOGICAL_SHA256: raise RuntimeError('V2 selection logical digest mismatch')
    famrec=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
    if famrec is None: raise RuntimeError('family missing')
    if sha256_file(state_path)!=famrec['state_sha256']: raise RuntimeError('state SHA mismatch')
    meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE' or sha256_file(state_path)!=meta['observable_state_sha256']: raise RuntimeError('observable state authority invalid')
    z=np.load(state_path,allow_pickle=False)
    PA=np.asarray(z['P_A'],np.float32); PB0=np.asarray(z['P_B'],np.float32); H=np.asarray(z['H_xyz'],np.float32); off=np.asarray(z['H_offsets'],np.int64)
    variable_ids=[int(x) for x in famrec['variable_ids']]
    if variable_ids_from_observation(z['V_A'],z['V_B'])!=variable_ids: raise RuntimeError('observable rank mask mismatch')
    selected={int(k):int(v) for k,v in famrec['selected_original_global_index'].items()}
    if sorted(selected)!=variable_ids: raise RuntimeError('selection ids != variable ids')
    PBv=PB0.copy()
    for i in variable_ids:
        gi=selected[i]
        if not (int(off[i])<=gi<int(off[i+1])): raise RuntimeError(('candidate outside H_i',i,gi))
        PBv[i]=H[gi]
    fixed=[i for i in range(64) if i not in set(variable_ids)]
    if fixed and not np.array_equal(PBv[fixed],PB0[fixed]): raise RuntimeError('fixed rank-3 carrier changed')
    base_world={k:np.asarray(z[k]) for k in ('P_A','P_B','N_A','N_B','V_A','V_B')}
    base_g=ev.gfdr.compute_gfdr_v2(**base_world,Z=None,U=None)
    if variable_ids:
        NBv,VBv,XYBv=obs.decorate_pb_observable(root,int(family),episode,meta['route'],PA,PBv)
        v2_world={'P_A':PA,'P_B':PBv,'N_A':np.asarray(z['N_A']),'N_B':NBv,'V_A':np.asarray(z['V_A']),'V_B':VBv}
        v2_g=ev.gfdr.compute_gfdr_v2(**v2_world,Z=None,U=None)
    else:
        NBv=np.asarray(z['N_B']); VBv=np.asarray(z['V_B']); XYBv=np.asarray(z['XY_B']); v2_g=base_g
    # Truth opens only here.
    t512=ev._truth(sidecar); ids,map_err=ev._map64(PA,t512); t=ev._subset_truth(t512,ids)
    sA=ev.local_scale(t512['P_A'],4)[ids]; sB=ev.local_scale(t512['P_B'],4)[ids]
    reliable=map_err<=2*sA; active=np.linalg.norm(np.asarray(t['P_B'])-np.asarray(t['P_A']),axis=1)>.005
    base_err=np.linalg.norm(PB0-np.asarray(t['P_B']),axis=1); v2_err=np.linalg.norm(PBv-np.asarray(t['P_B']),axis=1)
    true_g=ev.gfdr.compute_gfdr_v2(**t,Z=None,U=None)
    nearest_err=np.full(64,np.inf); contained2=np.zeros(64,bool)
    for i in range(64):
        a,b=int(off[i]),int(off[i+1]); HH=H[a:b]; er=np.linalg.norm(HH-np.asarray(t['P_B'])[i][None,:],axis=1)
        if len(er): nearest_err[i]=float(np.min(er)); contained2[i]=bool(np.min(er)<=2*sB[i])
    witness=[i for i in range(64) if reliable[i] and active[i] and contained2[i] and base_err[i]>2*sB[i]]
    rows=[]
    for i in witness:
        bt=mm.truth_block_errors(base_g,true_g,PA,i); vt=mm.truth_block_errors(v2_g,true_g,PA,i)
        bc=float(bt['composite_median_capped10']); vc=float(vt['composite_median_capped10']); rel,imp,deg=material_flags(bc,vc)
        rows.append({'carrier':int(i),'is_variable':bool(i in set(variable_ids)),'truth_surface_id':int(ids[i]),'baseline_error_over_scaleB':float(base_err[i]/max(sB[i],1e-12)),'v2_error_over_scaleB':float(v2_err[i]/max(sB[i],1e-12)),'geometry_improved_vs_baseline':bool(v2_err[i]<base_err[i]),'truth_relative':{'baseline':bt,'v2':vt,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg},'baseline_vs_v2_effect':mm.effect_blocks(base_g,v2_g,PA,i)})
    all_rows=[]
    for i in np.where(reliable&active)[0]:
        bt=mm.truth_block_errors(base_g,true_g,PA,int(i)); vt=mm.truth_block_errors(v2_g,true_g,PA,int(i)); bc=float(bt['composite_median_capped10']); vc=float(vt['composite_median_capped10']); rel,imp,deg=material_flags(bc,vc)
        all_rows.append({'carrier':int(i),'is_variable':bool(int(i) in set(variable_ids)),'baseline_composite':bc,'v2_composite':vc,'relative_improvement':rel,'material_improve':imp,'material_degrade':deg})
    return {'schema':SCHEMA,'family':int(family),'episode':episode,'authorities':{'state_sha256':sha256_file(state_path),'selection_logical_sha256':SELECTION_LOGICAL_SHA256,'sidecar_sha256':sha256_file(sidecar),'observable_panel_manifest_sha256':OBS_PANEL_SHA256,'v2_source_sha256':V2_SOURCE_SHA256},'rank':{'variable_ids':variable_ids,'fixed_n':64-len(variable_ids),'variable_n':len(variable_ids)},'population':{'mapping_reliable':int(reliable.sum()),'active':int(active.sum()),'reliable_active':int((reliable&active).sum()),'feasible_contained2':int(contained2.sum()),'hardtail_witnesses':len(witness),'hardtail_carriers':[int(x) for x in witness]},'hardtail_rows':rows,'reliable_active_rows':all_rows,'fixed_world_exact_preserved':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--science-dir',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True); ap.add_argument('--sidecar',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--episode',default='e01'); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError('refusing overwrite')
    r=evaluate_family(a.root,a.science_dir,a.state,a.selection,a.sidecar,a.family,a.episode); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(json_safe(r),indent=2,sort_keys=True,allow_nan=False)); print(json.dumps({'family':r['family'],'rank':r['rank'],'population':r['population']},indent=2,sort_keys=True))
if __name__=='__main__': main()
