from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V3B.GeometryWorld.v1'
CONTRACT_COMMIT='7668aff13fb87579bcfdbe113a18c5a46848be34'
V3A_SELECTION_SHA256='82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41'
V3A_SOLVER_SHA256='911cf2be0e06b502f7591f25168d5114a563324a276a3cf41392f9576e7566c1'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
PARITY_REPROJ_ATOL=1e-5
PARITY_DESC_ATOL=1e-6


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def logical_evidence_digest(z):
    h=hashlib.sha256()
    for name in ('coords','scores','valid','V_A','XY_A','output_idx'):
        a=np.ascontiguousarray(z[name]); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()
def qkeys(P): return [tuple(np.round(np.asarray(p)/.0015).astype(int)) for p in np.asarray(P)]

def lookup_assigned_score(ez,i,v,site):
    vals=[]
    target=tuple(map(int,site))
    for k in range(4):
        if not bool(ez['valid'][i,v,k]): continue
        q=tuple(map(int,np.rint(ez['coords'][i,v,k]).astype(int)))
        if q==target: vals.append(float(ez['scores'][i,v,k]))
    if not vals: raise RuntimeError(('assigned site absent from frozen top4',i,v,target))
    return max(vals)

def replay_h_parity(m,PA,seedB,output_idx,ez,state):
    off=np.asarray(state['H_offsets'],np.int64); H=np.asarray(state['H_xyz'],np.float32); HR=np.asarray(state['H_reproj_px'],np.float32); HD=np.asarray(state['H_desc_score'],np.float32)
    max_r=max_d=0.0
    for oi,gi0 in enumerate(output_idx):
        gi=int(gi0); cands=[]; scores=[]
        for v in range(8):
            mask=np.asarray(ez['valid'][oi,v],bool)
            cands.append(np.asarray(ez['coords'][oi,v][mask],np.float32)); scores.append(np.asarray(ez['scores'][oi,v][mask],np.float32))
        pool=m.candidate_pool(cands,scores,np.asarray(PA[gi],np.float32),np.asarray(seedB[gi],np.float32))
        a,b=int(off[oi]),int(off[oi+1])
        if len(pool)!=(b-a): raise RuntimeError(('H parity length',oi,len(pool),b-a))
        P=np.stack([x[0] for x in pool]).astype(np.float32); R=np.asarray([x[1] for x in pool],np.float64); D=np.asarray([x[2] for x in pool],np.float64)
        if qkeys(P)!=qkeys(H[a:b]): raise RuntimeError(('H parity key/order',oi))
        if not np.array_equal(P,H[a:b]): raise RuntimeError(('H parity xyz bytes',oi,float(np.max(np.abs(P-H[a:b])))))
        dr=float(np.max(np.abs(R-HR[a:b]))) if len(R) else 0.; dd=float(np.max(np.abs(D-HD[a:b]))) if len(D) else 0.
        max_r=max(max_r,dr); max_d=max(max_d,dd)
        if dr>PARITY_REPROJ_ATOL or dd>PARITY_DESC_ATOL: raise RuntimeError(('H parity scalar',oi,dr,dd))
    return {'carriers':64,'xyz_exact':True,'quantized_order_exact':True,'max_reproj_abs':max_r,'max_desc_abs':max_d,'reproj_atol':PARITY_REPROJ_ATOL,'desc_atol':PARITY_DESC_ATOL}

def build_family(root,science_dir,state_path,evidence_path,selection_path,family,episode,out_world):
    if sha256_file(selection_path)!=V3A_SELECTION_SHA256: raise RuntimeError('V3A selection SHA mismatch')
    sel=json.loads(selection_path.read_text())
    if sel.get('truth_access')!='NONE' or sel.get('source_sha256')!=V3A_SOLVER_SHA256 or sel.get('observable_panel_manifest_sha256')!=OBS_PANEL_SHA256: raise RuntimeError('V3A selection authority invalid')
    fam=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
    if fam is None: raise RuntimeError('family missing selection')
    if sha256_file(state_path)!=fam['state_sha256']: raise RuntimeError('state SHA mismatch')
    ez=np.load(evidence_path,allow_pickle=False)
    if logical_evidence_digest(ez)!=fam['proposal_evidence_content_sha256']: raise RuntimeError('proposal evidence digest mismatch')
    z=np.load(state_path,allow_pickle=False); meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE': raise RuntimeError('observable state truth contract invalid')
    obs=load_module('v3b_obs_exact',science_dir/'observable_phase.py')
    m,A,B,PA,VA,seedB,output_idx,hb,prb,diag,field,xyA=obs._model_context(root,int(family),episode,meta['route'])
    P_A=np.asarray(z['P_A'],np.float32); P_B=np.asarray(z['P_B'],np.float32)
    if not np.array_equal(np.asarray(PA[output_idx],np.float32),P_A): raise RuntimeError('P_A/model context mismatch')
    if not np.array_equal(np.asarray(output_idx,np.int32),np.asarray(ez['output_idx'],np.int32)): raise RuntimeError('output_idx evidence mismatch')
    parity=replay_h_parity(m,PA,seedB,output_idx,ez,z)
    viewmap={int(vv['view']):vv for vv in fam['views']}
    PBv=P_B.copy(); rows=[]
    for i in range(64):
        cands=[]; scores=[]; assigned_views=[]
        for v in range(8):
            vv=viewmap[v]; d=vv['assigned']; key=str(i)
            if key in d:
                site=tuple(map(int,d[key])); sc=lookup_assigned_score(ez,i,v,site)
                cands.append(np.asarray([site],np.float32)); scores.append(np.asarray([sc],np.float32)); assigned_views.append(v)
            else:
                cands.append(np.empty((0,2),np.float32)); scores.append(np.empty(0,np.float32))
        pool=m.candidate_pool(cands,scores,P_A[i],P_B[i])
        if not pool: raise RuntimeError(('empty V3B pool',i))
        p,e,d=pool[0]; p=np.asarray(p,np.float32); PBv[i]=p
        rows.append({'carrier':i,'assigned_view_n':len(assigned_views),'assigned_views':assigned_views,'pool_n':len(pool),'selected_xyz':p.tolist(),'selected_reproj_px':float(e),'selected_desc_score':float(d),'selected_is_baseline_exact':bool(np.array_equal(p,P_B[i])),'selected_is_PA_exact':bool(np.array_equal(p,P_A[i])),'distance_from_baseline':float(np.linalg.norm(p-P_B[i]))})
    out_world.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out_world,P_A=P_A,P_B_baseline=P_B,P_B_V3B=PBv)
    changed=np.linalg.norm(PBv-P_B,axis=1)
    return {'schema':SCHEMA,'contract_commit':CONTRACT_COMMIT,'family':int(family),'episode':episode,'truth_access':'NONE','authorities':{'state_sha256':sha256_file(state_path),'proposal_evidence_content_sha256':fam['proposal_evidence_content_sha256'],'selection_sha256':V3A_SELECTION_SHA256,'v3a_solver_sha256':V3A_SOLVER_SHA256,'observable_panel_manifest_sha256':OBS_PANEL_SHA256},'candidate_pool_parity':parity,'world':{'world_file_sha256':sha256_file(out_world),'changed_n_gt_1e_8':int(np.sum(changed>1e-8)),'changed_n_gt_1e_4':int(np.sum(changed>1e-4)),'baseline_exact_n':int(sum(r['selected_is_baseline_exact'] for r in rows)),'PA_exact_n':int(sum(r['selected_is_PA_exact'] for r in rows)),'mean_distance_from_baseline':float(np.mean(changed)),'max_distance_from_baseline':float(np.max(changed))},'rows':rows}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--science-dir',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--episode',default='e01'); ap.add_argument('--out-world',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists() or a.out_world.exists(): raise RuntimeError('refusing overwrite')
    r=build_family(a.root,a.science_dir,a.state,a.evidence,a.selection,a.family,a.episode,a.out_world); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'family':r['family'],'parity':r['candidate_pool_parity'],'world':r['world']},indent=2,sort_keys=True))
if __name__=='__main__':main()
