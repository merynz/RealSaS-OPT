from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V3A.TruthEvaluation.v1'
SELECTION_SHA256='82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41'
SOLVER_SHA256='911cf2be0e06b502f7591f25168d5114a563324a276a3cf41392f9576e7566c1'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'
RANK_TOL=1e-8
YAW_DEG=(0.,45.,90.,135.,180.,225.,270.,315.)

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
def eval_family(science_dir,state_path,evidence_path,selection_path,sidecar,family,episode):
    if sha256_file(selection_path)!=SELECTION_SHA256:raise RuntimeError('selection SHA mismatch')
    sel=json.loads(selection_path.read_text())
    if sel.get('truth_access')!='NONE' or sel.get('source_sha256')!=SOLVER_SHA256:raise RuntimeError('selection authority invalid')
    if sel.get('observable_panel_manifest_sha256')!=OBS_PANEL_SHA256:raise RuntimeError('panel authority mismatch')
    fam=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
    if fam is None:raise RuntimeError('family missing in selection')
    if sha256_file(state_path)!=fam['state_sha256']:raise RuntimeError('state SHA mismatch')
    ez=np.load(evidence_path,allow_pickle=False)
    if logical_evidence_digest(ez)!=fam['proposal_evidence_content_sha256']:raise RuntimeError('proposal evidence digest mismatch')
    z=np.load(state_path,allow_pickle=False);meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE':raise RuntimeError('state truth contract invalid')
    sys.path.insert(0,str(science_dir));ev=load_module('v3a_eval_exact',science_dir/'evaluation_phase.py');obs=load_module('v3a_obs_exact_eval',science_dir/'observable_phase.py')
    PA=np.asarray(z['P_A'],np.float32);PB=np.asarray(z['P_B'],np.float32);H=np.asarray(z['H_xyz'],np.float32);off=np.asarray(z['H_offsets'],np.int64)
    # All selection/proposal authority checks are complete before truth opens here.
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
    for vv in fam['views']:
        v=int(vv['view']);assigned={int(k):tuple(map(int,val)) for k,val in vv['assigned'].items()};gtop={int(k):tuple(map(int,val)) for k,val in vv['g_top1'].items()};ab=set(map(int,vv['abstained']))
        for i in np.where(VA[v]>0)[0]:
            i=int(i)
            if not reliable[i] or not bool(truthVB[v,i]):continue
            sites=[]
            for k in range(4):
                if valid[i,v,k]:
                    q=tuple(map(int,np.rint(coords[i,v,k]).astype(int)))
                    if q not in sites:sites.append(q)
            if not sites:continue
            target=np.asarray(truthXY[v,i],float);dist=np.asarray([np.linalg.norm(np.asarray(s,float)-target) for s in sites],float);best=float(np.min(dist));oracle={sites[k] for k in np.where(np.isclose(dist,best,rtol=0,atol=1e-9))[0]}
            gs=gtop[i];gd=float(np.linalg.norm(np.asarray(gs,float)-target));isab=i in ab
            rs=None if isab else assigned.get(i)
            rd=None if rs is None else float(np.linalg.norm(np.asarray(rs,float)-target))
            rows.append({'view':v,'carrier':i,'rank3':bool(i in rank3),'hardtail':bool(i in hardtail),'hardtail_rank3':bool(i in hardtail and i in rank3),'anchor':bool(i in set(vv['anchors'])),'oracle_best_sites':[list(x) for x in sorted(oracle)],'oracle_best_distance_px':best,'g_top1_site':list(gs),'g_top1_distance_px':gd,'g_top1_oracle_hit':bool(gs in oracle),'v3a_abstain':bool(isab),'v3a_site':None if rs is None else list(rs),'v3a_distance_px':rd,'v3a_oracle_hit':bool(False if rs is None else rs in oracle),'g_regret_px':float(gd-best),'v3a_regret_px':None if rd is None else float(rd-best)})
    return {'schema':SCHEMA,'family':int(family),'episode':episode,'authorities':{'state_sha256':sha256_file(state_path),'selection_sha256':SELECTION_SHA256,'proposal_evidence_content_sha256':fam['proposal_evidence_content_sha256'],'sidecar_sha256':sha256_file(sidecar),'solver_sha256':SOLVER_SHA256},'population':{'mapping_reliable':int(reliable.sum()),'hardtail_carriers':sorted(hardtail),'hardtail_n':len(hardtail),'rank3_hardtail_carriers':sorted(hardtail&rank3),'rank3_hardtail_n':len(hardtail&rank3),'applicable_rows':len(rows)},'rows':rows}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--science-dir',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',default='e01');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise RuntimeError('refusing overwrite')
    r=eval_family(a.science_dir,a.state,a.evidence,a.selection,a.sidecar,a.family,a.episode);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True));print(json.dumps({'family':r['family'],'population':r['population']},indent=2))
if __name__=='__main__':main()
