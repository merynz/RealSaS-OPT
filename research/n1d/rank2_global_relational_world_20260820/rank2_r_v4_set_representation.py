from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

SCHEMA='RealSaS.N1D.Rank2R.V4.SetRepresentation.v1'
CONTRACT_COMMIT='873c7c375226670416d7dedb0c0c3422745472e7'
V3C_SOURCE_SHA256='556a99090e90530a4642337d6ad562e356303d4cc6f99480bf6104e2737625ba'
V3A_SELECTION_SHA256='82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41'
OBS_PANEL_SHA256='b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef'

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def pareto_layers(g,r):
    g=np.asarray(g,float); r=np.asarray(r,float); n=len(g); layer=np.full(n,-1,np.int64); remaining=list(range(n)); L=0
    while remaining:
        front=[]
        for i in remaining:
            dominated=False
            for j in remaining:
                if i==j: continue
                if g[j] <= g[i] and r[j] <= r[i] and (g[j] < g[i] or r[j] < r[i]):
                    dominated=True; break
            if not dominated: front.append(i)
        if not front: raise RuntimeError('pareto decomposition stalled')
        for i in front: layer[i]=L
        fs=set(front); remaining=[i for i in remaining if i not in fs]; L+=1
    return layer

def solve_family(science_dir,state_path,evidence_path,selection_path,v3c_source,family,out_rep):
    if sha256_file(v3c_source)!=V3C_SOURCE_SHA256: raise RuntimeError('V3C source SHA mismatch')
    if sha256_file(selection_path)!=V3A_SELECTION_SHA256: raise RuntimeError('V3A selection SHA mismatch')
    v3=load_module('v4_v3c_exact',v3c_source)
    sel=json.loads(selection_path.read_text())
    if sel.get('truth_access')!='NONE' or sel.get('observable_panel_manifest_sha256')!=OBS_PANEL_SHA256: raise RuntimeError('selection authority invalid')
    famrec=next((x for x in sel['families'] if int(x['family'])==int(family)),None)
    if famrec is None: raise RuntimeError('family missing selection')
    if sha256_file(state_path)!=famrec['state_sha256']: raise RuntimeError('state SHA mismatch')
    z=np.load(state_path,allow_pickle=False); meta=json.loads(state_path.with_suffix('.json').read_text())
    if meta.get('truth_access')!='NONE': raise RuntimeError('state truth contract invalid')
    ez=np.load(evidence_path,allow_pickle=False)
    coords=np.asarray(ez['coords']); scores=np.asarray(ez['scores']); valid=np.asarray(ez['valid']); VA=np.asarray(ez['V_A']); XYA=np.asarray(ez['XY_A'])
    dig=v3.array_digest([('coords',coords),('scores',scores),('valid',valid),('V_A',VA),('XY_A',XYA),('output_idx',np.asarray(ez['output_idx'],np.int32))])
    if dig!=famrec['proposal_evidence_content_sha256']: raise RuntimeError('proposal evidence digest mismatch')
    if not np.array_equal(VA,np.asarray(z['V_A'])) or not np.array_equal(XYA,np.asarray(z['XY_A'])): raise RuntimeError('proposal/state observation mismatch')
    per_view,replay=v3.proposal_supports(coords,scores,valid,VA,XYA,famrec)
    obs=v3.load_obs_module(science_dir); m=obs._route_module(meta['route'])
    PB=np.asarray(z['P_B'],np.float32); H=np.asarray(z['H_xyz'],np.float32); HR=np.asarray(z['H_reproj_px'],np.float64); HD=np.asarray(z['H_desc_score'],np.float64); off=np.asarray(z['H_offsets'],np.int64)
    xs=[]; src=[]; base=[]; graws=[]; granks=[]; rraws=[]; rranks=[]; layers=[]; offsets=[0]; rows=[]
    for i in range(64):
        a,b=int(off[i]),int(off[i+1]); Q=[np.asarray(x,np.float32) for x in H[a:b]]; rp=[float(x) for x in HR[a:b]]; ds=[float(x) for x in HD[a:b]]; source_idx=[a+j for j in range(b-a)]
        exact=[j for j,p in enumerate(Q) if np.array_equal(p,PB[i])]
        if not exact:
            e,d=v3.baseline_observable_score(m,PB,coords,scores,valid,VA,i); Q.append(PB[i].copy()); rp.append(e); ds.append(d); source_idx.append(-1)
        rp=np.asarray(rp,float); ds=np.asarray(ds,float); graw=.75*v3.rank01(rp,False)+.25*v3.rank01(ds,True); gr=v3.rank01(graw,False)
        rr=np.asarray([v3.candidate_R_raw(m,p,i,VA,per_view) for p in Q],float); rk=v3.rank01(rr,False); pl=pareto_layers(gr,rk)
        bflags=np.array([np.array_equal(p,PB[i]) for p in Q],bool)
        qarr=np.stack(Q).astype(np.float32); n=len(Q)
        xs.append(qarr); src.extend(source_idx); base.extend(bflags.tolist()); graws.extend(graw.tolist()); granks.extend(gr.tolist()); rraws.extend(rr.tolist()); rranks.extend(rk.tolist()); layers.extend(pl.tolist()); offsets.append(offsets[-1]+n)
        rows.append({'carrier':i,'domain_n':n,'pareto_front_n':int(np.sum(pl==0)),'pareto_layers_n':int(pl.max()+1),'baseline_domain_index':int(np.where(bflags)[0][0]),'baseline_pareto_layer':int(pl[np.where(bflags)[0][0]]),'g_best_n':int(np.sum(gr==gr.min())),'r_best_n':int(np.sum(rk==rk.min()))})
    XYZ=np.concatenate(xs,axis=0)
    arrays={'offsets':np.asarray(offsets,np.int64),'xyz':XYZ,'source_global_index':np.asarray(src,np.int64),'baseline_flag':np.asarray(base,bool),'G_raw':np.asarray(graws,np.float64),'G_rank':np.asarray(granks,np.float64),'R_raw':np.asarray(rraws,np.float64),'R_rank':np.asarray(rranks,np.float64),'pareto_layer':np.asarray(layers,np.int64)}
    out_rep.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out_rep,**arrays)
    return {'schema':SCHEMA,'contract_commit':CONTRACT_COMMIT,'family':int(family),'truth_access':'NONE','authorities':{'state_sha256':sha256_file(state_path),'proposal_evidence_content_sha256':dig,'selection_sha256':V3A_SELECTION_SHA256,'v3c_source_sha256':V3C_SOURCE_SHA256,'observable_panel_manifest_sha256':OBS_PANEL_SHA256},'minmarginal_replay':replay,'representation':{'file_sha256':sha256_file(out_rep),'candidate_total':int(len(XYZ)),'pareto_front_total':int(sum(r['pareto_front_n'] for r in rows)),'pareto_front_mean':float(np.mean([r['pareto_front_n'] for r in rows])),'pareto_front_max':int(max(r['pareto_front_n'] for r in rows)),'baseline_front_n':int(sum(r['baseline_pareto_layer']==0 for r in rows))},'rows':rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--science-dir',type=Path,required=True); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True); ap.add_argument('--v3c-source',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--out-rep',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists() or a.out_rep.exists(): raise RuntimeError('refusing overwrite')
    r=solve_family(a.science_dir,a.state,a.evidence,a.selection,a.v3c_source,a.family,a.out_rep); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2,sort_keys=True)); print(json.dumps({'family':r['family'],'replay':r['minmarginal_replay'],'representation':r['representation']},indent=2,sort_keys=True))
if __name__=='__main__':main()
