from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

SCHEMA = "RealSaS.N1D.Rank2GlobalRelationalWorld.V1.TruthEvaluation.v1"
V0_SOURCE_SHA256 = "47fd12cd0d2370a10c1249573777f6266cc9df71b02540e4634cf33246941c6d"
OBS_PANEL_SHA256 = "b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef"


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name, str(path))
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


def json_safe(x):
    if isinstance(x, dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating,float)):
        v=float(x); return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,int)): return int(x)
    if isinstance(x, (np.bool_,bool)): return bool(x)
    return x


def material_flags(base_comp, new_comp):
    if not (np.isfinite(base_comp) and np.isfinite(new_comp)):
        return float('nan'), False, False
    rel=float((base_comp-new_comp)/(abs(base_comp)+1e-8))
    imp=bool(rel>=.20 and (base_comp-new_comp)>=.05)
    deg=bool(rel<=-.20 and (new_comp-base_comp)>=.05)
    return rel,imp,deg


def evaluate_family(root: Path, science_dir: Path, state_path: Path, selection_path: Path, sidecar: Path, family: int, episode: str):
    # Exact frozen R6.1 science modules.
    sys.path.insert(0, str(science_dir))
    ev=load_module("r6_eval_exact", science_dir/"evaluation_phase.py")
    mm=load_module("r6_metrics_exact", science_dir/"mechanics_metrics.py")
    obs=load_module("r6_observable_exact", science_dir/"observable_phase.py")

    sel=json.loads(selection_path.read_text())
    if sel.get("truth_access") != "NONE": raise RuntimeError("V1 observable selection truth contract invalid")
    if sel.get("source_sha256") != V0_SOURCE_SHA256: raise RuntimeError("V1 source authority mismatch")
    famrec=next((x for x in sel["families"] if int(x["family"])==int(family)), None)
    if famrec is None: raise RuntimeError(f"family {family} absent from V1 selection")
    if sha256_file(state_path) != famrec["state_sha256"]: raise RuntimeError("state SHA mismatch")

    meta_path=state_path.with_suffix('.json')
    meta=json.loads(meta_path.read_text())
    if meta.get("truth_access") != "NONE": raise RuntimeError("observable state truth contract invalid")
    if sha256_file(state_path) != meta["observable_state_sha256"]: raise RuntimeError("observable state meta SHA mismatch")

    z=np.load(state_path, allow_pickle=False)
    PA=np.asarray(z["P_A"], np.float32); PB0=np.asarray(z["P_B"], np.float32)
    H=np.asarray(z["H_xyz"], np.float32); off=np.asarray(z["H_offsets"], np.int64)
    gidx=np.asarray(famrec["selected_original_global_index"], np.int64)
    if gidx.shape != (64,): raise RuntimeError("V1 selection must have 64 candidate indices")
    for i in range(64):
        if not (int(off[i]) <= int(gidx[i]) < int(off[i+1])):
            raise RuntimeError(("selected candidate outside H_i", i, int(gidx[i]), int(off[i]), int(off[i+1])))
    PBv=H[gidx].astype(np.float32)

    base_world={k:np.asarray(z[k]) for k in ("P_A","P_B","N_A","N_B","V_A","V_B")}
    base_g=ev.gfdr.compute_gfdr_v2(**base_world, Z=None, U=None)

    # Full V1 world gets normals/visibility only from the real observable raster/model decorator.
    NBv,VBv,XYBv=obs.decorate_pb_observable(root, int(family), episode, meta["route"], PA, PBv)
    v0_world={"P_A":PA,"P_B":PBv,"N_A":np.asarray(z["N_A"]),"N_B":NBv,"V_A":np.asarray(z["V_A"]),"V_B":VBv}
    v0_g=ev.gfdr.compute_gfdr_v2(**v0_world, Z=None, U=None)

    # Truth opens only here, after V1 source and observable selection are frozen.
    t512=ev._truth(sidecar)
    ids,map_err=ev._map64(PA,t512)
    t=ev._subset_truth(t512,ids)
    sA=ev.local_scale(t512["P_A"],4)[ids]
    sB=ev.local_scale(t512["P_B"],4)[ids]
    reliable=map_err <= 2*sA
    active=np.linalg.norm(np.asarray(t["P_B"])-np.asarray(t["P_A"]),axis=1)>.005
    base_err=np.linalg.norm(PB0-np.asarray(t["P_B"]),axis=1)
    v0_err=np.linalg.norm(PBv-np.asarray(t["P_B"]),axis=1)
    true_g=ev.gfdr.compute_gfdr_v2(**t,Z=None,U=None)

    nearest=np.full(64,-1,np.int64); nearest_err=np.full(64,np.inf,np.float64); contained2=np.zeros(64,bool)
    for i in range(64):
        a,b=int(off[i]),int(off[i+1]); HH=H[a:b]
        er=np.linalg.norm(HH-np.asarray(t["P_B"])[i][None,:],axis=1)
        if len(er):
            j=int(np.argmin(er)); nearest[i]=a+j; nearest_err[i]=float(er[j]); contained2[i]=bool(er[j] <= 2*sB[i])

    witness=[i for i in range(64) if reliable[i] and active[i] and contained2[i] and base_err[i] > 2*sB[i]]
    rows=[]
    for i in witness:
        bt=mm.truth_block_errors(base_g,true_g,PA,i)
        vt=mm.truth_block_errors(v0_g,true_g,PA,i)
        bc=float(bt["composite_median_capped10"]); vc=float(vt["composite_median_capped10"])
        rel,imp,deg=material_flags(bc,vc)
        effect=mm.effect_blocks(base_g,v0_g,PA,i)
        rows.append({
            "carrier":int(i),"truth_surface_id":int(ids[i]),
            "mapping_error_over_scaleA":float(map_err[i]/max(sA[i],1e-12)),
            "baseline_error_over_scaleB":float(base_err[i]/max(sB[i],1e-12)),
            "v0_error_over_scaleB":float(v0_err[i]/max(sB[i],1e-12)),
            "teacher_nearest_error_over_scaleB":float(nearest_err[i]/max(sB[i],1e-12)),
            "v0_is_teacher_nearest":bool(int(gidx[i])==int(nearest[i])),
            "geometry_improved_vs_baseline":bool(v0_err[i] < base_err[i]),
            "geometry_delta_over_scaleB":float((base_err[i]-v0_err[i])/max(sB[i],1e-12)),
            "truth_relative":{"baseline":bt,"v0":vt,"relative_improvement":rel,"material_improve":imp,"material_degrade":deg},
            "baseline_vs_v0_effect":effect,
        })

    ra=np.where(reliable & active)[0]
    all_rows=[]
    for i in ra:
        bt=mm.truth_block_errors(base_g,true_g,PA,int(i)); vt=mm.truth_block_errors(v0_g,true_g,PA,int(i))
        bc=float(bt["composite_median_capped10"]); vc=float(vt["composite_median_capped10"])
        rel,imp,deg=material_flags(bc,vc)
        all_rows.append({"carrier":int(i),"baseline_composite":bc,"v0_composite":vc,"relative_improvement":rel,"material_improve":imp,"material_degrade":deg})

    return {
        "schema":SCHEMA,"family":int(family),"episode":episode,
        "authorities":{"state_sha256":sha256_file(state_path),"selection_sha256":sha256_file(selection_path),"sidecar_sha256":sha256_file(sidecar),"observable_panel_manifest_sha256":OBS_PANEL_SHA256,"v0_source_sha256":V0_SOURCE_SHA256},
        "population":{"mapping_reliable":int(reliable.sum()),"active":int(active.sum()),"reliable_active":int((reliable&active).sum()),"feasible_contained2":int(contained2.sum()),"hardtail_witnesses":len(witness),"hardtail_carriers":[int(x) for x in witness]},
        "world":{"selected_candidate_indices":[int(x) for x in gidx],"observable_decorator":{"route":meta["route"],"N_B_shape":list(NBv.shape),"V_B_shape":list(VBv.shape),"XY_B_shape":list(XYBv.shape)}},
        "hardtail_rows":rows,"reliable_active_rows":all_rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True); ap.add_argument('--science-dir',type=Path,required=True)
    ap.add_argument('--state',type=Path,required=True); ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--sidecar',type=Path,required=True); ap.add_argument('--family',type=int,required=True)
    ap.add_argument('--episode',default='e01'); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise RuntimeError(f"refusing overwrite: {a.out}")
    r=evaluate_family(a.root,a.science_dir,a.state,a.selection,a.sidecar,a.family,a.episode)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(json_safe(r),indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps({"family":r["family"],"population":r["population"]},indent=2,sort_keys=True))

if __name__=='__main__': main()
