from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict, rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict, normalization_domain_from_dict
from compiler.realsas_compiler_core.substrate.scene_first_signed import mesh_connected_component_labels_v1
from tools.demo.render_knight_motion_preview_v1 import _ctx

EXACT_CANDIDATE_SHA="0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def compact_inverse(points:np.ndarray, faces:np.ndarray, *, divisions:int, component_aware:bool):
    p=np.asarray(points,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    if divisions==0:
        return np.arange(len(p),dtype=np.int64)
    lo=p.min(axis=0); span=np.maximum(p.max(axis=0)-lo,1e-12)
    keys=np.floor((p-lo)/span*int(divisions)).astype(np.int64)
    keys=np.clip(keys,0,int(divisions)-1)
    if component_aware:
        labels=mesh_connected_component_labels_v1(len(p),f)
        keys=np.column_stack((labels,keys))
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    return np.asarray(inv,dtype=np.int64)

def compact_points(points, inv):
    n=int(inv.max())+1
    counts=np.bincount(inv,minlength=n).astype(np.float64)
    out=np.zeros((n,3),dtype=np.float64)
    np.add.at(out,inv,np.asarray(points,dtype=np.float64))
    out/=counts[:,None]
    return out

def mapped_faces(faces,inv):
    mf=inv[np.asarray(faces,dtype=np.int64)]
    keep=(mf[:,0]!=mf[:,1])&(mf[:,1]!=mf[:,2])&(mf[:,2]!=mf[:,0])
    mf=np.sort(mf[keep],axis=1)
    return np.unique(mf,axis=0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]

    zero_path=rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"
    norm_path=rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"
    surface_path=rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"
    candidate_path=rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"
    if sha(candidate_path)!=EXACT_CANDIDATE_SHA:
        raise RuntimeError(f"CANDIDATE_SHA_DRIFT:{sha(candidate_path)}")

    zero=signed_zero_surface_from_dict(json.loads(zero_path.read_text()))
    norm=normalization_domain_from_dict(json.loads(norm_path.read_text()))
    surface=rigging_surface_from_dict(json.loads(surface_path.read_text()))
    candidate=canonical_mesh_candidate_from_dict(json.loads(candidate_path.read_text()))

    npz=Path(zero.npz_path)
    if not npz.is_file() or sha(npz)!=zero.npz_sha256:
        raise RuntimeError("ZERO_NPZ_BYTES_DRIFT")
    with np.load(npz,allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        dense_faces=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)

    md=dict(surface.metadata or {})
    divisions=int(md.get("compact_voxel_divisions",-1))
    component_aware=bool(md.get("component_aware_compaction",False))
    if divisions<0: raise RuntimeError("COMPACTION_DIVISIONS_MISSING")
    inv=compact_inverse(world,dense_faces,divisions=divisions,component_aware=component_aware)
    cp=compact_points(world,inv)
    actual=np.asarray([n.P for n in surface.surface_nodes],dtype=np.float64)
    if cp.shape!=actual.shape:
        raise RuntimeError(f"COMPACT_COUNT_DRIFT:{cp.shape}:{actual.shape}")
    poserr=np.linalg.norm(cp-actual,axis=1)

    mf=mapped_faces(dense_faces,inv)
    mapped_set={tuple(map(int,row)) for row in mf.tolist()}
    sid_to_index={str(n.surface_id):i for i,n in enumerate(surface.surface_nodes)}

    current=[]
    non_identity=0
    for face in candidate.faces:
        row=[]
        for vid in face:
            v=next(x for x in candidate.vertices if str(x.candidate_vertex_id)==str(vid))
            coeff=tuple(v.support_binding.coefficients)
            if len(coeff)!=1 or abs(float(coeff[0][1])-1.0)>1e-12:
                non_identity+=1; row=[]; break
            row.append(sid_to_index[str(coeff[0][0])])
        if row:
            current.append(tuple(sorted(map(int,row))))
    if non_identity:
        raise RuntimeError(f"CURRENT_CANDIDATE_NONIDENTITY:{non_identity}")
    current_set=set(current)
    invented=sorted(current_set-mapped_set)
    supported=sorted(current_set&mapped_set)
    omitted=sorted(mapped_set-current_set)

    # Each invented 3-clique should still have all three pairwise mapped edges.
    mapped_edges=set()
    for a0,b0,c0 in mapped_set:
        mapped_edges.add(tuple(sorted((a0,b0))))
        mapped_edges.add(tuple(sorted((b0,c0))))
        mapped_edges.add(tuple(sorted((c0,a0))))
    invented_all_edges=sum(
        all(tuple(sorted(e)) in mapped_edges for e in ((a0,b0),(b0,c0),(c0,a0)))
        for a0,b0,c0 in invented
    )

    report={
        "schema":"RealSaS.KnightStage18DenseFaceReplayAudit.v1",
        "status":"MEASURED__NO_REPAIR",
        "bindings":{
            "zero_npz_sha256":zero.npz_sha256,
            "surface_lineage_hash":surface.geometry_lineage_hash,
            "candidate_sha256":EXACT_CANDIDATE_SHA,
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
        },
        "compaction_replay":{
            "dense_vertex_count":int(len(world)),
            "dense_face_count":int(len(dense_faces)),
            "component_aware":component_aware,
            "voxel_divisions":divisions,
            "compact_node_count":int(len(cp)),
            "surface_node_count":int(len(actual)),
            "position_error_max":float(poserr.max(initial=0.0)),
            "position_error_p99":float(np.quantile(poserr,.99)),
            "exact_index_order_replay":bool(float(poserr.max(initial=0.0))<1e-9),
        },
        "face_provenance":{
            "mapped_dense_face_count":int(len(mapped_set)),
            "current_candidate_face_count":int(len(current_set)),
            "current_face_supported_by_mapped_dense_face":int(len(supported)),
            "current_face_invented_by_edge_clique":int(len(invented)),
            "mapped_dense_faces_not_in_current_candidate":int(len(omitted)),
            "invented_face_fraction":float(len(invented)/max(1,len(current_set))),
            "invented_faces_with_all_three_edges_individually_supported":int(invented_all_edges),
        },
        "invented_face_indices_head":[list(x) for x in invented[:200]],
        "finding":{
            "stage14_compaction_can_be_replayed_without_surface_lineage_mutation":bool(float(poserr.max(initial=0.0))<1e-9),
            "stage18_contains_faces_absent_from_actual_mapped_dense_faces":bool(len(invented)>0),
            "edge_clique_reconstruction_is_directly_observed_minting_mechanism":bool(len(invented)>0 and invented_all_edges==len(invented)),
        },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE18_DENSE_FACE_REPLAY_AUDIT_PASS",json.dumps({**report["compaction_replay"],**report["face_provenance"],**report["finding"]},sort_keys=True))

if __name__=="__main__": main()
