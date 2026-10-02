from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict
from tools.audit_knight_topology_local_voxel_compaction_v1 import compact_metrics
from tools.audit_knight_vertex_link_fan_refinement_v1 import (
    loadj,compact_faces,link_components,refine_bad_vertex_links,
)

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--parent-inverse-npz",type=Path,required=True)
    ap.add_argument("--parent-seal",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--max-iterations",type=int,default=24)
    a=ap.parse_args();rr=a.run_root.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)

    parent_seal=loadj(a.parent_seal)
    if sha256(a.parent_inverse_npz)!=str(parent_seal["inverse_npz_sha256"]):
        raise RuntimeError("PARENT_REPAIRED_INVERSE_SHA_MISMATCH")
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        faces=np.asarray(z["faces"],dtype=np.int64)
        verts=np.asarray(z["vertices_normalized"],dtype=np.float64)
    with np.load(a.parent_inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],dtype=np.int64)
        base=np.asarray(z["base_inverse"],dtype=np.int64)

    current=inv.copy();history=[]
    for it in range(int(a.max_iterations)):
        cf=compact_faces(faces,current)
        bad=link_components(cf,int(current.max())+1)
        met=compact_metrics(verts,faces,current)
        if not bad:
            break
        nxt,meta=refine_bad_vertex_links(faces,current)
        cf2=compact_faces(faces,nxt)
        bad2=link_components(cf2,int(nxt.max())+1)
        met2=compact_metrics(verts,faces,nxt)
        history.append({
          "iteration":it,
          "before_nodes":int(current.max())+1,
          "before_illegal_vertex_links":len(bad),
          "before_nonmanifold_edges":met["nonmanifold_edge_count"],
          "after_nodes":int(nxt.max())+1,
          "after_illegal_vertex_links":len(bad2),
          "after_nonmanifold_edges":met2["nonmanifold_edge_count"],
          "multi_fan_dense_vertex_count":int(meta["multi_fan_dense_vertex_count"]),
          "unassigned_dense_vertex_count":int(meta["unassigned_dense_vertex_count"]),
        })
        if np.array_equal(nxt,current):
            current=nxt
            break
        current=nxt

    final_cf=compact_faces(faces,current)
    final_bad=link_components(final_cf,int(current.max())+1)
    final_metrics=compact_metrics(verts,faces,current)
    if final_metrics["nonmanifold_edge_count"]!=0 or final_bad:
        raise RuntimeError(
            f"REPAIRED_V2_NOT_MANIFOLD:{final_metrics['nonmanifold_edge_count']}:{len(final_bad)}"
        )

    pairs=np.unique(np.column_stack((base,current)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b):cross+=1
    if cross:
        raise RuntimeError(f"REPAIRED_V2_CROSS_BASE_MERGE:{cross}")

    inv_path=out/"repaired_inverse_v2.npz"
    np.savez_compressed(
      inv_path,
      final_inverse=np.asarray(current,dtype=np.int32),
      base_inverse=np.asarray(base,dtype=np.int32),
    )
    payload={
      "schema":"RealSaS.KnightRepairedCompactionSeal.v2",
      "status":"SEALED_AUDIT_CHILD__EDGE_AND_VERTEX_MANIFOLD__NO_MAINLINE_MUTATION",
      "source_run_id":"SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929",
      "parent_seal_schema":parent_seal.get("schema"),
      "parent_inverse_npz_sha256":parent_seal["inverse_npz_sha256"],
      "source_zero_surface_sha256":zero.npz_sha256,
      "repair_extension_operator":"ITERATIVE_DENSE_FACE_FAN_VERTEX_LINK_REFINEMENT_V1",
      "history":history,
      "final_metrics":final_metrics,
      "final_illegal_vertex_link_count":0,
      "cross_base_merge_count":0,
      "face_deletion":False,
      "teacher_or_subject_labels_used":False,
      "multi_fan_dense_vertex_count_observed":int(sum(r["multi_fan_dense_vertex_count"] for r in history)),
      "inverse_npz_sha256":sha256(inv_path),
    }
    (out/"SEAL_V2.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_COMPACTION_SEAL_V2="+json.dumps({
      "final_nodes":final_metrics["compact_node_count"],
      "final_nonmanifold_edges":final_metrics["nonmanifold_edge_count"],
      "final_illegal_vertex_links":0,
      "cross_base_merge_count":0,
      "iterations":len(history),
      "inverse_sha256":payload["inverse_npz_sha256"],
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
