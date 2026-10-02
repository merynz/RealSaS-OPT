from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import (
    split_by_edge_preimage_components,bad_edge_codes,
)
from tools.audit_knight_bad_edge_opposite_fan_refinement_v1 import (
    collect_bad_edge_opposite_tokens,refine_by_tokens,
)
from tools.audit_knight_residual_bad_edge_local_uncontract_v1 import (
    local_uncontract_bad_edge_preimage,
)

def loadj(p): return json.loads(Path(p).read_text())
def sha256(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)

    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)

    base=compact_inverse(world,faces,divisions=56,component_aware=True)
    inv,_=topology_local_refine(faces,base)
    history=[["TOPOLOGY_LOCAL",0,compact_metrics(world,faces,inv)["nonmanifold_edge_count"],compact_metrics(world,faces,inv)["compact_node_count"]]]

    for it in range(1,20):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]<=7: break
        inv,meta=split_by_edge_preimage_components(faces,inv)
        after=compact_metrics(world,faces,inv)
        history.append(["EDGE_PREIMAGE",it,after["nonmanifold_edge_count"],after["compact_node_count"]])

    for it in range(1,6):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0: break
        codes,_inc,n=bad_edge_codes(faces,inv)
        tokens,_w=collect_bad_edge_opposite_tokens(faces,inv,codes,n)
        nxt,_meta=refine_by_tokens(faces,inv,tokens)
        after=compact_metrics(world,faces,nxt)
        history.append(["BAD_EDGE_OPPOSITE_FAN_SIGNATURE",it,after["nonmanifold_edge_count"],after["compact_node_count"]])
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"]: break

    for it in range(1,8):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0: break
        inv,_meta=local_uncontract_bad_edge_preimage(faces,inv)
        after=compact_metrics(world,faces,inv)
        history.append(["RESIDUAL_BAD_EDGE_LOCAL_UNCONTRACT",it,after["nonmanifold_edge_count"],after["compact_node_count"]])

    final=compact_metrics(world,faces,inv)
    if final["nonmanifold_edge_count"]!=0:
        raise RuntimeError(f"SEALED_REPAIR_NOT_MANIFOLD::{final['nonmanifold_edge_count']}")

    # Strict refinement of exact historical division-56 base.
    pairs=np.unique(np.column_stack((base,inv)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b): cross+=1
    if cross:
        raise RuntimeError(f"SEALED_REPAIR_CROSS_BASE_MERGE::{cross}")

    inv_path=out/"repaired_inverse.npz"
    np.savez_compressed(
        inv_path,
        final_inverse=np.asarray(inv,dtype=np.int32),
        base_inverse=np.asarray(base,dtype=np.int32),
    )
    payload={
      "schema":"RealSaS.KnightRepairedCompactionSeal.v1",
      "status":"SEALED_AUDIT_CHILD__NO_MAINLINE_MUTATION",
      "source_run_id":"SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929",
      "source_zero_surface_sha256":zero.npz_sha256,
      "base_operator":"COMPONENT_AWARE_VOXEL_COMPACTION_DIVISION_56",
      "repair_operator_chain":[
        "TOPOLOGY_LOCAL_CONNECTIVITY",
        "EDGE_PREIMAGE_COMPONENT_SPLIT",
        "BAD_EDGE_OPPOSITE_FAN_SIGNATURE_REFINEMENT",
        "RESIDUAL_BAD_EDGE_DENSE_PREIMAGE_LOCAL_UNCONTRACT",
      ],
      "history":history,
      "final_metrics":final,
      "cross_base_merge_count":0,
      "face_deletion":False,
      "teacher_or_subject_labels_used":False,
      "inverse_npz_sha256":"",
    }
    payload["inverse_npz_sha256"]=sha256(inv_path)
    seal=out/"SEAL.json"
    seal.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_COMPACTION_SEAL="+json.dumps({
      "final_nonmanifold":final["nonmanifold_edge_count"],
      "final_nodes":final["compact_node_count"],
      "inverse_sha256":payload["inverse_npz_sha256"],
      "history":history,
    },sort_keys=True))

if __name__=="__main__":main()
