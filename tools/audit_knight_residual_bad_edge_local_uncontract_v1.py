from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import (
    split_by_edge_preimage_components,bad_edge_codes,collect_bad_preimage_edges,
)
from tools.audit_knight_bad_edge_opposite_fan_refinement_v1 import (
    collect_bad_edge_opposite_tokens,refine_by_tokens,
)

def loadj(p): return json.loads(Path(p).read_text())

def local_uncontract_bad_edge_preimage(faces,inv):
    bad_codes,bad_inc,n=bad_edge_codes(faces,inv)
    if len(bad_codes)==0:
        return inv.copy(),{
            "bad_edge_count_before":0,
            "released_dense_vertex_count":0,
            "released_from_cluster_count":0,
            "bad_incidences":[],
        }
    pre=collect_bad_preimage_edges(faces,inv,bad_codes,n)
    release=set()
    for code in map(int,bad_codes.tolist()):
        for u,v in pre.get(code,()):
            release.add(int(u));release.add(int(v))
    out=np.asarray(inv,dtype=np.int64).copy()
    old_clusters={int(inv[v]) for v in release}
    next_id=int(out.max())+1
    # Minimal fail-closed local decompaction: only vertices that are actual
    # dense-edge witnesses of an over-incident quotient edge become singleton
    # quotient classes. Unrelated members of the endpoint cluster remain compact.
    for vid in sorted(release):
        out[vid]=next_id
        next_id+=1
    _,out=np.unique(out,return_inverse=True)
    out=np.asarray(out,dtype=np.int64)
    # Removing witness vertices can disconnect the remainder of an old cluster;
    # restore the existing dense-edge connectivity invariant as a pure refinement.
    out,_=topology_local_refine(faces,out)
    return out,{
        "bad_edge_count_before":int(len(bad_codes)),
        "released_dense_vertex_count":int(len(release)),
        "released_from_cluster_count":int(len(old_clusters)),
        "bad_incidences":sorted(int(v) for v in bad_inc.values()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)

    # Exact smear-base compaction. Do not change the voxel grid.
    base=compact_inverse(world,faces,divisions=56,component_aware=True)
    inv,_=topology_local_refine(faces,base)
    history=[{"phase":"TOPOLOGY_LOCAL","iteration":0,"metrics":compact_metrics(world,faces,inv)}]

    # Existing generic preimage refinement until 7-edge regime/fixed point.
    for it in range(1,20):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]<=7:
            break
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        after=compact_metrics(world,faces,nxt)
        history.append({"phase":"EDGE_PREIMAGE","iteration":it,"metrics":after,"meta":meta})
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"] and meta["disconnected_bad_edge_count"]==0:
            break

    # Stronger face-fan signature refinement; previous court showed 7 -> 5.
    for it in range(1,6):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0:
            break
        bad_codes,bad_inc,n=bad_edge_codes(faces,inv)
        tokens,witness=collect_bad_edge_opposite_tokens(faces,inv,bad_codes,n)
        nxt,meta=refine_by_tokens(faces,inv,tokens)
        after=compact_metrics(world,faces,nxt)
        history.append({
            "phase":"BAD_EDGE_OPPOSITE_FAN_SIGNATURE",
            "iteration":it,"metrics":after,
            "meta":{**meta,"bad_edge_count_before":int(len(bad_codes)),
                    "witness_rows":int(sum(len(v) for v in witness.values()))},
        })
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"]:
            break

    # Final local fail-closed refinement. Iterate because splitting one quotient
    # edge may expose another previously collapsed quotient edge; every iteration
    # remains a strict refinement of the original smear-base compaction.
    for it in range(1,8):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0:
            break
        nxt,meta=local_uncontract_bad_edge_preimage(faces,inv)
        after=compact_metrics(world,faces,nxt)
        history.append({"phase":"RESIDUAL_BAD_EDGE_LOCAL_UNCONTRACT","iteration":it,"metrics":after,"meta":meta})
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"] and meta["released_dense_vertex_count"]==0:
            break

    final=compact_metrics(world,faces,inv)

    # Proof: pure refinement of the exact original division-56 compaction.
    pairs=np.unique(np.column_stack((base,inv)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b): cross+=1

    report={
      "schema":"RealSaS.KnightResidualBadEdgeLocalUncontractCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "base":compact_metrics(world,faces,base),
      "history":history,
      "final":final,
      "cross_base_merge_count":int(cross),
      "operator":{
        "id":"RESIDUAL_BAD_EDGE_DENSE_PREIMAGE_LOCAL_UNCONTRACT_V1",
        "generic_invariants":[
          "PURE_REFINEMENT_OF_EXACT_SMEAR_BASE_COMPACTION",
          "OVERINCIDENT_QUOTIENT_EDGE_DENSE_PREIMAGE_MAY_NOT_REMAIN_CONTRACTED",
          "ONLY_ACTUAL_DENSE_EDGE_WITNESS_VERTICES_ARE_RELEASED",
          "DENSE_EDGE_CONNECTED_CLUSTER_REMAINDERS_REQUIRED",
          "FACE_DELETION_FORBIDDEN",
          "TEACHER_OR_SUBJECT_LABELS_FORBIDDEN",
        ],
      },
      "success":bool(final["nonmanifold_edge_count"]==0 and cross==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_RESIDUAL_BAD_EDGE_LOCAL_UNCONTRACT="+json.dumps({
      "history":[[x["phase"],x["iteration"],x["metrics"]["nonmanifold_edge_count"],x["metrics"]["compact_node_count"]] for x in history],
      "final_nonmanifold":final["nonmanifold_edge_count"],
      "final_nodes":final["compact_node_count"],
      "cross_base_merge_count":cross,
      "success":report["success"],
    },sort_keys=True))
    if not report["success"]:
        raise SystemExit(2)

if __name__=="__main__":main()
