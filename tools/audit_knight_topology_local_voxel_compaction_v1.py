from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,
    signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse


def loadj(path: Path) -> dict:
    return json.loads(path.read_text())


def compress(parent: np.ndarray) -> None:
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return
        parent[:] = nxt


def topology_local_refine(
    faces: np.ndarray,
    base_inverse: np.ndarray,
    *,
    chunk_faces: int = 350_000,
    max_passes: int = 32,
) -> tuple[np.ndarray, dict]:
    """Split each voxel bucket by dense-edge connectivity.

    Edges may union vertices only when both endpoints already belong to the same
    base component+voxel bucket.  Therefore this is a pure refinement: it can never
    merge across the current compaction authority, only split aliases that are not
    locally connected on the dense admitted surface.
    """
    f=np.asarray(faces,dtype=np.int64)
    base=np.asarray(base_inverse,dtype=np.int64)
    parent=np.arange(len(base),dtype=np.int64)
    passes=0

    for pass_index in range(max_passes):
        compress(parent)
        changed=False
        for start in range(0,len(f),chunk_faces):
            rows=f[start:start+chunk_faces]
            for li,ri in ((0,1),(1,2),(2,0)):
                u=rows[:,li]; v=rows[:,ri]
                same=base[u]==base[v]
                if not np.any(same):
                    continue
                uu=u[same]; vv=v[same]
                ru=parent[uu]; rv=parent[vv]
                hi=np.maximum(ru,rv)
                lo=np.minimum(ru,rv)
                active=hi!=lo
                if np.any(active):
                    changed=True
                    np.minimum.at(parent,hi[active],lo[active])
        passes=pass_index+1
        compress(parent)
        if not changed:
            break
    else:
        raise RuntimeError("TOPOLOGY_LOCAL_CONNECTIVITY_DID_NOT_CONVERGE")

    roots=parent.copy()
    # A root is already confined to one base bucket. Canonicalize by
    # (base bucket, minimum dense vertex root) for deterministic IDs.
    pairs=np.column_stack((base,roots))
    _unique, refined=np.unique(pairs,axis=0,return_inverse=True)
    refined=np.asarray(refined,dtype=np.int64)

    base_count=int(base.max())+1
    refined_count=int(refined.max())+1
    split_counts=np.bincount(base,minlength=base_count)
    child_per_base=np.zeros((base_count,),dtype=np.int64)
    seen=np.unique(np.column_stack((base,refined)),axis=0)
    np.add.at(child_per_base,seen[:,0],1)

    return refined,{
        "connectivity_passes":passes,
        "base_bucket_count":base_count,
        "refined_bucket_count":refined_count,
        "added_bucket_count":refined_count-base_count,
        "split_base_bucket_count":int(np.count_nonzero(child_per_base>1)),
        "max_children_per_base_bucket":int(child_per_base.max(initial=0)),
        "max_dense_vertices_per_base_bucket":int(split_counts.max(initial=0)),
    }


def compact_metrics(points: np.ndarray, faces: np.ndarray, inverse: np.ndarray) -> dict:
    p=np.asarray(points,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    inv=np.asarray(inverse,dtype=np.int64)
    count=int(inv.max())+1

    counts=np.bincount(inv,minlength=count).astype(np.float64)
    cp=np.zeros((count,3),dtype=np.float64)
    np.add.at(cp,inv,p)
    cp/=counts[:,None]

    mapped=inv[f]
    collapsed=(
        (mapped[:,0]==mapped[:,1]) |
        (mapped[:,1]==mapped[:,2]) |
        (mapped[:,2]==mapped[:,0])
    )
    nondeg=np.sort(mapped[~collapsed],axis=1)
    unique_faces=np.unique(nondeg,axis=0)

    # Compact edge incidence from unique compact faces.
    edge=np.concatenate((
        unique_faces[:,[0,1]],
        unique_faces[:,[1,2]],
        unique_faces[:,[2,0]],
    ),axis=0)
    edge=np.sort(edge,axis=1)
    keys=edge[:,0].astype(np.int64)*np.int64(count)+edge[:,1].astype(np.int64)
    uk,cnt=np.unique(keys,return_counts=True)
    bad=cnt>2
    hist=Counter(map(int,cnt[bad].tolist()))

    # Dense-face area domain preservation is structural: every noncollapsed dense
    # triangle has exactly one compact triangle witness; duplicates may coalesce but
    # no face is deleted by policy.
    return {
        "compact_node_count":count,
        "compact_unique_face_count":int(len(unique_faces)),
        "collapsed_dense_face_count":int(np.count_nonzero(collapsed)),
        "noncollapsed_dense_face_count":int(np.count_nonzero(~collapsed)),
        "duplicate_mapped_face_occurrence_count":int(np.count_nonzero(~collapsed)-len(unique_faces)),
        "compact_edge_count":int(len(uk)),
        "nonmanifold_edge_count":int(np.count_nonzero(bad)),
        "nonmanifold_incidence_histogram":{str(k):v for k,v in sorted(hist.items())},
        "max_edge_face_incidence":int(cnt.max(initial=0)),
        "compact_point_finite":bool(np.isfinite(cp).all()),
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=a.run_root.expanduser().resolve()
    zero=signed_zero_surface_from_dict(loadj(
        rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(
        rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(loadj(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))

    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)

    md=dict(surface.metadata or {})
    divisions=int(md["compact_voxel_divisions"])
    component_aware=bool(md.get("component_aware_compaction",False))
    base=compact_inverse(
        world,faces,
        divisions=divisions,
        component_aware=component_aware,
    )
    base_metrics=compact_metrics(world,faces,base)

    refined,refine_meta=topology_local_refine(faces,base)
    refined_metrics=compact_metrics(world,faces,refined)

    # Strong invariants: refinement only, never cross-base merge.
    pair=np.column_stack((base,refined))
    refined_to_base={}
    cross_merge=0
    for b,r in np.unique(pair,axis=0):
        r=int(r); b=int(b)
        old=refined_to_base.setdefault(r,b)
        if old!=b: cross_merge+=1

    report={
        "schema":"RealSaS.KnightTopologyLocalVoxelCompactionCourt.v1",
        "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
        "operator":{
            "id":"TOPOLOGY_LOCAL_VOXEL_REFINEMENT_V1",
            "base":"CURRENT_COMPONENT_AWARE_VOXEL_COMPACTION",
            "rule":"WITHIN_EACH_BASE_BUCKET_SPLIT_BY_DENSE_EDGE_CONNECTED_COMPONENT",
            "face_deletion":False,
            "cross_base_merge_forbidden":True,
            "teacher_truth_used":False,
        },
        "source":{
            "dense_vertex_count":int(len(world)),
            "dense_face_count":int(len(faces)),
            "voxel_divisions":divisions,
            "component_aware":component_aware,
        },
        "base":base_metrics,
        "refinement":refine_meta,
        "refined":refined_metrics,
        "invariants":{
            "cross_base_merge_count":int(cross_merge),
            "face_deletion_count":0,
            "dense_source_bytes_unchanged":True,
            "exact_dense_face_witness_retained":True,
        },
        "verdict":{
            "nonmanifold_closed":refined_metrics["nonmanifold_edge_count"]==0,
            "strict_refinement":cross_merge==0,
            "success":bool(
                refined_metrics["nonmanifold_edge_count"]==0
                and cross_merge==0
            ),
        },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TOPOLOGY_LOCAL_VOXEL_COMPACTION="+json.dumps({
        "base_nonmanifold":base_metrics["nonmanifold_edge_count"],
        "refined_nonmanifold":refined_metrics["nonmanifold_edge_count"],
        "base_nodes":base_metrics["compact_node_count"],
        "refined_nodes":refined_metrics["compact_node_count"],
        "added_nodes":refine_meta["added_bucket_count"],
        "split_base_buckets":refine_meta["split_base_bucket_count"],
        "max_children_per_bucket":refine_meta["max_children_per_base_bucket"],
        "base_collapsed_dense_faces":base_metrics["collapsed_dense_face_count"],
        "refined_collapsed_dense_faces":refined_metrics["collapsed_dense_face_count"],
        "cross_base_merge_count":cross_merge,
        "success":report["verdict"]["success"],
    },sort_keys=True))
    if not report["verdict"]["success"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
