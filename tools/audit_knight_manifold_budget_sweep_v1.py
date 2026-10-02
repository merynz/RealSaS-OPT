from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict,signed_zero_surface_from_dict
from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact,mesh_connected_component_labels_v1
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components
from tools.audit_knight_link_safe_local_decompaction_v1 import bad_edges_and_links,locally_decompact_offenders

def loadj(p): return json.loads(Path(p).read_text())

def run_one(world,faces,normals,labels,target):
    cp,cn,edges,divisions,base=_adaptive_voxel_compact(
        world,faces,normals,
        target_nodes=int(target),
        preserve_connected_components=True,
        precomputed_component_labels=labels,
    )
    inv,meta0=topology_local_refine(faces,np.asarray(base,dtype=np.int64))
    history=[compact_metrics(world,faces,inv)["nonmanifold_edge_count"]]
    for _ in range(24):
        if history[-1]==0: break
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        nm=compact_metrics(world,faces,nxt)["nonmanifold_edge_count"]
        inv=nxt;history.append(nm)
        if len(history)>=2 and history[-1]==history[-2]:
            break
    pre=bad_edges_and_links(faces,inv)
    if pre:
        offenders=sorted({x for row in pre for x in row["edge"]})
        inv,decomp=locally_decompact_offenders(faces,np.asarray(base,dtype=np.int64),inv,offenders)
    else:
        decomp={"released_dense_vertex_count":0,"offender_cluster_count":0}
    final=compact_metrics(world,faces,inv)
    return {
      "target_nodes":int(target),
      "base_actual_nodes":int(len(cp)),
      "voxel_divisions":int(divisions),
      "after_topology_local_nodes":int(meta0["refined_bucket_count"]),
      "edge_preimage_history":history,
      "pre_link_bad_edge_count":len(pre),
      "released_dense_vertices":int(decomp["released_dense_vertex_count"]),
      "final_nodes":int(final["compact_node_count"]),
      "final_nonmanifold":int(final["nonmanifold_edge_count"]),
      "final_collapsed_dense_faces":int(final["collapsed_dense_face_count"]),
      "within_12288":bool(final["compact_node_count"]<=12288),
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    # Dummy normals are sufficient: this court measures topology/budget only.
    normals=np.zeros_like(world);normals[:,2]=1.0
    labels=mesh_connected_component_labels_v1(len(world),faces)
    targets=(4096,6144,8192,9216,10240,11008,11520,11776,12032,12288)
    rows=[]
    for t in targets:
        row=run_one(world,faces,normals,labels,t)
        rows.append(row)
        print("BUDGET_ROW="+json.dumps(row,sort_keys=True))
    feasible=[r for r in rows if r["final_nonmanifold"]==0 and r["within_12288"]]
    report={"schema":"RealSaS.KnightManifoldBudgetSweep.v1","status":"MEASURED_AUDIT_ONLY","rows":rows,
            "feasible":feasible,"success":bool(feasible)}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_MANIFOLD_BUDGET_SWEEP="+json.dumps({"success":bool(feasible),"feasible":feasible},sort_keys=True))
    if not feasible: raise SystemExit(2)
if __name__=="__main__":main()
