from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components

def loadj(p): return json.loads(Path(p).read_text())

def bad_edges_and_links(faces,inv):
    mapped=inv[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    canon=np.sort(mapped[valid],axis=1)
    uniq=np.unique(canon,axis=0)
    edge_rows=defaultdict(list)
    vertex_neighbors=defaultdict(set)
    for fi,f in enumerate(uniq):
        a,b,c=map(int,f)
        for x,y in ((a,b),(b,c),(c,a)):
            e=(x,y) if x<y else (y,x)
            edge_rows[e].append(fi)
            vertex_neighbors[x].add(y);vertex_neighbors[y].add(x)
    bad={e:r for e,r in edge_rows.items() if len(r)>2}
    rows=[]
    for (u,v),fis in sorted(bad.items()):
        opp=sorted({int(x) for fi in fis for x in uniq[fi] if int(x) not in (u,v)})
        common=sorted(vertex_neighbors[u]&vertex_neighbors[v])
        rows.append({
            "edge":[u,v],
            "incidence":len(fis),
            "opposite":opp,
            "common_neighbors":common,
            "link_condition_exact":set(common)==set(opp) and len(opp)<=2,
        })
    return rows

def locally_decompact_offenders(faces,base_inv,inv,offender_clusters):
    # Generic fail-closed fallback: only dense vertices currently mapped to an
    # offender quotient vertex are released to singleton classes. All other
    # equivalence classes remain untouched. This cannot invent or delete faces.
    inv=np.asarray(inv,dtype=np.int64)
    base=np.asarray(base_inv,dtype=np.int64)
    offenders=set(map(int,offender_clusters))
    next_id=int(inv.max())+1
    out=inv.copy()
    released=0
    for vid in range(len(out)):
        if int(inv[vid]) in offenders:
            out[vid]=next_id
            next_id+=1
            released+=1
    # Canonicalize ids and then allow only dense-edge connectivity inside each
    # remaining class. Released singletons stay singletons.
    _,out=np.unique(out,return_inverse=True)
    out=np.asarray(out,dtype=np.int64)
    out,_=topology_local_refine(faces,out)
    return out,{"released_dense_vertex_count":released,"offender_cluster_count":len(offenders)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    md=dict(surface.metadata or {})
    base=compact_inverse(world,faces,divisions=int(md["compact_voxel_divisions"]),component_aware=bool(md.get("component_aware_compaction",False)))
    inv,_=topology_local_refine(faces,base)
    hist=[]
    # First close all edge-preimage disconnected cases to fixed point.
    for i in range(32):
        m=compact_metrics(world,faces,inv)
        hist.append({"phase":"EDGE_PREIMAGE","iteration":i,"nonmanifold":m["nonmanifold_edge_count"],"nodes":m["compact_node_count"]})
        if m["nonmanifold_edge_count"]==0: break
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        nm=compact_metrics(world,faces,nxt)["nonmanifold_edge_count"]
        if nm==m["nonmanifold_edge_count"]:
            inv=nxt
            break
        inv=nxt

    link_before=bad_edges_and_links(faces,inv)
    offender_clusters=sorted({x for row in link_before for x in row["edge"]})
    inv2,decomp=locally_decompact_offenders(faces,base,inv,offender_clusters)
    final=compact_metrics(world,faces,inv2)
    link_after=bad_edges_and_links(faces,inv2)

    # Strict refinement of original voxel compaction.
    pairs=np.unique(np.column_stack((base,inv2)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b): cross+=1

    report={
      "schema":"RealSaS.KnightLinkSafeLocalDecompactionCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "operator":{
        "id":"LINK_CONDITION_FAIL_CLOSED_LOCAL_DECOMPACTION_V1",
        "generic_rule":"WHEN_QUOTIENT_MANIFOLD_LINK_CONDITION_FAILS_AFTER_TOPOLOGY_LOCAL_REFINEMENT, RELEASE_ONLY_OFFENDING_EQUIVALENCE_CLASSES_TO_DENSE_SINGLETONS",
        "face_deletion":False,
        "teacher_or_subject_labels_used":False,
        "dense_source_geometry_mutated":False,
      },
      "history":hist,
      "link_before":link_before,
      "decompaction":decomp,
      "final":final,
      "link_after":link_after,
      "cross_base_merge_count":cross,
      "success":bool(final["nonmanifold_edge_count"]==0 and cross==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_LINK_SAFE_LOCAL_DECOMPACTION="+json.dumps({
      "pre_link_bad":len(link_before),
      "offender_clusters":len(offender_clusters),
      "released_dense_vertices":decomp["released_dense_vertex_count"],
      "final_nonmanifold":final["nonmanifold_edge_count"],
      "final_nodes":final["compact_node_count"],
      "remaining_bad_edges":len(link_after),
      "cross_base_merge_count":cross,
      "success":report["success"],
    },sort_keys=True))
    if not report["success"]: raise SystemExit(2)
if __name__=="__main__":main()
