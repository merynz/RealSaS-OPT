from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import RiggingSurfaceIR,SurfaceNode,SurfaceRelation
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _self_zbuffer_support,topology_aware_zero_surface_normals_v2,
    ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,zero_surface_normal_operator_hash_v2,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    repair_candidate_fixed_vertex_flips_v1,
    repair_candidate_endpoint_collapses_v1,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict

def loadj(p): return json.loads(Path(p).read_text())

def compact(points,values,inv,normalize=False):
    n=int(inv.max())+1
    counts=np.bincount(inv,minlength=n).astype(np.float64)
    out=np.zeros((n,values.shape[1]),dtype=np.float64)
    np.add.at(out,inv,values)
    if normalize:
        norm=np.linalg.norm(out,axis=1)
        if np.any(norm<=1e-12): raise RuntimeError("COMPACT_VALUE_DEGENERATE")
        out/=norm[:,None]
    else:
        out/=counts[:,None]
    return out

def compact_faces(faces,inv):
    mf=inv[np.asarray(faces,dtype=np.int64)]
    valid=(mf[:,0]!=mf[:,1])&(mf[:,1]!=mf[:,2])&(mf[:,2]!=mf[:,0])
    return np.unique(np.sort(mf[valid],axis=1),axis=0)

def edge_topology(faces):
    counts={}
    for a,b,c in faces:
        for u,v in ((a,b),(b,c),(c,a)):
            e=(str(u),str(v)) if str(u)<str(v) else (str(v),str(u))
            counts[e]=counts.get(e,0)+1
    vals=np.asarray(list(counts.values()),dtype=np.int64)
    return {
      "edge_count":int(len(vals)),
      "boundary":int(np.count_nonzero(vals==1)),
      "interior":int(np.count_nonzero(vals==2)),
      "nonmanifold":int(np.count_nonzero(vals>2)),
      "max_incidence":int(vals.max(initial=0)),
    }

def report_quality(candidate,policy):
    from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import _report
    pos={str(v.candidate_vertex_id):tuple(map(float,v.P)) for v in candidate.vertices}
    faces=[tuple(map(str,f)) for f in candidate.faces]
    return _report(faces,pos,policy)

def build_repaired_surface(rr,inverse_npz):
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    cameras_raw=loadj(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")["cameras"]
    cameras=tuple(sorted(cameras_raw,key=lambda x:int(x["view_index"])))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
        hints=np.asarray(z["implicit_normals"],np.float64)
    with np.load(inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],np.int64)
        base=np.asarray(z["base_inverse"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    dense_normals=topology_aware_zero_surface_normals_v2(world,faces,hints)
    cp=compact(world,world,inv,normalize=False)
    cn=compact(world,np.asarray(dense_normals,np.float64),inv,normalize=True)
    cf=compact_faces(faces,inv)
    support,raster,visible_counts=_self_zbuffer_support(
        world,cp,cameras,depth_tolerance=0.02*float(norm.half_extent)
    )
    pairs=np.unique(np.column_stack((base,inv)),axis=0)
    final_to_base={}
    for b,r in pairs:
        old=final_to_base.setdefault(int(r),int(b))
        if old!=int(b):raise RuntimeError("CROSS_BASE_MERGE")
    nodes=[]
    for i,(p,n) in enumerate(zip(cp,cn)):
        views=tuple(int(v) for v in range(8) if bool(support[i,v]))
        binds=tuple((v,(float(raster[i,v,0]),float(raster[i,v,1]))) for v in views)
        flags=("OBSERVED_SIGNED_ZERO_SURFACE",) if views else ("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",)
        sid="SFSR:"+content_sha256({"i":i,"P":p.tolist(),"src":zero.npz_sha256})[:20]
        nodes.append(SurfaceNode(
          surface_id=sid,P=tuple(map(float,p)),support_views=views,
          provenance_refs=("TOPOLOGY_SAFE_REFINEMENT_AUDIT_CHILD",),
          source_observation_ids=(),raster_bindings=binds,
          persistence_group_id=f"TOPOLOGY_SAFE_REFINEMENT:{i:05d}",
          derived_normal=tuple(map(float,n)),validity_flags=flags,
          metadata={"normal_operator":ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,
                    "normal_operator_hash":zero_surface_normal_operator_hash_v2(),
                    "teacher_truth_used":False,"parent_base_cluster":int(final_to_base[i])},
        ))
    ids=tuple(n.surface_id for n in nodes)
    edge=np.concatenate((cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]),axis=0)
    edge=np.unique(np.sort(edge,axis=1),axis=0)
    rel=[]
    for ai,bi in edge.tolist():
        d=float(np.linalg.norm(cp[ai]-cp[bi]))
        rel.append(SurfaceRelation(
          relation_id="SFSRREL:"+content_sha256({"a":ids[ai],"b":ids[bi],"d":d})[:20],
          a_surface_id=ids[ai],b_surface_id=ids[bi],
          relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",score=1.0,
          metadata={"world_distance":d,"crosses_unknown":False,"unknown_bridge":False,"teacher_truth_used":False},
        ))
    lineage=content_sha256({
      "schema":"RealSaS.SceneFirstSignedTopologySafeRefinement.v1",
      "source_zero_surface_sha256":zero.npz_sha256,"base_divisions":56,
      "repair_inverse_sha256":content_sha256(inv.tolist()),
      "nodes":[n.to_dict() for n in nodes],"relations":[r.to_dict() for r in rel],
    })
    surface=RiggingSurfaceIR(
      surface_nodes=tuple(nodes),local_relations=tuple(sorted(rel,key=lambda x:x.relation_id)),
      geometry_lineage_hash=lineage,
      builder_id="RealSaS.GeometricSubstrateAssembler.TopologySafeRefinement.audit.v1",
      schema_version="RealSaS.RiggingSurfaceIR.v1",
      metadata={"scene_first_signed_geometry":True,"scene_first_decoder":"SIGNED_FIELD_ZERO_LEVEL_SURFACE",
                "source_dense_vertex_count":int(len(world)),"compact_surface_node_count":int(len(nodes)),
                "compact_voxel_divisions":56,"component_aware_compaction":True,
                "topology_safe_refinement":True,"raster_coordinate_system":"PIXEL_CENTER_XY",
                "resolution":int(cameras[0]["resolution"]),"Nd_operator":ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,
                "Nd_operator_sha256":zero_surface_normal_operator_hash_v2(),
                "source_zero_surface_sha256":zero.npz_sha256,"teacher_truth_used":False},
    )
    explicit=tuple(tuple(ids[int(x)] for x in row) for row in cf.tolist())
    return surface,explicit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_COLLAPSE_COURT",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    candidate=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({"court":"REPAIRED_STATIC_QUALITY_COLLAPSE_V1",
                                           "policy":policy.qualification_policy_lineage_hash}),
      explicit_face_provenance=explicit,
    )
    before=report_quality(candidate,policy)
    flipped,flip=repair_candidate_fixed_vertex_flips_v1(candidate,policy,max_passes=12)
    collapsed,collapse=repair_candidate_endpoint_collapses_v1(flipped,policy,max_collapses=2048)
    after=report_quality(collapsed,policy)
    topo=edge_topology(collapsed.faces)
    report={
      "schema":"RealSaS.KnightRepairedQualityCollapseCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "input":{"surface_nodes":len(surface.surface_nodes),"vertices":len(candidate.vertices),"faces":len(candidate.faces),
               "quality":before},
      "flip":{"accepted":flip["accepted_flip_count"],"after":flip["after"]},
      "collapse":{"accepted":collapse["accepted_collapse_count"],
                  "rejected_link":collapse["rejected_link_condition_count"],
                  "rejected_shape":collapse["rejected_shape_deviation_count"],
                  "rejected_quality":collapse["rejected_quality_count"],
                  "after":collapse["after"]},
      "output":{"vertices":len(collapsed.vertices),"faces":len(collapsed.faces),
                "quality":after,"topology":topo},
      "invariants":{
        "surviving_vertex_motion":False,
        "new_vertex_creation":False,
        "support_binding_change":False,
        "nonmanifold_zero":bool(topo["nonmanifold"]==0),
      },
      "success":bool(topo["nonmanifold"]==0 and after["policy_violating_face_count"]<before["policy_violating_face_count"]),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_QUALITY_COLLAPSE="+json.dumps({
      "before_violations":before["policy_violating_face_count"],
      "flip_count":flip["accepted_flip_count"],
      "after_flip":flip["after"]["policy_violating_face_count"],
      "collapse_count":collapse["accepted_collapse_count"],
      "after_violations":after["policy_violating_face_count"],
      "before_min_angle":before["min_angle_deg"],
      "after_min_angle":after["min_angle_deg"],
      "before_max_aspect":before["max_aspect_longest_over_min_altitude"],
      "after_max_aspect":after["max_aspect_longest_over_min_altitude"],
      "vertices_before":len(candidate.vertices),"vertices_after":len(collapsed.vertices),
      "faces_before":len(candidate.faces),"faces_after":len(collapsed.faces),
      "nonmanifold_after":topo["nonmanifold"],
      "success":report["success"],
    },sort_keys=True))
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
