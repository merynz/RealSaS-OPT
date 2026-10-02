from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import RiggingSurfaceIR,SurfaceNode,SurfaceRelation
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _self_zbuffer_support,mesh_connected_component_labels_v1,
    topology_aware_zero_surface_normals_v2,
    ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,zero_surface_normal_operator_hash_v2,
)
from compiler.realsas_compiler_core.substrate.adequacy_v1 import _metrics as substrate_metrics
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
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

def compact_face_indices(faces,inv):
    mf=np.asarray(inv,dtype=np.int64)[np.asarray(faces,dtype=np.int64)]
    valid=(mf[:,0]!=mf[:,1])&(mf[:,1]!=mf[:,2])&(mf[:,2]!=mf[:,0])
    raw=np.sort(mf[valid],axis=1)
    uniq=np.unique(raw,axis=0)
    return uniq,{"dense_face_count":int(len(faces)),"nondegenerate_mapped_dense_face_count":int(np.count_nonzero(valid)),
                "unique_compact_face_count":int(len(uniq)),
                "collapsed_or_duplicate_witness_count":int(len(faces)-len(uniq))}

def triangle_quality(points,faces):
    p=np.asarray(points,np.float64);f=np.asarray(faces,np.int64)
    tri=p[f]
    e0=np.linalg.norm(tri[:,1]-tri[:,0],axis=1)
    e1=np.linalg.norm(tri[:,2]-tri[:,1],axis=1)
    e2=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
    edges=np.stack((e0,e1,e2),axis=1)
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    deg=(~np.isfinite(area2))|(area2<=1e-15)
    safe=np.maximum(edges,1e-15)
    # angle at v0 opposite e1, at v1 opposite e2, at v2 opposite e0
    a=e1;b=e2;c=e0
    cos0=np.clip((b*b+c*c-a*a)/(2*b*c+1e-30),-1,1)
    cos1=np.clip((a*a+c*c-b*b)/(2*a*c+1e-30),-1,1)
    cos2=np.clip((a*a+b*b-c*c)/(2*a*b+1e-30),-1,1)
    angles=np.degrees(np.arccos(np.stack((cos0,cos1,cos2),axis=1)))
    minang=np.min(angles,axis=1)
    longest=np.max(edges,axis=1)
    min_alt=np.divide(area2,longest,out=np.zeros_like(area2),where=longest>1e-15)
    aspect=np.divide(longest,min_alt,out=np.full_like(longest,np.inf),where=min_alt>1e-15)
    edge_ratio=np.max(edges,axis=1)/np.maximum(np.min(edges,axis=1),1e-15)
    def q(x,pct): return float(np.percentile(x[np.isfinite(x)],pct)) if np.any(np.isfinite(x)) else None
    return {
      "face_count":int(len(f)),
      "degenerate_face_count":int(np.count_nonzero(deg)),
      "minimum_angle_deg":float(np.min(minang)) if len(minang) else None,
      "angle_p01_deg":q(minang,1),"angle_p05_deg":q(minang,5),"angle_p50_deg":q(minang,50),
      "maximum_aspect_longest_over_min_altitude":float(np.max(aspect)) if len(aspect) else None,
      "aspect_p95":q(aspect,95),"aspect_p99":q(aspect,99),
      "maximum_edge_ratio":float(np.max(edge_ratio)) if len(edge_ratio) else None,
      "edge_ratio_p99":q(edge_ratio,99),
      "minimum_double_area":float(np.min(area2)) if len(area2) else None,
      "double_area_p01":q(area2,1),
    }

def edge_topology(faces):
    counts={}
    for a,b,c in np.asarray(faces,dtype=np.int64):
        for u,v in ((a,b),(b,c),(c,a)):
            e=(int(u),int(v)) if u<v else (int(v),int(u))
            counts[e]=counts.get(e,0)+1
    vals=np.asarray(list(counts.values()),dtype=np.int64)
    return {
      "edge_count":int(len(counts)),
      "boundary_edge_count":int(np.count_nonzero(vals==1)),
      "manifold_interior_edge_count":int(np.count_nonzero(vals==2)),
      "nonmanifold_edge_count":int(np.count_nonzero(vals>2)),
      "maximum_edge_incidence":int(vals.max(initial=0)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    cameras_raw=loadj(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")["cameras"]
    cameras=tuple(sorted(cameras_raw,key=lambda x:int(x["view_index"])))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
        hints=np.asarray(z["implicit_normals"],np.float64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],np.int64)
        base=np.asarray(z["base_inverse"],np.int64)
    if inv.shape!=(len(vn),) or base.shape!=inv.shape:
        raise RuntimeError("REPAIRED_INVERSE_SHAPE_DRIFT")
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    dense_normals=topology_aware_zero_surface_normals_v2(world,faces,hints)
    cp=compact(world,world,inv,normalize=False)
    cn=compact(world,np.asarray(dense_normals,np.float64),inv,normalize=True)
    compact_faces,face_replay=compact_face_indices(faces,inv)

    # Strict refinement map final cluster -> one exact historical base cluster.
    final_to_base={}
    for b,r in np.unique(np.column_stack((base,inv)),axis=0):
        old=final_to_base.setdefault(int(r),int(b))
        if old!=int(b): raise RuntimeError("STATIC_COURT_CROSS_BASE_MERGE")

    support,raster,visible_counts=_self_zbuffer_support(
        world,cp,cameras,depth_tolerance=0.02*float(norm.half_extent)
    )
    nodes=[]
    for i,(p,n) in enumerate(zip(cp,cn)):
        views=tuple(int(v) for v in range(8) if bool(support[i,v]))
        binds=tuple((v,(float(raster[i,v,0]),float(raster[i,v,1]))) for v in views)
        flags=("OBSERVED_SIGNED_ZERO_SURFACE",) if views else ("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",)
        sid="SFSR:"+content_sha256({"i":int(i),"P":p.tolist(),"source":zero.npz_sha256,"base":int(final_to_base[i])})[:20]
        nodes.append(SurfaceNode(
            surface_id=sid,P=tuple(map(float,p)),support_views=views,
            provenance_refs=("IRIS_SCENE_FIRST_SIGNED_ZERO_SURFACE_V2","TOPOLOGY_SAFE_REFINEMENT_AUDIT_CHILD"),
            source_observation_ids=(),raster_bindings=binds,
            persistence_group_id=f"TOPOLOGY_SAFE_REFINEMENT:{i:05d}",
            derived_normal=tuple(map(float,n)),validity_flags=flags,
            metadata={"normal_operator":ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,
                      "normal_operator_hash":zero_surface_normal_operator_hash_v2(),
                      "teacher_truth_used":False,
                      "parent_base_cluster":int(final_to_base[i])},
        ))
    ids=tuple(n.surface_id for n in nodes)
    edge=np.concatenate((compact_faces[:,[0,1]],compact_faces[:,[1,2]],compact_faces[:,[2,0]]),axis=0)
    edge=np.unique(np.sort(edge,axis=1),axis=0)
    rel=[]
    for aidx,bidx in edge.tolist():
        d=float(np.linalg.norm(cp[aidx]-cp[bidx]))
        rel.append(SurfaceRelation(
            relation_id="SFSRREL:"+content_sha256({"a":ids[aidx],"b":ids[bidx],"d":d})[:20],
            a_surface_id=ids[aidx],b_surface_id=ids[bidx],
            relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",score=1.0,
            metadata={"world_distance":d,"crosses_unknown":False,"unknown_bridge":False,
                      "teacher_truth_used":False},
        ))
    lineage=content_sha256({
      "schema":"RealSaS.SceneFirstSignedTopologySafeRefinement.v1",
      "source_zero_surface_sha256":zero.npz_sha256,
      "base_divisions":56,
      "repair_inverse_sha256":content_sha256(inv.tolist()),
      "nodes":[n.to_dict() for n in nodes],
      "relations":[r.to_dict() for r in rel],
    })
    surface=RiggingSurfaceIR(
      surface_nodes=tuple(nodes),local_relations=tuple(sorted(rel,key=lambda x:x.relation_id)),
      geometry_lineage_hash=lineage,
      builder_id="RealSaS.GeometricSubstrateAssembler.TopologySafeRefinement.audit.v1",
      schema_version="RealSaS.RiggingSurfaceIR.v1",
      metadata={
        "scene_first_signed_geometry":True,"scene_first_decoder":"SIGNED_FIELD_ZERO_LEVEL_SURFACE",
        "source_dense_vertex_count":int(len(world)),"compact_surface_node_count":int(len(nodes)),
        "compact_voxel_divisions":56,"compact_target_nodes":12288,
        "component_aware_compaction":True,"topology_safe_refinement":True,
        "observed_node_count":int(np.any(support,axis=1).sum()),
        "completed_node_count":int((~np.any(support,axis=1)).sum()),
        "visibility_support_counts_by_view":visible_counts,
        "raster_coordinate_system":"PIXEL_CENTER_XY","resolution":int(cameras[0]["resolution"]),
        "Nd_operator":ZERO_SURFACE_NORMAL_OPERATOR_V2_ID,
        "Nd_operator_sha256":zero_surface_normal_operator_hash_v2(),
        "source_zero_surface_sha256":zero.npz_sha256,
        "teacher_truth_used":False,"full_3d_intermediate_allowed":True,
        "node_budget_exceeded_for_topology_closure":bool(len(nodes)>12288),
      },
    )

    dense_labels=mesh_connected_component_labels_v1(len(world),faces)
    dense_support,dense_raster,_=_self_zbuffer_support(
        world,world,cameras,depth_tolerance=0.02*float(norm.half_extent)
    )
    pdoc=loadj(Path("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V3_20260929.json"))
    adequacy=substrate_metrics(
      surface=surface,dense_world=world,dense_normals=np.asarray(dense_normals,np.float64),
      dense_labels=dense_labels,dense_support=dense_support,dense_raster=dense_raster,
      normalization_half_extent=float(norm.half_extent),policy=pdoc["gsa"]["adequacy_policy"],
    )

    explicit=tuple(tuple(ids[int(x)] for x in row) for row in compact_faces.tolist())
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("STATIC_REPAIRED_TOPOLOGY_COURT",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    candidate=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({"court":"REPAIRED_STATIC_MESH_QUALITY_V1","policy":policy.qualification_policy_lineage_hash}),
      explicit_face_provenance=explicit,
    )
    validate_canonical_mesh_candidate(candidate,surface=surface,partition=partition,carrier_policy=carrier)

    rawq=triangle_quality(cp,compact_faces)
    rawtop=edge_topology(compact_faces)
    cindex={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    cpoints=np.asarray([v.P for v in candidate.vertices],np.float64)
    cfaces=np.asarray([[cindex[str(x)] for x in f] for f in candidate.faces],np.int64)
    candq=triangle_quality(cpoints,cfaces)
    candtop=edge_topology(cfaces)
    min_angle=float(policy.g3_min_angle_deg)
    max_aspect=float(policy.g3_max_aspect_longest_over_min_altitude)
    static_policy_pass=bool(
      candq["degenerate_face_count"]==0 and
      candtop["nonmanifold_edge_count"]==0 and
      candq["minimum_angle_deg"]>=min_angle and
      candq["maximum_aspect_longest_over_min_altitude"]<=max_aspect and
      int(dict(candidate.metadata or {}).get("rejected_explicit_face_count",0))==0
    )
    report={
      "schema":"RealSaS.KnightRepairedStaticMeshQualityCourt.v1",
      "status":"PASS_STATIC_REPAIRED_MESH_QUALITY" if static_policy_pass else "FAIL_STATIC_REPAIRED_MESH_QUALITY",
      "source_run_id":"SUBJECT2_KNIGHT_SOLVED_LINEAGE_V1_20260929",
      "repaired_surface":{"lineage":surface.geometry_lineage_hash,"nodes":len(nodes),"relations":len(rel),
                          "node_budget_max":12288,"node_budget_pass":len(nodes)<=12288},
      "dense_face_replay":face_replay,
      "substrate_adequacy":adequacy,
      "raw_compact_mesh":{"quality":rawq,"topology":rawtop},
      "stage18_baseline_candidate":{
        "lineage":candidate.candidate_lineage_hash,"vertices":len(candidate.vertices),"faces":len(candidate.faces),
        "quality":candq,"topology":candtop,"metadata":dict(candidate.metadata or {}),
      },
      "policy":{"min_angle_deg":min_angle,"max_aspect_longest_over_min_altitude":max_aspect},
      "static_policy_pass":static_policy_pass,
      "claim_boundary":[
        "This court measures repaired static geometry before corrected skin-conditioned Stage17 repartition.",
        "Dynamic deformation quality is not claimed here.",
        "Node-budget policy is reported separately from geometric quality and is not relaxed.",
      ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_STATIC_MESH_QUALITY="+json.dumps({
      "status":report["status"],"nodes":len(nodes),
      "raw_nonmanifold":rawtop["nonmanifold_edge_count"],
      "candidate_nonmanifold":candtop["nonmanifold_edge_count"],
      "min_angle":candq["minimum_angle_deg"],"max_aspect":candq["maximum_aspect_longest_over_min_altitude"],
      "rejected_explicit_faces":int(dict(candidate.metadata or {}).get("rejected_explicit_face_count",0)),
      "adequacy_pass":bool(adequacy["passed"]),"node_budget_pass":len(nodes)<=12288,
    },sort_keys=True))
    if not static_policy_pass: raise SystemExit(2)

if __name__=="__main__":main()
