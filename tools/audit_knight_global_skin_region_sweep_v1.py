from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_holeless_partitioned_dense_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import run_skin_topology_compatibility_v1
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics, stress_arbitrary_weights
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def pair(a,b):
    a,b=str(a),str(b)
    return (a,b) if a<b else (b,a)


def skin_matrix(surface, skeleton, skin):
    jids=tuple(j.canonical_joint_id for j in skeleton.joints)
    ji={x:i for i,x in enumerate(jids)}
    rows={str(r.surface_id):r for r in skin.rows}
    sids=tuple(str(n.surface_id) for n in surface.surface_nodes)
    W=np.zeros((len(sids),len(jids)),dtype=np.float64)
    for i,sid in enumerate(sids):
        r=rows[sid]
        for jid,w in r.influences:
            W[i,ji[str(jid)]]=float(w)
    return sids,jids,W


def threshold_partition(surface,sids,W,threshold):
    idx={sid:i for i,sid in enumerate(sids)}
    adj={sid:set() for sid in sids}
    edge_l1={}
    for r in surface.local_relations:
        p=pair(r.a_surface_id,r.b_surface_id)
        if p in edge_l1:
            continue
        d=float(np.abs(W[idx[p[0]]]-W[idx[p[1]]]).sum())
        edge_l1[p]=d
        if d<=threshold:
            adj[p[0]].add(p[1]); adj[p[1]].add(p[0])

    comp={}
    groups=[]
    for sid in sorted(sids):
        if sid in comp: continue
        ci=len(groups); stack=[sid]; comp[sid]=ci; members=[]
        while stack:
            x=stack.pop(); members.append(x)
            for y in sorted(adj[x]):
                if y not in comp:
                    comp[y]=ci; stack.append(y)
        groups.append(tuple(sorted(members)))

    crossing=sorted(p for p in edge_l1 if comp[p[0]]!=comp[p[1]])
    overrides=[]
    for p in crossing:
        l1=edge_l1[p]
        overrides.append(ComponentBoundaryConstraintIR(
            constraint_id="GLCUT:"+content_sha256({"pair":p,"threshold":threshold})[:20],
            a_surface_id=p[0], b_surface_id=p[1], decision="SEPARATE",
            evidence_refs=(f"SKIN_L1:{l1:.17g}",),
            confidence=min(1.0,max(0.0,l1/2.0)),
            metadata={
                "evidence_class":"GLOBAL_SKIN_FIELD_GRAPH_PARTITION_SWEEP",
                "threshold":float(threshold),
                "pairwise_skin_l1":float(l1),
                "automatic":False,
                "audit_only":True,
            },
        ))
    return tuple(groups),tuple(overrides),edge_l1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--arm",required=True)
    ap.add_argument("--thresholds",default="0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(load_json(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    policy=mesh_policy_from_dict(load_json(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    if skin.surface_binding_hash!=surface.geometry_lineage_hash: raise RuntimeError("SKIN_SURFACE_DRIFT")
    if skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash: raise RuntimeError("SKIN_SKELETON_DRIFT")

    explicit_faces,prov=replay_compacted_dense_face_provenance(rr,surface)
    sids,jids,W=skin_matrix(surface,skeleton,skin)
    source_report=load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))

    rows=[]
    for threshold in tuple(float(x) for x in a.thresholds.split(",")):
        groups,overrides,edge_l1=threshold_partition(surface,sids,W,threshold)
        part=build_structural_partition(surface,boundary_overrides=overrides)
        if len(part.components)!=len(groups):
            raise RuntimeError(f"REGION_COUNT_DRIFT::{threshold}::{len(groups)}::{len(part.components)}")
        carrier=build_component_carrier_policy(
            partition=part,
            decisions=tuple(ComponentCarrierDecisionIR(
                c.component_id,"MESH",("GLOBAL_SKIN_FIELD_GRAPH_PARTITION_SWEEP",),
                metadata={"automatic":False,"audit_only":True}
            ) for c in part.components),
            metadata={"audit_only":True,"threshold":threshold},
        )
        candidate=build_holeless_partitioned_dense_candidate(
            surface,part,carrier,
            producer_policy_hash=content_sha256({"audit":"GLOBAL_SKIN_FIELD_GRAPH_PARTITION_SWEEP","threshold":threshold,"arm":a.arm}),
            explicit_face_provenance=explicit_faces,
        )
        validate_canonical_mesh_candidate(candidate,surface=surface,partition=part,carrier_policy=carrier)
        comp=run_skin_topology_compatibility_v1(
            candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy
        )

        rest,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
        stress=stress_arbitrary_weights(
            np.asarray(rest,dtype=np.float64),np.asarray(weights,dtype=np.float64),
            np.asarray(faces,dtype=np.int64),jids,skeleton,cameras,envelope,policy
        )
        motion=motion_metrics(
            np.asarray(rest,dtype=np.float64),np.asarray(weights,dtype=np.float64),
            np.asarray(faces,dtype=np.int64),jids,skeleton,cameras,rr,source_report
        )
        sizes=sorted((len(c.surface_ids) for c in part.components),reverse=True)
        l1vals=np.asarray(list(edge_l1.values()),dtype=np.float64)
        rows.append({
            "threshold":threshold,
            "component_count":len(part.components),
            "largest_component_size":sizes[0],
            "singleton_component_count":sum(x==1 for x in sizes),
            "small_le4_component_count":sum(x<=4 for x in sizes),
            "separate_boundary_count":sum(x.decision=="SEPARATE" for x in part.boundary_constraints),
            "edge_l1_p50":float(np.quantile(l1vals,.5)),
            "edge_l1_p90":float(np.quantile(l1vals,.9)),
            "edge_l1_p95":float(np.quantile(l1vals,.95)),
            "edge_l1_p99":float(np.quantile(l1vals,.99)),
            "candidate_vertex_count":len(candidate.vertices),
            "candidate_face_count":len(candidate.faces),
            "mixed_source_face_count":int(candidate.metadata["mixed_source_face_count"]),
            "face_deletion_count":int(candidate.metadata["face_deletion_count"]),
            "rest_area_relative_error":float(candidate.metadata["rest_area_relative_error"]),
            "g3b_passed":bool(comp["passed"]),
            "g3b_risky":int(comp["risky_face_count"]),
            "g3b_unsafe":int(comp["unsafe_face_count"]),
            "stress_unsafe":int(stress["unsafe_face_count"]),
            "motion_gt10":int(motion["max_edge_gt_10"]),
            "motion_gt4":int(motion["max_edge_gt_4"]),
            "motion_worst":float(motion["worst_edge_max"]),
            "motion_p99":float(motion["max_edge_p99"]),
        })
        print("GLOBAL_REGION_SWEEP_ROW="+json.dumps(rows[-1],sort_keys=True),flush=True)

    report={
        "schema":"RealSaS.KnightGlobalSkinFieldRegionSweep.v1",
        "status":"COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
        "arm":a.arm,
        "face_provenance_replay":prov,
        "rows":rows,
        "claim_boundary":[
            "This sweep tests a global graph partition owner for Stage35 evidence.",
            "Regions are connected components after retaining only local-relation edges with endpoint skin L1 <= threshold.",
            "Every graph edge crossing a derived region label is emitted as SEPARATE, satisfying Stage17 cut semantics by construction.",
            "No parent product state is mutated."
        ]
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
