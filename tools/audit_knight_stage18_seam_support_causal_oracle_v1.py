from __future__ import annotations

import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict, mesh_policy_from_dict,
    qualified_camera_set_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_holeless_partitioned_dense_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import run_g3_local_frame_micro_stress_v2
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO, run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, build_component_carrier_policy,
    canonical_mesh_candidate_lineage_hash, validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix
from tools.audit_knight_probe_conditioned_region_court_v1 import (
    probe_edge_risk, build_probe_partition,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics
from tools.demo.render_knight_motion_preview_v1 import _ctx


def interpolated_seam_skin(candidate):
    verts=[]
    changed=0
    for v in candidate.vertices:
        b=v.support_binding
        if str(b.mode)!="SEAM_GEOMETRY_INTERPOLATION":
            verts.append(v); continue
        md={**dict(b.metadata or {}), "skin_support_coefficients": tuple(b.coefficients),
            "audit_override":"GEOMETRY_INTERPOLATED_SKIN"}
        nb=replace(b, metadata=md)
        verts.append(replace(v, support_binding=nb))
        changed+=1
    provisional=replace(candidate, vertices=tuple(verts), candidate_lineage_hash="")
    return replace(provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional)), changed


def evaluate(name,candidate,*,surface,skeleton,skin,envelope,cameras,policy,jids,rr,source_report):
    full=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,
        cameras=cameras,policy=policy,stress_all_faces=True)
    rest,w,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    motion=motion_metrics(np.asarray(rest),np.asarray(w),np.asarray(faces),jids,
        skeleton,cameras,rr,source_report)
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,
        cameras=cameras,policy=policy)
    return {
        "name":name,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "all_face_unsafe":int(full["unsafe_face_count"]),
        "all_face_passed":bool(full["passed"]),
        "g3_passed":bool(g3.passed),
        "g3_failures":list(g3.failure_invariants),
        "motion_gt10":int(motion["max_edge_gt_10"]),
        "motion_gt4":int(motion["max_edge_gt_4"]),
        "motion_worst":float(motion["worst_edge_max"]),
        "motion_p99":float(motion["max_edge_p99"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(
        load_json(rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")
    ).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    policy=mesh_policy_from_dict(load_json(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)
    sids,jids,W=skin_matrix(surface,skeleton,skin)

    risk=probe_edge_risk(surface,skeleton,cameras,envelope,sids,W)
    part,_,_,closure=build_probe_partition(surface,risk,max_edge_ratio=DEFAULT_MAX_EDGE_RATIO)
    carrier=build_component_carrier_policy(
        partition=part,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("SEAM_SUPPORT_CAUSAL_ORACLE",),
            metadata={"automatic":False,"audit_only":True}) for c in part.components),
        metadata={"audit_only":True})
    owner=build_holeless_partitioned_dense_candidate(
        surface,part,carrier,
        producer_policy_hash=content_sha256({"audit":"SEAM_SUPPORT_CAUSAL_ORACLE","arm":"OWNER_COPY"}),
        explicit_face_provenance=explicit)
    validate_canonical_mesh_candidate(owner,surface=surface,partition=part,carrier_policy=carrier)
    interp,changed=interpolated_seam_skin(owner)
    validate_canonical_mesh_candidate(interp,surface=surface,partition=part,carrier_policy=carrier)

    source_report=load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    owner_eval=evaluate("OWNER_COPY",owner,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,jids=jids,rr=rr,source_report=source_report)
    interp_eval=evaluate("GEOMETRY_INTERPOLATED_SKIN",interp,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,jids=jids,rr=rr,source_report=source_report)

    report={
      "schema":"RealSaS.KnightStage18SeamSupportCausalOracle.v1",
      "status":"COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
      "partition":{"component_count":len(part.components),"closure":closure},
      "changed_seam_vertex_count":changed,
      "face_provenance_replay":prov,
      "owner_copy":owner_eval,
      "geometry_interpolated_skin":interp_eval,
      "delta":{
        "all_face_unsafe":interp_eval["all_face_unsafe"]-owner_eval["all_face_unsafe"],
        "motion_gt10":interp_eval["motion_gt10"]-owner_eval["motion_gt10"],
        "motion_gt4":interp_eval["motion_gt4"]-owner_eval["motion_gt4"],
        "motion_worst":interp_eval["motion_worst"]-owner_eval["motion_worst"],
      },
      "claim_boundary":[
        "Stage17 probe-conditioned partition is identical between arms.",
        "Geometry, faces, QualifiedSkinIR and skeleton are identical between arms.",
        "Only SEAM_GEOMETRY_INTERPOLATION mechanical skin transfer changes.",
        "This is an audit oracle, not product authority."
      ]
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SEAM_SUPPORT_CAUSAL_ORACLE="+json.dumps(report,sort_keys=True))


if __name__=="__main__": main()
