from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
    run_skin_topology_compatibility_v1,
)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path, codec):
    if not path.is_file():
        raise RuntimeError(f"STAGE35_REPLAY_ARTIFACT_MISSING::{path}")
    return codec(json.loads(path.read_text()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=a.authority_root/a.run_id
    surface=load(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",
        rigging_surface_from_dict,
    )
    partition=load(
        rr/"artifacts/17_MECHANICAL_PARTITION_QUALIFIED/mechanical_partition.json",
        mechanical_partition_from_dict,
    )
    candidate=load(
        rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",
        canonical_mesh_candidate_from_dict,
    )
    policy=load(
        rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",
        mesh_policy_from_dict,
    )
    skeleton=load(
        rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
        qualified_skeleton_from_dict,
    )
    skin=load(a.skin_json,qualified_skin_from_dict)
    cameras=load(
        rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",
        qualified_camera_set_from_dict,
    ).cameras
    envelope=load(
        rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",
        deformation_envelope_from_dict,
    )

    if skin.surface_binding_hash!=surface.geometry_lineage_hash:
        raise RuntimeError("STAGE35_REPLAY_SKIN_SURFACE_LINEAGE_DRIFT")
    if skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise RuntimeError("STAGE35_REPLAY_SKIN_SKELETON_LINEAGE_DRIFT")

    g3=run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    compatibility=run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    directive=None
    if not compatibility["passed"]:
        directive=propose_mechanical_repartition_directive_v2(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            partition=partition,
            compatibility_report=compatibility,
        )

    report={
        "schema":"RealSaS.KnightStage35ExactSkinReplay.v1",
        "status":"PASS_REPLAY_EXECUTED",
        "run_id":a.run_id,
        "skin_json_sha256":sha256(a.skin_json),
        "skin_lineage_hash":skin.skin_lineage_hash,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "partition_lineage_hash":partition.partition_lineage_hash,
        "g3":{
            "passed":bool(g3.passed),
            "report_hash":g3.report_hash,
            "unsafe_face_count":int(len(g3.unsafe_face_indices)),
        },
        "g3b":{
            "passed":bool(compatibility["passed"]),
            "report_hash":compatibility["report_hash"],
            "risky_face_count":int(compatibility["risky_face_count"]),
            "unsafe_face_count":int(compatibility["unsafe_face_count"]),
            "top_unsafe_faces":compatibility["top_unsafe_faces"],
            "weight_mutation":bool(compatibility["weight_mutation"]),
        },
        "repartition":(
            None if directive is None else {
                "status":directive["status"],
                "directive_hash":directive["directive_hash"],
                "candidate_separate_pair_count":int(directive["candidate_separate_pair_count"]),
                "unresolved_unsafe_face_count":int(directive["unresolved_unsafe_face_count"]),
                "face_deletion_count":int(directive["face_deletion_count"]),
                "weight_mutation":bool(directive["weight_mutation"]),
                "auto_apply_allowed":bool(directive["auto_apply_allowed"]),
                "requires_trustworthy_skin_reliability_authority":bool(
                    directive["requires_trustworthy_skin_reliability_authority"]
                ),
                "restart_from":directive["restart_from"],
                "mandatory_requalification_through":directive["mandatory_requalification_through"],
            }
        ),
        "claim_boundary":[
            "This replays the current Stage35 G3/G3B mechanics on the supplied QualifiedSkinIR.",
            "A generated repartition directive remains diagnostic and authorization-gated.",
            "No topology repair, weight mutation, face deletion, or product promotion is performed.",
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_EXACT_SKIN_REPLAY="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
