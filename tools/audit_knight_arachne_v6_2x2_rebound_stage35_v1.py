from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.skin import qualify_skin
from tools.training.materialize_dense_skin_proposal_v1 import materialize
from tools.training.rebind_dense_skin_to_semantically_equivalent_skeleton_v1 import (
    rebind_dense_skin,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path, codec):
    if not path.is_file():
        raise RuntimeError(f"ARACHNE_REBOUND_STAGE35_ARTIFACT_MISSING::{path}")
    return codec(json.loads(path.read_text()))


def compare_rebound_npz_to_skin(npz_path: Path, skin) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        weights = np.asarray(z["weights"], dtype=np.float64)
        surface_ids = tuple(map(str, z["surface_ids"].tolist()))
        joint_ids = tuple(map(str, z["canonical_joint_ids"].tolist()))

    row_ix = {sid: i for i, sid in enumerate(surface_ids)}
    col_ix = {jid: i for i, jid in enumerate(joint_ids)}
    if len(row_ix) != len(surface_ids) or len(col_ix) != len(joint_ids):
        raise RuntimeError("ARACHNE_REBOUND_STAGE35_NPZ_ID_DUPLICATE")

    if len(skin.rows) != len(surface_ids):
        raise RuntimeError("ARACHNE_REBOUND_STAGE35_SKIN_ROW_COUNT_DRIFT")

    max_abs = 0.0
    max_row_l1 = 0.0
    total_l1 = 0.0
    for row in skin.rows:
        sid = str(row.surface_id)
        if sid not in row_ix:
            raise RuntimeError(f"ARACHNE_REBOUND_STAGE35_SKIN_SURFACE_DRIFT::{sid}")
        expected = weights[row_ix[sid]]
        actual = np.zeros((len(joint_ids),), dtype=np.float64)
        for jid, w in row.influences:
            jid = str(jid)
            if jid not in col_ix:
                raise RuntimeError(f"ARACHNE_REBOUND_STAGE35_SKIN_JOINT_DRIFT::{jid}")
            actual[col_ix[jid]] = float(w)
        d = np.abs(expected - actual)
        if len(d):
            max_abs = max(max_abs, float(d.max()))
            max_row_l1 = max(max_row_l1, float(d.sum()))
            total_l1 += float(d.sum())

    return {
        "max_abs_weight_delta": max_abs,
        "max_row_l1_weight_delta": max_row_l1,
        "total_l1_weight_delta": total_l1,
        "numeric_field_preserved": bool(max_abs <= 1e-12 and max_row_l1 <= 1e-11),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--weights-npz", type=Path, required=True)
    ap.add_argument("--old-skeleton-json", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rr = (args.authority_root / "runs" / args.run_id).resolve()
    surface_path = rr / "artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"
    partition_path = rr / "artifacts/17_MECHANICAL_PARTITION_QUALIFIED/mechanical_partition.json"
    candidate_path = rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"
    policy_path = rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"
    skeleton_path = rr / "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"
    cameras_path = rr / "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    envelope_path = rr / "artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"

    surface = load(surface_path, rigging_surface_from_dict)
    partition = load(partition_path, mechanical_partition_from_dict)
    candidate = load(candidate_path, canonical_mesh_candidate_from_dict)
    policy = load(policy_path, mesh_policy_from_dict)
    skeleton = load(skeleton_path, qualified_skeleton_from_dict)
    cameras = load(cameras_path, qualified_camera_set_from_dict).cameras
    envelope = load(envelope_path, deformation_envelope_from_dict)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rebound_npz = args.work_dir / "weights_rebound_to_current_stage28.npz"
    rebind_receipt_path = args.work_dir / "semantic_id_rebind_receipt.json"
    proposal_path = args.work_dir / "skin_proposal_replay.json"
    qualified_skin_path = args.work_dir / "qualified_skin_rebound.json"
    directive_path = args.work_dir / "mechanical_repartition_directive.json"

    rebind_receipt = rebind_dense_skin(
        input_npz=args.weights_npz,
        old_skeleton_json=args.old_skeleton_json,
        new_skeleton_json=skeleton_path,
        output_npz=rebound_npz,
        receipt_json=rebind_receipt_path,
        expected_old_skeleton_lineage="c2727bcf4f4b7f4bfbc8e4af24b8d731dd0892beee10608862f0943e4daebcce",
        expected_new_skeleton_lineage="e54548b5dbd39e399fbd901edaef57efee167b665c9628f1f38820fa62c34850",
    )
    if rebind_receipt["numeric_weight_array_unchanged"] is not True:
        raise RuntimeError("ARACHNE_REBOUND_STAGE35_NUMERIC_REBIND_MUTATION")

    proposal = materialize(
        weights_npz=rebound_npz,
        surface_json=surface_path,
        skeleton_json=skeleton_path,
        output_json=proposal_path,
        model_provenance=f"KNIGHT_ARACHNE_V6_2X2_{args.arm}_SEMANTIC_REBOUND_DIAGNOSTIC",
        metadata={
            "candidate_architecture": "RealSaS.Arachne.A1.RawSurfaceK4AttentionDirectSimplex.v6",
            "dense_proposal": True,
            "compiler_owns_qualification": True,
            "teacher_input_used": False,
            "semantic_id_rebind": True,
            "numeric_skin_field_unchanged": True,
            "source_arm": args.arm,
            "supervision_coverage": 0.8776674937965261,
            "supervision_coverage_semantics": "HISTORICAL_TEACHER_VALID_MASK__PROJECTED_ROWS_DIAGNOSTIC_ONLY",
            "uncovered_row_semantics": "PROJECTED_TRAINING_ROWS_REQUIRE_DOWNSTREAM_MECHANICAL_COMPATIBILITY_PROOF",
        },
    )
    skin = qualify_skin(
        surface,
        skeleton,
        proposal,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=1e-4,
        negative_tolerance=1e-8,
        max_influences=None,
    )
    qualified_skin_path.write_text(json.dumps(skin.to_dict(), indent=2, sort_keys=True) + "\n")

    numeric = compare_rebound_npz_to_skin(rebound_npz, skin)
    if numeric["numeric_field_preserved"] is not True:
        raise RuntimeError(f"ARACHNE_REBOUND_STAGE35_COMPILER_WEIGHT_DRIFT::{numeric}")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise RuntimeError("ARACHNE_REBOUND_STAGE35_SKIN_SKELETON_BINDING_DRIFT")
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("ARACHNE_REBOUND_STAGE35_SKIN_SURFACE_BINDING_DRIFT")

    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    compatibility = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    directive = None
    if not compatibility["passed"]:
        directive = propose_mechanical_repartition_directive_v2(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            partition=partition,
            compatibility_report=compatibility,
        )
        directive_path.write_text(json.dumps(directive, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "RealSaS.KnightArachneV6ReboundStage35Diagnostic.v1",
        "status": "PASS_DIAGNOSTIC_EXECUTED",
        "arm": args.arm,
        "source_weights_npz_sha256": sha256(args.weights_npz),
        "old_skeleton_lineage_hash": rebind_receipt["old_skeleton_lineage_hash"],
        "current_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "semantic_tree_hash": rebind_receipt["semantic_tree_hash"],
        "semantic_rebind_mapping_hash": rebind_receipt["mapping_hash"],
        "numeric_weight_array_unchanged_by_rebind": True,
        "compiler_requalification": {
            "skin_lineage_hash": skin.skin_lineage_hash,
            "row_count": len(skin.rows),
            "corrected_row_count": int(skin.qualification_report["corrected_row_count"]),
            "total_correction_l1": float(skin.qualification_report["total_correction_l1"]),
            **numeric,
        },
        "topology_authority": {
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "envelope_lineage_hash": envelope.envelope_lineage_hash,
        },
        "g3": {
            "passed": bool(g3.passed),
            "report_hash": g3.report_hash,
            "unsafe_face_count": int(len(g3.unsafe_face_indices)),
        },
        "g3b": {
            "passed": bool(compatibility["passed"]),
            "report_hash": compatibility["report_hash"],
            "risky_face_count": int(compatibility["risky_face_count"]),
            "unsafe_face_count": int(compatibility["unsafe_face_count"]),
            "top_unsafe_faces": compatibility["top_unsafe_faces"],
            "weight_mutation": bool(compatibility["weight_mutation"]),
        },
        "repartition": None if directive is None else {
            "status": directive["status"],
            "directive_hash": directive["directive_hash"],
            "candidate_separate_pair_count": int(directive["candidate_separate_pair_count"]),
            "unresolved_unsafe_face_count": int(directive["unresolved_unsafe_face_count"]),
            "face_deletion_count": int(directive["face_deletion_count"]),
            "weight_mutation": bool(directive["weight_mutation"]),
            "auto_apply_allowed": bool(directive["auto_apply_allowed"]),
            "requires_trustworthy_skin_reliability_authority": bool(
                directive["requires_trustworthy_skin_reliability_authority"]
            ),
            "restart_from": directive["restart_from"],
            "mandatory_requalification_through": directive["mandatory_requalification_through"],
        },
        "artifacts": {
            "rebound_npz": str(rebound_npz),
            "rebind_receipt": str(rebind_receipt_path),
            "proposal": str(proposal_path),
            "qualified_skin": str(qualified_skin_path),
            "directive": None if directive is None else str(directive_path),
        },
        "claim_boundary": [
            "Semantic ID rebind preserves the F64 numeric skin field and changes only compiler joint identities.",
            "Compiler requalification is executed against the current Stage15 surface and current Stage28 skeleton.",
            "Stage35 G3/G3B diagnostics are measured on the current Stage18 candidate without mutating product state.",
            "Any repartition directive remains authorization-gated; no Stage17/18 mutation is performed here.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_ARACHNE_V6_REBOUND_STAGE35_DIAGNOSTIC=" + json.dumps({
        "arm": args.arm,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "g3_passed": report["g3"]["passed"],
        "g3_unsafe": report["g3"]["unsafe_face_count"],
        "g3b_passed": report["g3b"]["passed"],
        "g3b_risky": report["g3b"]["risky_face_count"],
        "g3b_unsafe": report["g3b"]["unsafe_face_count"],
        "separate_pairs": None if report["repartition"] is None else report["repartition"]["candidate_separate_pair_count"],
        "unresolved": None if report["repartition"] is None else report["repartition"]["unresolved_unsafe_face_count"],
        "max_abs_weight_delta": numeric["max_abs_weight_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
