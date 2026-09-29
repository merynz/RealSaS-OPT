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
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    validate_compacted_dense_face_provenance_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _tracks_for_clip
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
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_ARTIFACT_MISSING::{path}")
    return codec(json.loads(path.read_text()))


def stage_output_path(run_root: Path, stage_id: str, schema: str) -> Path:
    ledger = json.loads((run_root / "ACTIVE_RUN_V2.json").read_text())
    row = next((x for x in ledger["stages"] if x["id"] == stage_id), None)
    if row is None:
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_STAGE_MISSING::{stage_id}")
    rows = [x for x in row.get("outputs") or () if x.get("schema") == schema]
    if len(rows) != 1:
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_OUTPUT_CARDINALITY::{stage_id}::{schema}::{len(rows)}")
    path = Path(rows[0]["path"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_OUTPUT_MISSING::{path}")
    expected = str(rows[0].get("sha256") or "")
    if len(expected) == 64 and sha256(path) != expected:
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_OUTPUT_SHA_DRIFT::{stage_id}::{schema}")
    return path


def compare_rebound_npz_to_skin(npz_path: Path, skin) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        weights = np.asarray(z["weights"], dtype=np.float64)
        surface_ids = tuple(map(str, z["surface_ids"].tolist()))
        joint_ids = tuple(map(str, z["canonical_joint_ids"].tolist()))
    row_ix = {sid: i for i, sid in enumerate(surface_ids)}
    col_ix = {jid: i for i, jid in enumerate(joint_ids)}
    max_abs = 0.0
    max_row_l1 = 0.0
    total_l1 = 0.0
    for row in skin.rows:
        sid = str(row.surface_id)
        expected = weights[row_ix[sid]]
        actual = np.zeros((len(joint_ids),), dtype=np.float64)
        for jid, w in row.influences:
            actual[col_ix[str(jid)]] = float(w)
        d = np.abs(expected - actual)
        max_abs = max(max_abs, float(d.max(initial=0.0)))
        max_row_l1 = max(max_row_l1, float(d.sum()))
        total_l1 += float(d.sum())
    return {
        "max_abs_weight_delta": max_abs,
        "max_row_l1_weight_delta": max_row_l1,
        "total_l1_weight_delta": total_l1,
        "numeric_field_preserved": bool(max_abs <= 1e-12 and max_row_l1 <= 1e-11),
    }


def carrier_for_partition(partition):
    decisions = tuple(
        ComponentCarrierDecisionIR(
            component.component_id,
            "MESH",
            ("AUTOMATIC_CONSERVATIVE_MESH_CARRIER_V1",),
            metadata={
                "automatic": True,
                "semantic_recognition_used": False,
                "planar_optimization_deferred": True,
            },
        )
        for component in partition.components
    )
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=decisions,
        metadata={
            "default_carrier": "MESH",
            "automatic": True,
            "manual_carrier_authoring_used": False,
            "clip_is_presentation_only": True,
        },
    )
    validate_component_carrier_policy(carrier, partition)
    return carrier


def authorization_for(*, directive, surface, partition, evidence_hash: str) -> dict:
    auth = {
        "schema": "RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status": "PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash": directive["directive_hash"],
        "source_surface_lineage_hash": surface.geometry_lineage_hash,
        "source_partition_lineage_hash": partition.partition_lineage_hash,
        "source_skeleton_lineage_hash": directive["source_skeleton_lineage_hash"],
        "source_skin_lineage_hash": directive["source_skin_lineage_hash"],
        "teacher_inputs_used_by_predictor": False,
        "weight_reliability_closure_passed": True,
        "weight_reliability_evidence_hash": evidence_hash,
        "authorization_hash": "",
    }
    auth["authorization_hash"] = repartition_authorization_hash_v1(auth)
    return auth


def triangle_metrics(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> dict:
    r = rest[faces]
    p = posed[faces]
    rl = np.stack([
        np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
        np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
        np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
    ], axis=1)
    pl = np.stack([
        np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
        np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
    ], axis=1)
    edge = (pl / np.maximum(rl, 1e-15)).max(axis=1)

    r1 = r[:, 1] - r[:, 0]
    r2 = r[:, 2] - r[:, 0]
    l = np.linalg.norm(r1, axis=1)
    u = r1 / np.maximum(l[:, None], 1e-15)
    x2 = np.sum(r2 * u, axis=1)
    perp = r2 - x2[:, None] * u
    y2 = np.linalg.norm(perp, axis=1)
    inv = np.zeros((len(faces), 2, 2), dtype=np.float64)
    inv[:, 0, 0] = 1.0 / np.maximum(l, 1e-15)
    inv[:, 0, 1] = -x2 / np.maximum(l * y2, 1e-15)
    inv[:, 1, 1] = 1.0 / np.maximum(y2, 1e-15)
    s = np.linalg.svd(
        np.einsum(
            "nij,njk->nik",
            np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=2),
            inv,
        ),
        compute_uv=False,
    )
    cond = s[:, 0] / np.maximum(s[:, 1], 1e-15)
    area = s[:, 0] * s[:, 1]
    return {
        "face_count": int(len(faces)),
        "edge_p95": float(np.quantile(edge, 0.95)),
        "edge_p99": float(np.quantile(edge, 0.99)),
        "edge_max": float(edge.max()),
        "edge_gt_2": int(np.count_nonzero(edge > 2.0)),
        "edge_gt_4": int(np.count_nonzero(edge > 4.0)),
        "edge_gt_10": int(np.count_nonzero(edge > 10.0)),
        "condition_p95": float(np.quantile(cond, 0.95)),
        "condition_gt_16": int(np.count_nonzero(cond > 16.0)),
        "area_gt_20": int(np.count_nonzero(area > 20.0)),
    }


def pose_candidate(candidate, *, surface, skeleton, skin, tracks, time_seconds: float, cameras):
    rest, weights, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    matrices, _, _ = _joint_pose_v2(
        skeleton=skeleton,
        tracks=tracks,
        time_seconds=float(time_seconds),
        cameras=cameras,
    )
    hom = np.concatenate([rest, np.ones((len(rest), 1), dtype=np.float64)], axis=1)
    per = np.stack([(hom @ np.asarray(matrices[jid]).T)[:, :3] for jid in joint_ids], axis=1)
    posed = np.sum(per * weights[:, :, None], axis=1)
    return rest, posed, np.asarray(faces, dtype=np.int64)


def actual_motion_report(candidate, *, run_root, surface, skeleton, skin, cameras) -> dict:
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    clips = []
    aggregate = {
        "max_edge_gt_10": 0,
        "max_edge_gt_4": 0,
        "worst_edge_max": 0.0,
        "worst_edge_p99": 0.0,
    }
    for clip_id in ("demo_idle_v1", "demo_run_v1", "demo_slash_v1"):
        payload = json.loads(
            (run_root / "inputs/motion/quaternius_knight_v1" / f"{clip_id}.motion.json").read_text()
        )
        tracks, _ = _tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            4,
            endpoint=not bool(payload.get("loop")),
        )
        frames = []
        for fi, t in enumerate(times):
            rest, posed, faces = pose_candidate(
                candidate,
                surface=surface,
                skeleton=skeleton,
                skin=skin,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            m = triangle_metrics(rest, posed, faces)
            frames.append({"frame": fi, "time_seconds": float(t), **m})
            aggregate["max_edge_gt_10"] = max(aggregate["max_edge_gt_10"], m["edge_gt_10"])
            aggregate["max_edge_gt_4"] = max(aggregate["max_edge_gt_4"], m["edge_gt_4"])
            aggregate["worst_edge_max"] = max(aggregate["worst_edge_max"], m["edge_max"])
            aggregate["worst_edge_p99"] = max(aggregate["worst_edge_p99"], m["edge_p99"])
        clips.append({"clip_id": clip_id, "frames": frames})
    return {"clips": clips, "aggregate": aggregate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--weights-npz", type=Path, required=True)
    ap.add_argument("--old-skeleton-json", type=Path, required=True)
    ap.add_argument("--a100-audit-json", type=Path, required=True)
    ap.add_argument("--prereg-json", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prereg = json.loads(args.prereg_json.read_text())
    if prereg.get("status") != "FROZEN_BEFORE_CHILD_REPARTITION_ITERATION_1":
        raise RuntimeError("ARACHNE_TOPO_REPAIR_PREREG_STATUS_DRIFT")
    eligible = tuple(prereg["reliability_eligibility"]["expected_eligible_arms_by_already_sealed_a100_evidence"])
    if args.arm not in eligible:
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_ARM_NOT_PREREGISTERED::{args.arm}")

    a100 = json.loads(args.a100_audit_json.read_text())
    if a100.get("status") != "COMPLETE":
        raise RuntimeError("ARACHNE_TOPO_REPAIR_A100_COURT_NOT_COMPLETE")
    row = a100["arms"][args.arm]
    invalid = row["invalid"]
    gates = prereg["reliability_eligibility"]
    if row.get("final_valid_gate") is not True:
        raise RuntimeError("ARACHNE_TOPO_REPAIR_VALID_GATE_NOT_PASS")
    if float(invalid["row_l1_p95"]) > float(gates["invalid_row_l1_p95_max"]):
        raise RuntimeError("ARACHNE_TOPO_REPAIR_INVALID_P95_NOT_RELIABLE")
    if int(invalid["row_l1_gt_1"]) > int(gates["invalid_row_l1_gt_1_max"]):
        raise RuntimeError("ARACHNE_TOPO_REPAIR_INVALID_GT1_NOT_RELIABLE")
    if float(invalid["dominant_joint_accuracy"]) < float(gates["invalid_dominant_joint_accuracy_min"]):
        raise RuntimeError("ARACHNE_TOPO_REPAIR_INVALID_DOMINANT_NOT_RELIABLE")
    if sha256(args.weights_npz) != str(row["weights_sha256"]):
        raise RuntimeError("ARACHNE_TOPO_REPAIR_WEIGHT_SHA_DRIFT")

    rr = (args.authority_root / "runs" / args.run_id).resolve()
    surface_path = stage_output_path(rr, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1")
    provenance_path = stage_output_path(rr, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.CompactedDenseFaceProvenance.v1")
    partition_path = stage_output_path(rr, "17_MECHANICAL_PARTITION_QUALIFIED", "RealSaS.MechanicalPartitionIR.v1")
    candidate_path = stage_output_path(rr, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1")
    policy_path = stage_output_path(rr, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1")
    skeleton_path = stage_output_path(rr, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1")
    cameras_path = stage_output_path(rr, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")
    envelope_path = stage_output_path(rr, "34_DEFORMATION_CAPABILITY_ENVELOPE", "RealSaS.DeformationCapabilityEnvelopeIR.v1")

    surface = load(surface_path, rigging_surface_from_dict)
    parent_partition = load(partition_path, mechanical_partition_from_dict)
    current_candidate = load(candidate_path, canonical_mesh_candidate_from_dict)
    policy = load(policy_path, mesh_policy_from_dict)
    skeleton = load(skeleton_path, qualified_skeleton_from_dict)
    cameras = tuple(sorted(load(cameras_path, qualified_camera_set_from_dict).cameras, key=lambda c: int(c.view_index)))
    envelope = load(envelope_path, deformation_envelope_from_dict)

    provenance = json.loads(provenance_path.read_text())
    validate_compacted_dense_face_provenance_v1(provenance, surface=surface)
    explicit_faces = tuple(tuple(map(str, x)) for x in provenance["compact_faces"])
    if not explicit_faces:
        raise RuntimeError("ARACHNE_TOPO_REPAIR_FACE_PROVENANCE_EMPTY")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rebound_npz = args.work_dir / "weights_rebound_to_current_stage28.npz"
    rebind_receipt_path = args.work_dir / "semantic_id_rebind_receipt.json"
    proposal_path = args.work_dir / "skin_proposal_replay.json"
    qualified_skin_path = args.work_dir / "qualified_skin_rebound.json"

    rebind_receipt = rebind_dense_skin(
        input_npz=args.weights_npz,
        old_skeleton_json=args.old_skeleton_json,
        new_skeleton_json=skeleton_path,
        output_npz=rebound_npz,
        receipt_json=rebind_receipt_path,
        expected_old_skeleton_lineage="c2727bcf4f4b7f4bfbc8e4af24b8d731dd0892beee10608862f0943e4daebcce",
        expected_new_skeleton_lineage="e54548b5dbd39e399fbd901edaef57efee167b665c9628f1f38820fa62c34850",
    )
    proposal = materialize(
        weights_npz=rebound_npz,
        surface_json=surface_path,
        skeleton_json=skeleton_path,
        output_json=proposal_path,
        model_provenance=f"KNIGHT_ARACHNE_V6_TOPOLOGY_REPAIR_COURT::{args.arm}",
        metadata={
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
        surface, skeleton, proposal,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=1e-4,
        negative_tolerance=1e-8,
        max_influences=None,
    )
    qualified_skin_path.write_text(json.dumps(skin.to_dict(), indent=2, sort_keys=True) + "\n")
    numeric = compare_rebound_npz_to_skin(rebound_npz, skin)
    if numeric["numeric_field_preserved"] is not True:
        raise RuntimeError(f"ARACHNE_TOPO_REPAIR_COMPILER_WEIGHT_DRIFT::{numeric}")
    if float(numeric["max_abs_weight_delta"]) > float(gates["compiler_requalification_max_abs_weight_delta"]):
        raise RuntimeError("ARACHNE_TOPO_REPAIR_COMPILER_DELTA_ABOVE_PREREG")

    evidence_payload = {
        "schema": "RealSaS.KnightArachneV6WeightReliabilityEvidence.v1",
        "arm": args.arm,
        "a100_court_sha256": sha256(args.a100_audit_json),
        "a100_arm_result_sha256": row["result_sha256"],
        "source_weights_npz_sha256": row["weights_sha256"],
        "a100_invalid_metrics": {
            "row_l1_p95": invalid["row_l1_p95"],
            "row_l1_gt_1": invalid["row_l1_gt_1"],
            "dominant_joint_accuracy": invalid["dominant_joint_accuracy"],
        },
        "semantic_tree_hash": rebind_receipt["semantic_tree_hash"],
        "semantic_rebind_mapping_hash": rebind_receipt["mapping_hash"],
        "numeric_weight_array_unchanged": rebind_receipt["numeric_weight_array_unchanged"],
        "compiler_requalification_numeric": numeric,
        "current_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "model_retraining_used_for_identity_migration": False,
        "product_authority_claimed": False,
    }
    evidence_hash = content_sha256(evidence_payload)
    (args.work_dir / "weight_reliability_evidence.json").write_text(
        json.dumps({**evidence_payload, "evidence_hash": evidence_hash}, indent=2, sort_keys=True) + "\n"
    )

    baseline_motion = actual_motion_report(
        current_candidate,
        run_root=rr, surface=surface, skeleton=skeleton, skin=skin, cameras=cameras,
    )

    max_iterations = int(prereg["topology_repair_loop"]["max_iterations"])
    iterations = []
    partition = parent_partition
    candidate = current_candidate
    closed = False
    for iteration in range(max_iterations + 1):
        compatibility = run_skin_topology_compatibility_v1(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
        )
        step = {
            "evaluation_index": iteration,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "component_count": len(partition.components),
            "separate_boundary_count": sum(x.decision == "SEPARATE" for x in partition.boundary_constraints),
            "g3b": {
                "passed": bool(compatibility["passed"]),
                "report_hash": compatibility["report_hash"],
                "risky_face_count": int(compatibility["risky_face_count"]),
                "unsafe_face_count": int(compatibility["unsafe_face_count"]),
            },
        }
        if compatibility["passed"]:
            step["action"] = "CLOSED"
            iterations.append(step)
            closed = True
            break
        if iteration >= max_iterations:
            step["action"] = "REPAIR_BUDGET_EXHAUSTED"
            iterations.append(step)
            break

        directive = propose_mechanical_repartition_directive_v2(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            partition=partition,
            compatibility_report=compatibility,
        )
        if directive["status"] != "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY":
            step["action"] = "ABSTAIN_NO_IDENTITY_BOUNDARY_PROPOSAL"
            step["directive_status"] = directive["status"]
            iterations.append(step)
            break
        if int(directive["candidate_separate_pair_count"]) <= 0:
            raise RuntimeError("ARACHNE_TOPO_REPAIR_EMPTY_REPARTITION_PROPOSAL")

        auth = authorization_for(
            directive=directive,
            surface=surface,
            partition=partition,
            evidence_hash=evidence_hash,
        )
        child_partition = build_repartitioned_partition_v2(
            surface=surface,
            parent_partition=partition,
            directive=directive,
            authorization=auth,
        )
        validate_mechanical_partition(child_partition, surface)
        carrier = carrier_for_partition(child_partition)
        producer_hash = content_sha256({
            "schema": "RealSaS.KnightArachneV6CoverageTopologyRepairChildPolicy.v1",
            "arm": args.arm,
            "iteration": iteration + 1,
            "parent_candidate_lineage_hash": candidate.candidate_lineage_hash,
            "parent_partition_lineage_hash": partition.partition_lineage_hash,
            "directive_hash": directive["directive_hash"],
            "authorization_hash": auth["authorization_hash"],
            "weight_reliability_evidence_hash": evidence_hash,
        })
        child_candidate = build_holeless_partitioned_dense_candidate(
            surface,
            child_partition,
            carrier,
            producer_policy_hash=producer_hash,
            explicit_face_provenance=explicit_faces,
        )
        md = dict(child_candidate.metadata or {})
        if int(md.get("face_deletion_count", -1)) != 0:
            raise RuntimeError("ARACHNE_TOPO_REPAIR_FACE_DELETION_FORBIDDEN")
        if float(md.get("rest_area_relative_error", 1.0)) > 1e-10:
            raise RuntimeError("ARACHNE_TOPO_REPAIR_REST_AREA_DRIFT")

        idir = args.work_dir / f"iteration_{iteration + 1:02d}"
        idir.mkdir(parents=True, exist_ok=True)
        (idir / "directive.json").write_text(json.dumps(directive, indent=2, sort_keys=True) + "\n")
        (idir / "authorization.json").write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
        (idir / "mechanical_partition.json").write_text(json.dumps(child_partition.to_dict(), indent=2, sort_keys=True) + "\n")
        (idir / "component_carrier_policy.json").write_text(json.dumps(carrier.to_dict(), indent=2, sort_keys=True) + "\n")
        (idir / "canonical_mesh_candidate.json").write_text(json.dumps(child_candidate.to_dict(), indent=2, sort_keys=True) + "\n")

        step.update({
            "action": "REPARTITION_AND_HOLELESS_REBUILD",
            "directive_hash": directive["directive_hash"],
            "authorization_hash": auth["authorization_hash"],
            "candidate_separate_pair_count": int(directive["candidate_separate_pair_count"]),
            "unresolved_unsafe_face_count": int(directive["unresolved_unsafe_face_count"]),
            "child_partition_lineage_hash": child_partition.partition_lineage_hash,
            "child_candidate_lineage_hash": child_candidate.candidate_lineage_hash,
            "child_component_count": len(child_partition.components),
            "child_face_count": len(child_candidate.faces),
            "child_vertex_count": len(child_candidate.vertices),
            "child_mixed_source_face_count": int(md.get("mixed_source_face_count", 0)),
            "child_generated_seam_vertex_count": int(md.get("generated_seam_vertex_count", 0)),
            "child_generated_centroid_vertex_count": int(md.get("generated_centroid_vertex_count", 0)),
            "child_rest_area_relative_error": float(md.get("rest_area_relative_error", 0.0)),
            "child_face_deletion_count": int(md.get("face_deletion_count", 0)),
        })
        iterations.append(step)
        partition = child_partition
        candidate = child_candidate

    final_compatibility = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    final_motion = actual_motion_report(
        candidate,
        run_root=rr, surface=surface, skeleton=skeleton, skin=skin, cameras=cameras,
    )

    report = {
        "schema": "RealSaS.KnightArachneV6CoverageTopologyRepairCourtArm.v1",
        "status": "COMPLETE",
        "arm": args.arm,
        "reliability_evidence_hash": evidence_hash,
        "source_weights_npz_sha256": sha256(args.weights_npz),
        "semantic_rebind": {
            "old_skeleton_lineage_hash": rebind_receipt["old_skeleton_lineage_hash"],
            "new_skeleton_lineage_hash": rebind_receipt["new_skeleton_lineage_hash"],
            "semantic_tree_hash": rebind_receipt["semantic_tree_hash"],
            "mapping_hash": rebind_receipt["mapping_hash"],
            "numeric_weight_array_unchanged": rebind_receipt["numeric_weight_array_unchanged"],
        },
        "compiler_requalification": {
            "skin_lineage_hash": skin.skin_lineage_hash,
            "corrected_row_count": int(skin.qualification_report["corrected_row_count"]),
            "total_correction_l1": float(skin.qualification_report["total_correction_l1"]),
            **numeric,
        },
        "parent_topology": {
            "partition_lineage_hash": parent_partition.partition_lineage_hash,
            "candidate_lineage_hash": current_candidate.candidate_lineage_hash,
            "component_count": len(parent_partition.components),
            "face_count": len(current_candidate.faces),
            "vertex_count": len(current_candidate.vertices),
        },
        "iterations": iterations,
        "repair_closed_within_budget": bool(closed and final_compatibility["passed"]),
        "repairs_used": sum(1 for x in iterations if x.get("action") == "REPARTITION_AND_HOLELESS_REBUILD"),
        "final_topology": {
            "partition_lineage_hash": partition.partition_lineage_hash,
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "component_count": len(partition.components),
            "face_count": len(candidate.faces),
            "vertex_count": len(candidate.vertices),
            "candidate_metadata": dict(candidate.metadata or {}),
        },
        "final_g3b": {
            "passed": bool(final_compatibility["passed"]),
            "report_hash": final_compatibility["report_hash"],
            "risky_face_count": int(final_compatibility["risky_face_count"]),
            "unsafe_face_count": int(final_compatibility["unsafe_face_count"]),
            "weight_mutation": bool(final_compatibility["weight_mutation"]),
        },
        "final_g3": {
            "passed": bool(g3.passed),
            "report_hash": g3.report_hash,
            "failure_invariants": list(g3.failure_invariants),
            "probe_count": int(g3.probe_count),
            "face_count": int(g3.face_count),
            "minimum_area_ratio": float(g3.minimum_area_ratio),
            "maximum_area_ratio": float(g3.maximum_area_ratio),
            "maximum_condition_number": float(g3.maximum_condition_number),
            "minimum_edge_ratio": float(g3.minimum_edge_ratio),
            "maximum_edge_ratio": float(g3.maximum_edge_ratio),
        },
        "actual_motion": {
            "baseline": baseline_motion,
            "final": final_motion,
        },
        "claim_boundary": [
            "This court constructs isolated authorized Stage17 child partitions and holeless Stage18 candidates only.",
            "No product mainline ledger, run manifest, Stage17 authority, Stage18 authority, or Stage32 skin authority is mutated.",
            "The numeric Arachne field is losslessly semantic-ID rebound and independently requalified before topology evidence is used.",
            "Teacher inputs are not provided to the predictor or topology operator.",
            "A product integration attempt requires a separate mainline replay of the selected child repair.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_ARACHNE_V6_COVERAGE_TOPOLOGY_REPAIR_ARM=" + json.dumps({
        "arm": args.arm,
        "reliability_evidence_hash": evidence_hash,
        "closed": report["repair_closed_within_budget"],
        "repairs_used": report["repairs_used"],
        "final_g3b_passed": report["final_g3b"]["passed"],
        "final_g3b_unsafe": report["final_g3b"]["unsafe_face_count"],
        "final_g3_passed": report["final_g3"]["passed"],
        "baseline_motion": baseline_motion["aggregate"],
        "final_motion": final_motion["aggregate"],
        "face_count": report["final_topology"]["face_count"],
        "vertex_count": report["final_topology"]["vertex_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
