from __future__ import annotations

"""Exact Knight legacy-motion -> control-space Mechanics equivalence court.

This is a regression oracle only. It does not create product Rig authority and it
must never be used as an AXIS fallback. The one-control-per-joint oracle exists
only to prove that the new control-space Mechanics semantics can represent the
already accepted Knight joint-space motion/deformation without loss. A trained
AXIS V6 checkpoint must later predict its own generic control-ready rig program.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
    solve_control_pose_v1,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)

BASELINE_RUN_ID = "SUBJECT2_KNIGHT_DEMO_V2_20260924"
BASELINE_COMMIT = "62aebc388605e0f762f175dd3f929b5c09d1d70b"
FROZEN_RENDER_BLOB = "7917be02f325ba00b5dcd2b31a786898e45f640f"
LEGACY_MOTION_EVALUATOR_BLOB = "746a0bdb5c1be3cb5829a7dda2c4eac4b6310dd4"
EXPECTED_STAGE_SHA256 = {
    "05_CAMERA_CONTRACT_SOLVED": "312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a",
    "18_CANONICAL_MESH_ADDRESSING_BUILD": "0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3",
    "28_SKELETON_QUALIFIED": "e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987",
    "32_SKIN_QUALIFIED": "f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09",
}
CLIPS = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")
POSITION_TOLERANCE = 5.0e-8
ROTATION_TOLERANCE = 5.0e-8
VERTEX_TOLERANCE = 5.0e-8
NON_RIGID_LINEAR_TOLERANCE = 2.0e-6


def _git_blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], text=True).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise RuntimeError("KNIGHT_CONTROL_ORACLE_ROTATION_INVALID")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray((0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s))
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.asarray(((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s))
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.asarray(((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s))
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.asarray(((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s))
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12 or not math.isfinite(n):
        raise RuntimeError("KNIGHT_CONTROL_ORACLE_QUATERNION_DEGENERATE")
    q = q / n
    return tuple(map(float, q))


def _rotation_factor(linear: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(linear, dtype=np.float64)
    u, singular, vt = np.linalg.svd(a)
    r = u @ vt
    if float(np.linalg.det(r)) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    residual = max(float(np.max(np.abs(singular - 1.0))), float(np.max(np.abs(a - r))))
    return r, residual


def _oracle_rig(skeleton, cameras) -> tuple[CanonicalRigIRV2, dict[str, str]]:
    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    controls: list[CanonicalRigControlIRV2] = []
    control_by_joint: dict[str, str] = {}
    joints: list[CanonicalDeformJointIRV2] = []
    local_frames: list[CanonicalJointFrameIRV2] = []
    for joint in skeleton.joints:
        jid = str(joint.canonical_joint_id)
        cid = "ORACLE:C:" + jid
        control_by_joint[jid] = cid
        joints.append(CanonicalDeformJointIRV2(
            canonical_joint_id=jid,
            P=tuple(map(float, joint.position)),
            parent_canonical_joint_id=None if joint.parent_canonical_id is None else str(joint.parent_canonical_id),
            source_proposal_joint_id="REGRESSION_ORACLE:" + jid,
            metadata={"regression_oracle_only": True},
        ))
        local_frames.append(CanonicalJointFrameIRV2(
            canonical_joint_id=jid,
            basis=tuple(tuple(map(float, row)) for row in frames[jid].rotation_matrix),
            dof_labels=(),
            metadata={"regression_oracle_only": True},
        ))
        controls.append(CanonicalRigControlIRV2(
            canonical_control_id=cid,
            control_kind="FK",
            target_canonical_joint_ids=(jid,),
            parent_canonical_control_id=None,
            rest_P=tuple(map(float, joint.position)),
            source_proposal_control_id="REGRESSION_ORACLE:" + cid,
            metadata={"one_to_one_joint_control_oracle_not_product_target": True},
        ))
    roots = tuple(sorted(str(j.canonical_joint_id) for j in skeleton.joints if j.parent_canonical_id is None))
    rig = CanonicalRigIRV2(
        joints=tuple(joints),
        local_frames=tuple(local_frames),
        controls=tuple(controls),
        chains=(),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=roots,
        mesh_lineage_hash="0" * 64,
        feature_evidence_binding_hash="1" * 64,
        proposal_evidence_hash="2" * 64,
        rig_program_qualification_hash="3" * 64,
        deform_structure_hash="4" * 64,
        control_program_hash="5" * 64,
        rig_lineage_hash="6" * 64,
        metadata={
            "authority": "REGRESSION_ORACLE_ONLY",
            "product_authority_claimed": False,
            "axis_fallback_allowed": False,
        },
    )
    return rig, control_by_joint


def _sample_times(tracks: dict, duration: float) -> tuple[float, ...]:
    values = {float(x) for x in np.linspace(0.0, duration, 11)}
    for track in tracks.values():
        for key in track.keyframes:
            values.add(float(key.time_seconds))
    return tuple(sorted(x for x in values if -1.0e-12 <= x <= duration + 1.0e-12))


def _new_vertex_positions(rest: np.ndarray, weights: np.ndarray, joint_ids: tuple[str, ...], skeleton, pose) -> np.ndarray:
    rest_joint = {str(j.canonical_joint_id): np.asarray(j.position, dtype=np.float64) for j in skeleton.joints}
    solved = {str(row.canonical_joint_id): row for row in pose.joints}
    out = np.zeros_like(rest, dtype=np.float64)
    for column, jid in enumerate(joint_ids):
        row = solved[jid]
        p = np.asarray(row.solved_P, dtype=np.float64)
        r = np.asarray(row.rotation_basis, dtype=np.float64)
        transformed = p[None, :] + (rest - rest_joint[jid][None, :]) @ r.T
        out += weights[:, column, None] * transformed
    return out


def run(*, authority_root: Path, run_id: str, out_path: Path) -> dict:
    if run_id != BASELINE_RUN_ID:
        raise RuntimeError(f"KNIGHT_CONTROL_ORACLE_RUN_ID_DRIFT:{run_id}")
    if _git_blob("tools/demo/frozen/render_knight_motion_preview_v1_7917be02.py") != FROZEN_RENDER_BLOB:
        raise RuntimeError("KNIGHT_CONTROL_ORACLE_FROZEN_RENDER_BLOB_DRIFT")
    if _git_blob("compiler/realsas_compiler_core/motion_dynamic_proof_v2.py") != LEGACY_MOTION_EVALUATOR_BLOB:
        raise RuntimeError("KNIGHT_CONTROL_ORACLE_LEGACY_EVALUATOR_BLOB_DRIFT")

    run_root = authority_root / "runs" / run_id
    stage_paths = {
        "05_CAMERA_CONTRACT_SOLVED": run_root / "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",
        "18_CANONICAL_MESH_ADDRESSING_BUILD": run_root / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",
        "28_SKELETON_QUALIFIED": run_root / "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
        "32_SKIN_QUALIFIED": run_root / "artifacts/32_SKIN_QUALIFIED/qualified_skin.json",
    }
    stage_hashes = {sid: _sha256(path) for sid, path in stage_paths.items()}
    for sid, expected in EXPECTED_STAGE_SHA256.items():
        if stage_hashes[sid] != expected:
            raise RuntimeError(f"KNIGHT_CONTROL_ORACLE_STAGE_DRIFT:{sid}:{stage_hashes[sid]}:{expected}")

    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1"
    ))
    skeleton = qualified_skeleton_from_dict(stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
    ))
    skin = qualified_skin_from_dict(stage_output_payload(
        ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1"
    ))
    camera_set = qualified_camera_set_from_dict(stage_output_payload(
        ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
    ))
    cameras = tuple(sorted(camera_set.cameras, key=lambda row: int(row.view_index)))
    source_report = json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    joint_ids, weights = _candidate_skin_weights(candidate, skin, skeleton)
    joint_ids = tuple(map(str, joint_ids))
    rest = np.asarray([vertex.P for vertex in candidate.vertices], dtype=np.float64)
    oracle_rig, control_by_joint = _oracle_rig(skeleton, cameras)
    rest_joint = {str(j.canonical_joint_id): np.asarray(j.position, dtype=np.float64) for j in skeleton.joints}

    rows = []
    max_position = max_rotation = max_vertex = max_non_rigid = 0.0
    for clip_id in CLIPS:
        motion_path = run_root / "inputs/motion/quaternius_knight_v1" / f"{clip_id}.motion.json"
        payload = json.loads(motion_path.read_text())
        tracks, mapping_details = _tracks_for_clip(payload, skeleton, cameras, source_report)
        duration = float(payload["duration_seconds"])
        times = _sample_times(tracks, duration)
        clip_max = {"position": 0.0, "rotation": 0.0, "vertex": 0.0, "non_rigid": 0.0}
        for t in times:
            legacy_skin, legacy_positions, _ = _joint_pose_v2(
                skeleton=skeleton, tracks=tracks, time_seconds=float(t), cameras=cameras
            )
            controls = []
            target_rotations: dict[str, np.ndarray] = {}
            non_rigid = 0.0
            for jid in joint_ids:
                matrix = np.asarray(legacy_skin[jid], dtype=np.float64)
                rotation, residual = _rotation_factor(matrix[:3, :3])
                non_rigid = max(non_rigid, residual)
                target_rotations[jid] = rotation
                target_p = np.asarray(legacy_positions[jid], dtype=np.float64)
                controls.append(ControlTransformIRV1(
                    canonical_control_id=control_by_joint[jid],
                    translation=tuple(map(float, target_p - rest_joint[jid])),
                    rotation_quat_wxyz=_matrix_to_quat_wxyz(rotation),
                ))
            pose = solve_control_pose_v1(oracle_rig, tuple(controls))
            solved = {str(row.canonical_joint_id): row for row in pose.joints}
            position_error = max(
                float(np.linalg.norm(np.asarray(solved[jid].solved_P) - np.asarray(legacy_positions[jid])))
                for jid in joint_ids
            )
            rotation_error = max(
                float(np.max(np.abs(np.asarray(solved[jid].rotation_basis) - target_rotations[jid])))
                for jid in joint_ids
            )
            legacy_vertices = _skin(rest, weights, joint_ids, legacy_skin)
            new_vertices = _new_vertex_positions(rest, weights, joint_ids, skeleton, pose)
            vertex_error = float(np.max(np.abs(new_vertices - legacy_vertices)))
            clip_max["position"] = max(clip_max["position"], position_error)
            clip_max["rotation"] = max(clip_max["rotation"], rotation_error)
            clip_max["vertex"] = max(clip_max["vertex"], vertex_error)
            clip_max["non_rigid"] = max(clip_max["non_rigid"], non_rigid)
        max_position = max(max_position, clip_max["position"])
        max_rotation = max(max_rotation, clip_max["rotation"])
        max_vertex = max(max_vertex, clip_max["vertex"])
        max_non_rigid = max(max_non_rigid, clip_max["non_rigid"])
        rows.append({
            "clip_id": clip_id,
            "motion_input_sha256": _sha256(motion_path),
            "sample_count": len(times),
            "mapping_detail_count": len(mapping_details),
            "max_joint_position_residual": clip_max["position"],
            "max_joint_rotation_residual": clip_max["rotation"],
            "max_vertex_residual": clip_max["vertex"],
            "max_legacy_linear_non_rigid_residual": clip_max["non_rigid"],
        })

    passed = (
        max_position <= POSITION_TOLERANCE
        and max_rotation <= ROTATION_TOLERANCE
        and max_vertex <= VERTEX_TOLERANCE
        and max_non_rigid <= NON_RIGID_LINEAR_TOLERANCE
    )
    report = {
        "schema": "RealSaS.KnightControlSpaceEquivalenceRegression.v1",
        "status": "PASS" if passed else "FAIL",
        "authority_class": "REGRESSION_ORACLE_ONLY__NO_PRODUCT_AUTHORITY",
        "product_authority_claimed": False,
        "axis_fallback_allowed": False,
        "baseline": {
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "frozen_render_blob": FROZEN_RENDER_BLOB,
            "legacy_motion_evaluator_blob": LEGACY_MOTION_EVALUATOR_BLOB,
            "stage_sha256": stage_hashes,
        },
        "oracle": {
            "control_scheme": "ONE_FLAT_FK_CONTROL_PER_LEGACY_DEFORM_JOINT",
            "product_target": False,
            "purpose": "REPRESENTATIONAL_EQUIVALENCE_ONLY",
            "joint_count": len(joint_ids),
        },
        "thresholds": {
            "joint_position": POSITION_TOLERANCE,
            "joint_rotation": ROTATION_TOLERANCE,
            "vertex": VERTEX_TOLERANCE,
            "legacy_linear_non_rigid": NON_RIGID_LINEAR_TOLERANCE,
        },
        "aggregate": {
            "max_joint_position_residual": max_position,
            "max_joint_rotation_residual": max_rotation,
            "max_vertex_residual": max_vertex,
            "max_legacy_linear_non_rigid_residual": max_non_rigid,
        },
        "clips": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("KNIGHT_CONTROL_SPACE_EQUIVALENCE_FAIL:" + json.dumps(report["aggregate"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(authority_root=args.authority_root, run_id=args.run_id, out_path=args.out)
    print("KNIGHT_CONTROL_SPACE_EQUIVALENCE_PASS", json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
