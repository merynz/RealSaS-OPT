from __future__ import annotations

"""Measure authored-motion relation drift before admitting richer teacher truth.

This is a diagnostic/court producer, not a training-target producer. It compares
rest-pose and authored-pose first/second ray relations under the same admitted
Stage01 cameras. The goal is to decide whether motion-derived relation evidence
can be consumed by the current camera-conditioned relation head without adding a
new conditioning authority.

No result from this tool authorizes optimizer steps.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.build_fit8_vnext_shards_v1 import SUBJECT_SOURCE_IDS, _camera_rays, _seed, _sha
from tools.iris_vnext.build_fit8_vnext_shards_v2 import _RAY_BACKEND, _ray_triangle_first_two_parallel_grid

_SCHEMA = "RealSaS.IRISVNextAuthoredMotionRelationConditioningAudit.v1"
_STATUS = "MEASURED__COURT_ONLY__NO_TRAINING_AUTHORIZATION"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--stage01-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-rays-per-view", type=int, default=2048)
    parser.add_argument("--max-poses-per-subject", type=int, default=0)
    return parser.parse_args()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/cohort_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def _two_hit_mask(hits: dict[str, np.ndarray]) -> np.ndarray:
    first = np.asarray(hits["first_face"], dtype=np.int64)
    second = np.asarray(hits["second_face"], dtype=np.int64)
    return (first >= 0) & (second >= 0) & (first != second)


def compare_first_two_hits_v1(
    rest_hits: dict[str, np.ndarray],
    pose_hits: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Compare rest and pose first-two hit identity/order for identical rays."""

    rf = np.asarray(rest_hits["first_face"], dtype=np.int64).reshape(-1)
    rs = np.asarray(rest_hits["second_face"], dtype=np.int64).reshape(-1)
    pf = np.asarray(pose_hits["first_face"], dtype=np.int64).reshape(-1)
    ps = np.asarray(pose_hits["second_face"], dtype=np.int64).reshape(-1)
    if not (rf.shape == rs.shape == pf.shape == ps.shape):
        raise ValueError("VNEXT_MOTION_RELATION_HIT_SHAPE_DRIFT")

    rest_two = (rf >= 0) & (rs >= 0) & (rf != rs)
    pose_two = (pf >= 0) & (ps >= 0) & (pf != ps)
    joint = rest_two & pose_two

    same_orientation = joint & (rf == pf) & (rs == ps)
    reversed_orientation = joint & (rf == ps) & (rs == pf)
    same_unordered_pair = same_orientation | reversed_orientation
    changed_pair = joint & ~same_unordered_pair
    lost_two_hit = rest_two & ~pose_two
    gained_two_hit = ~rest_two & pose_two

    joint_count = int(np.count_nonzero(joint))
    same_pair_count = int(np.count_nonzero(same_unordered_pair))
    reversed_count = int(np.count_nonzero(reversed_orientation))
    changed_count = int(np.count_nonzero(changed_pair))

    return {
        "ray_count": int(len(rf)),
        "rest_two_hit": int(np.count_nonzero(rest_two)),
        "pose_two_hit": int(np.count_nonzero(pose_two)),
        "joint_two_hit": joint_count,
        "same_unordered_pair": same_pair_count,
        "same_pair_same_order": int(np.count_nonzero(same_orientation)),
        "same_pair_reversed_order": reversed_count,
        "changed_pair": changed_count,
        "lost_two_hit": int(np.count_nonzero(lost_two_hit)),
        "gained_two_hit": int(np.count_nonzero(gained_two_hit)),
        "same_pair_flip_fraction": float(reversed_count / same_pair_count) if same_pair_count else 0.0,
        "joint_pair_membership_drift_fraction": float(changed_count / joint_count) if joint_count else 0.0,
        "rest_two_hit_retention_fraction": float(joint_count / int(np.count_nonzero(rest_two))) if np.any(rest_two) else 0.0,
    }


def _sample_pixels(mask_path: Path, resolution: int, *, subject: str, view_index: int, budget: int) -> np.ndarray:
    if budget <= 0:
        raise ValueError("VNEXT_MOTION_RELATION_RAY_BUDGET_NONPOSITIVE")
    mask = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
    if mask.size != resolution * resolution:
        raise ValueError("VNEXT_MOTION_RELATION_MASK_SIZE_DRIFT")
    mask = mask.reshape(resolution, resolution)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("VNEXT_MOTION_RELATION_MASK_EMPTY")
    rng = np.random.default_rng(_seed(f"relation-motion-audit-v1:{subject}:view:{view_index}"))
    order = rng.permutation(len(xs))[: min(len(xs), int(budget))]
    return np.stack((xs[order], ys[order]), axis=1).astype(np.int64)


def _audit_subject(
    *,
    display_id: str,
    source_id: str,
    teacher_path: Path,
    teacher_receipt_path: Path,
    boot: dict[str, Any],
    stage01: Path,
    candidate_rays: int,
    max_poses: int,
) -> dict[str, Any]:
    receipt = json.loads(teacher_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("subject_id") != source_id:
        raise ValueError("VNEXT_MOTION_RELATION_TEACHER_SUBJECT_DRIFT:" + display_id)
    if receipt.get("source", {}).get("sha256") != boot["raw_source"]["sha256"]:
        raise ValueError("VNEXT_MOTION_RELATION_TEACHER_SOURCE_HASH_DRIFT:" + display_id)

    with np.load(teacher_path, allow_pickle=False) as data:
        required = {"rest_vertices_source_world", "posed_vertices_source_world", "faces"}
        if not required.issubset(set(data.files)):
            raise ValueError("VNEXT_MOTION_RELATION_TEACHER_FIELDS_MISSING:" + display_id)
        rest = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
        posed = np.asarray(data["posed_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    if rest.ndim != 2 or rest.shape[1] != 3 or posed.ndim != 3 or posed.shape[1:] != rest.shape:
        raise ValueError("VNEXT_MOTION_RELATION_TEACHER_SHAPE_INVALID:" + display_id)
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(rest)):
        raise ValueError("VNEXT_MOTION_RELATION_FACES_INVALID:" + display_id)
    if not np.isfinite(rest).all() or not np.isfinite(posed).all():
        raise ValueError("VNEXT_MOTION_RELATION_NONFINITE:" + display_id)

    frame_rows = list(receipt.get("sampled_frames") or [])
    if len(frame_rows) != len(posed):
        raise ValueError("VNEXT_MOTION_RELATION_FRAME_COUNT_DRIFT:" + display_id)
    pose_indices = np.arange(len(posed), dtype=np.int64)
    if max_poses > 0 and len(pose_indices) > max_poses:
        pose_indices = np.unique(np.linspace(0, len(posed) - 1, num=max_poses, dtype=np.int64))

    camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
    cameras = list(json.loads(camera_bundle_path.read_text(encoding="utf-8"))["cameras"])
    if len(cameras) != 8:
        raise ValueError("VNEXT_MOTION_RELATION_CAMERA_COUNT_DRIFT:" + display_id)

    base = stage01 / "observations" / source_id
    views: list[dict[str, Any]] = []
    aggregate = {
        "same_unordered_pair": 0,
        "same_pair_reversed_order": 0,
        "joint_two_hit": 0,
        "changed_pair": 0,
        "rest_two_hit": 0,
        "lost_two_hit": 0,
    }

    for view_index, camera in enumerate(cameras):
        resolution = int(camera["resolution"])
        pixels = _sample_pixels(
            base / f"V{view_index}.mask.bin",
            resolution,
            subject=source_id,
            view_index=view_index,
            budget=candidate_rays,
        )
        origins, directions = _camera_rays(camera, pixels)
        rest_hits = _ray_triangle_first_two_parallel_grid(origins, directions, rest, faces)
        pose_rows: list[dict[str, Any]] = []
        for pose_index in pose_indices.tolist():
            pose_hits = _ray_triangle_first_two_parallel_grid(origins, directions, posed[pose_index], faces)
            row = compare_first_two_hits_v1(rest_hits, pose_hits)
            row.update(
                {
                    "pose_index": int(pose_index),
                    "action": str(frame_rows[pose_index].get("action")),
                    "frame": int(frame_rows[pose_index].get("frame")),
                }
            )
            pose_rows.append(row)
            for key in aggregate:
                aggregate[key] += int(row[key])
        views.append(
            {
                "view_index": int(view_index),
                "sampled_ray_count": int(len(pixels)),
                "pose_count": int(len(pose_rows)),
                "poses": pose_rows,
            }
        )

    same_pair = int(aggregate["same_unordered_pair"])
    joint = int(aggregate["joint_two_hit"])
    rest_two = int(aggregate["rest_two_hit"])
    aggregate.update(
        {
            "same_pair_flip_fraction": float(aggregate["same_pair_reversed_order"] / same_pair) if same_pair else 0.0,
            "joint_pair_membership_drift_fraction": float(aggregate["changed_pair"] / joint) if joint else 0.0,
            "rest_two_hit_loss_fraction": float(aggregate["lost_two_hit"] / rest_two) if rest_two else 0.0,
        }
    )

    pose_dependence_observed = bool(
        aggregate["same_pair_reversed_order"]
        or aggregate["changed_pair"]
        or aggregate["lost_two_hit"]
    )
    return {
        "subject_id": display_id,
        "source_subject_id": source_id,
        "teacher_export_sha256": _sha(teacher_path),
        "teacher_receipt_sha256": _sha(teacher_receipt_path),
        "camera_bundle_sha256": _sha(camera_bundle_path),
        "exported_pose_count": int(len(posed)),
        "audited_pose_count": int(len(pose_indices)),
        "views": views,
        "aggregate": aggregate,
        "pose_dependence_observed": pose_dependence_observed,
        "training_disposition": "COURT_ONLY__CURRENT_HEAD_HAS_NO_EXPLICIT_POSE_CONDITIONING",
    }


def main() -> None:
    args = _args()
    if args.candidate_rays_per_view <= 0 or args.max_poses_per_subject < 0:
        raise ValueError("VNEXT_MOTION_RELATION_POLICY_INVALID")
    teacher_dir = args.teacher_dir.expanduser().resolve()
    stage01 = args.stage01_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    rows = {row["subject_id"]: row for row in bootstrap["subjects"]}

    subjects = []
    for display_id in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display_id]
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not teacher_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError("VNEXT_MOTION_RELATION_TEACHER_MISSING:" + display_id)
        subjects.append(
            _audit_subject(
                display_id=display_id,
                source_id=source_id,
                teacher_path=teacher_path,
                teacher_receipt_path=receipt_path,
                boot=rows[source_id],
                stage01=stage01,
                candidate_rays=int(args.candidate_rays_per_view),
                max_poses=int(args.max_poses_per_subject),
            )
        )

    total_same_pair = sum(int(s["aggregate"]["same_unordered_pair"]) for s in subjects)
    total_flips = sum(int(s["aggregate"]["same_pair_reversed_order"]) for s in subjects)
    total_joint = sum(int(s["aggregate"]["joint_two_hit"]) for s in subjects)
    total_changed = sum(int(s["aggregate"]["changed_pair"]) for s in subjects)
    pose_subjects = [s["subject_id"] for s in subjects if s["pose_dependence_observed"]]

    payload = {
        "schema": _SCHEMA,
        "status": _STATUS,
        "purpose": "QUALIFY_AUTHORED_MOTION_RELATION_EVIDENCE_BEFORE_ANY_V10_SUPERVISION",
        "ray_backend": _RAY_BACKEND,
        "teacher_inputs_in_inference_forward": False,
        "training_authorized": False,
        "scientific_fit_started": False,
        "policy": {
            "candidate_rays_per_view": int(args.candidate_rays_per_view),
            "max_poses_per_subject": int(args.max_poses_per_subject),
            "same_stage01_camera_per_rest_and_pose": True,
            "rest_triangle_connectivity_reused_for_all_poses": True,
        },
        "conditioning_fact": {
            "current_relation_head": "SEALED_H_QUERY_PLUS_CAMERA_FEATURES",
            "explicit_pose_or_motion_state": False,
            "admission_rule": "AUTHORED_MOTION_LABELS_REMAIN_COURT_ONLY_UNTIL_IDENTIFIABILITY_IS_PROVEN_OR_POSE_CONDITIONING_IS_ADDED",
        },
        "aggregate": {
            "same_unordered_pair": total_same_pair,
            "same_pair_reversed_order": total_flips,
            "same_pair_flip_fraction": float(total_flips / total_same_pair) if total_same_pair else 0.0,
            "joint_two_hit": total_joint,
            "changed_pair": total_changed,
            "joint_pair_membership_drift_fraction": float(total_changed / total_joint) if total_joint else 0.0,
            "pose_dependence_subject_count": int(len(pose_subjects)),
            "pose_dependence_subjects": pose_subjects,
        },
        "subjects": subjects,
        "source_bindings": {
            "stage01_bootstrap_sha256": _sha(bootstrap_path),
        },
        "source_provenance": _source_provenance(),
    }
    payload["value_sha256"] = _canonical_sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_AUTHORED_MOTION_RELATION_CONDITIONING_AUDIT_COMPLETE",
        f"subjects={len(subjects)}",
        f"pose_dependent_subjects={len(pose_subjects)}",
        f"same_pair_flip_fraction={payload['aggregate']['same_pair_flip_fraction']:.8f}",
        f"value_sha256={payload['value_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
