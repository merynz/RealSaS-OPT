from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)

EXPECTED = {
    "candidate_lineage_hash": "c1db3416db84ca49f06dec3a22f8864fb54a7898a098f99975937e764d732909",
    "skeleton_lineage_hash": "e54548b5dbd39e399fbd901edaef57efee167b665c9628f1f38820fa62c34850",
    "skin_lineage_hash": "b30417b9375a52c646618f0fb94ea8ddd7285fea959e460522dd5f8b6b70cb2a",
    "camera_set_hash": "79bf4b15bd1ecd1a1ace7a978237a294dddab8ca0b9402917407d2f767bb7224",
    "motion_json_sha256": "bfcfc1e44f4960e498f6f5183e224b72ed74e0352906c94f09989abab41b1b89",
    "mapping_sha256": "6d9e45124443a911f142e8a22c15c51ea455493591e9db26d728a21a1082044f",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _face_vertex_indices(candidate) -> np.ndarray:
    index = {
        str(v.candidate_vertex_id): i
        for i, v in enumerate(candidate.vertices)
    }
    out = []
    for face in candidate.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(x not in index for x in ids):
            raise RuntimeError("BASELINE_CAUSAL_FACE_INDEX_DRIFT")
        out.append(tuple(index[x] for x in ids))
    return np.asarray(out, dtype=np.int64)


def _signed_double_area(tri_xy: np.ndarray) -> np.ndarray:
    a = tri_xy[:, 1] - tri_xy[:, 0]
    b = tri_xy[:, 2] - tri_xy[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _edge_lengths(tri_xy: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 0], axis=1),
            np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 1], axis=1),
            np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 2], axis=1),
        ],
        axis=1,
    )


def _per_face_metrics(
    rest_tri_xy: np.ndarray,
    posed_tri_xy: np.ndarray,
    *,
    measurable_double_area_px2: float = 1.0,
) -> dict[str, np.ndarray]:
    rest_da = _signed_double_area(rest_tri_xy)
    posed_da = _signed_double_area(posed_tri_xy)
    valid = np.abs(rest_da) > 1.0e-9
    measurable = np.abs(rest_da) >= float(measurable_double_area_px2)

    area_ratio = np.full((len(rest_da),), np.nan, dtype=np.float64)
    area_ratio[valid] = np.abs(posed_da[valid]) / np.abs(rest_da[valid])
    flipped = valid & (rest_da * posed_da < 0.0)

    rest_edge = _edge_lengths(rest_tri_xy)
    posed_edge = _edge_lengths(posed_tri_xy)
    edge_ratio = posed_edge / np.maximum(rest_edge, 1.0e-9)
    max_edge_ratio = np.max(edge_ratio, axis=1)

    return {
        "valid": valid,
        "measurable": measurable,
        "flipped": flipped,
        "area_ratio": area_ratio,
        "max_edge_ratio": max_edge_ratio,
        "rest_double_area": rest_da,
        "posed_double_area": posed_da,
    }


def _summary(metric: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    select = np.asarray(metric["valid"], dtype=bool)
    if mask is not None:
        select &= np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(select))
    if count == 0:
        return {
            "face_count": 0,
            "flip_count": 0,
            "flip_rate": None,
            "area_ratio_p95": None,
            "area_ratio_p99": None,
            "area_ratio_max": None,
            "edge_ratio_p95": None,
            "edge_ratio_p99": None,
            "edge_ratio_max": None,
        }
    area = metric["area_ratio"][select]
    edge = metric["max_edge_ratio"][select]
    flip_count = int(np.count_nonzero(metric["flipped"] & select))
    return {
        "face_count": count,
        "flip_count": flip_count,
        "flip_rate": float(flip_count / count),
        "area_ratio_p95": float(np.nanpercentile(area, 95.0)),
        "area_ratio_p99": float(np.nanpercentile(area, 99.0)),
        "area_ratio_max": float(np.nanmax(area)),
        "edge_ratio_p95": float(np.nanpercentile(edge, 95.0)),
        "edge_ratio_p99": float(np.nanpercentile(edge, 99.0)),
        "edge_ratio_max": float(np.nanmax(edge)),
    }


def _skin_discontinuity(W: np.ndarray, face_index: np.ndarray) -> np.ndarray:
    wf = W[face_index]
    d01 = np.sum(np.abs(wf[:, 0] - wf[:, 1]), axis=1)
    d12 = np.sum(np.abs(wf[:, 1] - wf[:, 2]), axis=1)
    d20 = np.sum(np.abs(wf[:, 2] - wf[:, 0]), axis=1)
    return np.maximum(np.maximum(d01, d12), d20)


def _face_local_homogenized_triangles(
    *,
    rest: np.ndarray,
    face_index: np.ndarray,
    W: np.ndarray,
    joint_ids: tuple[str, ...],
    skin_matrices: dict,
) -> np.ndarray:
    hom = np.concatenate(
        [rest, np.ones((len(rest), 1), dtype=np.float64)], axis=1
    )
    per_joint = np.stack(
        [
            (hom @ np.asarray(skin_matrices[jid], dtype=np.float64).T)[:, :3]
            for jid in joint_ids
        ],
        axis=1,
    )
    face_mean_w = np.mean(W[face_index], axis=1)
    if not np.allclose(face_mean_w.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise RuntimeError("BASELINE_CAUSAL_FACE_MEAN_WEIGHT_NOT_SIMPLEX")

    out = np.empty((len(face_index), 3, 3), dtype=np.float64)
    for local in range(3):
        rows = per_joint[face_index[:, local]]
        out[:, local] = np.sum(rows * face_mean_w[:, :, None], axis=1)
    return out


def _project_triangles(tri_xyz: np.ndarray, camera) -> np.ndarray:
    flat = tri_xyz.reshape(-1, 3)
    projected = np.asarray(project_points_xyz_v3(flat, camera), dtype=np.float64)
    return projected[:, :2].reshape(len(tri_xyz), 3, 2)


def _risk_bins(discontinuity: np.ndarray) -> list[tuple[str, np.ndarray]]:
    d = np.asarray(discontinuity, dtype=np.float64)
    return [
        ("L1_LE_0_10", d <= 0.10),
        ("L1_0_10_TO_0_50", (d > 0.10) & (d <= 0.50)),
        ("L1_0_50_TO_1_00", (d > 0.50) & (d <= 1.00)),
        ("L1_1_00_TO_1_50", (d > 1.00) & (d <= 1.50)),
        ("L1_1_50_TO_1_90", (d > 1.50) & (d <= 1.90)),
        ("L1_GT_1_90", d > 1.90),
    ]


def run(*, authority_root: Path, run_id: str, out_dir: Path) -> None:
    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))

    actual_authority = {
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "camera_set_hash": camera_set.camera_set_hash,
    }
    for key, expected in EXPECTED.items():
        if key in actual_authority and str(actual_authority[key]) != expected:
            raise RuntimeError(
                f"BASELINE_CAUSAL_AUTHORITY_DRIFT:{key}:"
                f"{actual_authority[key]}:{expected}"
            )

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    motion_path = (
        ctx["run_root"]
        / "inputs"
        / "motion"
        / "quaternius_knight_v1"
        / "demo_run_v1.motion.json"
    )
    motion_sha = _sha256_file(motion_path)
    if motion_sha != EXPECTED["motion_json_sha256"]:
        raise RuntimeError("BASELINE_CAUSAL_MOTION_HASH_DRIFT")
    payload = json.loads(motion_path.read_text())

    tracks, mapping_details = _tracks_for_clip(
        payload, skeleton, cameras, source_report
    )
    mapping_sha = _json_hash(mapping_details)
    if mapping_sha != EXPECTED["mapping_sha256"]:
        raise RuntimeError("BASELINE_CAUSAL_MAPPING_HASH_DRIFT")

    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    face_index = _face_vertex_indices(candidate)
    discontinuity = _skin_discontinuity(W, face_index)

    duration = float(payload["duration_seconds"])
    sample_times = np.linspace(
        0.0,
        duration,
        4,
        endpoint=not bool(payload.get("loop")),
    )

    report = {
        "schema": "RealSaS.KnightBaselineMechanicalCausality.v1",
        "status": "MEASURED_CAUSAL_COUNTERFACTUAL",
        "claim_boundary": (
            "FACE_LOCAL_WEIGHT_HOMOGENIZATION_IS_DIAGNOSTIC_ONLY; "
            "MOTION, RETARGET, SKELETON, CAMERA, REST_GEOMETRY, AND JOINT "
            "TRANSFORMS ARE HELD FIXED."
        ),
        "authority": actual_authority,
        "motion_json_sha256": motion_sha,
        "mapping_sha256": mapping_sha,
        "face_count": int(len(face_index)),
        "skin_discontinuity": {
            "mean": float(np.mean(discontinuity)),
            "p95": float(np.percentile(discontinuity, 95.0)),
            "p99": float(np.percentile(discontinuity, 99.0)),
            "gt_0_5_count": int(np.count_nonzero(discontinuity > 0.5)),
            "gt_1_0_count": int(np.count_nonzero(discontinuity > 1.0)),
            "gt_1_5_count": int(np.count_nonzero(discontinuity > 1.5)),
            "gt_1_9_count": int(np.count_nonzero(discontinuity > 1.9)),
        },
        "frames": [],
    }

    rest_face_xyz = rest[face_index]
    for view in (0, 2):
        rest_tri_xy = _project_triangles(rest_face_xyz, cameras[view])
        for frame_index, time_seconds in enumerate(sample_times):
            skin_mats, _joint_pos, frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(time_seconds),
                cameras=cameras,
            )
            posed = _skin(rest, W, joint_ids, skin_mats)
            original_tri_xy = _project_triangles(posed[face_index], cameras[view])

            homogenized_face_xyz = _face_local_homogenized_triangles(
                rest=rest,
                face_index=face_index,
                W=W,
                joint_ids=joint_ids,
                skin_matrices=skin_mats,
            )
            homogenized_tri_xy = _project_triangles(
                homogenized_face_xyz, cameras[view]
            )

            original = _per_face_metrics(rest_tri_xy, original_tri_xy)
            homogenized = _per_face_metrics(rest_tri_xy, homogenized_tri_xy)
            measurable = original["measurable"]

            bins = []
            for label, mask in _risk_bins(discontinuity):
                bins.append(
                    {
                        "label": label,
                        "face_count": int(np.count_nonzero(mask)),
                        "original_all": _summary(original, mask),
                        "original_measurable": _summary(
                            original, mask & measurable
                        ),
                        "homogenized_all": _summary(homogenized, mask),
                        "homogenized_measurable": _summary(
                            homogenized, mask & measurable
                        ),
                    }
                )

            original_all = _summary(original)
            original_measurable = _summary(original, measurable)
            homogenized_all = _summary(homogenized)
            homogenized_measurable = _summary(homogenized, measurable)

            original_flip = original["flipped"] & measurable
            repaired_flip = homogenized["flipped"] & measurable
            flip_eliminated = original_flip & ~repaired_flip

            report["frames"].append(
                {
                    "view_index": int(view),
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                    "motion_frame_hash": frame_hash,
                    "original_all": original_all,
                    "original_measurable": original_measurable,
                    "homogenized_all": homogenized_all,
                    "homogenized_measurable": homogenized_measurable,
                    "measurable_flip_eliminated_count": int(
                        np.count_nonzero(flip_eliminated)
                    ),
                    "measurable_original_flip_count": int(
                        np.count_nonzero(original_flip)
                    ),
                    "measurable_flip_elimination_fraction": (
                        float(
                            np.count_nonzero(flip_eliminated)
                            / max(1, np.count_nonzero(original_flip))
                        )
                    ),
                    "risk_bins": bins,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "REPORT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "KNIGHT_BASELINE_MECHANICAL_CAUSALITY_PASS",
        json.dumps(
            {
                "candidate_lineage_hash": candidate.candidate_lineage_hash,
                "motion_json_sha256": motion_sha,
                "mapping_sha256": mapping_sha,
                "skin_discontinuity": report["skin_discontinuity"],
                "report": str(path),
            },
            sort_keys=True,
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir),
    )


if __name__ == "__main__":
    main()
