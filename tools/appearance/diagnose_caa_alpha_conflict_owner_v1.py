from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    caa_compile_artifact_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    _surface_sample_geometry_adaptive,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _load_source_inputs,
)


def _context(*, repo_root: Path, authority_root: Path, run_id: str):
    run_root = (authority_root / "runs" / run_id).resolve()
    ledger = json.loads((run_root / "ACTIVE_RUN_V2.json").read_text())
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    return {
        "repo_root": repo_root.resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": run_root,
        "run_id": run_id,
        "run_manifest_path": run_root / "run_manifest.json",
        "run_manifest": manifest,
        "stage": {"id": "CAA_ALPHA_CONFLICT_OWNER_DIAGNOSTIC"},
        "ledger": ledger,
    }


def _sample_distance(mask: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(xy, dtype=np.float64)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    ix = np.rint(points[:, 0]).astype(np.int64)
    iy = np.rint(points[:, 1]).astype(np.int64)
    ix = np.clip(ix, 0, mask.shape[1] - 1)
    iy = np.clip(iy, 0, mask.shape[0] - 1)
    cls = mask[iy, ix]
    dist = np.where(cls, inside[iy, ix], outside[iy, ix]).astype(np.float64)
    return dist, cls


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def diagnose(*, repo_root: Path, authority_root: Path, run_id: str) -> dict:
    ctx = _context(
        repo_root=repo_root,
        authority_root=authority_root,
        run_id=run_id,
    )
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    arrays = _load_compile_arrays(artifact)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    _rgba_by_view, masks_by_view = _load_source_inputs(ctx, observation)
    policy = json.loads(
        (
            repo_root
            / "canonical"
            / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
        ).read_text()
    )
    alpha_cut = float(
        policy["completion_quality_policy"]["cross_view_alpha_conflict_cut"]
    )

    valid = np.asarray(arrays["direct_valid"], dtype=bool)
    rgba = np.asarray(arrays["direct_rgba"], dtype=np.uint8)
    source_xy = np.asarray(arrays["source_xy"], dtype=np.float64)
    face_index = np.asarray(arrays["sample_face_index"], dtype=np.int32)
    resolutions = np.asarray(arrays["face_tile_resolutions"], dtype=np.int32)
    (
        _sample_positions,
        recomputed_face_index,
        _sample_component,
        _component_ids,
        face_normals,
        _offsets,
    ) = _surface_sample_geometry_adaptive(candidate, resolutions)
    recomputed_face_index = np.asarray(recomputed_face_index, dtype=np.int32)
    if not np.array_equal(recomputed_face_index, face_index):
        raise RuntimeError("CAA_ALPHA_DIAGNOSTIC_SAMPLE_FACE_RECOMPUTE_DRIFT")
    support = np.zeros((8, len(candidate.faces)), dtype=np.float64)
    camera_by_view = {int(camera.view_index): camera for camera in cameras.cameras}
    if set(camera_by_view) != set(range(8)):
        raise RuntimeError("CAA_ALPHA_DIAGNOSTIC_CAMERA_SET_INVALID")
    for view in range(8):
        forward = np.asarray(camera_by_view[view].forward, dtype=np.float64)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise RuntimeError("CAA_ALPHA_DIAGNOSTIC_CAMERA_FORWARD_INVALID")
        support[view] = np.abs(face_normals @ (forward / norm))
    if valid.ndim != 2 or valid.shape[0] != 8:
        raise RuntimeError("CAA_ALPHA_DIAGNOSTIC_DIRECT_VALID_SHAPE")
    sample_count = valid.shape[1]
    if (
        rgba.shape != (8, sample_count, 4)
        or source_xy.shape != (8, sample_count, 2)
        or face_index.shape != (sample_count,)
        or support.shape != (8, len(candidate.faces))
    ):
        raise RuntimeError("CAA_ALPHA_DIAGNOSTIC_ARRAY_SHAPE_DRIFT")

    distance_by_view = []
    class_by_view = []
    for view in range(8):
        dist, cls = _sample_distance(masks_by_view[view], source_xy[view])
        distance_by_view.append(dist)
        class_by_view.append(cls)
    distance_by_view = np.asarray(distance_by_view, dtype=np.float64)
    class_by_view = np.asarray(class_by_view, dtype=bool)

    pairs = []
    total_conflicts = 0
    total_shared = 0
    all_min_dist = []
    all_opposite = []
    top_examples = []

    for left in range(8):
        right = (left + 1) % 8
        shared = valid[left] & valid[right]
        ids = np.flatnonzero(shared)
        total_shared += int(len(ids))
        if not len(ids):
            pairs.append(
                {
                    "left_view_index": left,
                    "right_view_index": right,
                    "shared_direct_sample_count": 0,
                    "alpha_conflict_count": 0,
                    "alpha_conflict_fraction": 0.0,
                }
            )
            continue

        alpha = np.abs(
            rgba[left, ids, 3].astype(np.float64)
            - rgba[right, ids, 3].astype(np.float64)
        ) / 255.0
        left_class_all = class_by_view[left, ids]
        right_class_all = class_by_view[right, ids]
        same_class_all = left_class_all == right_class_all
        fg_fg_all = left_class_all & right_class_all
        bg_bg_all = (~left_class_all) & (~right_class_all)
        opposite_class_all = ~same_class_all

        conflict_local = alpha > alpha_cut
        conflict_ids = ids[conflict_local]
        conflict_alpha = alpha[conflict_local]
        conflict_count = int(len(conflict_ids))
        total_conflicts += conflict_count

        if conflict_count:
            ld = distance_by_view[left, conflict_ids]
            rd = distance_by_view[right, conflict_ids]
            md = np.minimum(ld, rd)
            opposite = (
                class_by_view[left, conflict_ids]
                != class_by_view[right, conflict_ids]
            )
            all_min_dist.append(md)
            all_opposite.append(opposite)
            bins = {
                "le_1px": _fraction(md <= 1.0),
                "le_2px": _fraction(md <= 2.0),
                "le_4px": _fraction(md <= 4.0),
                "le_8px": _fraction(md <= 8.0),
                "gt_8px": _fraction(md > 8.0),
            }
            order = np.argsort(-conflict_alpha, kind="stable")[:8]
            for local in order:
                sid = int(conflict_ids[int(local)])
                fid = int(face_index[sid])
                top_examples.append(
                    {
                        "left_view_index": left,
                        "right_view_index": right,
                        "sample_index": sid,
                        "face_index": fid,
                        "alpha_abs": float(conflict_alpha[int(local)]),
                        "left_alpha_u8": int(rgba[left, sid, 3]),
                        "right_alpha_u8": int(rgba[right, sid, 3]),
                        "left_xy": list(map(float, source_xy[left, sid])),
                        "right_xy": list(map(float, source_xy[right, sid])),
                        "left_silhouette_distance_px": float(
                            distance_by_view[left, sid]
                        ),
                        "right_silhouette_distance_px": float(
                            distance_by_view[right, sid]
                        ),
                        "left_foreground_class": bool(
                            class_by_view[left, sid]
                        ),
                        "right_foreground_class": bool(
                            class_by_view[right, sid]
                        ),
                        "left_face_support": float(support[left, fid]),
                        "right_face_support": float(support[right, fid]),
                    }
                )
        else:
            md = np.empty((0,), dtype=np.float64)
            opposite = np.empty((0,), dtype=bool)
            bins = {
                "le_1px": 0.0,
                "le_2px": 0.0,
                "le_4px": 0.0,
                "le_8px": 0.0,
                "gt_8px": 0.0,
            }

        def category_row(category_mask: np.ndarray) -> dict:
            category_count = int(np.count_nonzero(category_mask))
            category_conflicts = int(
                np.count_nonzero(conflict_local & category_mask)
            )
            return {
                "shared_sample_count": category_count,
                "alpha_conflict_count": category_conflicts,
                "alpha_conflict_fraction": (
                    float(category_conflicts) / float(category_count)
                    if category_count
                    else 0.0
                ),
            }

        pairs.append(
            {
                "left_view_index": left,
                "right_view_index": right,
                "shared_direct_sample_count": int(len(ids)),
                "alpha_conflict_count": conflict_count,
                "alpha_conflict_fraction": (
                    float(conflict_count) / float(len(ids))
                ),
                "source_silhouette_class_partition": {
                    "same_class": category_row(same_class_all),
                    "foreground_foreground": category_row(fg_fg_all),
                    "background_background": category_row(bg_bg_all),
                    "opposite_class": category_row(opposite_class_all),
                },
                "alpha_conflict_cut": alpha_cut,
                "conflict_min_silhouette_distance_px": {
                    "median": float(np.median(md)) if len(md) else None,
                    "p95": float(np.quantile(md, 0.95)) if len(md) else None,
                    **bins,
                },
                "opposite_foreground_class_fraction": (
                    _fraction(opposite) if len(opposite) else 0.0
                ),
            }
        )

    if all_min_dist:
        global_dist = np.concatenate(all_min_dist)
        global_opposite = np.concatenate(all_opposite)
    else:
        global_dist = np.empty((0,), dtype=np.float64)
        global_opposite = np.empty((0,), dtype=bool)

    top_examples.sort(
        key=lambda row: (
            -row["alpha_abs"],
            row["left_view_index"],
            row["sample_index"],
        )
    )
    return {
        "schema": "RealSaS.CAAAlphaConflictOwnerDiagnostic.v1",
        "status": "MEASURED",
        "run_id": run_id,
        "source_stage": "21_CAA_COMPILE",
        "alpha_conflict_cut": alpha_cut,
        "shared_direct_sample_count": int(total_shared),
        "alpha_conflict_count": int(total_conflicts),
        "alpha_conflict_fraction": (
            float(total_conflicts) / float(total_shared)
            if total_shared
            else 0.0
        ),
        "global_conflict_min_silhouette_distance_px": {
            "median": (
                float(np.median(global_dist)) if len(global_dist) else None
            ),
            "p95": (
                float(np.quantile(global_dist, 0.95))
                if len(global_dist)
                else None
            ),
            "le_1px": _fraction(global_dist <= 1.0),
            "le_2px": _fraction(global_dist <= 2.0),
            "le_4px": _fraction(global_dist <= 4.0),
            "le_8px": _fraction(global_dist <= 8.0),
            "gt_8px": _fraction(global_dist > 8.0),
        },
        "opposite_foreground_class_fraction": (
            _fraction(global_opposite) if len(global_opposite) else 0.0
        ),
        "per_pair": pairs,
        "top_conflict_examples": top_examples[:32],
        "interpretation_contract": {
            "near_silhouette_conflicts_implicate": (
                "VISIBILITY_OR_PROJECTION_OR_SOURCE_SAMPLING"
            ),
            "far_interior_conflicts_implicate": (
                "CANONICAL_GEOMETRY_OR_CORRESPONDENCE_OR_SOURCE_ART_VIEW_DEPENDENCE"
            ),
            "diagnostic_does_not_assign_root_cause_by_itself": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = diagnose(
        repo_root=Path(args.repo_root).resolve(),
        authority_root=Path(args.authority_root).resolve(),
        run_id=args.run_id,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("CAA_ALPHA_CONFLICT_OWNER_DIAGNOSTIC", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
