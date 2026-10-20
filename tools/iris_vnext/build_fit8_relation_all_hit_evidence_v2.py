from __future__ import annotations

"""Build tie-safe, source-defensible all-hit relation evidence for Fit8.

V2 preserves the exact V1 ray sampling/intersection machinery and changes only
label authority at equal depth:

- pairwise front is valid only when the two depths differ beyond the subject
  depth-tie epsilon;
- global visibility is valid-visible only for a unique minimum-depth hit;
- minimum-depth ties abstain for every tied minimum;
- deeper hits remain valid-invisible even when the minimum is tied.

No static owner/cut/attachment truth is inferred. Occlusion is positive same-ray
support only. This tool never authorizes optimizer/backward steps.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from tools.iris_vnext.build_fit8_vnext_shards_v1 import SUBJECT_SOURCE_IDS, _camera_rays, _normalize_source_world, _sha
from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v1 import (
    _canonical_sha,
    _face_edge_components,
    _ray_triangle_all_parallel_grid,
    _sample_pixels,
)

_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidence.v2"
_MANIFEST_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidenceManifest.v2"
_STATUS = "MATERIALIZED__TIE_SAFE_EVIDENCE_ONLY__NO_TRAINING_AUTHORIZATION"
_GRID_SIZE = 32


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-shard-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--stage01-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-rays-per-view", type=int, default=8192)
    return parser.parse_args()


def _source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "tools/iris_vnext/build_fit8_relation_all_hit_evidence_v1.py",
        repo / "models/iris/vnext/relation_truth_v3_adapter_v1.py",
        repo / "models/iris/vnext/cohort_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def depth_tie_epsilon(vertices: np.ndarray) -> float:
    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError("VNEXT_ALL_HIT_V2_REST_INVALID")
    scale = max(1.0, float(np.max(value) - np.min(value)), float(np.max(np.abs(value))))
    return float(scale * 1.0e-10)


def stack_visibility_evidence(depths: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depths, dtype=np.float64).reshape(-1)
    if not np.isfinite(depth).all() or np.any(depth <= 0.0) or np.any(np.diff(depth) < 0.0):
        raise ValueError("VNEXT_ALL_HIT_V2_STACK_DEPTH_INVALID")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("VNEXT_ALL_HIT_V2_TIE_EPS_INVALID")
    probability = np.zeros(len(depth), dtype=np.float32)
    valid = np.ones(len(depth), dtype=np.bool_)
    tied_minimum = np.zeros(len(depth), dtype=np.bool_)
    if not len(depth):
        return probability, valid, tied_minimum
    minimum_mask = np.abs(depth - depth[0]) <= float(epsilon)
    if int(np.count_nonzero(minimum_mask)) == 1:
        probability[0] = 1.0
    else:
        valid[minimum_mask] = False
        tied_minimum[minimum_mask] = True
    return probability, valid, tied_minimum


def _flat_stack_visibility(all_hits: dict[str, np.ndarray], epsilon: float) -> dict[str, np.ndarray]:
    offsets = np.asarray(all_hits["offsets"], dtype=np.int64)
    depths = np.asarray(all_hits["depth"], dtype=np.float64)
    probability = np.zeros(len(depths), dtype=np.float32)
    valid = np.zeros(len(depths), dtype=np.bool_)
    tied = np.zeros(len(depths), dtype=np.bool_)
    for ray in range(len(offsets) - 1):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        p, v, t = stack_visibility_evidence(depths[start:end], epsilon)
        probability[start:end] = p
        valid[start:end] = v
        tied[start:end] = t
    return {
        "stack_visibility_probability": probability,
        "stack_visibility_valid": valid,
        "stack_tied_minimum": tied,
    }


def oriented_pair_evidence_v2(
    *, all_hits: dict[str, np.ndarray], pixels: np.ndarray, center: np.ndarray, half: float, epsilon: float
) -> dict[str, np.ndarray]:
    offsets = np.asarray(all_hits["offsets"], dtype=np.int64)
    faces = np.asarray(all_hits["face"], dtype=np.int64)
    points = np.asarray(all_hits["point"], dtype=np.float64)
    depths = np.asarray(all_hits["depth"], dtype=np.float64)
    if offsets.shape != (len(pixels) + 1,) or len(faces) != len(points) or len(faces) != len(depths):
        raise ValueError("VNEXT_ALL_HIT_V2_CSR_SHAPE_DRIFT")
    vis = _flat_stack_visibility(all_hits, epsilon)

    columns: dict[str, list[Any]] = {
        key: [] for key in (
            "point_a", "point_b", "face_a", "face_b", "rank_a", "rank_b", "ray_index", "pixel_xy",
            "depth_a", "depth_b", "front_probability", "front_valid", "visibility_probability_a",
            "visibility_probability_b", "visibility_valid_a", "visibility_valid_b", "occlusion_positive_support",
            "base_pair_id",
        )
    }
    pair_id = 0
    for ray in range(len(pixels)):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        count = end - start
        if count < 2:
            continue
        stack_faces, stack_points, stack_depths = faces[start:end], points[start:end], depths[start:end]
        for left in range(count - 1):
            for right in range(left + 1, count):
                for reverse in (False, True):
                    ra, rb = (right, left) if reverse else (left, right)
                    da, db = float(stack_depths[ra]), float(stack_depths[rb])
                    front_valid = abs(da - db) > float(epsilon)
                    pa = _normalize_source_world(stack_points[ra], center, half).astype(np.float32)
                    pb = _normalize_source_world(stack_points[rb], center, half).astype(np.float32)
                    flat_a, flat_b = start + ra, start + rb
                    columns["point_a"].append(pa); columns["point_b"].append(pb)
                    columns["face_a"].append(int(stack_faces[ra])); columns["face_b"].append(int(stack_faces[rb]))
                    columns["rank_a"].append(int(ra)); columns["rank_b"].append(int(rb))
                    columns["ray_index"].append(int(ray)); columns["pixel_xy"].append(tuple(int(x) for x in pixels[ray]))
                    columns["depth_a"].append(da); columns["depth_b"].append(db)
                    columns["front_probability"].append(float(front_valid and da < db)); columns["front_valid"].append(front_valid)
                    columns["visibility_probability_a"].append(float(vis["stack_visibility_probability"][flat_a]))
                    columns["visibility_probability_b"].append(float(vis["stack_visibility_probability"][flat_b]))
                    columns["visibility_valid_a"].append(bool(vis["stack_visibility_valid"][flat_a]))
                    columns["visibility_valid_b"].append(bool(vis["stack_visibility_valid"][flat_b]))
                    columns["occlusion_positive_support"].append(1.0)
                    columns["base_pair_id"].append(pair_id)
                pair_id += 1

    n = len(columns["front_probability"])
    if n % 2:
        raise RuntimeError("VNEXT_ALL_HIT_V2_ORIENTATION_CARDINALITY_DRIFT")
    result = {
        "point_a": np.asarray(columns["point_a"], np.float32).reshape(n, 3),
        "point_b": np.asarray(columns["point_b"], np.float32).reshape(n, 3),
        "face_a": np.asarray(columns["face_a"], np.int64), "face_b": np.asarray(columns["face_b"], np.int64),
        "rank_a": np.asarray(columns["rank_a"], np.int16), "rank_b": np.asarray(columns["rank_b"], np.int16),
        "ray_index": np.asarray(columns["ray_index"], np.int32),
        "pixel_xy": np.asarray(columns["pixel_xy"], np.int32).reshape(n, 2),
        "depth_a": np.asarray(columns["depth_a"], np.float64), "depth_b": np.asarray(columns["depth_b"], np.float64),
        "front_probability": np.asarray(columns["front_probability"], np.float32),
        "front_valid": np.asarray(columns["front_valid"], np.bool_),
        "visibility_probability_a": np.asarray(columns["visibility_probability_a"], np.float32),
        "visibility_probability_b": np.asarray(columns["visibility_probability_b"], np.float32),
        "visibility_valid_a": np.asarray(columns["visibility_valid_a"], np.bool_),
        "visibility_valid_b": np.asarray(columns["visibility_valid_b"], np.bool_),
        "occlusion_positive_support": np.asarray(columns["occlusion_positive_support"], np.float32),
        "base_pair_id": np.asarray(columns["base_pair_id"], np.int64),
    }
    result.update(vis)
    return result


def _view_stats(all_hits: dict[str, np.ndarray], pairs: dict[str, np.ndarray], ray_count: int) -> dict[str, int | float]:
    stack = np.diff(np.asarray(all_hits["offsets"], np.int64))
    rows = len(pairs["front_probability"])
    fv = np.asarray(pairs["front_valid"], bool)
    va = np.asarray(pairs["visibility_valid_a"], bool); vb = np.asarray(pairs["visibility_valid_b"], bool)
    front = np.asarray(pairs["front_probability"]) >= 0.5
    visa = np.asarray(pairs["visibility_probability_a"]) >= 0.5; visb = np.asarray(pairs["visibility_probability_b"]) >= 0.5
    comparable = fv & va & vb
    decoupled = int(np.count_nonzero(comparable & ((front & ~visa) | ((~front) & ~visb))))
    return {
        "sampled_rays": int(ray_count), "foreground_hit_rays": int(np.count_nonzero(stack >= 1)),
        "two_or_more_hit_rays": int(np.count_nonzero(stack >= 2)), "three_or_more_hit_rays": int(np.count_nonzero(stack >= 3)),
        "maximum_stack_depth": int(stack.max()) if len(stack) else 0, "total_hits": int(np.asarray(all_hits["offsets"])[-1]),
        "base_pair_count": int(rows // 2), "oriented_pair_rows": int(rows),
        "front_abstain_rows": int(np.count_nonzero(~fv)),
        "visibility_abstain_pair_slots": int(np.count_nonzero(~va) + np.count_nonzero(~vb)),
        "deep_hidden_pair_rows": int(np.count_nonzero((pairs["rank_a"] > 0) & (pairs["rank_b"] > 0))),
        "front_visibility_decoupled_rows": decoupled,
        "front_visibility_decoupled_fraction": float(decoupled / max(1, int(np.count_nonzero(comparable)))),
    }


def main() -> None:
    args = _args()
    if int(args.candidate_rays_per_view) <= 0:
        raise ValueError("VNEXT_ALL_HIT_V2_POLICY_NONPOSITIVE")
    legacy_dir, teacher_dir = args.legacy_shard_dir.resolve(), args.teacher_dir.resolve()
    stage01, output = args.stage01_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("VNEXT_ALL_HIT_V2_OUTPUT_DIR_NOT_EMPTY")
    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    subjects: list[dict[str, Any]] = []

    for display_id in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display_id]
        legacy_path = legacy_dir / f"{source_id}.pt"
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not legacy_path.is_file() or not teacher_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError("VNEXT_ALL_HIT_V2_INPUT_MISSING:" + display_id)
        boot = bootstrap_rows[source_id]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("subject_id") != source_id or receipt.get("source", {}).get("sha256") != boot["raw_source"]["sha256"]:
            raise ValueError("VNEXT_ALL_HIT_V2_TEACHER_PROVENANCE_DRIFT:" + display_id)
        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
        center = np.asarray(legacy["center_xyz"], np.float64).reshape(3)
        half = float(torch.as_tensor(legacy["normalization_half_extent"]).item())
        with np.load(teacher_path, allow_pickle=False) as data:
            rest = np.asarray(data["rest_vertices_source_world"], np.float64)
            faces = np.asarray(data["faces"], np.int64)
        epsilon = depth_tie_epsilon(rest)
        face_component = _face_edge_components(faces)
        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        cameras = list(json.loads(camera_bundle_path.read_text(encoding="utf-8"))["cameras"])
        if len(cameras) != 8:
            raise ValueError("VNEXT_ALL_HIT_V2_CAMERA_COUNT_DRIFT:" + display_id)
        arrays: dict[str, np.ndarray] = {"subject_depth_tie_epsilon": np.asarray([epsilon], np.float64)}
        views: list[dict[str, Any]] = []
        observation_root = stage01 / "observations" / source_id
        for view_index, camera in enumerate(cameras):
            resolution = int(camera["resolution"])
            pixels = _sample_pixels(observation_root / f"V{view_index}.mask.bin", resolution, source_id=source_id, view_index=view_index, budget=int(args.candidate_rays_per_view))
            origins, directions = _camera_rays(camera, pixels)
            all_hits = _ray_triangle_all_parallel_grid(origins, directions, rest, faces, face_component)
            pairs = oriented_pair_evidence_v2(all_hits=all_hits, pixels=pixels, center=center, half=half, epsilon=epsilon)
            prefix = f"v{view_index}_"
            arrays[prefix + "pixels_xy"] = pixels.astype(np.int32)
            arrays[prefix + "stack_offsets"] = all_hits["offsets"].astype(np.int64)
            arrays[prefix + "stack_face"] = all_hits["face"].astype(np.int64)
            arrays[prefix + "stack_depth"] = all_hits["depth"].astype(np.float64)
            for name, value in pairs.items(): arrays[prefix + name] = value
            stats = _view_stats(all_hits, pairs, len(pixels)); stats["view_index"] = view_index
            stats["camera_sha256"] = _canonical_sha(camera); stats["mask_sha256"] = _sha(observation_root / f"V{view_index}.mask.bin")
            views.append(stats)
        out_path = output / f"{display_id.lower().replace('-', '').replace(' ', '')}.all_hit_relation_evidence_v2.npz"
        np.savez_compressed(out_path, **arrays)
        subjects.append({
            "subject_id": display_id, "source_subject_id": source_id, "file": out_path.name, "sha256": _sha(out_path), "bytes": out_path.stat().st_size,
            "depth_tie_epsilon": epsilon, "teacher_export_sha256": _sha(teacher_path), "teacher_receipt_sha256": _sha(receipt_path),
            "legacy_shard_sha256": _sha(legacy_path), "source_asset_sha256": boot["raw_source"]["sha256"], "camera_bundle_sha256": _sha(camera_bundle_path),
            "normalization": {"authority": "EXACT_LEGACY_IRIS_SHARD", "center_xyz": [float(x) for x in center], "half_extent": half}, "views": views,
        })
    manifest = {
        "schema": _MANIFEST_SCHEMA, "status": _STATUS, "subject_order": list(SUBJECT_IDS), "subjects": subjects,
        "policy": {"candidate_rays_per_view": int(args.candidate_rays_per_view), "grid_size": _GRID_SIZE,
                   "same_component_same_depth_duplicates_merged": True, "distinct_component_equal_depth_hits_retained": True,
                   "all_unordered_stack_pairs_emitted_both_orientations": True, "pair_point_coordinate_domain": "EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD",
                   "depth_tie_epsilon": "SUBJECT_SCALE_MAX_1_SPAN_MAXABS_TIMES_1E-10"},
        "authority": {"front": "PAIRWISE_STRICT_DEPTH_ORDER__EQUAL_DEPTH_WITHIN_EPS_ABSTAIN",
                      "visibility": "UNIQUE_MINIMUM_VISIBLE__TIED_MINIMA_ABSTAIN__DEEPER_VALID_INVISIBLE",
                      "occlusion": "POSITIVE_SUPPORT_ONLY__SAME_RAY_PAIR__NO_BINARY_NEGATIVE_AUTHORITY",
                      "projected_boundary": "NOT_EMITTED__KEEP_SEPARATE_CURRENT_VIEW_RGBA_AUTHORITY",
                      "static_same_cut_attachment": "NOT_EMITTED__NO_NEW_AUTHORITY"},
        "training_authorized": False, "scientific_fit_started": False, "teacher_inputs_in_inference_forward": False,
        "source_bindings": {"stage01_bootstrap_sha256": _sha(bootstrap_path)}, "source_provenance": _source_provenance(),
    }
    manifest["manifest_value_sha256"] = _canonical_sha(manifest)
    path = output / "ALL_HIT_RELATION_EVIDENCE_V2_MANIFEST.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rows = sum(v["oriented_pair_rows"] for s in subjects for v in s["views"])
    abstain = sum(v["front_abstain_rows"] for s in subjects for v in s["views"])
    decoupled = sum(v["front_visibility_decoupled_rows"] for s in subjects for v in s["views"])
    print("VNEXT_FIT8_ALL_HIT_RELATION_EVIDENCE_V2_PASS", f"subjects={len(subjects)}", f"oriented_rows={rows}", f"front_abstain_rows={abstain}", f"decoupled_rows={decoupled}", flush=True)


if __name__ == "__main__":
    main()
