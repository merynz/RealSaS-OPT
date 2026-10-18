from __future__ import annotations

"""Build source-audited IRIS vNext relation-truth V2 sidecars.

This producer intentionally does not reinterpret or overwrite historical Fit8
shards. It creates a new relation-only corpus whose target validity matches the
post-V8 truth/conditioning contract:

- static same/cut/attachment targets abstain unless a future source authority is
  added explicitly;
- front/occlusion/visibility come from exact first/second ray ordering;
- projected alpha/appearance boundaries come from the current source RGBA view;
- target values, validity, evidence authority and provenance remain explicit;
- every two-hit ray is emitted in both pair orientations, giving exact balanced
  directional front/occlusion/visibility supervision without fabricated fallback
  negatives.

The output is training evidence only. Teacher source data is never placed on an
inference forward path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.iris.vnext.boundary_causality_v1 import EvidenceAuthority
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.graphics_truth_v2 import STATIC_RELATION_KEYS_V2, VIEW_RELATION_KEYS_V2
from tools.iris_vnext.build_fit8_vnext_shards_v1 import (
    SUBJECT_SOURCE_IDS,
    _boundary_flags,
    _camera_rays,
    _normalize_source_world,
    _seed,
    _sha,
)
from tools.iris_vnext.build_fit8_vnext_shards_v2 import (
    _RAY_BACKEND,
    _ray_triangle_first_two_parallel_grid,
)

_SCHEMA = "RealSaS.IRISVNextFit8RelationTruth.v2"
_MANIFEST_SCHEMA = "RealSaS.IRISVNextFit8RelationTruthManifest.v2"
_STATIC_AUTHORITY = EvidenceAuthority.DIAGNOSTIC_ONLY.value
_VIEW_AUTHORITY = EvidenceAuthority.DETERMINISTIC_DERIVED.value


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-shard-dir", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-rays-per-view", type=int, default=8192)
    p.add_argument("--max-oriented-pairs-per-view", type=int, default=2048)
    return p.parse_args()


def _source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/graphics_truth_v2.py",
        repo / "models/iris/vnext/boundary_causality_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
    )
    return {str(path.relative_to(repo)): _sha(path) for path in paths}


def _empty_static_targets(count: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = {name: np.zeros(count, dtype=np.float32) for name in STATIC_RELATION_KEYS_V2}
    validity = {name: np.zeros(count, dtype=np.bool_) for name in STATIC_RELATION_KEYS_V2}
    return values, validity


def _orient_two_hit_rows_v2(
    *,
    point_front: np.ndarray,
    point_back: np.ndarray,
    alpha_boundary: np.ndarray,
    appearance_boundary: np.ndarray,
    max_oriented_pairs: int,
) -> dict[str, np.ndarray]:
    """Emit front/back rows in both orientations with exact balanced labels."""

    front = np.asarray(point_front, dtype=np.float32)
    back = np.asarray(point_back, dtype=np.float32)
    alpha = np.asarray(alpha_boundary, dtype=np.float32).reshape(-1)
    appearance = np.asarray(appearance_boundary, dtype=np.float32).reshape(-1)
    if front.ndim != 2 or front.shape[1] != 3 or back.shape != front.shape:
        raise ValueError("VNEXT_RELATION_TRUTH_V2_POINT_SHAPE_INVALID")
    if len(alpha) != len(front) or len(appearance) != len(front):
        raise ValueError("VNEXT_RELATION_TRUTH_V2_BOUNDARY_SHAPE_INVALID")
    if int(max_oriented_pairs) <= 0 or int(max_oriented_pairs) % 2:
        raise ValueError("VNEXT_RELATION_TRUTH_V2_ORIENTED_PAIR_BUDGET_MUST_BE_POSITIVE_EVEN")

    base_count = min(len(front), int(max_oriented_pairs) // 2)
    if base_count == 0:
        return {
            "point_a": np.zeros((0, 3), dtype=np.float32),
            "point_b": np.zeros((0, 3), dtype=np.float32),
            **{name: np.zeros(0, dtype=np.float32) for name in VIEW_RELATION_KEYS_V2},
            **{f"{name}__valid": np.zeros(0, dtype=np.bool_) for name in VIEW_RELATION_KEYS_V2},
        }

    front = front[:base_count]
    back = back[:base_count]
    alpha = alpha[:base_count]
    appearance = appearance[:base_count]

    point_a = np.empty((base_count * 2, 3), dtype=np.float32)
    point_b = np.empty((base_count * 2, 3), dtype=np.float32)
    point_a[0::2], point_b[0::2] = front, back
    point_a[1::2], point_b[1::2] = back, front

    ones = np.ones(base_count, dtype=np.float32)
    zeros = np.zeros(base_count, dtype=np.float32)

    def interleave(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        out = np.empty(base_count * 2, dtype=np.float32)
        out[0::2], out[1::2] = pos, neg
        return out

    projected_alpha = np.repeat(alpha, 2).astype(np.float32, copy=False)
    projected_appearance = np.repeat(appearance, 2).astype(np.float32, copy=False)
    count = base_count * 2
    valid = np.ones(count, dtype=np.bool_)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "front_probability": interleave(ones, zeros),
        "occlusion_probability": interleave(ones, zeros),
        "visibility_probability_a": interleave(ones, zeros),
        "visibility_probability_b": interleave(zeros, ones),
        "projected_alpha_boundary_probability": projected_alpha,
        "projected_appearance_boundary_probability": projected_appearance,
        **{f"{name}__valid": valid.copy() for name in VIEW_RELATION_KEYS_V2},
    }


def _binary_stats(value: np.ndarray, valid: np.ndarray) -> dict[str, int | float]:
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    m = np.asarray(valid, dtype=np.bool_).reshape(-1)
    if len(v) != len(m):
        raise ValueError("VNEXT_RELATION_TRUTH_V2_STATS_SHAPE_INVALID")
    active = v[m]
    positives = int(np.count_nonzero(active >= 0.5))
    negatives = int(len(active) - positives)
    return {
        "total": int(len(v)),
        "valid": int(len(active)),
        "abstained": int(len(v) - len(active)),
        "positive": positives,
        "negative": negatives,
        "positive_fraction": float(positives / len(active)) if len(active) else 0.0,
    }


def _view_truth(
    *,
    source_id: str,
    view_index: int,
    camera: dict,
    png_path: Path,
    mask_path: Path,
    rest_source: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    half: float,
    candidate_rays: int,
    max_oriented_pairs: int,
) -> tuple[dict, dict]:
    rgba = np.asarray(Image.open(png_path).convert("RGBA"), dtype=np.uint8)
    resolution = int(camera["resolution"])
    if rgba.shape != (resolution, resolution, 4):
        raise ValueError("VNEXT_RELATION_TRUTH_V2_RGBA_DIMENSION_DRIFT")
    mask = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8).reshape(resolution, resolution)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("VNEXT_RELATION_TRUTH_V2_SOURCE_MASK_EMPTY")

    rng = np.random.default_rng(_seed(f"relation-v2:{source_id}:view:{view_index}"))
    order = rng.permutation(len(xs))
    choice = order[: min(len(order), int(candidate_rays))]
    pixels = np.stack((xs[choice], ys[choice]), axis=1).astype(np.int64)
    origins, directions = _camera_rays(camera, pixels)
    hits = _ray_triangle_first_two_parallel_grid(origins, directions, rest_source, faces)

    two_hit = np.flatnonzero(
        (hits["first_face"] >= 0)
        & (hits["second_face"] >= 0)
        & (hits["first_face"] != hits["second_face"])
    )
    selected = two_hit[: int(max_oriented_pairs) // 2]
    alpha, appearance = _boundary_flags(rgba, pixels[selected])
    point_front = _normalize_source_world(hits["first_point"][selected], center, half)
    point_back = _normalize_source_world(hits["second_point"][selected], center, half)
    view = _orient_two_hit_rows_v2(
        point_front=point_front,
        point_back=point_back,
        alpha_boundary=alpha,
        appearance_boundary=appearance,
        max_oriented_pairs=max_oriented_pairs,
    )
    count = int(len(view["point_a"]))
    static_values, static_validity = _empty_static_targets(count)

    payload = {
        "schema": _SCHEMA,
        "view_index": int(view_index),
        "camera_conditioned": True,
        "point_a": torch.from_numpy(view.pop("point_a")),
        "point_b": torch.from_numpy(view.pop("point_b")),
        "static_truth": {name: torch.from_numpy(static_values[name]) for name in STATIC_RELATION_KEYS_V2},
        "static_validity": {name: torch.from_numpy(static_validity[name]) for name in STATIC_RELATION_KEYS_V2},
        "static_authority": {name: _STATIC_AUTHORITY for name in STATIC_RELATION_KEYS_V2},
        "view_truth": {name: torch.from_numpy(view[name]) for name in VIEW_RELATION_KEYS_V2},
        "view_validity": {
            name: torch.from_numpy(view[f"{name}__valid"]) for name in VIEW_RELATION_KEYS_V2
        },
        "view_authority": {name: _VIEW_AUTHORITY for name in VIEW_RELATION_KEYS_V2},
        "source_binding": {
            "rgba_sha256": _sha(png_path),
            "mask_sha256": _sha(mask_path),
        },
        "provenance": {
            "static_truth": {
                "same_drawable_affinity": "ABSTAIN:generic_FIT8_3D_does_not_identify_authoritative_drawable_owner",
                "cut_probability": "ABSTAIN:generic_FIT8_3D_does_not_identify_authoritative_drawable_cut",
                "attachment_affinity": "ABSTAIN:no_independent_attachment_truth_in_relation_v2_source_contract",
            },
            "view_truth": {
                "front_probability": "DETERMINISTIC_DERIVED:first_second_exact_ray_order",
                "occlusion_probability": "DETERMINISTIC_DERIVED:same_pixel_first_hit_occludes_second_hit",
                "visibility_probability_a": "DETERMINISTIC_DERIVED:pair_orientation_and_exact_ray_order",
                "visibility_probability_b": "DETERMINISTIC_DERIVED:pair_orientation_and_exact_ray_order",
                "projected_alpha_boundary_probability": "DETERMINISTIC_DERIVED:current_view_source_RGBA_3x3_alpha_range_gt_0.20",
                "projected_appearance_boundary_probability": "DETERMINISTIC_DERIVED:current_view_source_RGBA_3x3_RGB_distance_gt_0.20",
            },
        },
    }
    stats = {
        "view_index": int(view_index),
        "candidate_rays": int(len(pixels)),
        "two_hit_rays": int(len(two_hit)),
        "selected_two_hit_rays": int(len(selected)),
        "oriented_rows": count,
        "source_binding": dict(payload["source_binding"]),
        "static": {
            name: _binary_stats(static_values[name], static_validity[name])
            for name in STATIC_RELATION_KEYS_V2
        },
        "view": {
            name: _binary_stats(
                payload["view_truth"][name].numpy(),
                payload["view_validity"][name].numpy(),
            )
            for name in VIEW_RELATION_KEYS_V2
        },
    }
    return payload, stats


def main():
    args = _args()
    if int(args.candidate_rays_per_view) <= 0:
        raise ValueError("VNEXT_RELATION_TRUTH_V2_CANDIDATE_RAYS_NONPOSITIVE")
    if int(args.max_oriented_pairs_per_view) <= 0 or int(args.max_oriented_pairs_per_view) % 2:
        raise ValueError("VNEXT_RELATION_TRUTH_V2_PAIR_BUDGET_MUST_BE_POSITIVE_EVEN")

    legacy_dir = args.legacy_shard_dir.resolve()
    teacher_dir = args.teacher_dir.resolve()
    stage01 = args.stage01_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    manifest_subjects = []

    for display in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display]
        legacy_path = legacy_dir / f"{source_id}.pt"
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        teacher_receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not legacy_path.is_file() or not teacher_path.is_file() or not teacher_receipt_path.is_file():
            raise FileNotFoundError(f"VNEXT_RELATION_TRUTH_V2_INPUT_MISSING:{display}")

        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
        if str(legacy.get("subject_id")) != source_id:
            raise ValueError("VNEXT_RELATION_TRUTH_V2_LEGACY_SUBJECT_DRIFT:" + display)
        boot = bootstrap_rows[source_id]
        receipt = json.loads(teacher_receipt_path.read_text(encoding="utf-8"))
        if receipt["source"]["sha256"] != boot["raw_source"]["sha256"]:
            raise ValueError("VNEXT_RELATION_TRUTH_V2_TEACHER_SOURCE_HASH_DRIFT:" + display)

        with np.load(teacher_path, allow_pickle=False) as data:
            rest_source = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int64)

        center = np.asarray(legacy["center_xyz"], dtype=np.float64).reshape(3)
        half = float(torch.as_tensor(legacy["normalization_half_extent"]).item())
        if not np.isfinite(half) or half <= 0:
            raise ValueError("VNEXT_RELATION_TRUTH_V2_NORMALIZATION_INVALID")

        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        cameras = list(json.loads(camera_bundle_path.read_text(encoding="utf-8"))["cameras"])
        if len(cameras) != 8:
            raise ValueError("VNEXT_RELATION_TRUTH_V2_CAMERA_COUNT_DRIFT")

        base = stage01 / "observations" / source_id
        views = []
        view_stats = []
        for view_index, camera in enumerate(cameras):
            payload, stats = _view_truth(
                source_id=source_id,
                view_index=view_index,
                camera=camera,
                png_path=base / f"V{view_index}.png",
                mask_path=base / f"V{view_index}.mask.bin",
                rest_source=rest_source,
                faces=faces,
                center=center,
                half=half,
                candidate_rays=args.candidate_rays_per_view,
                max_oriented_pairs=args.max_oriented_pairs_per_view,
            )
            views.append(payload)
            view_stats.append(stats)

        subject_payload = {
            "schema": _SCHEMA,
            "subject_id": display,
            "source_subject_id": source_id,
            "teacher_inputs_in_inference_forward": False,
            "conditioning": {
                "static_relations": "VIEW_INVARIANT_ONLY",
                "view_relations": "CAMERA_CONDITIONED",
                "projected_boundaries": "VIEW_RELATIONS_ONLY",
            },
            "views": views,
        }
        path = output / f"{display.lower().replace('-', '').replace(' ', '')}.relation_truth_v2.pt"
        torch.save(subject_payload, path)
        manifest_subjects.append(
            {
                "subject_id": display,
                "source_subject_id": source_id,
                "file": path.name,
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "legacy_shard_sha256": _sha(legacy_path),
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_receipt_sha256": _sha(teacher_receipt_path),
                "stage01_bootstrap_sha256": _sha(bootstrap_path),
                "camera_bundle_sha256": _sha(camera_bundle_path),
                "views": view_stats,
            }
        )

    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "builder": "tools/iris_vnext/build_fit8_relation_truth_v2.py",
        "source_provenance": _source_provenance(),
        "ray_backend": _RAY_BACKEND,
        "subjects": manifest_subjects,
        "subject_order": list(SUBJECT_IDS),
        "teacher_inputs_in_inference_forward": False,
        "authority_policy": {
            "static": _STATIC_AUTHORITY,
            "view": _VIEW_AUTHORITY,
        },
        "static_target_policy": "ABSTAIN_UNLESS_INDEPENDENT_SOURCE_AUTHORITY_EXISTS",
        "view_target_policy": "EXACT_RAY_ORDER_PLUS_CURRENT_VIEW_PROJECTED_BOUNDARY_EVIDENCE",
        "pair_orientation_policy": "TWO_HIT_RAY_BOTH_ORIENTATIONS_EXACT_BALANCE",
        "policy": {
            "candidate_rays_per_view": int(args.candidate_rays_per_view),
            "max_oriented_pairs_per_view": int(args.max_oriented_pairs_per_view),
        },
    }
    manifest["manifest_value_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = output / "RELATION_TRUTH_V2_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    print(
        "VNEXT_FIT8_RELATION_TRUTH_V2_PASS",
        "subjects=8",
        f"ray_backend={_RAY_BACKEND}",
        f"manifest={manifest['manifest_value_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
