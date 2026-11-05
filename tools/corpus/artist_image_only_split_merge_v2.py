#!/usr/bin/env python3
"""Image-only split/merge precommit. Never inspect original artist author slots.

The observation is exactly one flattened RGBA image; the source/native project,
meshes, controls, part count and artist interventions are never model inputs.
The output is *candidate* image regions, not graphics ownership or a puppet.

Split: connected Felzenszwalb appearance regions at two precommitted scales.
Merge: deterministic adjacent appearance/edge-consistency objective without
target-K, semantics, learned labels or any private edit-reference optimization.
Court: separate executable action, AFTER SHA-locking predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure
from skimage.measure import label as connected_label
from skimage.segmentation import felzenszwalb

from tools.corpus.artist_image_only_edit_baseline_v1 import (
    INPUT_SCHEMA, COURT_SCHEMA as V1_COURT_SCHEMA, PRED_SCHEMA as V1_PRED_SCHEMA,
    canonical_labels, court as v1_court, digest, read_image,
)

PRED_SCHEMA = "RealSaS.ImageOnlyGenericSplitMergeCandidate.v2"
COURT_SCHEMA = "RealSaS.ImageOnlyGenericSplitMergeEditSupportCourt.v2"
PARAMS = {
    "seed_graph": {"scale": 180, "sigma": 0.8, "min_size": 140},
    "fine_graph": {"scale": 75, "sigma": 0.8, "min_size": 55},
    "merge_conservative": {
        "max_score": 0.085, "max_border": 0.11, "max_mean": 0.18
    },
    "merge_adaptive": {
        "max_score": 0.135, "max_border": 0.18, "max_mean": 0.26
    },
    "fine_split_merge": {
        "max_score": 0.105, "max_border": 0.14, "max_mean": 0.23
    },
    "score": "0.55_boundary_RGB_jump+0.30_region_RGB_mean_jump+0.15_normalized_Ward",
    "alpha_foreground": 0,
    "adjacency": "FOUR_CONNECTED_FOREGROUND_ONLY",
    "max_visible_regions": 16384,
}
METHODS = ("merge_conservative", "merge_adaptive", "fine_split_merge")
MAX_REGIONS = PARAMS["max_visible_regions"]


def require(ok, code):
    if not ok:
        raise ValueError(code)


def supported_labels(a, foreground):
    a = np.asarray(a, dtype=np.int64)
    require(a.shape == foreground.shape, "GENERIC_SPLIT_REGION_SHAPE")
    # Explicitly cut transparency bridges, including graph components whose
    # foreground fragments share one segment via invisible background.
    # Label by 4-connected equal-value components; the negative value is BG.
    image = np.where(foreground, a, -1)
    split = connected_label(
        image, background=-1, connectivity=1
    )
    result, count = canonical_labels(split, foreground)
    require(count <= MAX_REGIONS, "GENERIC_SPLIT_TOO_MANY_VISIBLE_REGIONS")
    return result, count


def build_seed(rgba, params):
    require(rgba.ndim == 3 and rgba.shape[2] == 4 and rgba.dtype == np.uint8,
            "GENERIC_SPLIT_ONLY_RGBA8888")
    foreground = rgba[..., 3] > PARAMS["alpha_foreground"]
    require(bool(foreground.any()), "GENERIC_SPLIT_EMPTY_FOREGROUND")
    rgb = np.array(rgba[..., :3], copy=True)
    rgb[~foreground] = 0
    raw = felzenszwalb(rgb,
                       scale=params["scale"],
                       sigma=params["sigma"],
                       min_size=params["min_size"],
                       channel_axis=-1)
    return supported_labels(raw, foreground)[0]


def _region_stats(labels, rgb):
    valid = labels >= 0
    ids = labels[valid]
    n = int(ids.max(initial=-1)) + 1
    require(0 < n <= MAX_REGIONS, "GENERIC_SPLIT_REGION_CEILING")
    pixels = rgb[valid].astype(np.float64) / 255.
    area = np.bincount(ids, minlength=n).astype(np.int64)
    require(np.all(area > 0), "GENERIC_SPLIT_REGION_GAP")
    totals = np.stack(
        [np.bincount(ids, weights=pixels[:, i], minlength=n) for i in range(3)],
        axis=-1
    )
    return area, totals


def _adjacency(labels, rgb):
    """Per native pixel-boundary: sum RGB discontinuities and edge lengths."""
    n = int(labels.max(initial=-1)) + 1
    pairs = []
    jumps = []
    f = rgb.astype(np.float32) / 255.
    for left, right, c1, c2 in (
        (labels[:, :-1], labels[:, 1:], f[:, :-1], f[:, 1:]),
        (labels[:-1, :], labels[1:, :], f[:-1, :], f[1:, :]),
    ):
        boundary = (left >= 0) & (right >= 0) & (left != right)
        if not np.any(boundary):
            continue
        first = np.minimum(left[boundary], right[boundary]).astype(np.int64)
        second = np.maximum(left[boundary], right[boundary]).astype(np.int64)
        pairs.append(first * n + second)
        jumps.append(
            np.sqrt(np.mean((c1[boundary] - c2[boundary]) ** 2, axis=-1))
        )
    if not pairs:
        return n, {}
    key = np.concatenate(pairs)
    delta = np.concatenate(jumps)
    distinct, inverse = np.unique(key, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    edges = {
        (int(k // n), int(k % n)): (float(s), int(c))
        for k, s, c in zip(distinct, sums, counts)
    }
    return n, edges


def merge_regions(seed, rgba, policy):
    """Adjacent-graph greedy merge, no target count and no hidden edit signals.

    The border discontinuity is measured directly on original adjacent image
    pixels. Updating region means after each union avoids trusting original
    Felzenszwalb segment identity as a target.
    """
    fg = rgba[..., 3] > PARAMS["alpha_foreground"]
    labels, _ = supported_labels(seed, fg)
    n, edges = _adjacency(labels, rgba[..., :3])
    area, sums = _region_stats(labels, rgba[..., :3])
    require(n == len(area), "GENERIC_SPLIT_REGION_STATS_MISMATCH")
    active = np.ones(n, dtype=bool)
    version = np.zeros(n, dtype=np.int64)
    parents = np.arange(n, dtype=np.int32)
    neighbors = [dict() for _ in range(n)]
    for (a, b), edge in edges.items():
        neighbors[a][b] = edge
        neighbors[b][a] = edge

    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = int(parents[a])
        return int(a)

    def proposal(a, b):
        if not active[a] or not active[b] or b not in neighbors[a]:
            return None
        edge_sum, edge_count = neighbors[a][b]
        boundary = edge_sum / edge_count
        mean_a = sums[a] / area[a]
        mean_b = sums[b] / area[b]
        mean_dist = float(np.sqrt(np.mean((mean_a - mean_b) ** 2)))
        # Normalized Ward penalty; each candidate judged by how much
        # mean color disparity it introduces relative to both areas.
        ward = float(
            np.sqrt(area[a] * area[b]) / (area[a] + area[b]) * mean_dist
        )
        score = 0.55 * boundary + 0.30 * mean_dist + 0.15 * ward
        if (boundary > policy["max_border"] or
                mean_dist > policy["max_mean"] or
                score > policy["max_score"]):
            return None
        return float(score)

    heap = []

    def offer(a, b):
        a, b = min(a, b), max(a, b)
        score = proposal(a, b)
        if score is not None:
            heapq.heappush(
                heap, (score, a, b, int(version[a]), int(version[b]))
            )

    for a, b in edges:
        offer(a, b)

    accepted = 0
    while heap:
        score, a, b, va, vb = heapq.heappop(heap)
        if not active[a] or not active[b] or version[a] != va or version[b] != vb:
            continue
        if proposal(a, b) is None:
            continue

        # Always retain the smaller root for deterministic equivalence classes.
        active[b] = False
        parents[b] = a
        area[a] += area[b]
        sums[a] += sums[b]
        version[a] += 1
        version[b] += 1
        neighbors[a].pop(b, None)

        for c, (edge_sum, count) in list(neighbors[b].items()):
            if c == a or not active[c]:
                continue
            neighbors[c].pop(b, None)
            old_sum, old_count = neighbors[a].get(c, (0., 0))
            combined = (old_sum + edge_sum, old_count + count)
            neighbors[a][c] = combined
            neighbors[c][a] = combined
        neighbors[b].clear()

        for c in sorted(neighbors[a]):
            offer(a, c)
        accepted += 1
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    merged = np.full(labels.shape, -1, dtype=np.int32)
    merged[fg] = roots[labels[fg]]
    result, regions = canonical_labels(merged, fg)
    require(regions == n - accepted, "GENERIC_SPLIT_REGION_MERGE_ARITHMETIC")
    return result, {
        "input_regions": n,
        "accepted_source_image_only_merges": accepted,
        "output_regions": regions,
        "initial_visible_4_neighbor_region_boundaries": len(edges),
        "merge_policy": dict(policy),
    }


def predict_candidates(rgba):
    graph = build_seed(rgba, PARAMS["seed_graph"])
    fine = build_seed(rgba, PARAMS["fine_graph"])
    arrays = {}
    reports = {}
    for method in METHODS:
        seed = fine if method == "fine_split_merge" else graph
        arrays[method], reports[method] = merge_regions(
            seed, rgba, PARAMS[method]
        )
    return arrays, reports


def write_candidate(rgba_file, public_file, base_npz, base_manifest,
                    candidate_npz, candidate_manifest):
    rgba, public = read_image(rgba_file, public_file)
    parent = json.loads(Path(base_manifest).read_text())
    require(parent.get("schema") == V1_PRED_SCHEMA
            and parent.get("image_RGBA_SHA256") == public["image_RGBA_SHA256"]
            and parent.get("prediction_arrays_sha256") ==
                digest(Path(base_npz).read_bytes())
            and parent.get("reference_used_during_prediction") is False
            and parent.get("training_authorized") is False
            and parent.get("drawable_ownership_authorized") is False,
            "GENERIC_SPLIT_PARENT_BLIND_BASELINE_SEAL")
    with np.load(base_npz, allow_pickle=False) as d:
        require(set(d.files) == {"alpha_connected", "image_graph"}
                and d["image_graph"].shape == rgba.shape[:2],
                "GENERIC_SPLIT_PARENT_REGION_FORMAT")
    arrays, reports = predict_candidates(rgba)
    candidate_npz = Path(candidate_npz)
    candidate_manifest = Path(candidate_manifest)
    candidate_npz.parent.mkdir(parents=True, exist_ok=True)
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate_npz, **arrays)
    result = {
        "schema": PRED_SCHEMA,
        "status": "V2_IMAGE_ONLY_SPLIT_MERGE_PREDICTION_SEALED__NO_OWNER",
        "input_RGBA_SHA256": public["image_RGBA_SHA256"],
        "parent_blind_v1_prediction_sha256": parent["prediction_arrays_sha256"],
        "v2_candidate_arrays_sha256": digest(candidate_npz.read_bytes()),
        "image_dimensions": list(rgba.shape[:2]),
        "precommitted_algorithm_parameters": PARAMS,
        "algorithms": list(METHODS),
        "image_only_region_reports": reports,
        "source_mesh_face_incidence_read": False,
        "source_artist_slot_name_or_count_read": False,
        "original_artist_edit_reference_read_during_prediction": False,
        "native_source_controls_as_model_prior": False,
        "external_training_art_rights_verified": False,
        "artist_same_drawable_pairs_minted": 0,
        "artist_different_drawable_pairs_minted": 0,
        "graphics_owner_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
    }
    candidate_manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def score_edit_support(supports, labels, visible):
    """Same metrics as V1 (one-to-one Hungarian support matching)."""
    from scipy.optimize import linear_sum_assignment

    n = int(labels.max(initial=-1)) + 1
    sizes = np.bincount(labels[visible], minlength=n)
    target = np.stack(supports, axis=0)
    intersection = np.zeros((len(supports), n), np.int64)
    iou = np.zeros((len(supports), n), np.float64)
    flat = labels.ravel()
    for i, m in enumerate(target):
        valid_ids = flat[m.ravel()]
        hit = np.bincount(valid_ids[valid_ids >= 0], minlength=n)
        intersection[i] = hit
        denom = sizes + int(m.sum()) - hit
        np.divide(hit, denom, out=iou[i], where=denom > 0)
    r, c = linear_sum_assignment(-iou)
    return {
        "predicted_region_count": n,
        "matched_observable_source_edits": len(r),
        "observable_source_edits": len(supports),
        "mean_one_to_one_edit_support_iou_over_all_observable_edits":
            float(iou[r, c].sum() / len(supports)),
        "matched_edit_effect_pixel_recall_over_all_edit_effect_instances":
            float(intersection[r, c].sum() / target.sum()),
        "edit_support_pixel_instances": int(target.sum()),
        "mean_area_per_predicted_region": float(sizes.mean()),
        "max_area_fraction_among_foreground_regions":
            float(sizes.max() / visible.sum()),
    }


def read_private_edit_effects(rgba, public, private_json, private_dir):
    """Private information is court-only. No call path from predict."""
    d = json.loads(Path(private_json).read_text())
    require(d.get("schema") ==
                "RealSaS.IndependentOriginalArtistSlotRGBAHideIsolateCourtOnly.v1"
            and d.get("official_spine_runtime_reference") is True
            and d.get("original_model_input_RGBA_SHA256") ==
                public["image_RGBA_SHA256"]
            and d.get("original_hide_isolate_restoration_exact") is True
            and d.get("training_authorized") is False
            and d.get("native_editable_source_artist_project_file_verified") is False,
            "GENERIC_SPLIT_PRIVATE_SOURCE_COURT_ORIGINAL_AUTHORITY")
    rows = d["source_author_slot_hide_isolate_references_teacher_only"]
    require(len(rows) == d["active_original_author_slot_count_teacher_only"]
            and len(rows) > 0, "GENERIC_SPLIT_PRIVATE_SOURCE_EDIT_COUNT")
    ref = rgba.reshape(-1, 4)
    root = Path(private_dir).resolve()
    masks = []
    unseen = 0
    seen = set()
    for row in rows:
        i = row["original_slot_index_teacher_only"]
        require(type(i) is int and i >= 0 and i not in seen,
                "GENERIC_SPLIT_PRIVATE_EDIT_CASE_DUPLICATE")
        seen.add(i)
        hidden_path = (root / ("slot" + str(i) + "_hidden.rgba")).resolve()
        isolate_path = (root / ("slot" + str(i) + "_isolated.rgba")).resolve()
        require(hidden_path.parent == root and isolate_path.parent == root,
                "GENERIC_SPLIT_PRIVATE_EDIT_PATH_ESCAPE")
        hidden_raw = hidden_path.read_bytes()
        isolate_raw = isolate_path.read_bytes()
        require(len(hidden_raw) == len(isolate_raw) == ref.size
                and digest(hidden_raw) == row["original_hidden_source_RGBA_sha256"]
                and digest(isolate_raw) ==
                    row["original_isolated_source_RGBA_sha256"],
                "GENERIC_SPLIT_PRIVATE_SOURCE_EDIT_SHA")
        observed = np.any(
            np.frombuffer(hidden_raw, dtype=np.uint8).reshape(-1, 4) != ref,
            axis=1,
        ).reshape(rgba.shape[:2])
        require(int(observed.sum()) ==
                    row["original_hide_changes_composite_RGBA_pixel_count"]
                and int(np.count_nonzero(np.frombuffer(isolate_raw, dtype=np.uint8)
                                        .reshape(-1, 4)[:, 3])) ==
                    row["original_isolated_alpha_pixel_count"],
                "GENERIC_SPLIT_PRIVATE_SOURCE_EDIT_PIXEL_COUNTS")
        if observed.any():
            masks.append(observed)
        else:
            unseen += 1
    require(len(masks) > 0, "GENERIC_SPLIT_NO_OBSERVABLE_EDIT_CASES")
    return masks, unseen, len(rows)


def court_candidates(rgba_file, public_file, base_npz, base_manifest,
                     base_receipt, candidate_npz, candidate_manifest,
                     private_json, private_dir):
    rgba, public = read_image(rgba_file, public_file)
    parent = json.loads(Path(base_manifest).read_text())
    v1 = json.loads(Path(base_receipt).read_text())
    candidate = json.loads(Path(candidate_manifest).read_text())
    require(candidate.get("schema") == PRED_SCHEMA
            and candidate.get("status") ==
                "V2_IMAGE_ONLY_SPLIT_MERGE_PREDICTION_SEALED__NO_OWNER"
            and candidate.get("input_RGBA_SHA256") ==
                public["image_RGBA_SHA256"]
            and candidate.get("parent_blind_v1_prediction_sha256") ==
                parent.get("prediction_arrays_sha256") ==
                digest(Path(base_npz).read_bytes())
            and candidate.get("v2_candidate_arrays_sha256") ==
                digest(Path(candidate_npz).read_bytes())
            and candidate.get("precommitted_algorithm_parameters") == PARAMS
            and candidate.get("algorithms") == list(METHODS)
            and candidate.get("original_artist_edit_reference_read_during_prediction")
                is False
            and candidate.get("source_mesh_face_incidence_read") is False
            and candidate.get("source_artist_slot_name_or_count_read") is False
            and candidate.get("native_source_controls_as_model_prior") is False
            and candidate.get("artist_same_drawable_pairs_minted") == 0
            and candidate.get("artist_different_drawable_pairs_minted") == 0
            and candidate.get("graphics_owner_authorized") is False
            and candidate.get("training_authorized") is False
            and candidate.get("materialization_authorized") is False,
            "GENERIC_SPLIT_CANDIDATE_PREDICTION_NOT_FROZEN_OR_TRUTH_FORGERY")
    require(v1.get("schema") == V1_COURT_SCHEMA
            and v1.get("source_RGBA_SHA256") ==
                public["image_RGBA_SHA256"]
            and v1.get("frozen_prediction_sha256") ==
                parent.get("prediction_arrays_sha256")
            and v1.get("status") ==
                "SOURCE_ARTIST_EDIT_SUPPORT_BASELINES_MEASURED__NOT_DRAWABLE_TRUTH"
            and v1.get("training_authorized") is False
            and v1.get("graphics_owner_authorized") is False,
            "GENERIC_SPLIT_PARENT_REAL_INDEPENDENT_EDIT_COURT_SEAL")
    supports, unseen, total = read_private_edit_effects(
        rgba, public, private_json, private_dir
    )
    require(v1["author_hide_observable_cases"] == len(supports)
            and v1["author_hide_unobservable_masked_cases"] == unseen,
            "GENERIC_SPLIT_PARENT_V1_COURT_OBSERVATIONS_DRIFT")
    metrics = {}
    foreground = rgba[..., 3] > PARAMS["alpha_foreground"]
    with np.load(candidate_npz, allow_pickle=False) as data:
        require(set(data.files) == set(METHODS),
                "GENERIC_SPLIT_WRONG_METHOD_PREDICTIONS")
        for method in METHODS:
            labels = np.asarray(data[method])
            require(labels.dtype == np.int32
                    and labels.shape == rgba.shape[:2]
                    and np.all(labels[~foreground] == -1)
                    and np.all(labels[foreground] >= 0),
                    "GENERIC_SPLIT_INVALID_FOREGROUND_SUPPORT")
            n = int(labels.max(initial=-1)) + 1
            require(n == candidate["image_only_region_reports"][method][
                "output_regions"]
                and np.array_equal(np.unique(labels[foreground]), np.arange(n))
                and n <= MAX_REGIONS,
                "GENERIC_SPLIT_INCONSISTENT_SEALED_REGIONS")
            metrics[method] = score_edit_support(
                supports, labels, foreground
            )
    metrics0 = v1["metrics_mean_edit_support_iou_not_artist_membership_accuracy"]
    baseline = {
        method: {
            "predicted_region_count": metrics0[method]["predicted_region_count"],
            "mean_one_to_one_edit_support_iou_over_all_observable_edits":
                metrics0[method][
                    "mean_one_to_one_edit_support_iou_over_all_observable_edits"
                ],
            "matched_edit_effect_pixel_recall_over_all_edit_effect_instances":
                metrics0[method][
                    "matched_edit_effect_pixel_recall_over_all_edit_effect_instances"
                ]
        }
        for method in ("alpha_connected", "image_graph")
    }
    for method in METHODS:
        metrics[method]["delta_mean_edit_support_iou_vs_v1_image_graph"] = (
            metrics[method][
                "mean_one_to_one_edit_support_iou_over_all_observable_edits"
            ] - baseline["image_graph"][
                "mean_one_to_one_edit_support_iou_over_all_observable_edits"
            ]
        )
        metrics[method]["delta_effect_recall_vs_v1_image_graph"] = (
            metrics[method][
                "matched_edit_effect_pixel_recall_over_all_edit_effect_instances"
            ] - baseline["image_graph"][
                "matched_edit_effect_pixel_recall_over_all_edit_effect_instances"
            ]
        )
    return {
        "schema": COURT_SCHEMA,
        "status": "REAL_IMAGE_ONLY_GENERIC_SPLIT_MERGE_V2_EVALUATED__NO_ARTIST_OWNER",
        "source_RGBA_SHA256": public["image_RGBA_SHA256"],
        "source_composite_width_height": [
            int(rgba.shape[1]), int(rgba.shape[0])
        ],
        "v1_prediction_parent_sha256": parent["prediction_arrays_sha256"],
        "v2_sealed_prediction_sha256":
            candidate["v2_candidate_arrays_sha256"],
        "observed_original_author_hide_effects": len(supports),
        "masked_original_author_hide_effects": unseen,
        "source_active_attachments_count_court_only": total,
        "candidate_methods_precommitted": list(METHODS),
        "v1_image_only_baselines": baseline,
        "v2_image_only_candidate_results": metrics,
        "artist_layer_names_or_count_entered_prediction": False,
        "source_mechanics_control_identity_entered_prediction": False,
        "original_artist_edit_support_entered_prediction": False,
        "gold_artist_drawable_membership_measured": False,
        "source_draw_order_and_alpha_CAA_proven": False,
        "new_same_drawable_pairs_minted": 0,
        "new_different_drawable_pairs_minted": 0,
        "graphics_owner_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--rgba", required=True)
    pred.add_argument("--public-manifest", required=True)
    pred.add_argument("--baseline-npz", required=True)
    pred.add_argument("--baseline-manifest", required=True)
    pred.add_argument("--prediction-npz", required=True)
    pred.add_argument("--prediction-manifest", required=True)
    c = sub.add_parser("court")
    for arg in ("rgba", "public-manifest", "baseline-npz",
                "baseline-manifest", "baseline-receipt", "prediction-npz",
                "prediction-manifest", "private-reference", "private-dir",
                "receipt"):
        c.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    if args.cmd == "predict":
        result = write_candidate(
            args.rgba, args.public_manifest, args.baseline_npz,
            args.baseline_manifest, args.prediction_npz,
            args.prediction_manifest
        )
    else:
        result = court_candidates(
            args.rgba, args.public_manifest, args.baseline_npz,
            args.baseline_manifest, args.baseline_receipt,
            args.prediction_npz, args.prediction_manifest,
            args.private_reference, args.private_dir
        )
        Path(args.receipt).write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
