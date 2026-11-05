#!/usr/bin/env python3
"""Original-art image-only *baseline*, followed by a separately invoked blind court.

This is NOT a RealSaS learned owner/puppet inference system. Inference sees only
one RGBA composite and its public dimensions/hash. It never reads an editor
project, source mesh, art slots, attachments, artist layer count or the court.
The downstream independent court tests visible HIDE consequences, not the
artist's true layer membership or an authoritative SAME_DRAWABLE label.

Predict before opening any private original-artist reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT_SCHEMA = "RealSaS.IndependentHumanArtNativeRGBAImageOnlyInput.v1"
PRED_SCHEMA = "RealSaS.ArtistBlindImageOnlyRegionBaselines.v1"
COURT_SCHEMA = "RealSaS.ArtistBlindImageOnlyEditSupportDiagnostic.v1"
PARAMS = {"graph_scale": 180, "graph_sigma": 0.8, "graph_min_size": 140,
          "foreground_alpha_threshold": 0}
MAX_REGIONS = 16384


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_image(image_path, public_path):
    public = json.loads(Path(public_path).read_text())
    require(public.get("schema") == INPUT_SCHEMA
            and public.get("consumer_image_only_source_input") is True
            and public.get("source_mesh_face_incidence_present") is False
            and public.get("original_artist_slot_names_present") is False
            and public.get("original_artist_layer_count_present") is False
            and public.get("source_bone_rig_or_attachment_names_present") is False
            and public.get("training_authorized") is False,
            "IMAGE_ONLY_REQUIRES_EXACT_PUBLIC_INPUT")
    w, h = public.get("width"), public.get("height")
    require(type(w) is int and type(h) is int and 0 < w <= 2048
            and 0 < h <= 2048
            and public.get("format") == "RGBA8888_UNPREMULTIPLIED",
            "IMAGE_ONLY_RGBA_FORMAT")
    raw = Path(image_path).read_bytes()
    require(len(raw) == w * h * 4
            and digest(raw) == public.get("image_RGBA_SHA256"),
            "IMAGE_ONLY_INPUT_BYTES_SHA")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).copy(), public


def canonical_labels(raw_labels, valid):
    a = np.asarray(raw_labels, np.int64)
    require(a.shape == valid.shape and np.isfinite(a).all(),
            "IMAGE_ONLY_BAD_REGION_SHAPE")
    distinct, mapped = np.unique(a[valid], return_inverse=True)
    require(len(distinct) > 0 and len(distinct) <= MAX_REGIONS,
            "IMAGE_ONLY_REGION_COUNT_OUT_OF_BOUNDS")
    out = np.full(a.shape, -1, dtype=np.int32)
    out[valid] = mapped.astype(np.int32)
    return out, int(len(distinct))


def predict(rgba):
    """Only the composite enters either baseline; no preset part count."""
    from scipy.ndimage import label, generate_binary_structure
    from skimage.segmentation import felzenszwalb

    require(rgba.ndim == 3 and rgba.shape[2] == 4
            and rgba.dtype == np.uint8, "IMAGE_ONLY_RGBA_LAYOUT")
    valid = rgba[:, :, 3] > PARAMS["foreground_alpha_threshold"]
    require(int(valid.sum()) > 0, "IMAGE_ONLY_NO_VISIBLE_ART")
    cc_raw, _ = label(valid, structure=generate_binary_structure(2, 1))
    alpha_cc, n_cc = canonical_labels(cc_raw, valid)
    # Transparent RGB is suppressed before graph segmentation.
    rgb = rgba[:, :, :3].copy()
    rgb[~valid] = 0
    graph_raw = felzenszwalb(
        rgb, scale=PARAMS["graph_scale"], sigma=PARAMS["graph_sigma"],
        min_size=PARAMS["graph_min_size"], channel_axis=-1,
    )
    graph, n_graph = canonical_labels(graph_raw, valid)
    return {"alpha_connected": alpha_cc, "image_graph": graph}, {
        "alpha_connected": n_cc, "image_graph": n_graph
    }


def write_prediction(image, public_path, npz_path, manifest_path):
    rgba, public = read_image(image, public_path)
    arrays, counts = predict(rgba)
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    require(npz_path.resolve() != manifest_path.resolve(),
            "IMAGE_ONLY_PREDICTION_DESTINATIONS_COLLIDE")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    result = {
        "schema": PRED_SCHEMA,
        "status": "IMAGE_ONLY_PREDICTION_LOCKED_BEFORE_ARTIST_REFERENCE",
        "image_RGBA_SHA256": public["image_RGBA_SHA256"],
        "prediction_arrays_sha256": digest(npz_path.read_bytes()),
        "image_dimensions": [public["height"], public["width"]],
        "image_only_algorithms": {
            "alpha_connected": "4_CONNECTED_NONZERO_ALPHA_NO_FIXED_K",
            "image_graph": "FELZENSZWALB_RGB_GRAPH_NO_FIXED_K",
        },
        "parameters_frozen": PARAMS,
        "predicted_region_counts": counts,
        "artist_slot_names_or_layer_count_exposed": False,
        "artist_original_mesh_face_incidence_exposed": False,
        "reference_used_during_prediction": False,
        "artist_edit_equivalence_proven": False,
        "drawable_ownership_authorized": False,
        "training_authorized": False,
    }
    manifest_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def court(image_path, public_path, prediction_npz, prediction_manifest,
          artist_reference_path, artist_reference_dir):
    """Compare *after* predictions are sealed. Read only original HIDE effects."""
    from scipy.optimize import linear_sum_assignment

    rgba, public = read_image(image_path, public_path)
    predicted = json.loads(Path(prediction_manifest).read_text())
    require(predicted.get("schema") == PRED_SCHEMA
            and predicted.get("status") ==
                "IMAGE_ONLY_PREDICTION_LOCKED_BEFORE_ARTIST_REFERENCE"
            and predicted.get("image_RGBA_SHA256") ==
                public["image_RGBA_SHA256"]
            and predicted.get("prediction_arrays_sha256") ==
                digest(Path(prediction_npz).read_bytes())
            and predicted.get("artist_slot_names_or_layer_count_exposed") is False
            and predicted.get("artist_original_mesh_face_incidence_exposed") is False
            and predicted.get("reference_used_during_prediction") is False
            and predicted.get("training_authorized") is False
            and predicted.get("drawable_ownership_authorized") is False,
            "ARTIST_IMAGE_COURT_UNSEALED_OR_PRIVILEGED_PREDICTION")
    ref = json.loads(Path(artist_reference_path).read_text())
    require(ref.get("schema") ==
                "RealSaS.IndependentOriginalArtistSlotRGBAHideIsolateCourtOnly.v1"
            and ref.get("original_model_input_RGBA_SHA256") ==
                public["image_RGBA_SHA256"]
            and ref.get("official_spine_runtime_reference") is True
            and ref.get("original_hide_isolate_restoration_exact") is True
            and ref.get("training_authorized") is False
            and ref.get("native_editable_source_artist_project_file_verified") is False,
            "ARTIST_IMAGE_COURT_UNQUALIFIED_INDEPENDENT_REFERENCE")
    rows = ref["source_author_slot_hide_isolate_references_teacher_only"]
    require(len(rows) == ref["active_original_author_slot_count_teacher_only"]
            and len(rows) > 0, "ARTIST_IMAGE_COURT_INVALID_AUTHOR_CASE_COUNT")
    source = rgba.reshape(-1, 4)
    supports = []
    masked = 0
    all_indices = []
    root = Path(artist_reference_dir).resolve()
    seen = set()
    for entry in rows:
        index = entry["original_slot_index_teacher_only"]
        require(type(index) is int and index >= 0 and index not in seen,
                "ARTIST_IMAGE_COURT_DUPLICATED_AUTHOR_SOURCE_SLOT")
        seen.add(index)
        path_hidden = (root / ("slot" + str(index) + "_hidden.rgba")).resolve()
        path_isolate = (root / ("slot" + str(index) + "_isolated.rgba")).resolve()
        require(path_hidden.parent == root and path_isolate.parent == root,
                "ARTIST_IMAGE_COURT_PRIVATE_PATH_ESCAPE")
        hidden_raw, isolate_raw = path_hidden.read_bytes(), path_isolate.read_bytes()
        require(len(hidden_raw) == len(isolate_raw) == source.size
                and digest(hidden_raw) ==
                    entry["original_hidden_source_RGBA_sha256"]
                and digest(isolate_raw) ==
                    entry["original_isolated_source_RGBA_sha256"],
                "ARTIST_IMAGE_COURT_SOURCE_EDIT_REFERENCE_SHA")
        hidden = np.frombuffer(hidden_raw, dtype=np.uint8).reshape(-1, 4)
        isolated = np.frombuffer(isolate_raw, dtype=np.uint8).reshape(-1, 4)
        observed = np.any(hidden != source, axis=1)
        require(int(np.count_nonzero(isolated[:, 3])) ==
                    entry["original_isolated_alpha_pixel_count"]
                and int(observed.sum()) ==
                    entry["original_hide_changes_composite_RGBA_pixel_count"],
                "ARTIST_IMAGE_COURT_ACTUAL_EDIT_MEASUREMENTS")
        if observed.any():
            supports.append(observed)
            all_indices.append(index)
        else:
            masked += 1
    result = {
        "schema": COURT_SCHEMA,
        "status": "SOURCE_ARTIST_EDIT_SUPPORT_BASELINES_MEASURED__NOT_DRAWABLE_TRUTH",
        "source_RGBA_SHA256": public["image_RGBA_SHA256"],
        "frozen_prediction_sha256": predicted["prediction_arrays_sha256"],
        "algorithm_inference_read_original_art_names": False,
        "algorithm_inference_read_artist_part_count": False,
        "algorithm_inference_read_artist_hide_isolate": False,
        "original_author_active_slot_count_teacher_only": len(rows),
        "author_hide_observable_cases": len(supports),
        "author_hide_unobservable_masked_cases": masked,
        "artist_project_original_editor_file_verified": False,
        "metrics_mean_edit_support_iou_not_artist_membership_accuracy": {},
        "exact_artist_drawable_identity_proven": False,
        "image_only_hallucinated_invisible_art": False,
        "graphics_owner_authorized": False,
        "materialization_authorized": False,
        "training_authorized": False,
    }
    require(len(supports) > 0, "ARTIST_IMAGE_COURT_NO_OBSERVABLE_SOURCE_EDIT_ABSTAIN")
    support = np.stack(supports, axis=0)
    total_support = int(support.sum())
    with np.load(prediction_npz, allow_pickle=False) as data:
        require(set(data.files) == {"alpha_connected", "image_graph"},
                "ARTIST_IMAGE_COURT_PREDICTION_ARRAY_NAMES")
        for algorithm in ("alpha_connected", "image_graph"):
            labels = np.asarray(data[algorithm], np.int64).reshape(-1)
            require(labels.shape == (source.shape[0],)
                    and np.all(labels[(source[:, 3] == 0)] == -1)
                    and np.all(labels[(source[:, 3] > 0)] >= 0)
                    and labels.max(initial=-1) < MAX_REGIONS,
                    "ARTIST_IMAGE_COURT_INVALID_PIXEL_REGION_LABELS")
            n = int(labels.max(initial=-1)) + 1
            require(n == predicted["predicted_region_counts"][algorithm]
                    and set(np.unique(labels[labels >= 0])) == set(range(n)),
                    "ARTIST_IMAGE_COURT_REGION_COUNTER_OR_GAP")
            region_size = np.bincount(labels[labels >= 0], minlength=n)
            iou = np.zeros((len(support), n), dtype=np.float64)
            intersections = np.zeros_like(iou)
            for i, target in enumerate(support):
                hits = labels[target]
                hits = hits[hits >= 0]
                inter = np.bincount(hits, minlength=n)
                intersections[i] = inter
                union = region_size + int(target.sum()) - inter
                np.divide(inter, union, out=iou[i], where=union > 0)
            rr, cc = linear_sum_assignment(-iou)
            sum_iou = float(iou[rr, cc].sum())
            intersect = int(intersections[rr, cc].sum())
            result["metrics_mean_edit_support_iou_not_artist_membership_accuracy"][algorithm] = {
                "predicted_region_count": n,
                "matched_observable_source_edits": int(len(rr)),
                "observable_source_edits": len(support),
                "mean_one_to_one_edit_support_iou_over_all_observable_edits":
                    sum_iou / len(support),
                "matched_edit_effect_pixel_recall_over_all_edit_effect_instances":
                    intersect / total_support,
                "edit_support_pixel_instances": total_support,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--rgba", required=True)
    p.add_argument("--public-manifest", required=True)
    p.add_argument("--prediction-npz", required=True)
    p.add_argument("--prediction-manifest", required=True)
    c = sub.add_parser("court")
    c.add_argument("--rgba", required=True)
    c.add_argument("--public-manifest", required=True)
    c.add_argument("--prediction-npz", required=True)
    c.add_argument("--prediction-manifest", required=True)
    c.add_argument("--private-reference", required=True)
    c.add_argument("--private-dir", required=True)
    c.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.command == "predict":
        result = write_prediction(args.rgba, args.public_manifest,
                                  args.prediction_npz, args.prediction_manifest)
    else:
        result = court(args.rgba, args.public_manifest, args.prediction_npz,
                       args.prediction_manifest, args.private_reference,
                       args.private_dir)
        Path(args.receipt).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
