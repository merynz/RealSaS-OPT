#!/usr/bin/env python3
"""Numeric pilot gate for source-matched private lookdev images.

A numeric image gate does NOT make anatomical camera, native RGB, geometry or
semantic truth. Visual reviewer must still inspect and approve each contact sheet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

SCHEMA = "RealSaS.Corpus4008LookdevPixelAudit.v1"
MIN_OCCUPIED = 0.07
MIN_BBOX_SPAN = 0.42
MAX_BBOX_SPAN = 0.95
MIN_FLAT_CEL_RGB_DELTA = 0.027
MIN_ALPHA_JACCARD = 0.98


def require(flag, code):
    if not flag:
        raise ValueError("CORPUS4008_LOOKDEV_PIXEL_" + code)


def alpha_stats(image):
    alpha = image.getchannel("A")
    binary = alpha.point(lambda x: 255 if x > 20 else 0)
    bbox = binary.getbbox()
    require(bbox is not None, "EMPTY_CHARACTER_ALPHA")
    w, h = image.size
    count = sum(bool(x) for x in binary.getdata())
    return {
        "bbox_xyxy": list(bbox),
        "bbox_w_fraction": (bbox[2] - bbox[0]) / w,
        "bbox_h_fraction": (bbox[3] - bbox[1]) / h,
        "occupied_alpha_fraction": count / (w * h),
        "alpha_pixel_count": count,
    }


def pair_stats(a, b):
    require(a.size == b.size, "IMAGE_SIZE_MISMATCH")
    p = a.tobytes()
    q = b.tobytes()
    intersection = union = count = 0
    delta = 0
    for i in range(0, len(p), 4):
        active_a = p[i + 3] > 20
        active_b = q[i + 3] > 20
        if active_a or active_b:
            union += 1
        if active_a and active_b:
            intersection += 1
            count += 1
            delta += (abs(p[i] - q[i]) +
                      abs(p[i + 1] - q[i + 1]) +
                      abs(p[i + 2] - q[i + 2]))
    require(union > 0 and count > 0, "NO_SHARED_SOURCE_MASK")
    return {
        "shared_opaque_pixels": count,
        "alpha_jaccard": intersection / union,
        "mean_absolute_rgb_delta_0_1": delta / (count * 3 * 255),
    }


def audit(root, *, enforce_thresholds=True):
    doc = json.loads((root / "LOOKDEV_STUDY_V1.json").read_text())
    require(doc["training_authorized"] is False and
            doc["source_rgba_truth_authorized"] is False,
            "RENDER_AUTHORITY")
    modes = {entry["mode"]: entry["path_private"] for entry in doc["outputs"]}
    require(set(modes) == {
        "SOURCE_MATERIAL_DIAGNOSTIC",
        "STYLIZED_FLAT_DIAGNOSTIC",
        "STYLIZED_CEL_3BAND_DIAGNOSTIC",
    }, "MODES")
    renders = {key: Image.open(root / name).convert("RGBA")
               for key, name in modes.items()}
    require(len({im.size for im in renders.values()}) == 1,
            "MIXED_RENDER_SIZE")
    summary = {k: alpha_stats(im) for k, im in renders.items()}
    baseline = renders["SOURCE_MATERIAL_DIAGNOSTIC"]
    flat = renders["STYLIZED_FLAT_DIAGNOSTIC"]
    cel = renders["STYLIZED_CEL_3BAND_DIAGNOSTIC"]
    contrasts = {
        "native_vs_flat": pair_stats(baseline, flat),
        "native_vs_cel": pair_stats(baseline, cel),
        "flat_vs_cel": pair_stats(flat, cel),
    }
    metrics_pass = (
        all(x["occupied_alpha_fraction"] >= MIN_OCCUPIED
            and MIN_BBOX_SPAN <= x["bbox_w_fraction"] <= MAX_BBOX_SPAN
            and MIN_BBOX_SPAN <= x["bbox_h_fraction"] <= MAX_BBOX_SPAN
            for x in summary.values())
        and contrasts["flat_vs_cel"]["mean_absolute_rgb_delta_0_1"] >=
        MIN_FLAT_CEL_RGB_DELTA
        and contrasts["flat_vs_cel"]["alpha_jaccard"] >= MIN_ALPHA_JACCARD
    )
    result = {
        "schema": SCHEMA, "asset_id": doc["asset_id"],
        "source_sha256": doc["source_sha256"],
        "metrics": summary,
        "pair_metrics": contrasts,
        "minimum_occupied_alpha_fraction": MIN_OCCUPIED,
        "minimum_bbox_span_fraction": MIN_BBOX_SPAN,
        "minimum_flat_cel_rgb_delta": MIN_FLAT_CEL_RGB_DELTA,
        "minimum_flat_cel_alpha_jaccard": MIN_ALPHA_JACCARD,
        "numeric_pilot_gate_pass": metrics_pass,
        "human_visual_review_courted": False,
        "style_training_authorized": False,
        "native_source_rgb_qualified": False,
        "training_authorized": False,
    }
    if enforce_thresholds:
        require(metrics_pass, "PILOT_COMPOSITION_OR_STYLE_UNDER_THRESHOLD")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lookdev-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.lookdev_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "numeric_pilot_gate_pass", "pair_metrics", "training_authorized",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
