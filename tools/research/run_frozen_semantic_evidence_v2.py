#!/usr/bin/env python3
from __future__ import annotations

"""Segmentation-first frozen semantic evidence for RealSaS.

V2 fixes the first Knight oracle's text-box->SAM collapse by reversing the
responsibilities:

1. SAM2 segment-everything proposes label-free masks.
2. CLIPSeg produces text-conditioned dense semantic fields.
3. Each SAM candidate receives at most one semantic class when the class margin
   is strong enough; ambiguous candidates abstain.
4. Provider-collapse and whole-subject gates run before evidence is serialized.

The script never imports or mutates IRIS geometry. Output remains the existing
provider-neutral FrozenSemanticMaskEvidenceV1 contract consumed by M fusion.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.iris.semantic_evidence_v1 import build_mask_evidence_v1, write_mask_evidence_v1
from models.iris.semantic_provider_quality_v1 import audit_semantic_proposals_v1


SAM_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SAM_SAFETENSORS_SHA256 = "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60"

CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
CLIPSEG_SAFETENSORS_SHA256 = "d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55"

# Canonical labels are intentionally narrower than V1. Generic aliases such as
# `weapon` and `prop` were whole-character attractors in the failed oracle.
PROMPTS = {
    "head": ("head", "face"),
    "arm": ("arm",),
    "forearm": ("forearm", "lower arm"),
    "hand": ("hand", "glove"),
    "cape": ("cape", "cloak"),
    "hair": ("hair",),
    "cloth": ("cloth", "fabric"),
    "armor": ("armor plate", "armor"),
    "helmet": ("helmet",),
    "horn": ("horn",),
    "sword": ("sword", "blade"),
    "shield": ("shield",),
}


def _read_views(source_zip: Path) -> list[Image.Image]:
    rows = []
    with zipfile.ZipFile(source_zip) as archive:
        for view in range(8):
            raw = archive.read(f"observations/V{view}.png")
            rows.append(Image.open(io.BytesIO(raw)).convert("RGBA").copy())
    if len({x.size for x in rows}) != 1:
        raise RuntimeError("source views must have equal dimensions")
    return rows


def _hash_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hf_weights(repo_id: str, revision: str, expected_sha256: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
        )
    )
    actual = _hash_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"MODEL_WEIGHT_HASH_DRIFT:{repo_id}:{actual}!={expected_sha256}")
    return path


def _resolve_clipseg_revision() -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(CLIPSEG_MODEL_ID, revision="main")
    if not info.sha:
        raise RuntimeError("CLIPSEG_RESOLVED_REVISION_MISSING")
    return str(info.sha)


def _to_bool_mask(value, shape) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != tuple(shape):
        raise RuntimeError(f"SAM_MASK_SHAPE_DRIFT:{arr.shape}!={tuple(shape)}")
    return arr.astype(bool, copy=False)


def _nms_masks(rows, iou_threshold: float):
    from models.iris.semantic_provider_quality_v1 import mask_iou

    keep = []
    for row in sorted(rows, key=lambda x: (-float(x["sam_score"]), -int(x["mask"].sum()))):
        if any(mask_iou(row["mask"], old["mask"]) >= iou_threshold for old in keep):
            continue
        keep.append(row)
    return keep


def _softmax(value, temperature: float) -> np.ndarray:
    z = np.asarray(value, dtype=np.float64) / max(float(temperature), 1e-6)
    z -= np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _clipseg_fields(model, processor, rgb, *, device, height: int, width: int):
    import torch
    import torch.nn.functional as F

    canonical = tuple(PROMPTS)
    flat_prompts = []
    owners = []
    for li, label in enumerate(canonical):
        for phrase in PROMPTS[label]:
            flat_prompts.append(phrase)
            owners.append(li)
    inputs = processor(
        text=flat_prompts,
        images=[rgb] * len(flat_prompts),
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = F.interpolate(
        logits[:, None], size=(height, width), mode="bilinear", align_corners=False
    )[:, 0]
    # Max-pool prompt aliases into one dense field per canonical label.
    fields = []
    for li in range(len(canonical)):
        ids = [i for i, owner in enumerate(owners) if owner == li]
        fields.append(torch.max(logits[ids], dim=0).values)
    return canonical, torch.stack(fields).detach().cpu().numpy()


def _standardize_fields(fields: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    result = np.zeros_like(fields, dtype=np.float32)
    idx = foreground.astype(bool)
    if int(idx.sum()) < 8:
        raise RuntimeError("SOURCE_FOREGROUND_TOO_SMALL")
    for li, field in enumerate(fields):
        values = np.asarray(field, dtype=np.float64)[idx]
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * 1.4826)
        if scale < 1e-5:
            scale = float(np.std(values))
        scale = max(scale, 1e-5)
        result[li] = ((field - center) / scale).astype(np.float32)
    return result


def run(args) -> dict:
    import torch
    from transformers import AutoProcessor, CLIPSegForImageSegmentation, pipeline

    images = _read_views(args.source_zip)
    clipseg_revision = _resolve_clipseg_revision()
    _verify_hf_weights(SAM_MODEL_ID, SAM_REVISION, SAM_SAFETENSORS_SHA256)
    _verify_hf_weights(CLIPSEG_MODEL_ID, clipseg_revision, CLIPSEG_SAFETENSORS_SHA256)

    sam = pipeline(
        "mask-generation",
        model=SAM_MODEL_ID,
        revision=SAM_REVISION,
        device=0 if str(args.device).startswith("cuda") else -1,
    )
    clip_processor = AutoProcessor.from_pretrained(CLIPSEG_MODEL_ID, revision=clipseg_revision)
    clip_model = CLIPSegForImageSegmentation.from_pretrained(
        CLIPSEG_MODEL_ID,
        revision=clipseg_revision,
        use_safetensors=True,
    ).to(args.device)
    clip_model.eval()

    proposals = []
    audit_views = []
    subject_masks = []
    for view, rgba in enumerate(images):
        rgb = rgba.convert("RGB")
        alpha = np.asarray(rgba)[:, :, 3] > 0
        subject_masks.append(alpha)
        h, w = alpha.shape
        generated = sam(
            rgb,
            points_per_batch=int(args.points_per_batch),
            pred_iou_thresh=float(args.sam_pred_iou_threshold),
        )
        raw_masks = list(generated.get("masks") or [])
        raw_scores = generated.get("scores")
        if hasattr(raw_scores, "detach"):
            raw_scores = raw_scores.detach().cpu().numpy()
        raw_scores = np.asarray(raw_scores if raw_scores is not None else np.ones(len(raw_masks)), dtype=float)
        candidates = []
        for ci, value in enumerate(raw_masks):
            mask = _to_bool_mask(value, (h, w)) & alpha
            pixels = int(mask.sum())
            subject_pixels = int(alpha.sum())
            coverage = pixels / max(subject_pixels, 1)
            if pixels < int(args.minimum_mask_pixels):
                continue
            if coverage >= float(args.maximum_subject_coverage):
                continue
            candidates.append(
                {
                    "mask": mask,
                    "sam_score": float(raw_scores[ci]) if ci < len(raw_scores) else 1.0,
                    "subject_coverage": coverage,
                }
            )
        candidates = _nms_masks(candidates, float(args.candidate_nms_iou))

        labels, dense = _clipseg_fields(
            clip_model,
            clip_processor,
            rgb,
            device=args.device,
            height=h,
            width=w,
        )
        dense = _standardize_fields(dense, alpha)
        view_rows = []
        for ci, row in enumerate(candidates):
            mask = row["mask"]
            raw = np.asarray([float(np.mean(dense[li][mask])) for li in range(len(labels))])
            prob = _softmax(raw, float(args.semantic_temperature))
            order = np.argsort(-prob)
            top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            top_prob = float(prob[top])
            margin = float(prob[top] - prob[second]) if len(order) > 1 else top_prob
            if (
                top_prob < float(args.minimum_class_probability)
                or margin < float(args.minimum_class_margin)
                or raw[top] < float(args.minimum_localization_z)
            ):
                continue
            label = labels[top]
            score = float(np.sqrt(max(row["sam_score"], 0.0) * top_prob))
            view_rows.append(
                {
                    "view_index": view,
                    "label": label,
                    "score": score,
                    "proposal_id": f"V{view}:{label}:SAM{ci:03d}",
                    "mask": mask,
                    "sam_score": row["sam_score"],
                    "class_probability": top_prob,
                    "class_margin": margin,
                    "localization_z": float(raw[top]),
                    "subject_coverage": row["subject_coverage"],
                }
            )
        # Bound repeated masks of one class without creating cross-class duplicates.
        bounded = []
        for label in labels:
            chosen = sorted(
                [x for x in view_rows if x["label"] == label],
                key=lambda x: (-x["score"], x["proposal_id"]),
            )[: int(args.max_per_label)]
            bounded.extend(chosen)
        proposals.extend(bounded)
        audit_views.append(
            {
                "view_index": view,
                "sam_raw_mask_count": len(raw_masks),
                "sam_local_candidate_count": len(candidates),
                "accepted_semantic_count": len(bounded),
                "accepted_labels": sorted({x["label"] for x in bounded}),
            }
        )
        print(
            f"V{view}: SAM {len(raw_masks)} -> local {len(candidates)} -> semantic {len(bounded)} "
            f"labels={audit_views[-1]['accepted_labels']}",
            flush=True,
        )

    quality = audit_semantic_proposals_v1(
        proposals,
        subject_masks=subject_masks,
        near_duplicate_iou=float(args.collapse_iou),
        maximum_duplicate_pair_fraction=float(args.maximum_duplicate_pair_fraction),
        whole_subject_coverage=float(args.maximum_subject_coverage),
        maximum_whole_subject_fraction=float(args.maximum_whole_subject_fraction),
        minimum_distinct_labels=int(args.minimum_distinct_labels),
        minimum_proposals=int(args.minimum_total_proposals),
    )
    audit = {
        "schema": "RealSaS.FrozenSemanticProviderAudit.v2",
        "status": quality.status,
        "failure_code": quality.failure_code,
        "provider": "SAM2_SEGMENT_EVERYTHING__CLIPSEG_REGION_CLASSIFICATION",
        "sam_model": f"{SAM_MODEL_ID}@{SAM_REVISION}",
        "clipseg_model": f"{CLIPSEG_MODEL_ID}@{clipseg_revision}",
        "quality": asdict(quality),
        "views": audit_views,
        "geometry_model_loaded": False,
        "canonical_geometry_authority": "UNCHANGED",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if quality.failure_code is not None:
        raise RuntimeError(quality.failure_code)

    source_identity = "SHA256:" + sha256(args.source_zip.read_bytes()).hexdigest()
    provider_id = (
        "FROZEN_SEGMENTATION_FIRST_SEMANTIC_STACK_V2|"
        f"{SAM_MODEL_ID}@{SAM_REVISION}|"
        f"{CLIPSEG_MODEL_ID}@{clipseg_revision}"
    )
    h, w = np.asarray(images[0]).shape[:2]
    evidence = build_mask_evidence_v1(
        proposals,
        labels=tuple(PROMPTS),
        provider_id=provider_id,
        source_identity=source_identity,
        image_shape=(h, w),
    )
    ref = write_mask_evidence_v1(args.output, evidence)
    summary = {
        **ref,
        "provider_id": provider_id,
        "proposal_count": evidence.proposal_count,
        "labels": list(evidence.labels),
        "source_identity": source_identity,
        "sam_safetensors_sha256": SAM_SAFETENSORS_SHA256,
        "clipseg_safetensors_sha256": CLIPSEG_SAFETENSORS_SHA256,
        "provider_quality": asdict(quality),
        "provider_audit": str(args.audit_output),
        "geometry_model_loaded": False,
        "geometry_checkpoint_modified": False,
        "canonical_geometry_authority": "UNCHANGED",
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--sam-pred-iou-threshold", type=float, default=0.86)
    p.add_argument("--minimum-mask-pixels", type=int, default=48)
    p.add_argument("--maximum-subject-coverage", type=float, default=0.90)
    p.add_argument("--candidate-nms-iou", type=float, default=0.95)
    p.add_argument("--semantic-temperature", type=float, default=0.35)
    p.add_argument("--minimum-class-probability", type=float, default=0.30)
    p.add_argument("--minimum-class-margin", type=float, default=0.08)
    p.add_argument("--minimum-localization-z", type=float, default=0.20)
    p.add_argument("--max-per-label", type=int, default=3)
    p.add_argument("--collapse-iou", type=float, default=0.97)
    p.add_argument("--maximum-duplicate-pair-fraction", type=float, default=0.10)
    p.add_argument("--maximum-whole-subject-fraction", type=float, default=0.25)
    p.add_argument("--minimum-distinct-labels", type=int, default=3)
    p.add_argument("--minimum-total-proposals", type=int, default=6)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
