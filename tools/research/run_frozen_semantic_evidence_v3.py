#!/usr/bin/env python3
from __future__ import annotations

"""SAM2 local masks + official PartCLIPSeg frozen naming evidence.

V3 keeps the successful V2 localization strategy but replaces generic CLIPSeg
naming with dense fields produced by the official zero-shot PartCLIPSeg
checkpoint.  PartCLIPSeg runs in an isolated worker because its official code
ships a legacy Transformers fork.  This runner remains on the current SAM2
research environment.

No output is semantic truth.  Bad/ambiguous regions abstain before canonical-M
fusion and no IRIS geometry model is imported or modified.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from models.iris.semantic_evidence_v1 import build_mask_evidence_v1, write_mask_evidence_v1
from models.iris.semantic_part_naming_v1 import classify_part_region_v1
from models.iris.semantic_provider_quality_v1 import audit_semantic_proposals_v1, mask_iou


SAM_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SAM_SAFETENSORS_SHA256 = "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60"
PARTCLIPSEG_UPSTREAM_COMMIT = "8abceefff960920019096982688e3e8209ed39de"


def _hash_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hf_weights(repo_id: str, revision: str, expected_sha256: str) -> None:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision))
    actual = _hash_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"MODEL_WEIGHT_HASH_DRIFT:{repo_id}:{actual}!={expected_sha256}")


def _read_views(source_zip: Path) -> list[Image.Image]:
    rows = []
    with zipfile.ZipFile(source_zip) as archive:
        for view in range(8):
            rows.append(Image.open(io.BytesIO(archive.read(f"observations/V{view}.png"))).convert("RGBA").copy())
    if len({x.size for x in rows}) != 1:
        raise RuntimeError("source views must have equal dimensions")
    return rows


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
    keep = []
    for row in sorted(rows, key=lambda x: (-float(x["sam_score"]), -int(x["mask"].sum()))):
        if any(mask_iou(row["mask"], old["mask"]) >= iou_threshold for old in keep):
            continue
        keep.append(row)
    return keep


def _load_part_fields(path: Path):
    with np.load(path, allow_pickle=False) as data:
        fields = data["part_probabilities"].astype(np.float32)
        background = data["background_probability"].astype(np.float32)
        meta = json.loads(str(data["metadata_json"]))
    if meta.get("schema") != "RealSaS.PartCLIPSegDenseFields.v1":
        raise RuntimeError("PARTCLIPSEG_DENSE_FIELD_SCHEMA_DRIFT")
    if meta.get("upstream_commit") != PARTCLIPSEG_UPSTREAM_COMMIT:
        raise RuntimeError("PARTCLIPSEG_UPSTREAM_COMMIT_DRIFT")
    labels = tuple(str(x) for x in meta.get("labels", []))
    if fields.ndim != 4 or fields.shape[0] != 8 or fields.shape[1] != len(labels):
        raise RuntimeError("PARTCLIPSEG_DENSE_FIELD_SHAPE_DRIFT")
    if background.shape != (fields.shape[0], fields.shape[2], fields.shape[3]):
        raise RuntimeError("PARTCLIPSEG_BACKGROUND_SHAPE_DRIFT")
    if not np.isfinite(fields).all() or not np.isfinite(background).all():
        raise RuntimeError("PARTCLIPSEG_DENSE_FIELD_NONFINITE")
    if np.any(fields < 0) or np.any(fields > 1) or np.any(background < 0) or np.any(background > 1):
        raise RuntimeError("PARTCLIPSEG_DENSE_FIELD_RANGE_DRIFT")
    return labels, fields, background, meta


def run(args) -> dict:
    from transformers import pipeline

    images = _read_views(args.source_zip)
    labels, fields, background, field_meta = _load_part_fields(args.part_fields)
    _verify_hf_weights(SAM_MODEL_ID, SAM_REVISION, SAM_SAFETENSORS_SHA256)

    sam = pipeline(
        "mask-generation",
        model=SAM_MODEL_ID,
        revision=SAM_REVISION,
        device=0 if str(args.device).startswith("cuda") else -1,
    )

    all_rows = []
    subject_masks = []
    audit_views = []
    rejection_totals = Counter()
    for view, rgba in enumerate(images):
        rgb = rgba.convert("RGB")
        alpha = np.asarray(rgba)[:, :, 3] > 0
        subject_masks.append(alpha)
        h, w = alpha.shape
        if fields.shape[2:] != (h, w):
            raise RuntimeError(f"PARTCLIPSEG_SOURCE_SHAPE_DRIFT:{fields.shape[2:]}!={(h, w)}")

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
        subject_pixels = int(alpha.sum())
        for ci, value in enumerate(raw_masks):
            mask = _to_bool_mask(value, (h, w)) & alpha
            pixels = int(mask.sum())
            coverage = pixels / max(subject_pixels, 1)
            if pixels < int(args.minimum_mask_pixels):
                continue
            if coverage >= float(args.maximum_subject_coverage):
                continue
            candidates.append({
                "mask": mask,
                "sam_score": float(raw_scores[ci]) if ci < len(raw_scores) else 1.0,
                "subject_coverage": coverage,
            })
        candidates = _nms_masks(candidates, float(args.candidate_nms_iou))

        accepted = []
        view_rejections = Counter()
        for ci, row in enumerate(candidates):
            decision = classify_part_region_v1(
                part_probabilities=fields[view],
                background_probability=background[view],
                mask=row["mask"],
                foreground=alpha,
                temperature=float(args.semantic_temperature),
                minimum_top_probability=float(args.minimum_class_probability),
                minimum_class_margin=float(args.minimum_class_margin),
                minimum_localization_margin=float(args.minimum_localization_margin),
                minimum_part_vs_background_margin=float(args.minimum_part_vs_background_margin),
            )
            if not decision.accepted:
                code = str(decision.failure_code or "PART_NAMING_UNKNOWN_REJECTION")
                view_rejections[code] += 1
                rejection_totals[code] += 1
                continue
            label = labels[int(decision.label_index)]
            score = float(np.sqrt(max(row["sam_score"], 0.0) * decision.top_probability))
            accepted.append({
                "view_index": view,
                "label": label,
                "score": score,
                "proposal_id": f"V{view}:{label}:PCS{ci:03d}",
                "mask": row["mask"],
                "sam_score": row["sam_score"],
                "naming_probability": decision.top_probability,
                "naming_margin": decision.margin,
                "localization_margin": decision.localization_margin,
                "background_probability": decision.background_probability,
                "subject_coverage": row["subject_coverage"],
            })

        bounded = []
        for label in labels:
            chosen = sorted(
                [x for x in accepted if x["label"] == label],
                key=lambda x: (-x["score"], x["proposal_id"]),
            )[: int(args.max_per_label)]
            bounded.extend(chosen)
        all_rows.extend(bounded)
        audit_views.append({
            "view_index": view,
            "sam_raw_mask_count": len(raw_masks),
            "sam_local_candidate_count": len(candidates),
            "accepted_semantic_count": len(bounded),
            "accepted_labels": sorted({x["label"] for x in bounded}),
            "rejections": dict(sorted(view_rejections.items())),
        })
        print(
            f"V{view}: SAM {len(raw_masks)} -> local {len(candidates)} -> PartCLIPSeg semantic {len(bounded)} "
            f"labels={audit_views[-1]['accepted_labels']}",
            flush=True,
        )

    # Generic multiview presence sanity: a semantic label seen in only one source
    # view is too weak to become provider evidence on an 8-view subject.
    label_views = defaultdict(set)
    for row in all_rows:
        label_views[str(row["label"])].add(int(row["view_index"]))
    stable_labels = {
        label for label, views in label_views.items()
        if len(views) >= int(args.minimum_label_support_views)
    }
    proposals = [row for row in all_rows if str(row["label"]) in stable_labels]

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
        "schema": "RealSaS.FrozenSemanticProviderAudit.v3",
        "status": quality.status,
        "failure_code": quality.failure_code,
        "provider": "SAM2_SEGMENT_EVERYTHING__OFFICIAL_PARTCLIPSEG_REGION_NAMING",
        "sam_model": f"{SAM_MODEL_ID}@{SAM_REVISION}",
        "partclipseg_upstream_commit": PARTCLIPSEG_UPSTREAM_COMMIT,
        "partclipseg_checkpoint_sha256": field_meta.get("checkpoint_sha256"),
        "partclipseg_loaded_parameter_numel_fraction": field_meta.get("loaded_parameter_numel_fraction"),
        "partclipseg_object_contexts": field_meta.get("object_contexts"),
        "quality": asdict(quality),
        "views": audit_views,
        "label_support_views": {k: sorted(v) for k, v in sorted(label_views.items())},
        "stable_labels": sorted(stable_labels),
        "rejection_totals": dict(sorted(rejection_totals.items())),
        "geometry_model_loaded": False,
        "canonical_geometry_authority": "UNCHANGED",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if quality.failure_code is not None:
        raise RuntimeError(quality.failure_code)

    source_identity = "SHA256:" + sha256(args.source_zip.read_bytes()).hexdigest()
    provider_id = (
        "FROZEN_PART_SEMANTIC_STACK_V3|"
        f"{SAM_MODEL_ID}@{SAM_REVISION}|"
        f"PartCLIPSeg@{PARTCLIPSEG_UPSTREAM_COMMIT}|"
        f"checkpoint:{field_meta.get('checkpoint_sha256')}"
    )
    h, w = np.asarray(images[0]).shape[:2]
    evidence = build_mask_evidence_v1(
        proposals,
        labels=labels,
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
        "partclipseg_dense_fields_sha256": _hash_file(args.part_fields),
        "partclipseg_checkpoint_sha256": field_meta.get("checkpoint_sha256"),
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
    p.add_argument("--part-fields", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--sam-pred-iou-threshold", type=float, default=0.86)
    p.add_argument("--minimum-mask-pixels", type=int, default=48)
    p.add_argument("--maximum-subject-coverage", type=float, default=0.90)
    p.add_argument("--candidate-nms-iou", type=float, default=0.95)
    p.add_argument("--semantic-temperature", type=float, default=0.55)
    p.add_argument("--minimum-class-probability", type=float, default=0.34)
    p.add_argument("--minimum-class-margin", type=float, default=0.07)
    p.add_argument("--minimum-localization-margin", type=float, default=0.10)
    p.add_argument("--minimum-part-vs-background-margin", type=float, default=0.02)
    p.add_argument("--max-per-label", type=int, default=3)
    p.add_argument("--minimum-label-support-views", type=int, default=2)
    p.add_argument("--collapse-iou", type=float, default=0.97)
    p.add_argument("--maximum-duplicate-pair-fraction", type=float, default=0.05)
    p.add_argument("--maximum-whole-subject-fraction", type=float, default=0.20)
    p.add_argument("--minimum-distinct-labels", type=int, default=3)
    p.add_argument("--minimum-total-proposals", type=int, default=6)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
