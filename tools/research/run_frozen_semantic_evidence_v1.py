#!/usr/bin/env python3
from __future__ import annotations

"""Run frozen open-vocabulary detection + prompted segmentation on 8 source views.

Research-only evidence producer.  It does not import, alter, fine-tune or replace
IRIS geometry.  Output is the provider-neutral FrozenSemanticMaskEvidenceV1
bundle consumed by Compiler research adapters.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.iris.semantic_evidence_v1 import build_mask_evidence_v1, write_mask_evidence_v1


GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_SAFETENSORS_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
SEGMENTATION_MODEL_ID = "facebook/sam2.1-hiera-small"
SEGMENTATION_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SEGMENTATION_SAFETENSORS_SHA256 = "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60"

DEFAULT_LABELS = (
    "body",
    "head",
    "arm",
    "forearm",
    "hand",
    "leg",
    "foot",
    "cape",
    "cloak",
    "hair",
    "cloth",
    "armor",
    "helmet",
    "horn",
    "sword",
    "weapon",
    "shield",
    "prop",
)


def _read_views(source_zip: Path) -> list[Image.Image]:
    rows = []
    with zipfile.ZipFile(source_zip) as archive:
        for view in range(8):
            raw = archive.read(f"observations/V{view}.png")
            rows.append(Image.open(io.BytesIO(raw)).convert("RGBA").copy())
    size = {x.size for x in rows}
    if len(size) != 1:
        raise RuntimeError(f"source views must have equal dimensions:{sorted(size)}")
    return rows


def _device_index(device: str) -> int:
    if device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return -1


def _normalise_mask_tensor(value, height: int, width: int) -> np.ndarray:
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        # multimask_output=False should leave one mask; be fail-closed otherwise.
        if arr.shape[0] != 1:
            raise RuntimeError(f"unexpected segmentation mask shape:{arr.shape}")
        arr = arr[0]
    if arr.shape != (height, width):
        raise RuntimeError(f"segmentation mask size drift:{arr.shape}!={(height, width)}")
    return arr > 0


def run(args) -> dict:
    import torch
    from transformers import AutoProcessor, pipeline
    try:
        from transformers import Sam2Model as SegmentationModel
    except ImportError:  # compatibility fallback only
        from transformers import AutoModel as SegmentationModel

    images = _read_views(args.source_zip)
    labels = tuple(dict.fromkeys(x.strip().lower() for x in args.labels.split(",") if x.strip()))
    if not labels:
        labels = DEFAULT_LABELS
    prompts = [f"a {x}" for x in labels]
    prompt_to_label = {f"a {x}": x for x in labels}

    detector = pipeline(
        "zero-shot-object-detection",
        model=GROUNDING_MODEL_ID,
        revision=GROUNDING_REVISION,
        device=_device_index(args.device),
    )
    segment_processor = AutoProcessor.from_pretrained(
        SEGMENTATION_MODEL_ID, revision=SEGMENTATION_REVISION
    )
    segment_model = SegmentationModel.from_pretrained(
        SEGMENTATION_MODEL_ID,
        revision=SEGMENTATION_REVISION,
    ).to(args.device)
    segment_model.eval()

    proposals = []
    for view, rgba in enumerate(images):
        rgb = rgba.convert("RGB")
        detections = detector(
            rgb,
            candidate_labels=prompts,
            threshold=float(args.detection_threshold),
        )
        by_label = defaultdict(list)
        for row in detections:
            raw_label = str(row["label"]).strip().lower().rstrip(".")
            label = prompt_to_label.get(raw_label, raw_label.removeprefix("a "))
            if label not in labels:
                continue
            by_label[label].append(row)
        selected = []
        for label in labels:
            selected.extend(
                sorted(by_label[label], key=lambda r: -float(r["score"]))[: args.max_per_label]
            )
        if not selected:
            print(f"V{view}: no detections", flush=True)
            continue
        boxes = []
        labels_for_boxes = []
        scores = []
        for row in selected:
            box = row["box"]
            boxes.append(
                [
                    float(box["xmin"]),
                    float(box["ymin"]),
                    float(box["xmax"]),
                    float(box["ymax"]),
                ]
            )
            raw_label = str(row["label"]).strip().lower().rstrip(".")
            labels_for_boxes.append(prompt_to_label.get(raw_label, raw_label.removeprefix("a ")))
            scores.append(float(row["score"]))
        inputs = segment_processor(
            images=rgb,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(args.device)
        with torch.no_grad():
            outputs = segment_model(**inputs, multimask_output=False)
        masks = segment_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
        )[0]
        alpha = np.asarray(rgba)[:, :, 3] > 0
        for ordinal, (label, score) in enumerate(zip(labels_for_boxes, scores)):
            mask = _normalise_mask_tensor(masks[ordinal], rgba.height, rgba.width)
            mask &= alpha
            if int(mask.sum()) < int(args.minimum_mask_pixels):
                continue
            proposals.append(
                {
                    "view_index": view,
                    "label": label,
                    "score": score,
                    "proposal_id": f"V{view}:{label}:{ordinal:03d}",
                    "mask": mask,
                }
            )
        print(f"V{view}: {len(selected)} boxes -> {sum(p['view_index']==view for p in proposals)} masks", flush=True)

    if not proposals:
        raise RuntimeError("semantic provider produced no accepted masks")
    source_identity = "SHA256:" + sha256(args.source_zip.read_bytes()).hexdigest()
    provider_id = (
        f"FROZEN_OPEN_VOCABULARY_MASK_STACK_V1|"
        f"{GROUNDING_MODEL_ID}@{GROUNDING_REVISION}|"
        f"{SEGMENTATION_MODEL_ID}@{SEGMENTATION_REVISION}"
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
        "grounding_safetensors_sha256": GROUNDING_SAFETENSORS_SHA256,
        "segmentation_safetensors_sha256": SEGMENTATION_SAFETENSORS_SHA256,
        "geometry_model_loaded": False,
        "geometry_checkpoint_modified": False,
        "canonical_geometry_authority": "UNCHANGED",
    }
    print(summary, flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    p.add_argument("--detection-threshold", type=float, default=0.16)
    p.add_argument("--max-per-label", type=int, default=2)
    p.add_argument("--minimum-mask-pixels", type=int, default=16)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
