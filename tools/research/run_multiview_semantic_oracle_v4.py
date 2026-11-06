#!/usr/bin/env python3
from __future__ import annotations

"""Multiview semantic localization -> prompted SAM2 -> multiview mask adjudication.

V4 deliberately separates three responsibilities:
  1. a frozen multimodal model reasons over all eight source views and proposes
     stable semantic instances plus per-view boxes/points;
  2. SAM2 receives those semantic prompts and only performs pixel segmentation;
  3. the multimodal model judges the resulting mask overlays across views.

Only masks that pass both deterministic shape checks and multimodal adjudication
are serialized as FrozenSemanticMaskEvidenceV1.  Nothing in this script imports
or mutates IRIS geometry, canonical M, rig, skin or motion.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.iris.semantic_evidence_v1 import build_mask_evidence_v1, write_mask_evidence_v1
from models.iris.semantic_mask_court_v1 import (
    CANONICAL_SEMANTIC_LABELS,
    audit_prompted_mask_v1,
    normalized_box_to_pixels,
    normalized_point_to_pixels,
    validate_multiview_oracle_payload_v1,
)
from models.iris.semantic_provider_quality_v1 import audit_semantic_proposals_v1


SAM_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
SAM_SAFETENSORS_SHA256 = "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60"
QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_views(source_zip: Path) -> list[Image.Image]:
    rows = []
    with zipfile.ZipFile(source_zip) as archive:
        for view in range(8):
            raw = archive.read(f"observations/V{view}.png")
            rows.append(Image.open(io.BytesIO(raw)).convert("RGBA").copy())
    if len({x.size for x in rows}) != 1:
        raise RuntimeError("source views must have equal dimensions")
    return rows


def _resolve_hf_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision="main")
    if not info.sha:
        raise RuntimeError(f"HF_REVISION_MISSING:{repo_id}")
    return str(info.sha)


def _verify_hf_weight(repo_id: str, revision: str, filename: str, expected_sha256: str) -> str:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"MODEL_WEIGHT_HASH_DRIFT:{repo_id}:{actual}!={expected_sha256}")
    return actual


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("MULTIMODAL_JSON_NOT_FOUND")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MULTIMODAL_JSON_INVALID:{exc}") from exc


def _load_qwen(model_id: str, revision: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map="auto" if str(device).startswith("cuda") else None,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model.eval()
    return model, processor


def _qwen_generate(model, processor, messages, *, max_new_tokens: int) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _localization_prompt() -> str:
    labels = ", ".join(CANONICAL_SEMANTIC_LABELS)
    return f"""
You are the non-authoritative multiview semantic oracle for an animation compiler.
The eight images V0..V7 are orthographic views of the SAME stylized character at equal scale.
Reason across ALL views before deciding. Do not force every allowed label to exist.

Allowed coarse labels only: {labels}.
If a concept is not visibly supported, omit it entirely. Distinguish physical parts from silhouette resemblance.
A pointed helmet/decorative contour is not automatically a horn. A long rigid held object is not an arm merely because it touches the hand.

Identify stable physical instances that matter for animation. Keep the same instance_id across views.
For every instance provide every view key "0".."7". If visible, give a tight box and an interior point in normalized 0..1000 coordinates. The box must cover ONLY that visible part, not the full character and not neighboring parts. If hidden/absent/too uncertain, set present=false and bbox_1000/center_1000=null.

Output JSON only, exactly this shape:
{{
  "schema":"RealSaS.MultiviewSemanticOracle.v1",
  "subject_summary":"short description",
  "instances":[
    {{
      "instance_id":"I00",
      "label":"sword",
      "confidence":0.0,
      "views":{{
        "0":{{"present":true,"bbox_1000":[x1,y1,x2,y2],"center_1000":[x,y],"confidence":0.0}},
        "1":{{"present":false,"bbox_1000":null,"center_1000":null,"confidence":0.0}},
        "2":{{...}},"3":{{...}},"4":{{...}},"5":{{...}},"6":{{...}},"7":{{...}}
      }},
      "relations":[{{"type":"HELD_BY|IN_FRONT_OF|BEHIND|ATTACHED_TO","target_instance_id":"Ixx","confidence":0.0}}]
    }}
  ]
}}

Confidence must be in [0,1]. Prefer UNKNOWN-by-omission over guessing. No markdown.
""".strip()


def _prepare_view_files(images: list[Image.Image], work: Path) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    paths = []
    for v, image in enumerate(images):
        path = work / f"V{v}.png"
        image.save(path)
        paths.append(path)
    return paths


def _run_localization_oracle(model, processor, paths: list[Path], *, max_new_tokens: int) -> tuple[dict, str]:
    content = []
    for v, path in enumerate(paths):
        content.append({"type": "text", "text": f"View V{v}:"})
        content.append({"type": "image", "image": path.resolve().as_uri(), "resized_height": 512, "resized_width": 512})
    content.append({"type": "text", "text": _localization_prompt()})
    messages = [{"role": "user", "content": content}]
    raw = _qwen_generate(model, processor, messages, max_new_tokens=max_new_tokens)
    payload = _extract_json(raw)
    validate_multiview_oracle_payload_v1(payload)
    return payload, raw


def _sam2_masks(images, oracle, *, device: str):
    import torch
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(SAM_MODEL_ID, revision=SAM_REVISION)
    model = Sam2Model.from_pretrained(SAM_MODEL_ID, revision=SAM_REVISION).to(device)
    model.eval()

    rows = []
    by_view = {v: [] for v in range(8)}
    for inst in oracle["instances"]:
        if float(inst["confidence"]) < 0.40:
            continue
        for v in range(8):
            item = inst["views"][str(v)]
            if not bool(item["present"]) or float(item.get("confidence", 0.0)) < 0.35:
                continue
            by_view[v].append((inst, item))

    for view, items in by_view.items():
        if not items:
            continue
        rgba = images[view]
        rgb = rgba.convert("RGB")
        alpha = np.asarray(rgba)[:, :, 3] > 0
        h, w = alpha.shape
        boxes, points = [], []
        for inst, item in items:
            boxes.append(list(normalized_box_to_pixels(item["bbox_1000"], width=w, height=h)))
            points.append(list(normalized_point_to_pixels(item["center_1000"], width=w, height=h)))
        inputs = processor(
            images=rgb,
            input_boxes=[boxes],
            input_points=[[[p] for p in points]],
            input_labels=[[[1] for _ in points]],
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.detach().float().cpu().numpy()
        arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        while arr.ndim > 3 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[0] != len(items):
            raise RuntimeError(f"SAM2_PROMPTED_MASK_SHAPE_DRIFT:{arr.shape}:{len(items)}")
        flat_scores = np.asarray(scores).reshape(-1)
        for idx, ((inst, item), mask, box, center) in enumerate(zip(items, arr, boxes, points)):
            mask = np.asarray(mask, dtype=bool) & alpha
            court = audit_prompted_mask_v1(
                mask,
                subject_mask=alpha,
                prompt_box=box,
                prompt_center=center,
            )
            rows.append(
                {
                    "instance_id": inst["instance_id"],
                    "label": inst["label"],
                    "instance_confidence": float(inst["confidence"]),
                    "view_index": view,
                    "view_confidence": float(item.get("confidence", 0.0)),
                    "bbox_1000": item["bbox_1000"],
                    "center_1000": item["center_1000"],
                    "prompt_box_pixels": box,
                    "prompt_center_pixels": center,
                    "sam_iou_score": float(flat_scores[idx]) if idx < len(flat_scores) else 0.0,
                    "shape_court": asdict(court),
                    "mask": mask,
                }
            )
    return rows


def _overlay_sheet(images, rows, *, instance_id: str, label: str) -> Image.Image:
    panel = 256
    sheet = Image.new("RGB", (panel * 4, panel * 2), "white")
    by_view = {int(r["view_index"]): r for r in rows if r["instance_id"] == instance_id}
    for v in range(8):
        base = images[v].convert("RGBA").resize((panel, panel), Image.Resampling.BILINEAR)
        row = by_view.get(v)
        if row is not None:
            mask = Image.fromarray((row["mask"].astype(np.uint8) * 255), mode="L").resize((panel, panel), Image.Resampling.NEAREST)
            red = Image.new("RGBA", (panel, panel), (255, 0, 0, 115))
            base = Image.composite(red, base, mask)
            draw = ImageDraw.Draw(base)
            x1, y1, x2, y2 = [int(round(x * panel / images[v].size[0])) for x in row["prompt_box_pixels"]]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=2)
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, 120, 22], fill=(0, 0, 0, 180))
        draw.text((4, 3), f"V{v} {instance_id} {label}", fill=(255, 255, 255, 255))
        sheet.paste(base.convert("RGB"), ((v % 4) * panel, (v // 4) * panel))
    return sheet


def _adjudication_prompt(instance_order) -> str:
    refs = ", ".join(f"{iid}:{label}" for iid, label in instance_order)
    return f"""
You are judging segmentation masks for a multiview animation compiler. Each supplied image is an 8-view sheet for one proposed physical instance; red is the SAM2 mask and yellow is the semantic localization box.
Instances in image order: {refs}.

For each instance decide whether the RED masks, across the views where present, actually follow ONE coherent physical part matching the proposed label. Reject masks that are fragments, swallow neighboring parts, merge multiple parts, follow only arbitrary texture/edge islands, or systematically target the wrong semantic part. Do not reward a plausible label if the mask shape is wrong. If evidence is insufficient, UNKNOWN.

Return JSON only:
{{"schema":"RealSaS.MultiviewMaskAdjudication.v1","instances":[
  {{"instance_id":"I00","label":"sword","verdict":"PASS|FAIL|UNKNOWN","semantic_match":true,"confidence":0.0,"bad_views":[0],"reason_code":"GOOD|FRAGMENT|MULTI_PART|WRONG_PART|BOUNDARY_POOR|INSUFFICIENT"}}
]}}
No markdown.
""".strip()


def _run_mask_adjudication(model, processor, images, rows, work: Path, *, max_new_tokens: int):
    instances = []
    seen = set()
    for row in rows:
        key = (row["instance_id"], row["label"])
        if key not in seen:
            seen.add(key)
            instances.append(key)
    if not instances:
        raise RuntimeError("NO_PROMPTED_INSTANCES_FOR_ADJUDICATION")

    work.mkdir(parents=True, exist_ok=True)
    content = []
    for index, (iid, label) in enumerate(instances):
        sheet = _overlay_sheet(images, rows, instance_id=iid, label=label)
        path = work / f"{index:02d}_{iid}_{label}.png"
        sheet.save(path)
        content.append({"type": "text", "text": f"Instance sheet {iid}:{label}"})
        content.append({"type": "image", "image": path.resolve().as_uri(), "resized_height": 512, "resized_width": 1024})
    content.append({"type": "text", "text": _adjudication_prompt(instances)})
    raw = _qwen_generate(model, processor, [{"role": "user", "content": content}], max_new_tokens=max_new_tokens)
    payload = _extract_json(raw)
    if payload.get("schema") != "RealSaS.MultiviewMaskAdjudication.v1":
        raise RuntimeError("MASK_ADJUDICATION_SCHEMA_MISMATCH")
    decisions = payload.get("instances")
    if not isinstance(decisions, list):
        raise RuntimeError("MASK_ADJUDICATION_INSTANCES_MISSING")
    out = {}
    expected = {iid: label for iid, label in instances}
    for item in decisions:
        iid = str(item.get("instance_id", ""))
        if iid not in expected or iid in out:
            continue
        verdict = str(item.get("verdict", "UNKNOWN")).upper()
        if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
            verdict = "UNKNOWN"
        out[iid] = {
            "instance_id": iid,
            "label": expected[iid],
            "verdict": verdict,
            "semantic_match": bool(item.get("semantic_match", False)),
            "confidence": min(max(float(item.get("confidence", 0.0)), 0.0), 1.0),
            "bad_views": [int(v) for v in item.get("bad_views", []) if str(v).isdigit() and 0 <= int(v) <= 7],
            "reason_code": str(item.get("reason_code", "INSUFFICIENT")),
        }
    for iid, label in instances:
        out.setdefault(iid, {
            "instance_id": iid,
            "label": label,
            "verdict": "UNKNOWN",
            "semantic_match": False,
            "confidence": 0.0,
            "bad_views": [],
            "reason_code": "INSUFFICIENT",
        })
    payload["instances"] = [out[iid] for iid, _ in instances]
    return payload, raw, list(work.glob("*.png"))


def run(args) -> dict:
    images = _read_views(args.source_zip)
    qwen_revision = _resolve_hf_revision(args.qwen_model)
    _verify_hf_weight(SAM_MODEL_ID, SAM_REVISION, "model.safetensors", SAM_SAFETENSORS_SHA256)

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    view_paths = _prepare_view_files(images, work / "source_views")

    qwen_model, qwen_processor = _load_qwen(args.qwen_model, qwen_revision, args.device)
    oracle, oracle_raw = _run_localization_oracle(
        qwen_model, qwen_processor, view_paths, max_new_tokens=args.max_oracle_tokens
    )
    args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
    args.oracle_output.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    (work / "oracle_raw.txt").write_text(oracle_raw, encoding="utf-8")

    rows = _sam2_masks(images, oracle, device=args.device)
    if not rows:
        raise RuntimeError("PROMPTED_SAM2_NO_MASKS")

    adjudication, adjudication_raw, overlay_paths = _run_mask_adjudication(
        qwen_model,
        qwen_processor,
        images,
        rows,
        work / "mask_adjudication_sheets",
        max_new_tokens=args.max_adjudication_tokens,
    )
    args.adjudication_output.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    (work / "adjudication_raw.txt").write_text(adjudication_raw, encoding="utf-8")
    by_iid = {x["instance_id"]: x for x in adjudication["instances"]}

    proposals = []
    rejected = {}
    subject_masks = [np.asarray(x)[:, :, 3] > 0 for x in images]
    for row in rows:
        shape = row["shape_court"]
        decision = by_iid.get(row["instance_id"], {})
        failure = None
        if shape["status"] != "PASS":
            failure = shape.get("failure_code") or "SHAPE_COURT_FAIL"
        elif decision.get("verdict") != "PASS":
            failure = "MULTIVIEW_MASK_" + str(decision.get("verdict", "UNKNOWN"))
        elif not bool(decision.get("semantic_match", False)):
            failure = "MULTIVIEW_SEMANTIC_MISMATCH"
        elif float(decision.get("confidence", 0.0)) < float(args.minimum_adjudication_confidence):
            failure = "MULTIVIEW_ADJUDICATION_LOW_CONFIDENCE"
        elif int(row["view_index"]) in set(decision.get("bad_views", [])):
            failure = "MULTIVIEW_BAD_VIEW"
        if failure is not None:
            rejected[failure] = rejected.get(failure, 0) + 1
            continue
        confidence = float(np.sqrt(max(row["sam_iou_score"], 0.0) * max(decision["confidence"], 0.0)))
        confidence *= float(np.sqrt(max(row["view_confidence"], 0.0) * max(row["instance_confidence"], 0.0)))
        proposals.append({
            "view_index": int(row["view_index"]),
            "label": str(row["label"]),
            "score": float(min(max(confidence, 0.0), 1.0)),
            "proposal_id": f"{row['instance_id']}:V{row['view_index']}:{row['label']}",
            "mask": row["mask"],
        })

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

    labels = tuple(sorted({p["label"] for p in proposals}))
    audit = {
        "schema": "RealSaS.FrozenSemanticProviderAudit.v4",
        "status": quality.status,
        "failure_code": quality.failure_code,
        "provider": "QWEN3VL_8VIEW_LOCALIZATION__PROMPTED_SAM2__QWEN3VL_MASK_COURT",
        "qwen_model": f"{args.qwen_model}@{qwen_revision}",
        "sam_model": f"{SAM_MODEL_ID}@{SAM_REVISION}",
        "oracle_instance_count": len(oracle["instances"]),
        "prompted_mask_count": len(rows),
        "accepted_proposal_count": len(proposals),
        "accepted_labels": list(labels),
        "rejection_totals": rejected,
        "quality": asdict(quality),
        "geometry_model_loaded": False,
        "canonical_geometry_authority": "UNCHANGED",
        "training_executed": False,
        "runtime_behavior_changed": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if quality.failure_code is not None:
        raise RuntimeError(quality.failure_code)

    source_identity = "SHA256:" + sha256(args.source_zip.read_bytes()).hexdigest()
    provider_id = (
        "FROZEN_MULTIVIEW_SEMANTIC_ORACLE_V4|"
        f"{args.qwen_model}@{qwen_revision}|{SAM_MODEL_ID}@{SAM_REVISION}"
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
        "source_identity": source_identity,
        "qwen_revision": qwen_revision,
        "sam_safetensors_sha256": SAM_SAFETENSORS_SHA256,
        "oracle_output": str(args.oracle_output),
        "adjudication_output": str(args.adjudication_output),
        "provider_audit": str(args.audit_output),
        "overlay_sheet_count": len(overlay_paths),
        "proposal_count": evidence.proposal_count,
        "labels": list(evidence.labels),
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
    p.add_argument("--oracle-output", type=Path, required=True)
    p.add_argument("--adjudication-output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--qwen-model", default=QWEN_MODEL_ID)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-oracle-tokens", type=int, default=4096)
    p.add_argument("--max-adjudication-tokens", type=int, default=2048)
    p.add_argument("--minimum-adjudication-confidence", type=float, default=0.60)
    p.add_argument("--collapse-iou", type=float, default=0.97)
    p.add_argument("--maximum-duplicate-pair-fraction", type=float, default=0.10)
    p.add_argument("--maximum-subject-coverage", type=float, default=0.92)
    p.add_argument("--maximum-whole-subject-fraction", type=float, default=0.20)
    p.add_argument("--minimum-distinct-labels", type=int, default=3)
    p.add_argument("--minimum-total-proposals", type=int, default=6)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
