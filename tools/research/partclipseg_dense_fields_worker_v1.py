#!/usr/bin/env python3
from __future__ import annotations

"""Run the official frozen PartCLIPSeg checkpoint as an isolated dense scorer.

This worker intentionally runs in a subprocess whose PYTHONPATH begins with the
pinned upstream PartCLIPSeg repository.  That repository carries its own
Transformers fork.  Keeping it isolated prevents the legacy research runtime
from replacing the current Transformers/SAM2 environment used by RealSaS.

Output is dense non-authoritative evidence only.  The worker never imports IRIS
geometry and never writes canonical M/rig/skin/motion state.
"""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image


UPSTREAM_REPO = "https://github.com/kaist-cvml/part-clipseg.git"
UPSTREAM_COMMIT = "8abceefff960920019096982688e3e8209ed39de"
BASE_MODEL = "CIDAS/clipseg-rd64-refined"

LABELS = (
    "head",
    "arm",
    "forearm",
    "hand",
    "cape",
    "hair",
    "cloth",
    "armor",
    "helmet",
    "horn",
    "sword",
    "shield",
)
DEFAULT_OBJECT_CONTEXTS = ("person", "knight", "character")


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_views(path: Path) -> list[Image.Image]:
    out = []
    with zipfile.ZipFile(path) as z:
        for view in range(8):
            out.append(Image.open(io.BytesIO(z.read(f"observations/V{view}.png"))).convert("RGB").copy())
    if len({im.size for im in out}) != 1:
        raise RuntimeError("source views must have equal dimensions")
    return out


def _verify_upstream(repo: Path) -> None:
    if not (repo / ".git").exists():
        raise RuntimeError("PARTCLIPSEG_UPSTREAM_REPO_MISSING")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"PARTCLIPSEG_UPSTREAM_COMMIT_DRIFT:{head}!={UPSTREAM_COMMIT}")


def _extract_model_state(checkpoint: Path) -> tuple[dict, str]:
    import torch

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise RuntimeError("PARTCLIPSEG_CHECKPOINT_STATE_INVALID")

    extracted = {}
    marker = "partclipseg_model."
    for key, value in state.items():
        key = str(key)
        if marker not in key:
            continue
        extracted[key.split(marker, 1)[1]] = value
    if not extracted:
        raise RuntimeError("PARTCLIPSEG_MODEL_KEYS_NOT_FOUND")
    return extracted, _sha256(checkpoint)


def _initialize_official_extra_layers(model) -> None:
    # Mirrors the official PartCLIPSeg wrapper before Detectron2 Checkpointer
    # applies the trained zero-shot checkpoint.
    decoder = model.decoder
    decoder.part_obj_with_cond_embed.load_state_dict(decoder.transposed_convolution.state_dict())
    decoder.cond_film_add.load_state_dict(decoder.film_add.state_dict())
    decoder.cond_film_mul.load_state_dict(decoder.film_mul.state_dict())


def _load_model(checkpoint: Path, device: str):
    import torch
    from transformers import PartCLIPSegForImageSegmentation, PartCLIPSegProcessor

    processor = PartCLIPSegProcessor.from_pretrained(BASE_MODEL)
    model = PartCLIPSegForImageSegmentation.from_pretrained(BASE_MODEL)
    _initialize_official_extra_layers(model)
    state, checkpoint_sha = _extract_model_state(checkpoint)

    target = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in target and tuple(target[k].shape) == tuple(v.shape)
    }
    if not compatible:
        raise RuntimeError("PARTCLIPSEG_NO_COMPATIBLE_CHECKPOINT_TENSORS")
    loaded_numel = sum(int(target[k].numel()) for k in compatible)
    target_numel = sum(int(v.numel()) for v in target.values())
    coverage = loaded_numel / max(target_numel, 1)
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    required_prefixes = (
        "decoder.part_obj_with_cond_embed",
        "decoder.cond_film_add",
        "decoder.cond_film_mul",
    )
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in compatible):
            raise RuntimeError(f"PARTCLIPSEG_ADAPTED_WEIGHTS_MISSING:{prefix}")
    if coverage < 0.90:
        raise RuntimeError(f"PARTCLIPSEG_CHECKPOINT_COVERAGE_TOO_LOW:{coverage:.6f}")

    model = model.to(device)
    model.eval()
    return processor, model, {
        "checkpoint_sha256": checkpoint_sha,
        "compatible_tensor_count": len(compatible),
        "checkpoint_tensor_count": len(state),
        "target_tensor_count": len(target),
        "loaded_parameter_numel_fraction": coverage,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
    }


def _maps(labels: tuple[str, ...], contexts: tuple[str, ...], device: str):
    import torch

    part_index = []
    obj_index = []
    # Object-major ordering: each requested object context gets all requested
    # generalized parts.  The output part_obj_logits follows this pair order.
    for oi, _ in enumerate(contexts):
        for pi, _ in enumerate(labels):
            part_index.append(pi)
            obj_index.append(oi)
    part_index = torch.tensor(part_index, dtype=torch.long, device=device)[None, None, None, :]
    obj_index = torch.tensor(obj_index, dtype=torch.long, device=device)[None, None, None, :]
    return part_index, obj_index


def _dense_for_image(processor, model, image: Image.Image, *, device: str, contexts: tuple[str, ...]):
    import torch
    import torch.nn.functional as F

    labels = LABELS
    pair_count = len(labels) * len(contexts)
    part_index, obj_index = _maps(labels, contexts, device)

    pixel_inputs = processor(images=[image], return_tensors="pt").to(device)
    text_inputs = processor.tokenizer(
        list(labels) + list(contexts),
        return_tensors="pt",
        padding="max_length",
    ).to(device)
    pixel_inputs.update(text_inputs)
    pixel_inputs["part_index_map"] = part_index
    pixel_inputs["obj_index_map"] = obj_index
    pixel_inputs["num_text_classes"] = pair_count
    pixel_inputs["num_part_classes"] = len(labels)
    pixel_inputs["output_attentions"] = False

    with torch.no_grad():
        outputs = model(**pixel_inputs)

    h, w = image.size[1], image.size[0]
    generalized = outputs.logits[:, : len(labels)]
    generalized = torch.sigmoid(
        F.interpolate(generalized, size=(h, w), mode="bilinear", align_corners=False)
    )[0]

    contextual_with_bg = torch.sigmoid(
        F.interpolate(outputs.part_obj_logits, size=(h, w), mode="bilinear", align_corners=False)
    )[0]
    if contextual_with_bg.shape[0] != pair_count + 1:
        raise RuntimeError(
            f"PARTCLIPSEG_PAIR_LOGIT_SHAPE_DRIFT:{contextual_with_bg.shape[0]}!={pair_count + 1}"
        )
    pair = contextual_with_bg[:-1].reshape(len(contexts), len(labels), h, w)
    contextual = pair.max(dim=0).values
    background = contextual_with_bg[-1]

    # Generalized-part evidence and object-conditioned part evidence are both
    # required.  Geometric mean penalizes a class that is strong in only one.
    dense = torch.sqrt(torch.clamp(generalized * contextual, min=0.0, max=1.0))
    return (
        dense.detach().cpu().numpy().astype(np.float16),
        background.detach().cpu().numpy().astype(np.float16),
    )


def run(args) -> dict:
    _verify_upstream(args.upstream_repo)
    processor, model, load_report = _load_model(args.checkpoint, args.device)
    contexts = tuple(x.strip() for x in args.object_contexts.split(",") if x.strip())
    if not contexts:
        raise RuntimeError("PARTCLIPSEG_OBJECT_CONTEXTS_EMPTY")

    views = _read_views(args.source_zip)
    fields, backgrounds = [], []
    for view, image in enumerate(views):
        dense, bg = _dense_for_image(processor, model, image, device=args.device, contexts=contexts)
        fields.append(dense)
        backgrounds.append(bg)
        print(f"V{view}: PartCLIPSeg dense fields ready", flush=True)

    metadata = {
        "schema": "RealSaS.PartCLIPSegDenseFields.v1",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_source": str(args.checkpoint_source),
        "base_model": BASE_MODEL,
        "labels": list(LABELS),
        "object_contexts": list(contexts),
        "geometry_model_loaded": False,
        "canonical_geometry_authority": "UNCHANGED",
        **load_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        part_probabilities=np.stack(fields),
        background_probability=np.stack(backgrounds),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        **metadata,
        "path": str(args.output),
        "sha256": _sha256(args.output),
        "shape": list(np.stack(fields).shape),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-zip", type=Path, required=True)
    p.add_argument("--upstream-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-source", default="OFFICIAL_GOOGLE_DRIVE")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--object-contexts", default=",".join(DEFAULT_OBJECT_CONTEXTS))
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
