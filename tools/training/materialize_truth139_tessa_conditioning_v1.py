#!/usr/bin/env python3
"""Materialize per-subject TESSA conditioning caches from a sealed Truth139 IRIS checkpoint.

Run this independently for the TRAIN manifest and, only after fit closure, for the physically
separate UNSEEN manifest.  Teacher topology files are neither accepted nor opened by this tool.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from models.iris.training.cohort_v1 import file_hash, value_hash
from models.iris.v31.evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2
from models.iris.v31.tessa_bridge_v1 import (
    Truth139TESSABridgeConfigV1,
    Truth139TESSASourceBuilderV1,
)
from models.iris.v31.tessa_conditioning_cache_v1 import (
    Truth139TESSAConditioningPolicyV1,
    build_truth139_tessa_conditioning_v1,
)
from models.iris.v31.truth139_model_v1 import IRISV31Truth139V1
from models.iris.v31.truth139_trainer_v1 import Truth139SubjectV1


TRAIN_MANIFEST_SCHEMA = "RealSaS.IRISV31Truth139TrainManifest.v1"
UNSEEN_MANIFEST_SCHEMA = "RealSaS.IRISV31Truth139UnseenEvalManifest.v1"
TRAIN_RESULT_SCHEMA = "RealSaS.IRISV31Truth139TrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.IRISV31Truth139Checkpoint.v1"
INDEX_SCHEMA = "RealSaS.Truth139TESSAConditioningIndex.v1"


def _require(value, code: str) -> None:
    if not value:
        raise ValueError(code)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def materialize(
    *,
    subject_manifest: dict,
    training_result: dict,
    recipe: dict,
    checkpoint_path,
    output_root,
    scope: str,
    device: str = "cuda",
    policy: Truth139TESSAConditioningPolicyV1 = Truth139TESSAConditioningPolicyV1(),
) -> dict:
    if scope not in {"TRAIN", "UNSEEN"}:
        raise ValueError("TRUTH139_TESSA_CONDITIONING_SCOPE")
    expected_schema = TRAIN_MANIFEST_SCHEMA if scope == "TRAIN" else UNSEEN_MANIFEST_SCHEMA
    _require(subject_manifest.get("schema") == expected_schema, "TRUTH139_TESSA_CONDITIONING_MANIFEST_SCHEMA")
    _require(training_result.get("schema") == TRAIN_RESULT_SCHEMA, "TRUTH139_TESSA_CONDITIONING_TRAIN_RESULT_SCHEMA")
    split = str(subject_manifest.get("split_identity_sha256") or "")
    _require(split == str(training_result.get("split_identity_sha256") or ""), "TRUTH139_TESSA_CONDITIONING_SPLIT_DRIFT")
    if scope == "TRAIN":
        _require(subject_manifest.get("contains_unseen_subject_ids") is False, "TRUTH139_TESSA_CONDITIONING_TRAIN_UNSEEN_IDS")
        _require(subject_manifest.get("contains_unseen_file_refs") is False, "TRUTH139_TESSA_CONDITIONING_TRAIN_UNSEEN_REFS")
    else:
        _require(subject_manifest.get("optimizer_updates_allowed") is False, "TRUTH139_TESSA_CONDITIONING_UNSEEN_OPTIMIZER")
        _require(subject_manifest.get("checkpoint_selection_allowed") is False, "TRUTH139_TESSA_CONDITIONING_UNSEEN_SELECTION")

    checkpoint = Path(checkpoint_path).resolve()
    _require(
        checkpoint.is_file()
        and file_hash(checkpoint) == str(training_result.get("selected_checkpoint_sha256") or ""),
        "TRUTH139_TESSA_CONDITIONING_CHECKPOINT_DRIFT",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == CHECKPOINT_SCHEMA, "TRUTH139_TESSA_CONDITIONING_CHECKPOINT_SCHEMA")
    _require(payload.get("foundation_identity") == training_result.get("foundation_identity"), "TRUTH139_TESSA_CONDITIONING_FOUNDATION_DRIFT")

    config_row = dict(payload["config"])
    config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
    config_row["pair_channels"] = tuple(config_row["pair_channels"])
    iris_config = IRISEvidenceAdapterConfigV2(**config_row)
    dev = torch.device(device)
    _require(dev.type == "cuda" and torch.cuda.is_available(), "TRUTH139_TESSA_CONDITIONING_CUDA_REQUIRED")
    model = IRISV31Truth139V1(iris_config).to(dev)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    source_builder = Truth139TESSASourceBuilderV1(
        Truth139TESSABridgeConfigV1(
            scene_channels=iris_config.carrier_channels,
            iris_query_width=iris_config.query_width,
            pooled_scene_resolution=12,
        )
    )
    policy.validate()
    output = Path(output_root).resolve()
    cache_root = output / "conditioning"
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = list(subject_manifest.get("subjects") or ())
    ids = [str(row.get("subject_id") or "") for row in rows]
    _require(ids and len(ids) == len(set(ids)) == int(subject_manifest["subject_count"]), "TRUTH139_TESSA_CONDITIONING_SUBJECT_COUNT")
    subject_rows = []
    for row in rows:
        subject = Truth139SubjectV1(row, device=dev, recipe=recipe)
        final, layer2, alpha, cameras = subject.observation()
        with torch.no_grad():
            carrier = model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras)
            cache = build_truth139_tessa_conditioning_v1(
                iris_model=model,
                carrier=carrier,
                source_builder=source_builder,
                policy=policy,
            )
        cache["subject_id"] = subject.subject_id
        cache["source_sha256"] = str(row["source_sha256"])
        cache["observation_binding"] = str(row["observation_binding"])
        cache["iris_checkpoint_sha256"] = str(training_result["selected_checkpoint_sha256"])
        cache["iris_foundation_identity"] = str(training_result["foundation_identity"])
        path = cache_root / f"{subject.subject_id}.pt"
        staged = path.with_suffix(".pt.partial")
        torch.save(cache, staged)
        torch.load(staged, map_location="cpu", weights_only=True)
        staged.replace(path)
        subject_rows.append({
            "subject_id": subject.subject_id,
            "source_sha256": str(row["source_sha256"]),
            "observation_binding": str(row["observation_binding"]),
            "conditioning_cache": {"path": str(path), "sha256": file_hash(path)},
            "conditioning_tensor_sha256": cache["conditioning_tensor_sha256"],
            "surface_witness_count": cache["surface_witness_count"],
        })
        del carrier
        torch.cuda.empty_cache()

    index = {
        "schema": INDEX_SCHEMA,
        "scope": scope,
        "split_identity_sha256": split,
        "subject_manifest_identity_sha256": subject_manifest["manifest_identity_sha256"],
        "iris_training_binding_sha256": training_result["binding_sha256"],
        "iris_checkpoint_sha256": training_result["selected_checkpoint_sha256"],
        "iris_foundation_identity": training_result["foundation_identity"],
        "source_builder": "RealSaS.IRISV31.Truth139.TESSASourceBuilder.v1",
        "conditioning_policy": asdict(policy),
        "subject_count": len(subject_rows),
        "subjects": subject_rows,
        "teacher_topology_used": False,
        "optimizer_created": False,
        "checkpoint_selection_performed": False,
        "semantic_part_taxonomy_used": False,
    }
    index["index_identity_sha256"] = value_hash(index)
    _atomic_json(output / "TESSA_CONDITIONING_INDEX.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scope", choices=("TRAIN", "UNSEEN"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zero-surface-resolution", type=int, default=128)
    parser.add_argument("--maximum-surface-witnesses", type=int, default=8192)
    args = parser.parse_args()
    manifest = json.loads(Path(args.subject_manifest).resolve().read_text())
    result = json.loads(Path(args.training_result).resolve().read_text())
    recipe = json.loads(Path(args.recipe).resolve().read_text())
    index = materialize(
        subject_manifest=manifest,
        training_result=result,
        recipe=recipe,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        scope=args.scope,
        device=args.device,
        policy=Truth139TESSAConditioningPolicyV1(
            zero_surface_resolution=args.zero_surface_resolution,
            maximum_surface_witnesses=args.maximum_surface_witnesses,
        ),
    )
    print(json.dumps({
        "scope": index["scope"],
        "subject_count": index["subject_count"],
        "index_identity_sha256": index["index_identity_sha256"],
        "iris_checkpoint_sha256": index["iris_checkpoint_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
