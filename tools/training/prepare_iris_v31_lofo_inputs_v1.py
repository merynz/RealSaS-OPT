#!/usr/bin/env python3
"""Prepare exact IRIS V3.1 LOFO inputs from existing FIT8 geometry + Stage48 truth.

This is the bridge that was missing between the frozen V3.1 apparatus and real
FIT8 training.  It intentionally reuses the already-qualified image-derived F*
query shards for geometry while building the V3.1 FINAL384/L2-64/alpha-64 frozen
feature cache from the exact admitted observations.  Stage48 presentation targets
are additive children; neither source is regenerated or mutated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2
from models.iris.v31.alpha_resampling_v1 import policy_sha256_v1, resample_source_alpha_v1
from models.iris.v31.frozen_features_v31 import cache_identity, get_or_build_cache
from models.iris.v31.numerical_environment_v1 import numerical_environment_sha256_v1, numerical_environment_v1
from models.training_runtime.durable_state_v1 import file_sha256
from tools.training.prepare_iris_cohort_v1 import verify_observations

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LEGACY_MANIFEST_SHA256 = "088283fbca5ac6c95dc8d9fedadaafbf9bf82603cc6bee28948af995945e5053"


def require(value, code):
    if not value:
        raise ValueError(code)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _native_observation(folder):
    rgb, alpha = [], []
    for view in range(8):
        rgba = np.asarray(Image.open(Path(folder) / f"V{view}.png").convert("RGBA"), dtype=np.uint8)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        # Preserve the existing admitted white-composite DINO observation law.
        composed = np.rint(rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
        rgb.append(torch.from_numpy(composed.copy()).permute(2, 0, 1))
        alpha.append(torch.from_numpy(rgba[..., 3].copy())[None])
    return torch.stack(rgb, dim=0)[None], torch.stack(alpha, dim=0)[None]


def prepare(
    *,
    input_root,
    legacy_manifest,
    legacy_shard_root,
    presentation_bundle,
    presentation_root,
    source_dir,
    weight_path,
    cache_root,
    output,
    device="cuda",
    view_batch_size=8,
):
    input_root = Path(input_root).resolve()
    legacy_manifest = Path(legacy_manifest).resolve()
    legacy_shard_root = Path(legacy_shard_root).resolve()
    presentation_bundle = Path(presentation_bundle).resolve()
    presentation_root = Path(presentation_root).resolve()
    cache_root = Path(cache_root).resolve()
    output = Path(output).resolve()
    require(file_sha256(legacy_manifest) == EXPECTED_LEGACY_MANIFEST_SHA256, "IRIS_V31_LOFO_LEGACY_MANIFEST_DRIFT")
    legacy = json.loads(legacy_manifest.read_text())
    require(
        legacy.get("schema") == "RealSaS.SharedIrisCohortShards.v1"
        and legacy.get("teacher_inputs_in_forward") is False
        and len(legacy.get("subjects") or ()) == 8,
        "IRIS_V31_LOFO_LEGACY_MANIFEST_INVALID",
    )
    presentation = json.loads(presentation_bundle.read_text())
    require(
        presentation.get("schema") == "RealSaS.Fit8PresentationTruthBundle.v1"
        and presentation.get("status") == "READY_FOR_LOFO_RECIPE_BINDING"
        and presentation.get("semantic_part_taxonomy_used") is False
        and len(presentation.get("cohort") or ()) == 8,
        "IRIS_V31_LOFO_PRESENTATION_BUNDLE_INVALID",
    )
    legacy_rows = {row["subject_id"]: row for row in legacy["subjects"]}
    presentation_rows = {row["subject_id"]: row for row in presentation["cohort"]}
    require(set(legacy_rows) == set(presentation_rows), "IRIS_V31_LOFO_COHORT_IDENTITY_DRIFT")

    roster = json.loads((ROOT / "canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
    selected = roster["selected"]
    require(tuple(row["subject_key"] for row in selected) == tuple(legacy_rows), "IRIS_V31_LOFO_ROSTER_ORDER_DRIFT")
    verified = [verify_observations(input_root, row) for row in selected]

    model, authority = load_exact_dinov2_s_v2(source_dir, weight_path, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    env = numerical_environment_v1(device)
    env_sha = numerical_environment_sha256_v1(device)
    alpha_policy_sha = policy_sha256_v1()
    subject_rows = []
    try:
        for row, (folder, proof, binding) in zip(selected, verified):
            subject = row["subject_key"]
            legacy_row = legacy_rows[subject]
            presentation_row = presentation_rows[subject]
            require(binding == legacy_row["observation_binding"], "IRIS_V31_LOFO_OBSERVATION_BINDING_DRIFT:" + subject)
            legacy_shard = legacy_shard_root / legacy_row["shard_file"]
            require(legacy_shard.is_file() and file_sha256(legacy_shard) == legacy_row["shard_sha256"], "IRIS_V31_LOFO_GEOMETRY_SHARD_DRIFT:" + subject)
            target_path = presentation_root / presentation_row["v31_target_file"]
            require(target_path.is_file() and file_sha256(target_path) == presentation_row["v31_target_sha256"], "IRIS_V31_LOFO_PRESENTATION_TARGET_DRIFT:" + subject)
            native_rgb, native_alpha = _native_observation(folder)
            alpha_518 = resample_source_alpha_v1(native_alpha)
            identity = cache_identity(
                observation_bundle_sha256=binding,
                camera_bundle_sha256=str(proof["camera_sha256"]),
                alpha_resampling_policy_sha256=alpha_policy_sha,
                numerical_environment_sha256=env_sha,
            )
            features, cache_hit = get_or_build_cache(
                cache_root,
                identity,
                build=lambda rgb=native_rgb, a=alpha_518: __import__(
                    "models.iris.v31.frozen_features_v31", fromlist=["extract_frozen_features"]
                ).extract_frozen_features(
                    model,
                    rgb.to(device),
                    a,
                    view_batch_size=view_batch_size,
                ),
            )
            # Validate readback and locate the content-addressed cache directory.
            cache_key = digest_json(identity)
            receipt_path = cache_root / cache_key / "CACHE.json"
            feature_path = cache_root / cache_key / "features.pt"
            require(receipt_path.is_file() and feature_path.is_file(), "IRIS_V31_LOFO_CACHE_MISSING:" + subject)
            receipt = json.loads(receipt_path.read_text())
            require(receipt.get("identity") == identity and file_sha256(feature_path) == receipt["sha256"], "IRIS_V31_LOFO_CACHE_RECEIPT_DRIFT:" + subject)
            require(set(features) == {"final", "layer2", "alpha"}, "IRIS_V31_LOFO_CACHE_FIELDS_DRIFT:" + subject)
            # The legacy shard camera features are still used because V3.1 admits
            # the same factorized 13-D camera contract.  Read only enough to prove
            # the packet is compatible; do not duplicate the giant query pools.
            shard = torch.load(legacy_shard, map_location="cpu", weights_only=True)
            require(
                shard.get("subject_id") == subject
                and tuple(shard["camera_features"].shape) == (1, 8, 13)
                and tuple(shard["image_tokens"].shape) == (1, 8, 1369, 384)
                and shard["observation_binding"] == binding,
                "IRIS_V31_LOFO_LEGACY_SHARD_LAYOUT_DRIFT:" + subject,
            )
            subject_rows.append(
                {
                    "subject_id": subject,
                    "source_sha256": legacy_row["source_sha256"],
                    "observation_binding": binding,
                    "geometry_shard": {"path": str(legacy_shard), "sha256": legacy_row["shard_sha256"]},
                    "presentation_targets": {"path": str(target_path), "sha256": presentation_row["v31_target_sha256"]},
                    "frozen_feature_cache": {
                        "identity_sha256": cache_key,
                        "receipt_path": str(receipt_path),
                        "receipt_sha256": file_sha256(receipt_path),
                        "features_path": str(feature_path),
                        "features_sha256": receipt["sha256"],
                        "cache_hit": bool(cache_hit),
                    },
                    "camera_features_contract": "LEGACY_FIT8_FACTOR13__SAME_ADMITTED_CAMERA_NORMALIZATION",
                }
            )
            print(f"[IRIS_V31_LOFO_INPUT] subject={subject} cache_hit={int(cache_hit)} cache={cache_key[:12]}", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    order = tuple(row["subject_id"] for row in subject_rows)
    folds = []
    for index, heldout in enumerate(order):
        folds.append(
            {
                "fold_index": index,
                "fold_id": f"LOFO_{index:02d}_{heldout}",
                "train_subjects": [subject for subject in order if subject != heldout],
                "heldout_subject": heldout,
                "checkpoint_selection_may_read_heldout": False,
                "heldout_gradient_updates": False,
            }
        )
    manifest = {
        "schema": "RealSaS.IRISV31LofoInputManifest.v1",
        "legacy_geometry_manifest": {"path": str(legacy_manifest), "sha256": EXPECTED_LEGACY_MANIFEST_SHA256},
        "presentation_truth_bundle": {"path": str(presentation_bundle), "sha256": file_sha256(presentation_bundle)},
        "foundation_authority_sha256": authority.authority.authority_hash,
        "alpha_resampling_policy_sha256": alpha_policy_sha,
        "numerical_environment": env,
        "numerical_environment_sha256": env_sha,
        "subjects": subject_rows,
        "folds": folds,
        "subject_count": 8,
        "train_subjects_per_fold": 7,
        "knight_in_training": False,
        "teacher_inputs_in_forward": False,
        "semantic_part_taxonomy_used": False,
        "heldout_selection_blind": True,
        "training_started": False,
        "status": "LOFO_INPUTS_SEALED__TRAINING_NOT_STARTED",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "IRIS_V31_LOFO_INPUT_MANIFEST_V1.json"
    atomic_json(path, manifest)
    print(json.dumps({"manifest": str(path), "sha256": file_sha256(path), "folds": len(folds)}, sort_keys=True), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root", "legacy-manifest", "legacy-shard-root", "presentation-bundle",
        "presentation-root", "source-dir", "weight-path", "cache-root", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-batch-size", type=int, default=8)
    args = parser.parse_args()
    prepare(
        input_root=args.input_root,
        legacy_manifest=args.legacy_manifest,
        legacy_shard_root=args.legacy_shard_root,
        presentation_bundle=args.presentation_bundle,
        presentation_root=args.presentation_root,
        source_dir=args.source_dir,
        weight_path=args.weight_path,
        cache_root=args.cache_root,
        output=args.output,
        device=args.device,
        view_batch_size=args.view_batch_size,
    )
