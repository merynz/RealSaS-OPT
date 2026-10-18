from __future__ import annotations

"""Migrate the exact authority-bound completed V8 relation head to V2.

This is artifact migration only: zero optimizer steps, no historical checkpoint
mutation, and no scientific PASS claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v1 import GraphicsRelationHeadV1
from models.iris.vnext.heads_v2 import migrate_graphics_relation_head_v1_to_v2

SCHEMA = "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1"
V8_SOURCE_SHA = "5083c9915533ec8ec6530c7d229c635887abfea7"
V8_DRIVE_FOLDER_ID = "1N6pfo41BaMBEItVJQYKGcjPQSa8WiX38"
V8_CHECKPOINT_DRIVE_FILE_ID = "1xttwuxgC8y9EVmf7xxMEN4yj84yhm4zH"
V8_LATEST_DRIVE_FILE_ID = "1t5Wk4Xj1SGRmqAq3-eo5wR5NTSt-7sij"
V8_COMPLETION_DRIVE_FILE_ID = "1AHDGBtViB8KDGRPpn7uF53OWqM-DBQwS"
V8_START_DRIVE_FILE_ID = "1iQcLM3oKloIxw5YpybavM4ekxOX75rg3"
V8_BINDING_SHA256 = "735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2"
V8_GLOBAL_STEP = 12000
V8_CHECKPOINT_SHA256 = "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
V8_CHECKPOINT_FILE = "step_00012000_ac23a9ea448e465cb668773a131981be.pt"
V8_LATEST_SHA256 = "6238c2a792364beb241dd6ee044778da6280854483303a1502626562515c3a9f"
V8_COMPLETION_SHA256 = "61c207adcbdb9ab398c58acf4ca7afa55eb95aa3a1ef23dc859e7deb8ab4059d"
V8_START_SHA256 = "6bcb92c830ae918978bb60b23ccde528cde477c56bb41ac1ade60ef5065aee55"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_exact(path: Path, expected_sha: str, error: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha:
        raise ValueError(error)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(error + "_JSON_OBJECT_REQUIRED")
    return value


def migrate_v8_relation_checkpoint(
    checkpoint_path: Path,
    latest_pointer_path: Path,
    completion_receipt_path: Path,
    scientific_start_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint_path.resolve()
    latest_path = latest_pointer_path.resolve()
    completion_path = completion_receipt_path.resolve()
    start_path = scientific_start_path.resolve()

    if checkpoint.name != V8_CHECKPOINT_FILE:
        raise ValueError("VNEXT_V8_RELATION_CHECKPOINT_FILENAME_DRIFT")
    if _sha256(checkpoint) != V8_CHECKPOINT_SHA256:
        raise ValueError("VNEXT_V8_RELATION_CHECKPOINT_SHA_DRIFT")

    pointer = _load_json_exact(
        latest_path, V8_LATEST_SHA256, "VNEXT_V8_RELATION_LATEST_POINTER_SHA_DRIFT"
    )
    completion = _load_json_exact(
        completion_path,
        V8_COMPLETION_SHA256,
        "VNEXT_V8_RELATION_COMPLETION_RECEIPT_SHA_DRIFT",
    )
    start = _load_json_exact(
        start_path, V8_START_SHA256, "VNEXT_V8_RELATION_SCIENTIFIC_START_SHA_DRIFT"
    )

    if pointer.get("schema") != "RealSaS.IRISVNextFit8CheckpointPointer.v2":
        raise ValueError("VNEXT_V8_RELATION_LATEST_POINTER_SCHEMA_INVALID")
    if pointer.get("file") != V8_CHECKPOINT_FILE:
        raise ValueError("VNEXT_V8_RELATION_LATEST_POINTER_FILE_DRIFT")
    if pointer.get("sha256") != V8_CHECKPOINT_SHA256:
        raise ValueError("VNEXT_V8_RELATION_LATEST_POINTER_CHECKPOINT_SHA_DRIFT")
    if int(pointer.get("step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("VNEXT_V8_RELATION_LATEST_POINTER_STEP_DRIFT")
    if pointer.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("VNEXT_V8_RELATION_LATEST_POINTER_BINDING_DRIFT")

    if completion.get("schema") != "RealSaS.IRISVNextFit8TrainingComplete.v8":
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_SCHEMA_INVALID")
    if int(completion.get("global_step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_STEP_INVALID")
    if completion.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_BINDING_DRIFT")
    if completion.get("scene_core_sealed_post_m1") is not True:
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_SCENE_CORE_SEAL_MISSING")
    if completion.get("training_complete") is not True:
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_NOT_COMPLETE")
    if completion.get("scientific_fit_pass_claimed") is not False:
        raise ValueError("VNEXT_V8_RELATION_COMPLETION_PASS_OVERCLAIM")

    if start.get("schema") != "RealSaS.IRISVNextFit8ScientificContinuation.v8":
        raise ValueError("VNEXT_V8_RELATION_START_SCHEMA_INVALID")
    if start.get("authority_commit") != V8_SOURCE_SHA:
        raise ValueError("VNEXT_V8_RELATION_AUTHORITY_COMMIT_DRIFT")
    if start.get("trainer_binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("VNEXT_V8_RELATION_START_BINDING_DRIFT")
    if start.get("post_m1_scene_core_sealed") is not True:
        raise ValueError("VNEXT_V8_RELATION_START_SCENE_CORE_SEAL_MISSING")
    if start.get("scientific_fit_pass_claimed") is not False:
        raise ValueError("VNEXT_V8_RELATION_START_PASS_OVERCLAIM")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("VNEXT_V8_RELATION_CHECKPOINT_STEP_INVALID")
    if payload.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("VNEXT_V8_RELATION_CHECKPOINT_BINDING_HASH_DRIFT")
    binding = payload.get("binding")
    if not isinstance(binding, dict) or binding.get("schema") != "RealSaS.IRISVNextFit8Binding.v10":
        raise ValueError("VNEXT_V8_RELATION_CHECKPOINT_BINDING_SCHEMA_INVALID")
    authority = binding.get("post_m1_parameter_authority")
    if not isinstance(authority, dict) or not str(authority.get("scene_core", "")).startswith(
        "SEALED_LR_ZERO"
    ):
        raise ValueError("VNEXT_V8_RELATION_SCENE_CORE_SEAL_MISSING")
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("VNEXT_V8_RELATION_MODEL_STATE_MISSING")

    relation_state = {
        key[len("relations.") :]: tensor
        for key, tensor in model_state.items()
        if isinstance(key, str) and key.startswith("relations.")
    }
    source = GraphicsRelationHeadV1(IRISVNextConfigV1())
    expected_keys = set(source.state_dict())
    if set(relation_state) != expected_keys:
        missing = sorted(expected_keys - set(relation_state))
        extra = sorted(set(relation_state) - expected_keys)
        raise ValueError(
            "VNEXT_V8_RELATION_STATE_KEYS_DRIFT:"
            f"missing={missing}:extra={extra}"
        )
    source.load_state_dict(relation_state, strict=True)
    target, migration = migrate_graphics_relation_head_v1_to_v2(source)

    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "GRAPHICS_RELATION_HEAD_V2_FROM_V8.pt"
    torch.save(
        {
            "schema": "RealSaS.IRISGraphicsRelationHeadV2Artifact.v1",
            "architecture": "GraphicsRelationHeadV2",
            "state_dict": target.state_dict(),
            "source_v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "source_v8_binding_sha256": V8_BINDING_SHA256,
            "source_v8_commit_sha": V8_SOURCE_SHA,
            "source_v8_global_step": V8_GLOBAL_STEP,
            "scene_core_included": False,
            "optimizer_state_included": False,
            "training_steps_performed": 0,
        },
        artifact_path,
    )
    artifact_sha = _sha256(artifact_path)

    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "migration": migration.to_dict(),
        "source_v8": {
            "source_commit_sha": V8_SOURCE_SHA,
            "drive_folder_id": V8_DRIVE_FOLDER_ID,
            "checkpoint_drive_file_id": V8_CHECKPOINT_DRIVE_FILE_ID,
            "latest_pointer_drive_file_id": V8_LATEST_DRIVE_FILE_ID,
            "completion_drive_file_id": V8_COMPLETION_DRIVE_FILE_ID,
            "scientific_start_drive_file_id": V8_START_DRIVE_FILE_ID,
            "binding_sha256": V8_BINDING_SHA256,
            "checkpoint_file": V8_CHECKPOINT_FILE,
            "checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "latest_pointer_sha256": V8_LATEST_SHA256,
            "completion_receipt_sha256": V8_COMPLETION_SHA256,
            "scientific_start_sha256": V8_START_SHA256,
            "global_step": V8_GLOBAL_STEP,
            "scene_core_sealed_post_m1": True,
            "scientific_fit_pass_claimed": False,
        },
        "target": {
            "file": artifact_path.name,
            "sha256": artifact_sha,
            "architecture": "GraphicsRelationHeadV2",
            "new_projected_boundaries_camera_conditioned": True,
            "new_boundary_initial_probability": 0.5,
        },
        "historical_v8_checkpoint_mutated": False,
        "optimizer_state_migrated": False,
        "scene_core_migrated": False,
        "training_steps_performed": 0,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
    }
    receipt_path = out / "V8_RELATION_HEAD_TO_V2_MIGRATION_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--latest-pointer", type=Path, required=True)
    p.add_argument("--completion-receipt", type=Path, required=True)
    p.add_argument("--scientific-start", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = migrate_v8_relation_checkpoint(
        args.checkpoint,
        args.latest_pointer,
        args.completion_receipt,
        args.scientific_start,
        output_dir=args.output_dir,
    )
    print(
        "VNEXT_V8_RELATION_HEAD_TO_V2_MIGRATION_PASS",
        f"artifact={report['target']['sha256']}",
        "training_steps=0",
        "scientific_fit_pass=false",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
