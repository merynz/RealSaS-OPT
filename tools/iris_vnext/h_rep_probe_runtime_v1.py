from __future__ import annotations

"""Shared runtime helpers for H-REP-P0 training and independent court."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.surface_rescue_training_v1 import module_state_sha256_v1
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    load_exact_v8_model,
    load_json,
    load_subject_source,
    sha256_file,
)

DATASET_SCHEMA = "RealSaS.HRepP0ProbeDataset.v1"


def load_h_rep_binding_v1(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HRepP0InputBinding.v1":
        raise ValueError("H_REP_P0_BINDING_SCHEMA_INVALID")
    return value


def _load_split_arrays(dataset_path: Path, split: str) -> dict[str, torch.Tensor]:
    if split not in ("train", "holdout"):
        raise ValueError("H_REP_P0_SPLIT_INVALID")
    with np.load(dataset_path, allow_pickle=False) as data:
        keys = (
            "xyz_iris",
            "nearest_reference_iris",
            "distance_iris",
            "offset_to_reference_iris",
            "normal_unit",
            "normal_valid",
            "near_label",
        )
        out: dict[str, torch.Tensor] = {}
        for key in keys:
            name = f"{split}_{key}"
            if name not in data:
                raise ValueError("H_REP_P0_DATASET_FIELD_MISSING:" + name)
            array = np.asarray(data[name])
            if key in ("normal_valid", "near_label"):
                out[key] = torch.from_numpy(array.astype(np.bool_))
            else:
                out[key] = torch.from_numpy(array.astype(np.float32))
    n = int(out["xyz_iris"].shape[0])
    if (
        out["xyz_iris"].shape != (n, 3)
        or out["nearest_reference_iris"].shape != (n, 3)
        or out["offset_to_reference_iris"].shape != (n, 3)
        or out["normal_unit"].shape != (n, 3)
    ):
        raise ValueError("H_REP_P0_DATASET_VECTOR_SHAPE_INVALID")
    if (
        out["distance_iris"].shape != (n,)
        or out["normal_valid"].shape != (n,)
        or out["near_label"].shape != (n,)
    ):
        raise ValueError("H_REP_P0_DATASET_SCALAR_SHAPE_INVALID")
    if n <= 0:
        raise ValueError("H_REP_P0_DATASET_EMPTY")
    for key, value in out.items():
        if value.dtype != torch.bool and not bool(torch.isfinite(value).all().item()):
            raise ValueError("H_REP_P0_DATASET_NONFINITE:" + key)
    return out


def cache_frozen_v8_probe_features_v1(
    *,
    v8_checkpoint: Path,
    prepared_root: Path,
    prepared_manifest: Path,
    dataset_root: Path,
    dataset_receipt: Path,
    binding: dict[str, Any],
    binding_sha256: str,
    split: str,
    device: torch.device,
    query_chunk: int = 8192,
) -> tuple[dict[str, dict[str, torch.Tensor]], str]:
    prepared_root = prepared_root.resolve()
    dataset_root = dataset_root.resolve()
    prepared_manifest = prepared_manifest.resolve()
    dataset_receipt = dataset_receipt.resolve()
    if int(query_chunk) <= 0:
        raise ValueError("H_REP_P0_QUERY_CHUNK_INVALID")

    manifest = load_json(prepared_manifest)
    rows = list(manifest.get("subjects") or ())
    if (
        manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4"
        or tuple(r.get("subject_id") for r in rows) != SUBJECT_IDS
    ):
        raise ValueError("H_REP_P0_MANIFEST_INVALID")
    manifest_sha = sha256_file(prepared_manifest)
    if manifest_sha != binding["prepared"]["manifest"]["sha256"]:
        raise ValueError("H_REP_P0_MANIFEST_SHA_DRIFT")

    receipt = load_json(dataset_receipt)
    if (
        receipt.get("schema") != DATASET_SCHEMA
        or receipt.get("status") != "MATERIALIZED_H_REP_P0_POINT_PROXY_DATASET"
        or tuple(receipt.get("subject_order") or ()) != SUBJECT_IDS
    ):
        raise ValueError("H_REP_P0_DATASET_RECEIPT_INVALID")
    if receipt.get("input_binding_sha256") != str(binding_sha256).lower():
        raise ValueError("H_REP_P0_DATASET_BINDING_SHA_DRIFT")
    if receipt.get("prepared_manifest_sha256") != manifest_sha:
        raise ValueError("H_REP_P0_DATASET_MANIFEST_SHA_DRIFT")
    if receipt.get("teacher_truth_in_probe_input") is not False:
        raise ValueError("H_REP_P0_DATASET_INPUT_CONTAMINATION")
    if receipt.get("mesh_extraction_invoked") is not False or receipt.get("tessa_invoked") is not False:
        raise ValueError("H_REP_P0_DATASET_SCOPE_CONTAMINATION")

    receipt_rows_list = list(receipt.get("subjects") or ())
    if tuple(r.get("subject_id") for r in receipt_rows_list) != SUBJECT_IDS:
        raise ValueError("H_REP_P0_DATASET_ROW_ORDER_DRIFT")
    receipt_rows = {str(r["subject_id"]): r for r in receipt_rows_list}
    prepared_rows = {str(r["subject_id"]): r for r in rows}

    v8 = load_exact_v8_model(v8_checkpoint.resolve(), device)
    scene_hash = module_state_sha256_v1(v8.scene_core)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for subject_index, subject_id in enumerate(SUBJECT_IDS):
            row = prepared_rows[subject_id]
            shard = (prepared_root / str(row["shard_file"])).resolve()
            if not shard.is_relative_to(prepared_root):
                raise ValueError("H_REP_P0_PREPARED_PATH_ESCAPE:" + subject_id)
            source = load_subject_source(shard, str(row["shard_sha256"]), device)
            planes = v8.encode_scene(
                source["image_tokens"], source["token_xy"], source["camera_features"]
            ).float().detach()

            ds_row = receipt_rows[subject_id]
            if str(ds_row.get("prepared_shard_sha256")) != str(row["shard_sha256"]):
                raise ValueError("H_REP_P0_DATASET_SHARD_BINDING_DRIFT:" + subject_id)
            dataset_path = (dataset_root / str(ds_row["payload_file"])).resolve()
            if not dataset_path.is_relative_to(dataset_root):
                raise ValueError("H_REP_P0_DATASET_PATH_ESCAPE:" + subject_id)
            if sha256_file(dataset_path) != str(ds_row["payload_sha256"]):
                raise ValueError("H_REP_P0_DATASET_SHA_DRIFT:" + subject_id)

            arrays = _load_split_arrays(dataset_path, split)
            xyz = arrays["xyz_iris"]
            h_rows = []
            field_rows = []
            for start in range(0, len(xyz), int(query_chunk)):
                q = xyz[start : start + int(query_chunk)].to(
                    device=device, dtype=torch.float32
                )[None]
                local = v8.scene_core.query(planes, q)
                h_rows.append(local["H"].squeeze(0).detach().cpu().float())
                field_rows.append(local["field"].squeeze(0).detach().cpu().float())
            arrays["H"] = torch.cat(h_rows, dim=0)
            arrays["raw_field"] = torch.cat(field_rows, dim=0)
            if arrays["H"].shape != (len(xyz), 384) or arrays["raw_field"].shape != (len(xyz),):
                raise RuntimeError("H_REP_P0_FROZEN_FEATURE_SHAPE_DRIFT:" + subject_id)
            if not bool(torch.isfinite(arrays["H"]).all().item()) or not bool(
                torch.isfinite(arrays["raw_field"]).all().item()
            ):
                raise RuntimeError("H_REP_P0_FROZEN_FEATURE_NONFINITE:" + subject_id)

            g = torch.Generator(device="cpu")
            g.manual_seed(26103980 + subject_index + (0 if split == "train" else 1000))
            arrays["shuffle_index"] = torch.randperm(len(xyz), generator=g)
            cache[subject_id] = arrays
            print(
                f"H_REP_P0_CACHE split={split} subject={subject_id} rows={len(xyz)}",
                flush=True,
            )

    if module_state_sha256_v1(v8.scene_core) != scene_hash:
        raise RuntimeError("H_REP_P0_SCENE_CORE_HASH_DRIFT")
    return cache, scene_hash
