#!/usr/bin/env python3
from __future__ import annotations

"""Hydrate exact read-only inputs for the direct IRIS surface baseline.

Only immutable Drive file IDs from the reviewed binding are read. The tool has no
remote write operation and emits a local byte-verification receipt. It does not
instantiate a model, invoke TESSA, Compiler, trainer or optimizer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
    V8_SOURCE_COMMIT,
)


BINDING_SCHEMA = "RealSaS.IRISVNextDirectSurfaceInputBinding.v1"
RECEIPT_SCHEMA = "RealSaS.IRISVNextDirectSurfaceHydrationReceipt.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_SURFACE_HYDRATION_JSON_OBJECT_REQUIRED")
    return value


def require_sha64(value: Any, code: str) -> str:
    out = str(value or "").lower()
    if len(out) != 64 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError(code)
    return out


def validate_binding(binding: dict[str, Any]) -> None:
    if binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("DIRECT_SURFACE_INPUT_BINDING_SCHEMA_INVALID")
    if binding.get("training_allowed") is not False:
        raise ValueError("DIRECT_SURFACE_INPUT_BINDING_TRAINING_MUST_BE_FALSE")
    if binding.get("external_storage_write_allowed") is not False:
        raise ValueError("DIRECT_SURFACE_INPUT_BINDING_REMOTE_WRITE_MUST_BE_FALSE")
    if binding.get("scientific_pass_claimed") is not False:
        raise ValueError("DIRECT_SURFACE_INPUT_BINDING_SCIENTIFIC_CLAIM_INVALID")

    stage01 = binding.get("stage01") or {}
    if stage01.get("file") != "FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.zip" or not stage01.get("file_id"):
        raise ValueError("DIRECT_SURFACE_STAGE01_BINDING_INVALID")
    require_sha64(stage01.get("sha256"), "DIRECT_SURFACE_STAGE01_SHA_INVALID")

    prepared = binding.get("prepared") or {}
    if prepared.get("source_commit_sha") != "7029ee3f48787aa9f7e27eecef89155eb9368a96":
        raise ValueError("DIRECT_SURFACE_PREPARED_SOURCE_DRIFT")
    manifest = prepared.get("manifest") or {}
    require_sha64(manifest.get("sha256"), "DIRECT_SURFACE_MANIFEST_SHA_INVALID")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4" or not manifest.get("file_id"):
        raise ValueError("DIRECT_SURFACE_MANIFEST_BINDING_INVALID")
    prep_receipt = prepared.get("preparation_receipt") or {}
    require_sha64(prep_receipt.get("sha256"), "DIRECT_SURFACE_PREPARATION_RECEIPT_SHA_INVALID")
    if prep_receipt.get("schema") != "RealSaS.IRISVNextFit8PreparationReceipt.v2" or not prep_receipt.get("file_id"):
        raise ValueError("DIRECT_SURFACE_PREPARATION_RECEIPT_BINDING_INVALID")

    subjects = list(prepared.get("subjects") or ())
    if tuple(row.get("subject_id") for row in subjects) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_INPUT_SUBJECT_ORDER_DRIFT")
    if len({str(row.get("file")) for row in subjects}) != len(SUBJECT_IDS):
        raise ValueError("DIRECT_SURFACE_INPUT_SHARD_FILE_DUPLICATE")
    for row in subjects:
        if not row.get("file_id"):
            raise ValueError("DIRECT_SURFACE_INPUT_SHARD_FILE_ID_MISSING")
        require_sha64(row.get("sha256"), "DIRECT_SURFACE_INPUT_SHARD_SHA_INVALID")

    v8 = binding.get("v8") or {}
    if v8.get("source_commit_sha") != V8_SOURCE_COMMIT:
        raise ValueError("DIRECT_SURFACE_V8_SOURCE_DRIFT")
    if int(v8.get("global_step", -1)) != V8_GLOBAL_STEP:
        raise ValueError("DIRECT_SURFACE_V8_STEP_DRIFT")
    if v8.get("binding_sha256") != V8_BINDING_SHA256:
        raise ValueError("DIRECT_SURFACE_V8_BINDING_DRIFT")
    checkpoint = v8.get("checkpoint") or {}
    if checkpoint.get("sha256") != V8_CHECKPOINT_SHA256 or not checkpoint.get("file_id"):
        raise ValueError("DIRECT_SURFACE_V8_CHECKPOINT_BINDING_INVALID")

    surface = binding.get("surface_policy") or {}
    if int(surface.get("base_cells", -1)) != 512 or int(surface.get("fine_cells", -1)) != 1024:
        raise ValueError("DIRECT_SURFACE_INPUT_FULL_RESOLUTION_POLICY_REQUIRED")
    if surface.get("numeric_policy") != "FP32_NO_AUTOCAST" or surface.get("tessa_invoked") is not False:
        raise ValueError("DIRECT_SURFACE_INPUT_SURFACE_POLICY_DRIFT")


def copyid(file_id: str, destination: Path) -> None:
    if not file_id or "/" in file_id or " " in file_id:
        raise ValueError("DIRECT_SURFACE_HYDRATION_FILE_ID_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "backend", "copyid", "gdrive:", file_id, str(destination)], check=True)


def hydrate(binding_path: Path, output_root: Path) -> dict[str, Any]:
    binding_path = binding_path.resolve()
    binding = load_json(binding_path)
    validate_binding(binding)
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("DIRECT_SURFACE_HYDRATION_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    stage01_out = output_root / "stage01"
    prepared_out = output_root / "prepared"
    v8_out = output_root / "v8"
    hydrated: list[dict[str, Any]] = []

    def hydrate_one(spec: dict[str, Any], destination: Path, role: str) -> None:
        copyid(str(spec["file_id"]), destination)
        expected = str(spec["sha256"]).lower()
        actual = sha256_file(destination)
        if actual != expected:
            raise RuntimeError(f"DIRECT_SURFACE_HYDRATION_SHA_DRIFT:{role}:{expected}:{actual}")
        hydrated.append({"role": role, "file": destination.relative_to(output_root).as_posix(), "file_id": str(spec["file_id"]), "sha256": actual, "bytes": destination.stat().st_size})

    stage01 = binding["stage01"]
    hydrate_one(stage01, stage01_out / str(stage01["file"]), "STAGE01_BUNDLE")

    prepared = binding["prepared"]
    hydrate_one(prepared["manifest"], prepared_out / str(prepared["manifest"]["file"]), "PREPARED_MANIFEST")
    hydrate_one(prepared["preparation_receipt"], prepared_out / str(prepared["preparation_receipt"]["file"]), "PREPARATION_RECEIPT")
    for row in prepared["subjects"]:
        hydrate_one(row, prepared_out / str(row["file"]), "PREPARED_SHARD:" + str(row["subject_id"]))
    hydrate_one(binding["v8"]["checkpoint"], v8_out / str(binding["v8"]["checkpoint"]["file"]), "V8_CHECKPOINT")

    manifest = load_json(prepared_out / str(prepared["manifest"]["file"]))
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_MANIFEST_SCHEMA_INVALID")
    manifest_rows = list(manifest.get("subjects") or ())
    expected_rows = list(prepared["subjects"])
    if tuple(row.get("subject_id") for row in manifest_rows) != SUBJECT_IDS:
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_MANIFEST_SUBJECT_ORDER_DRIFT")
    for manifest_row, bound_row in zip(manifest_rows, expected_rows, strict=True):
        if manifest_row.get("subject_id") != bound_row.get("subject_id") or manifest_row.get("shard_file") != bound_row.get("file") or manifest_row.get("shard_sha256") != bound_row.get("sha256"):
            raise RuntimeError("DIRECT_SURFACE_HYDRATED_MANIFEST_BINDING_DRIFT")

    prep_receipt = load_json(prepared_out / str(prepared["preparation_receipt"]["file"]))
    if prep_receipt.get("source_commit_bound") is not True:
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_PREPARATION_SOURCE_UNBOUND")
    if prep_receipt.get("source_commit_sha") != prepared.get("source_commit_sha"):
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_PREPARATION_SOURCE_DRIFT")
    if prep_receipt.get("cohort_manifest_sha256") != prepared["manifest"]["sha256"]:
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_PREPARATION_MANIFEST_DRIFT")
    stage_receipt = ((prep_receipt.get("verified_inputs") or {}).get("stage01") or {})
    if stage_receipt.get("stage01_bundle_sha256") != stage01["sha256"]:
        raise RuntimeError("DIRECT_SURFACE_HYDRATED_STAGE01_PREPARATION_DRIFT")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS__EXACT_INPUT_BYTES_HYDRATED__NO_TRAINING__NO_REMOTE_WRITE",
        "input_binding_sha256": sha256_file(binding_path),
        "files": hydrated,
        "stage01_bundle_sha256": stage01["sha256"],
        "prepared_source_commit_sha": prepared["source_commit_sha"],
        "v8_source_commit_sha": V8_SOURCE_COMMIT,
        "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
        "v8_binding_sha256": V8_BINDING_SHA256,
        "v8_global_step": V8_GLOBAL_STEP,
        "training_allowed": False,
        "remote_write_performed": False,
        "scientific_pass_claimed": False,
        "next_action": "A100_DIRECT_SURFACE_ENGINEERING_PREFLIGHT_THEN_FULL_T512_T1024_MATERIALIZATION_AND_SOURCE_FIDELITY_COURTS",
    }
    (output_root / "DIRECT_SURFACE_HYDRATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", default="canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json")
    p.add_argument("--output-root", required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    result = hydrate(Path(args.binding), Path(args.output_root))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
