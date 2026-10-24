#!/usr/bin/env python3
from __future__ import annotations

"""Materialize SourceObservation.v1 from an exact hash-bound run manifest.

The command writes canonical source-observation bytes and a diagnostic receipt.
It does not import/register the bytes, run learned inference, qualify a stage, or
mint product authority.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from compiler.realsas_compiler_services.platform_worker.source_observation_materializer_v1 import (
    materialize_source_observation_from_manifest_v1,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_bytes(path, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    manifest_raw = args.run_manifest.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("SOURCE_OBSERVATION_MATERIALIZE_MANIFEST_OBJECT_REQUIRED")
    result = materialize_source_observation_from_manifest_v1(manifest)
    _atomic_bytes(args.out, result.payload)
    receipt = {
        "schema": "RealSaS.SourceObservationMaterializationReceipt.v1",
        "run_manifest_path": str(args.run_manifest.resolve()),
        "run_manifest_sha256": _sha256(manifest_raw),
        "output_path": str(args.out.resolve()),
        **result.diagnostics,
        "artifact_import_performed": False,
        "registry_mutation_performed": False,
        "model_inference_executed": False,
        "qualification_minted": False,
        "product_authority_minted": False,
    }
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
