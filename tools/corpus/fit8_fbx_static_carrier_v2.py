from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.corpus.build_ufbx_static_census_probe_v3 import validate_build_receipt_v1
from tools.corpus.fit8_fbx_static_carrier_v1 import (
    qualify_ufbx_probe_payload_v1,
    run_ufbx_static_carrier_v1,
)
from tools.corpus.fit8_f_qualified_fa_bundle_v1 import write_deterministic_npz_v1

SCHEMA_VERSION = "RealSaS.FIT8FBXStaticCarrier.v2"
PROBE_SOURCE = Path(__file__).with_name("ufbx_static_census_probe_v3.c")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_ufbx_static_carrier_v2(
    *,
    source_path: Path,
    probe_executable: Path,
    build_receipt_path: Path,
    timeout_seconds: int = 120,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    build_receipt_path = build_receipt_path.resolve()
    if not build_receipt_path.is_file():
        raise ValueError("FIT8_FBX_CARRIER_V2_BUILD_RECEIPT_MISSING")
    try:
        receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("FIT8_FBX_CARRIER_V2_BUILD_RECEIPT_JSON_INVALID") from exc
    validate_build_receipt_v1(
        receipt,
        expected_probe_source_sha256=_sha256_file(PROBE_SOURCE),
        binary_path=probe_executable,
    )
    manifest, arrays = run_ufbx_static_carrier_v1(
        source_path=source_path,
        probe_executable=probe_executable,
        build_receipt_sha256=_sha256_file(build_receipt_path),
        build_receipt_verifies_expected_ufbx_revision=True,
        timeout_seconds=timeout_seconds,
    )
    manifest = dict(manifest)
    manifest["schema"] = SCHEMA_VERSION
    manifest["build_receipt_identity_sha256"] = receipt["receipt_sha256"]
    manifest["build_receipt_file_sha256"] = _sha256_file(build_receipt_path)
    manifest["ufbx_c_sha256"] = receipt["ufbx_c_sha256"]
    manifest["ufbx_h_sha256"] = receipt["ufbx_h_sha256"]
    manifest["probe_source_sha256"] = receipt["probe_source_sha256"]
    manifest["compiler_sha256"] = receipt["compiler_sha256"]
    manifest["claims"] = dict(manifest["claims"])
    manifest["claims"].update({
        "ufbx_revision_verified": True,
        "probe_binary_bound_to_receipt": True,
        "probe_source_bound_to_repository_v3": True,
    })
    return manifest, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--probe-executable", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    manifest, arrays = run_ufbx_static_carrier_v2(
        source_path=args.source,
        probe_executable=args.probe_executable,
        build_receipt_path=args.build_receipt,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest["carrier_npz_sha256"] = write_deterministic_npz_v1(args.output_npz, arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


__all__ = ["SCHEMA_VERSION", "run_ufbx_static_carrier_v2"]

if __name__ == "__main__":
    main()
