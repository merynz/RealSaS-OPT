from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.platform_observation_truth_cohort_bootstrap_v2 import safe_extract_verified_v2


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _archive(tmp_path: Path, *, manifest_schema: str = "RealSaS.Corpus370PlatformFamilyInputBundle.v2") -> Path:
    manifest = _json(
        {
            "schema": manifest_schema,
            "candidate_subject_count": 370,
            "observation_qualified_count": 294,
            "observation_failed_count": 76,
            "training_authorized": False,
        }
    )
    evidence = b"exact-evidence-bytes\n"
    rows = [
        {"path": "BOOTSTRAP_MANIFEST.json", "sha256": _digest(manifest), "size_bytes": len(manifest)},
        {"path": "evidence.bin", "sha256": _digest(evidence), "size_bytes": len(evidence)},
    ]
    receipt = _json(
        {
            "schema": "RealSaS.Corpus370PlatformFamilyInputBundleReceipt.v2",
            "file_count": len(rows),
            "files": rows,
            "manifest_sha256": _digest(manifest),
            "training_authorized": False,
        }
    )
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("BOOTSTRAP_MANIFEST.json", manifest)
        zipped.writestr("evidence.bin", evidence)
        zipped.writestr("BUNDLE_RECEIPT.json", receipt)
    return archive


def _descriptor(archive: Path) -> dict:
    return {
        "schema": "RealSaS.PlatformObservationTruthCohortBootstrapDescriptor.v2",
        "cohort_bundle": {
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bootstrap_manifest_member": "BOOTSTRAP_MANIFEST.json",
            "bundle_receipt_member": "BUNDLE_RECEIPT.json",
        },
    }


def test_extracts_exact_v2_schema_and_rechecks_every_receipted_byte(tmp_path: Path):
    archive = _archive(tmp_path)
    manifest, receipt, root = safe_extract_verified_v2(archive, _descriptor(archive), tmp_path / "out")

    assert manifest["schema"] == "RealSaS.Corpus370PlatformFamilyInputBundle.v2"
    assert manifest["observation_qualified_count"] == 294
    assert manifest["observation_failed_count"] == 76
    assert receipt["schema"] == "RealSaS.Corpus370PlatformFamilyInputBundleReceipt.v2"
    assert (root / "evidence.bin").read_bytes() == b"exact-evidence-bytes\n"


def test_rejects_historical_or_wrong_manifest_schema(tmp_path: Path):
    archive = _archive(tmp_path, manifest_schema="RealSaS.SubjectCohortPlatformStage01InputBundle.v1")
    with pytest.raises(RuntimeError, match="OBS_TRUTH_V2_BUNDLE_SCHEMA"):
        safe_extract_verified_v2(archive, _descriptor(archive), tmp_path / "out")
