import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tools.platform_cohort_bootstrap_v1 import (
    derive_observation_binding,
    safe_extract_verified,
    verify_descriptor,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_bootstrap_executor_has_no_fit8_subject_branching():
    root = Path(__file__).resolve().parents[2]
    source = (root / "tools/platform_cohort_bootstrap_v1.py").read_text(encoding="utf-8").lower()
    for token in ("snake", "bat", "trex", "horse", "spider", "greenblob", "shark", "george", "knight"):
        assert token not in source


def test_descriptor_contract_requires_generic_research_rules():
    descriptor = {
        "schema": "RealSaS.PlatformCohortBootstrapDescriptor.v1",
        "purpose": "RESEARCH",
        "source_license_defaults": {"license_name": "X", "license_ref": "urn:test"},
        "rules": {
            "descriptor_is_data_not_subject_specific_code": True,
            "subject_specific_code_forbidden": True,
            "all_bundle_members_verified_against_bundle_receipt": True,
            "observation_binding_must_match_shared_cohort_manifest": True,
        },
    }
    verify_descriptor(descriptor, {"canonical_branch": "main"})
    bad = json.loads(json.dumps(descriptor))
    bad["rules"]["subject_specific_code_forbidden"] = False
    with pytest.raises(RuntimeError, match="COHORT_BOOTSTRAP_RULE_REQUIRED"):
        verify_descriptor(bad, {"canonical_branch": "main"})


def test_bundle_receipt_is_exact_and_path_safe(tmp_path):
    manifest = {"schema": "RealSaS.TestPlatformStage01InputBundle.v1", "subjects": []}
    manifest_raw = json.dumps(manifest).encode()
    member = b"evidence"
    receipt = {
        "schema": "RealSaS.TestPlatformStage01InputBundleReceipt.v1",
        "file_count": 2,
        "manifest_sha256": _sha(manifest_raw),
        "files": [
            {"path": "BOOTSTRAP_MANIFEST.json", "sha256": _sha(manifest_raw), "size_bytes": len(manifest_raw)},
            {"path": "evidence.bin", "sha256": _sha(member), "size_bytes": len(member)},
        ],
    }
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("BOOTSTRAP_MANIFEST.json", manifest_raw)
        z.writestr("evidence.bin", member)
        z.writestr("BUNDLE_RECEIPT.json", json.dumps(receipt))
    descriptor = {"cohort_bundle": {
        "sha256": _sha(archive.read_bytes()),
        "bootstrap_manifest_member": "BOOTSTRAP_MANIFEST.json",
        "bundle_receipt_member": "BUNDLE_RECEIPT.json",
    }}
    got, got_receipt, root = safe_extract_verified(archive, descriptor, tmp_path / "out")
    assert got == manifest
    assert got_receipt["file_count"] == 2
    assert (root / "evidence.bin").read_bytes() == member


def test_observation_binding_matches_canonical_hash_shape():
    proof = {"source_sha256": "a" * 64, "camera_sha256": "b" * 64,
             "views": [{"view_index": 0, "raster_sha256": "c" * 64}]}
    expected = hashlib.sha256(json.dumps({
        "proof_sha256": "d" * 64,
        "source_sha256": proof["source_sha256"],
        "camera_sha256": proof["camera_sha256"],
        "views": proof["views"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert derive_observation_binding(proof, "d" * 64) == expected
