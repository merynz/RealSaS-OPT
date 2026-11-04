"""Stage50 bootstrap uses exact V3 transport bytes, not regenerated imports.

The sealed 370-source universe remains whole; only observation PASS rows enter
the observation-dependent research cohort. These are input bindings, NOT truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.corpus.prepare_corpus370_platform_truth_bundle_v3 import prepare
from tools.platform_cohort_bootstrap_v1 import receipt_index
from tools.platform_source_surface_cohort_bootstrap_v3 import (
    safe_extract_verified_v3, subject_sections,
)
from test_corpus370_platform_family_input_bundle_v3 import fixture, sha


def transport(tmp_path):
    roster, source_manifest, extracted, observations, provenance, packet = fixture(tmp_path)
    manifest, receipt, archive = prepare(
        roster, source_manifest, extracted, observations, provenance,
        tmp_path/"bundle-v3.zip",
    )
    descriptor = {"cohort_bundle": {
        "sha256": archive["sha256"],
        "bootstrap_manifest_member": "BOOTSTRAP_MANIFEST.json",
        "bundle_receipt_member": "BUNDLE_RECEIPT.json",
    }}
    returned_manifest, returned_receipt, bundle_root = safe_extract_verified_v3(
        Path(archive["path"]), descriptor, tmp_path/"extracted",
    )
    assert returned_manifest == manifest
    assert returned_receipt == receipt
    return manifest, receipt_index(receipt), bundle_root, packet


def test_stage50_uses_exact_v4_import_bytes_without_calibration_dependency(tmp_path):
    manifest, index, root, original = transport(tmp_path)
    subject = next(r for r in manifest["subjects"] if r["subject_id"] == "subject_000")
    sections = subject_sections(
        subject, root, index,
        {"truth_foundry": {"scope": "SYNTHETIC_NO_TARGETS"}},
    )
    v = sections["source_surface"]
    assert v["mode"] == "EXACT_ADMITTED_IMPORT_SURFACE_ONLY"
    assert Path(v["admitted_import_geometry"]["path"]).read_bytes() == (
        original/"imported.json"
    ).read_bytes()
    assert v["admitted_import_geometry"]["sha256"] == sha(original/"imported.json")
    assert sections["source_audit"]["iris_observation_admission"]["sha256"] == sha(
        original/"IRIS_OBSERVATION_ADMISSION.json"
    )
    assert len(sections["observation"]["source_rasters"]) == 8
    assert len(sections["observation"]["source_foreground_masks"]) == 8
    assert sections["source_audit"]["iris_observation_admission_required"] is True
    assert "truth_foundry" in sections


def test_non_observation_subject_cannot_enter_stage50(tmp_path):
    manifest, index, root, packet = transport(tmp_path)
    failed = next(r for r in manifest["subjects"] if r["subject_id"] == "subject_001")
    with pytest.raises((ValueError,RuntimeError), match="SUBJECT_NOT_QUALIFIED"):
        subject_sections(failed, root, index, {"truth_foundry": {}})


def test_v3_stage50_plan_can_be_validated_without_training_or_product_authority():
    from compiler.realsas_compiler_services.orchestrator import mainline
    from tools.platform_truth_cohort_bootstrap_v1 import plan_manifest_keys
    plan = json.loads(Path("canonical/CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V3.json").read_text())
    campaign = json.loads(Path("canonical/CORPUS370_SOURCE_SURFACE_TRUTH_CAMPAIGN_V3.json").read_text())
    mainline.validate_plan(plan, require_product_pass_authority=False)
    assert campaign["compiler_plan"].endswith("_V3.json")
    assert campaign["required_stage_count"] == len(plan["stages"]) == 11
    assert campaign["required_target_stage_id"] == plan["stages"][-1]["id"]
    assert "source_surface" in plan_manifest_keys(plan)
    assert "iris_fit" not in plan_manifest_keys(plan)
