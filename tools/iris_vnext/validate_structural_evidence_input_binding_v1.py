from __future__ import annotations

"""Fail-closed admission validator for STRUCTURAL-EVIDENCE input materialization.

The static binding can be validated in source CI without touching Drive or running
scientific work.  When a zero-optimizer materialization is later executed, the
same validator can additionally verify its hashed output directory.

This tool does not train, score a scientific hypothesis, or authorize a court.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "RealSaS.StructuralEvidenceInputBinding.v1"
OUTPUT_SCHEMA = "RealSaS.IRISVNextFit8AllHitRelationEvidenceManifest.v1"
OUTPUT_STATUS = "MATERIALIZED__EVIDENCE_ONLY__NO_TRAINING_AUTHORIZATION"
SUBJECTS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _require_sha256(value: Any, code: str) -> None:
    _require(isinstance(value, str) and bool(SHA256_RE.fullmatch(value)), code)


def _subject_ids(rows: Any, code: str) -> tuple[str, ...]:
    _require(isinstance(rows, list) and len(rows) == len(SUBJECTS), code + "_COUNT")
    ids = tuple(str(row.get("subject_id")) for row in rows if isinstance(row, dict))
    _require(ids == SUBJECTS, code + "_ORDER")
    return ids


def validate_binding(value: dict[str, Any]) -> dict[str, Any]:
    _require(value.get("schema") == SCHEMA, "STRUCTURAL_EVIDENCE_BINDING_SCHEMA_DRIFT")
    _require(value.get("experiment_id") == "STRUCTURAL-EVIDENCE-INPUT-AUTHORITY", "STRUCTURAL_EVIDENCE_BINDING_EXPERIMENT_DRIFT")
    _require(value.get("scientific_execution_authorized") is False, "STRUCTURAL_EVIDENCE_BINDING_SCIENTIFIC_EXECUTION_AUTHORIZED")
    _require(value.get("optimizer_or_training_authorized") is False, "STRUCTURAL_EVIDENCE_BINDING_OPTIMIZER_AUTHORIZED")
    _require(value.get("notebook_execution_authorized") is False, "STRUCTURAL_EVIDENCE_BINDING_NOTEBOOK_AUTHORIZED")

    status = value.get("status")
    _require(status in {
        "PENDING_SOURCE_FREEZE__MATERIALIZATION_NOT_RUN",
        "READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS",
    }, "STRUCTURAL_EVIDENCE_BINDING_STATUS_INVALID")
    implementation_sha = value.get("implementation_source_sha")
    if status.startswith("READY_"):
        _require(isinstance(implementation_sha, str) and bool(GIT_SHA_RE.fullmatch(implementation_sha)), "STRUCTURAL_EVIDENCE_BINDING_SOURCE_SHA_MISSING")
    else:
        _require(implementation_sha is None, "STRUCTURAL_EVIDENCE_BINDING_PENDING_SOURCE_SHA_MUST_BE_NULL")

    producer = value.get("producer", {})
    _require(producer.get("path") == "tools/iris_vnext/build_fit8_relation_all_hit_evidence_v1.py", "STRUCTURAL_EVIDENCE_BINDING_PRODUCER_DRIFT")
    _require(producer.get("candidate_rays_per_view") == 8192, "STRUCTURAL_EVIDENCE_BINDING_RAY_POLICY_DRIFT")
    _require(producer.get("grid_size") == 32, "STRUCTURAL_EVIDENCE_BINDING_GRID_POLICY_DRIFT")
    _require(producer.get("output_manifest") == "ALL_HIT_RELATION_EVIDENCE_V1_MANIFEST.json", "STRUCTURAL_EVIDENCE_BINDING_MANIFEST_NAME_DRIFT")

    inputs = value.get("source_inputs", {})
    stage01 = inputs.get("stage01", {})
    _require(stage01.get("file") == "FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.zip", "STRUCTURAL_EVIDENCE_BINDING_STAGE01_FILE_DRIFT")
    _require_sha256(stage01.get("sha256"), "STRUCTURAL_EVIDENCE_BINDING_STAGE01_SHA_INVALID")
    _require_sha256(stage01.get("bootstrap_sha256"), "STRUCTURAL_EVIDENCE_BINDING_STAGE01_BOOTSTRAP_SHA_INVALID")

    legacy = inputs.get("legacy_fit8", {})
    _require(legacy.get("binding_path") == "canonical/VNEXT_FIT8_INPUT_BINDING_V1.json", "STRUCTURAL_EVIDENCE_BINDING_LEGACY_BINDING_DRIFT")
    _require_sha256(legacy.get("binding_sha256"), "STRUCTURAL_EVIDENCE_BINDING_LEGACY_BINDING_SHA_INVALID")
    _require_sha256(legacy.get("cohort_manifest_sha256"), "STRUCTURAL_EVIDENCE_BINDING_LEGACY_MANIFEST_SHA_INVALID")
    _subject_ids(legacy.get("subjects"), "STRUCTURAL_EVIDENCE_BINDING_LEGACY_SUBJECT")
    for row in legacy["subjects"]:
        _require_sha256(row.get("sha256"), "STRUCTURAL_EVIDENCE_BINDING_LEGACY_SHARD_SHA_INVALID")
        _require(str(row.get("shard_file", "")).endswith(".pt"), "STRUCTURAL_EVIDENCE_BINDING_LEGACY_SHARD_FILE_INVALID")

    teachers = inputs.get("authored_teachers", {})
    _require(bool(teachers.get("folder_id")), "STRUCTURAL_EVIDENCE_BINDING_TEACHER_FOLDER_MISSING")
    _require(teachers.get("batch_receipt_file") == "TEACHER_BATCH_RECEIPT.json", "STRUCTURAL_EVIDENCE_BINDING_TEACHER_RECEIPT_NAME_DRIFT")
    _require_sha256(teachers.get("batch_receipt_sha256"), "STRUCTURAL_EVIDENCE_BINDING_TEACHER_BATCH_SHA_INVALID")
    _subject_ids(teachers.get("subjects"), "STRUCTURAL_EVIDENCE_BINDING_TEACHER_SUBJECT")
    for row in teachers["subjects"]:
        _require_sha256(row.get("teacher_sha256"), "STRUCTURAL_EVIDENCE_BINDING_TEACHER_SHA_INVALID")
        _require_sha256(row.get("receipt_sha256"), "STRUCTURAL_EVIDENCE_BINDING_TEACHER_RECEIPT_SHA_INVALID")
        _require(int(row.get("sampled_frame_count", 0)) > 0, "STRUCTURAL_EVIDENCE_BINDING_TEACHER_FRAME_COUNT_INVALID")

    prepared = inputs.get("prepared_fit8_receipt", {})
    _require(bool(prepared.get("folder_id")), "STRUCTURAL_EVIDENCE_BINDING_PREPARED_FOLDER_MISSING")
    _require_sha256(prepared.get("preparation_receipt_sha256"), "STRUCTURAL_EVIDENCE_BINDING_PREPARATION_RECEIPT_SHA_INVALID")
    _require_sha256(prepared.get("cohort_manifest_sha256"), "STRUCTURAL_EVIDENCE_BINDING_PREPARED_MANIFEST_SHA_INVALID")

    law = value.get("authority_law", {})
    _require(law.get("front") == "PAIRWISE_DEPTH_ORDER_FROM_COMPLETE_ORDERED_RAY_STACK", "STRUCTURAL_EVIDENCE_BINDING_FRONT_AUTHORITY_DRIFT")
    _require(law.get("global_visibility") == "RANK_ZERO_ONLY_FROM_COMPLETE_ORDERED_RAY_STACK", "STRUCTURAL_EVIDENCE_BINDING_VISIBILITY_AUTHORITY_DRIFT")
    _require(law.get("incomplete_stack_visibility_policy") == "ABSTAIN", "STRUCTURAL_EVIDENCE_BINDING_ABSTAIN_POLICY_DRIFT")
    _require(law.get("occlusion") == "POSITIVE_SAME_RAY_OVERLAP_SUPPORT_ONLY__NO_FABRICATED_NEGATIVES", "STRUCTURAL_EVIDENCE_BINDING_OCCLUSION_AUTHORITY_DRIFT")
    _require(law.get("projected_boundary") == "SEPARATE_CURRENT_VIEW_RGBA_ALPHA_AUTHORITY", "STRUCTURAL_EVIDENCE_BINDING_BOUNDARY_AUTHORITY_DRIFT")
    _require(law.get("material_id_as_owner_truth") is False, "STRUCTURAL_EVIDENCE_BINDING_MATERIAL_OWNER_SHORTCUT")
    _require(law.get("named_semantic_taxonomy_as_structural_authority") is False, "STRUCTURAL_EVIDENCE_BINDING_SEMANTIC_SHORTCUT")
    _require(law.get("teacher_topology_or_rig_in_inference_forward") is False, "STRUCTURAL_EVIDENCE_BINDING_TEACHER_INFERENCE_SHORTCUT")

    required = value.get("required_output", {})
    _require(required.get("manifest_schema") == OUTPUT_SCHEMA, "STRUCTURAL_EVIDENCE_BINDING_OUTPUT_SCHEMA_DRIFT")
    _require(required.get("manifest_status") == OUTPUT_STATUS, "STRUCTURAL_EVIDENCE_BINDING_OUTPUT_STATUS_DRIFT")
    _require(tuple(required.get("subject_order", ())) == SUBJECTS, "STRUCTURAL_EVIDENCE_BINDING_OUTPUT_SUBJECT_ORDER_DRIFT")
    _require(required.get("subject_count") == len(SUBJECTS), "STRUCTURAL_EVIDENCE_BINDING_OUTPUT_SUBJECT_COUNT_DRIFT")
    flags = required.get("required_manifest_flags", {})
    _require(flags == {
        "training_authorized": False,
        "scientific_fit_started": False,
        "teacher_inputs_in_inference_forward": False,
    }, "STRUCTURAL_EVIDENCE_BINDING_OUTPUT_FLAGS_DRIFT")

    post = value.get("post_materialization_gate", {})
    for key in ("materialization_is_scientific_pass", "scientific_court_authorized", "relation_training_authorized", "tessa_training_authorized"):
        _require(post.get(key) is False, "STRUCTURAL_EVIDENCE_BINDING_POST_GATE_AUTHORIZATION_DRIFT:" + key)
    return value


def validate_materialized(binding: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_binding(binding)
    _require(binding.get("status") == "READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS", "STRUCTURAL_EVIDENCE_OUTPUT_BINDING_NOT_FROZEN")
    manifest_path = output_dir / binding["producer"]["output_manifest"]
    _require(manifest_path.is_file(), "STRUCTURAL_EVIDENCE_OUTPUT_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = binding["required_output"]
    _require(manifest.get("schema") == required["manifest_schema"], "STRUCTURAL_EVIDENCE_OUTPUT_SCHEMA_DRIFT")
    _require(manifest.get("status") == required["manifest_status"], "STRUCTURAL_EVIDENCE_OUTPUT_STATUS_DRIFT")
    for key, expected in required["required_manifest_flags"].items():
        _require(manifest.get(key) is expected, "STRUCTURAL_EVIDENCE_OUTPUT_FLAG_DRIFT:" + key)
    _require(tuple(manifest.get("subject_order", ())) == SUBJECTS, "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT_ORDER_DRIFT")

    policy = manifest.get("policy", {})
    _require(policy.get("candidate_rays_per_view") == binding["producer"]["candidate_rays_per_view"], "STRUCTURAL_EVIDENCE_OUTPUT_RAY_POLICY_DRIFT")
    _require(policy.get("grid_size") == binding["producer"]["grid_size"], "STRUCTURAL_EVIDENCE_OUTPUT_GRID_POLICY_DRIFT")
    _require(policy.get("all_unordered_stack_pairs_emitted_both_orientations") is True, "STRUCTURAL_EVIDENCE_OUTPUT_PAIR_ORIENTATION_DRIFT")
    _require(policy.get("pair_point_coordinate_domain") == "EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD", "STRUCTURAL_EVIDENCE_OUTPUT_COORDINATE_AUTHORITY_DRIFT")

    authority = manifest.get("authority", {})
    _require(authority.get("front") == "DETERMINISTIC_DERIVED__FULL_RAY_STACK_DEPTH_ORDER", "STRUCTURAL_EVIDENCE_OUTPUT_FRONT_AUTHORITY_DRIFT")
    _require(authority.get("visibility") == "DETERMINISTIC_DERIVED__GLOBAL_STACK_RANK_ZERO_ONLY", "STRUCTURAL_EVIDENCE_OUTPUT_VISIBILITY_AUTHORITY_DRIFT")
    _require(authority.get("occlusion") == "POSITIVE_SUPPORT_ONLY__SAME_RAY_PAIR__NO_BINARY_NEGATIVE_AUTHORITY", "STRUCTURAL_EVIDENCE_OUTPUT_OCCLUSION_AUTHORITY_DRIFT")
    _require(authority.get("projected_boundary") == "NOT_EMITTED__KEEP_SEPARATE_CURRENT_VIEW_RGBA_AUTHORITY", "STRUCTURAL_EVIDENCE_OUTPUT_BOUNDARY_AUTHORITY_DRIFT")
    _require(authority.get("static_same_cut_attachment") == "NOT_EMITTED__NO_NEW_AUTHORITY", "STRUCTURAL_EVIDENCE_OUTPUT_STATIC_AUTHORITY_DRIFT")

    source_bindings = manifest.get("source_bindings", {})
    _require(source_bindings.get("stage01_bootstrap_sha256") == binding["source_inputs"]["stage01"]["bootstrap_sha256"], "STRUCTURAL_EVIDENCE_OUTPUT_STAGE01_BOOTSTRAP_DRIFT")

    rows = manifest.get("subjects")
    _subject_ids(rows, "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT")
    suffix = required["per_subject_file_suffix"]
    for row in rows:
        filename = str(row.get("file", ""))
        _require(filename.endswith(suffix), "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT_FILE_NAME_DRIFT")
        path = output_dir / filename
        _require(path.is_file(), "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT_FILE_MISSING:" + str(row.get("subject_id")))
        _require_sha256(row.get("sha256"), "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT_SHA_INVALID")
        _require(_sha256_file(path) == row["sha256"], "STRUCTURAL_EVIDENCE_OUTPUT_SUBJECT_HASH_MISMATCH:" + str(row.get("subject_id")))
        _require(isinstance(row.get("views"), list) and len(row["views"]) == 8, "STRUCTURAL_EVIDENCE_OUTPUT_VIEW_COUNT_DRIFT:" + str(row.get("subject_id")))
        _require(row.get("normalization", {}).get("authority") == "EXACT_LEGACY_IRIS_SHARD", "STRUCTURAL_EVIDENCE_OUTPUT_NORMALIZATION_AUTHORITY_DRIFT:" + str(row.get("subject_id")))

    return {
        "schema": "RealSaS.StructuralEvidenceMaterializationAdmission.v1",
        "status": "ADMITTED_HASHED_EVIDENCE_ONLY__NO_SCIENTIFIC_PASS",
        "subject_count": len(rows),
        "manifest_sha256": _sha256_file(manifest_path),
        "implementation_source_sha": binding["implementation_source_sha"],
        "training_authorized": False,
        "scientific_court_authorized": False,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _args()
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    validate_binding(binding)
    if args.materialized_dir is None:
        print("STRUCTURAL_EVIDENCE_INPUT_BINDING_PASS", binding["status"], flush=True)
        return
    result = validate_materialized(binding, args.materialized_dir)
    print("STRUCTURAL_EVIDENCE_MATERIALIZATION_ADMISSION_PASS", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
