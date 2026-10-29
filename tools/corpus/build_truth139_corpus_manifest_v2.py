#!/usr/bin/env python3
"""Build the capability-aware Truth139 corpus manifest after sealed structural audit.

Historical raw capability names remain compatibility vocabulary only. Current teacher projection is
IRIS Geometry + Graphics + Relations + optional Response Evidence + TESSA topology. In particular,
Graphics/Relations materialization must not be blocked by absence of Stage47 response evidence.
Historical-reproduction mismatch is recorded as a court finding rather than used as a global
operational stop after the structural byte/parse audit has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

AUDIT_SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"
AUDIT_STATUS = "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
COURT_SCHEMA = "RealSaS.Truth139HistoricalReproductionCourt.v1"
COURT_STATUS = "PASS_HISTORICAL_139_REPRODUCTION"
OUTPUT_SCHEMA = "RealSaS.Truth139CorpusManifest.v2"
EXPECTED_SUBJECT_COUNT = 139

GRAPHICS_CONTRACT = "canonical/TRUTH139_DENSE_GRAPHICS_TEACHER_V1.json"
LAYER_CONTRACT = "canonical/TRUTH139_GRAPHICS_LAYER_DECOMPOSITION_TEACHER_V1.json"
RELATIONS_CONTRACT = "canonical/TRUTH139_RELATIONS_TEACHER_V1.json"
RESPONSE_CONTRACT = "canonical/TRUTH139_IRIS_RESPONSE_EVIDENCE_TEACHER_V1.json"
LEGACY_APPEARANCE_CONTRACT = "canonical/TRUTH139_DENSE_APPEARANCE_AUTHORITY_V1.json"

RAW_CAPABILITY_NAMES = (
    "OBSERVATION_FSTAR",
    "SURFACE_NORMAL_TRUTH",
    "STAGE47_MECHANICS",
    "STAGE48_APPEARANCE_PRESENTATION",
    "TESSA_TOPOLOGY_TEACHER",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_binding(row: dict) -> dict:
    source = dict(row.get("source") or {})
    required = ("relative_path", "sha256", "size_bytes", "format", "creator", "pack_id")
    missing = [name for name in required if source.get(name) in (None, "")]
    if missing:
        raise ValueError("TRUTH139_CORPUS_SOURCE_BINDING_MISSING:" + ",".join(missing))
    deps = list(source.get("dependencies") or ())
    if int(source.get("dependency_count", len(deps))) != len(deps):
        raise ValueError("TRUTH139_CORPUS_DEPENDENCY_COUNT_DRIFT:" + str(row.get("subject_id")))
    snapshot = dict(source.get("source_page_snapshot") or {})
    if not str(snapshot.get("path") or "") or len(str(snapshot.get("sha256") or "")) != 64:
        raise ValueError("TRUTH139_CORPUS_PROVENANCE_SNAPSHOT_BINDING_MISSING:" + str(row.get("subject_id")))
    return {
        "corpus_relative_path": str(source["relative_path"]),
        "sha256": str(source["sha256"]),
        "byte_size": int(source["size_bytes"]),
        "format": str(source["format"]),
        "creator": str(source["creator"]),
        "pack_id": str(source["pack_id"]),
        "dependencies": deps,
        "dependency_count": len(deps),
        "source_page_snapshot": snapshot,
        "historical_source_page_snapshot_byte_available": bool(source.get("historical_source_page_snapshot_byte_available")),
        "provenance_status": str(source.get("provenance_status") or ""),
        "official_url": str(source.get("official_url") or ""),
        "drive_folder_id": str(source.get("drive_folder_id") or ""),
        "source_family_signature": list(source.get("source_family_signature") or ()),
    }


def _materialization_plan(raw: dict) -> dict[str, bool]:
    observation = bool(raw.get("OBSERVATION_FSTAR"))
    normals = bool(raw.get("SURFACE_NORMAL_TRUTH"))
    response_capable = bool(raw.get("STAGE47_MECHANICS"))
    stage48_witness = bool(raw.get("STAGE48_APPEARANCE_PRESENTATION"))
    tessa = bool(raw.get("TESSA_TOPOLOGY_TEACHER"))

    # Graphics comes from source scene + observations. Response evidence is an optional parallel
    # IRIS branch and may never be a prerequisite for source color, depth, visibility or occlusion.
    dense_graphics = observation
    return {
        "CONTROLLED_8VIEW_SOURCE_OBSERVATIONS": observation,
        "DINO_AND_IMAGE_DERIVED_FSTAR_SHARDS": observation,
        "IRIS_GEOMETRY_NORMAL_DEPTH_TEACHER": observation and normals,
        "IRIS_RESPONSE_EVIDENCE": response_capable,
        "GRAPHICS_FRONT_SURFACE_GBUFFER": dense_graphics,
        "GRAPHICS_VISIBILITY_OCCLUSION_TEACHER": dense_graphics,
        "GRAPHICS_LAYER_PEEL_GEOMETRY": dense_graphics,
        "IRIS_RELATIONS_TEACHER": dense_graphics,
        "STAGE48_RENDERER_INTERVENTION_WITNESS": stage48_witness,
        "GRAPHICS_FULL_COURT_CANDIDATE": dense_graphics and stage48_witness,
        "TESSA_TOPOLOGY_TEACHER_CACHE": tessa,
    }


def build(audit: dict, court: dict) -> dict:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != AUDIT_STATUS:
        raise ValueError("TRUTH139_CORPUS_V2_STRUCTURAL_AUDIT_NOT_PASS")
    if audit.get("structural_core_passed") is not True:
        raise ValueError("TRUTH139_CORPUS_V2_STRUCTURAL_CORE_NOT_PASS")
    if court.get("schema") != COURT_SCHEMA:
        raise ValueError("TRUTH139_CORPUS_V2_HISTORICAL_REPRODUCTION_SCHEMA")
    if str(court.get("source_audit_identity_sha256") or "") != str(audit.get("audit_identity_sha256") or ""):
        raise ValueError("TRUTH139_CORPUS_V2_PARENT_IDENTITY_MISMATCH")
    corpus = dict(audit.get("corpus") or {})
    if corpus.get("seal_verified") is not True:
        raise ValueError("TRUTH139_CORPUS_V2_SEAL_NOT_VERIFIED")

    subjects = list(audit.get("subjects") or ())
    if not subjects:
        raise ValueError("TRUTH139_CORPUS_V2_NO_SUBJECTS")
    historical_reproduction_passed = court.get("status") == COURT_STATUS
    subject_count_matches_historical = len(subjects) == EXPECTED_SUBJECT_COUNT

    rows = []
    raw_capability_counts = Counter()
    materialization_counts = Counter()
    development_count = 0
    historical_full_family_clean_count = 0
    iris_tessa_full_teacher_clean_count = 0
    pack_counts = Counter()

    for ordinal, row in enumerate(sorted(subjects, key=lambda item: str(item.get("subject_id") or "")), 1):
        subject_id = str(row.get("subject_id") or "")
        if not subject_id:
            raise ValueError("TRUTH139_CORPUS_V2_SUBJECT_ID_MISSING")
        raw = {name: bool((row.get("capabilities") or {}).get(name)) for name in RAW_CAPABILITY_NAMES}
        for name, enabled in raw.items():
            raw_capability_counts[name] += int(enabled)

        development_exclusion = row.get("development_exclusion")
        development_taint = list(row.get("duplicate_family_development_taint") or ())
        is_development = bool(development_exclusion or development_taint)
        development_count += int(is_development)

        historical_full = all(raw.values())
        historical_full_clean = historical_full and not is_development
        historical_full_family_clean_count += int(historical_full_clean)

        plan = _materialization_plan(raw)
        for name, enabled in plan.items():
            materialization_counts[name] += int(bool(enabled))
        iris_tessa_full = bool(
            plan["IRIS_GEOMETRY_NORMAL_DEPTH_TEACHER"]
            and plan["GRAPHICS_VISIBILITY_OCCLUSION_TEACHER"]
            and plan["IRIS_RELATIONS_TEACHER"]
            and plan["TESSA_TOPOLOGY_TEACHER_CACHE"]
        )
        iris_tessa_full_teacher_clean_count += int(iris_tessa_full and not is_development)

        source = _source_binding(row)
        pack_counts[source["pack_id"]] += 1
        rows.append({
            "ordinal": ordinal,
            "subject_id": subject_id,
            "subject_identity_signature": str(row.get("subject_identity_signature") or ""),
            "exact_geometry_identity_signature": str(row.get("exact_geometry_identity_signature") or row.get("subject_identity_signature") or ""),
            "historical_variant_family_signature": str(row.get("historical_variant_family_signature") or ""),
            "source": source,
            "lineage": dict(row.get("lineage") or {}),
            "raw_structure": dict(row.get("raw_structure") or {}),
            "duplicate_source_file_count": int(row.get("duplicate_source_file_count", 1)),
            "duplicate_source_files": list(row.get("duplicate_source_files") or ()),
            "raw_capabilities": raw,
            "raw_capability_vocabulary": "HISTORICAL_COMPATIBILITY__NOT_CANONICAL_MODEL_DOMAIN_NAMES",
            "raw_eligibility_blockers": list(row.get("raw_eligibility_blockers") or ()),
            "development_exclusion": development_exclusion,
            "duplicate_family_development_taint": development_taint,
            "development_or_regression_tainted": is_development,
            "materialization_plan": plan,
            "historical_full_family_raw_capable": historical_full,
            "iris_tessa_full_teacher_raw_candidate": iris_tessa_full,
            "platform_truth_admission": "PENDING",
            "graphics_front_surface_teacher": "PENDING" if plan["GRAPHICS_FRONT_SURFACE_GBUFFER"] else "NOT_RAW_CAPABLE",
            "graphics_visibility_occlusion_teacher": "PENDING" if plan["GRAPHICS_VISIBILITY_OCCLUSION_TEACHER"] else "NOT_RAW_CAPABLE",
            "graphics_layer_peel_teacher": "PENDING" if plan["GRAPHICS_LAYER_PEEL_GEOMETRY"] else "NOT_RAW_CAPABLE",
            "relations_teacher": "PENDING" if plan["IRIS_RELATIONS_TEACHER"] else "NOT_RAW_CAPABLE",
            "iris_response_evidence": "PENDING" if plan["IRIS_RESPONSE_EVIDENCE"] else "NOT_RAW_CAPABLE",
            "stage48_renderer_intervention_witness": "PENDING" if plan["STAGE48_RENDERER_INTERVENTION_WITNESS"] else "NOT_RAW_CAPABLE",
            "graphics_full_court": "PENDING" if plan["GRAPHICS_FULL_COURT_CANDIDATE"] else "NOT_RAW_CAPABLE",
            "train_membership": "UNASSIGNED",
            "unseen_membership": "UNASSIGNED",
            "training_authorized": False,
        })

    status = (
        "SEALED_RAW_139_REPRODUCED__CAPABILITY_AWARE_TEACHER_MATERIALIZATION_PENDING"
        if historical_reproduction_passed and subject_count_matches_historical
        else "SEALED_RAW_STRUCTURAL_PASS__HISTORICAL_REPRODUCTION_FINDING__CAPABILITY_MATERIALIZATION_MAY_CONTINUE"
    )
    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": status,
        "historical_reproduction": {
            "passed": historical_reproduction_passed,
            "court_status": str(court.get("status") or ""),
            "findings": list(court.get("findings") or ()),
            "expected_subject_count": EXPECTED_SUBJECT_COUNT,
            "actual_subject_count": len(rows),
            "subject_count_exact": subject_count_matches_historical,
            "operational_blocker": False,
        },
        "parents": {
            "sealed_raw_audit_identity_sha256": str(audit["audit_identity_sha256"]),
            "historical_reproduction_court_identity_sha256": str(court["court_identity_sha256"]),
        },
        "corpus": {
            "root_sha256": str(corpus.get("root_sha256") or ""),
            "manifest_sha256": str(corpus.get("manifest_sha256") or ""),
            "ledger_sha256": str(corpus.get("ledger_sha256") or ""),
            "seal_json_sha256": str(corpus.get("seal_json_sha256") or ""),
            "historical_subject_count": EXPECTED_SUBJECT_COUNT,
            "subject_count": len(rows),
            "development_or_regression_tainted_subject_count": development_count,
            "historical_clean_full_family_raw_candidate_count": historical_full_family_clean_count,
            "iris_tessa_clean_full_teacher_raw_candidate_count": iris_tessa_full_teacher_clean_count,
            "raw_capability_counts": dict(sorted(raw_capability_counts.items())),
            "capability_counts": dict(sorted(raw_capability_counts.items())),
            "materialization_candidate_counts": dict(sorted(materialization_counts.items())),
            "pack_counts": dict(sorted(pack_counts.items())),
        },
        "teacher_contracts": {
            "graphics": GRAPHICS_CONTRACT,
            "graphics_layer_decomposition": LAYER_CONTRACT,
            "relations": RELATIONS_CONTRACT,
            "iris_response_evidence": RESPONSE_CONTRACT,
            "legacy_dense_appearance_contract": LEGACY_APPEARANCE_CONTRACT,
        },
        "policy": {
            "corpus_identity_is_not_experiment_projection": True,
            "historical_reproduction_is_independent_nonblocking_court_after_structural_pass": True,
            "historical_stage47_mechanics_name_is_compatibility_vocabulary_only": True,
            "historical_stage48_appearance_presentation_name_is_compatibility_vocabulary_only": True,
            "legacy_dense_appearance_contract_is_not_global_graphics_authority": True,
            "graphics_is_parent_domain": True,
            "graphics_does_not_depend_on_response_evidence": True,
            "response_evidence_is_optional_parallel_iris_branch": True,
            "source_color_authority_is_partial_and_provenance_bound": True,
            "relations_are_primitive_nonsemantic_truth": True,
            "layer_masks_are_depth_or_drawable_evidence_not_humanoid_part_labels": True,
            "development_taint_does_not_delete_corpus_rows": True,
            "missing_optional_truth_family_does_not_delete_corpus_rows": True,
            "heavy_truth_materialization_must_follow_per_subject_capability_plan": True,
            "stage48_alone_is_not_graphics_training_truth": True,
            "split_before_platform_and_derived_truth_admission_forbidden": True,
            "training_before_split_and_unseen_seal_forbidden": True,
            "subject_specific_code_forbidden": True,
        },
        "external_worker": {
            "contract": "canonical/TRUTH139_EXTERNAL_WORKER_NOTEBOOK_CONTRACT_V1.json",
            "intended_notebook_repo_path": "notebooks/Truth139_Corpus_Preparation_T4_CPU_v1.ipynb",
            "authority": "NON_AUTHORITATIVE_COMPUTE_WORKER",
            "platform_remains_lineage_authority": True,
            "expected_heavy_families": [
                "8X1024_GRAPHICS_FRONT_SURFACE",
                "VISIBILITY_OCCLUSION",
                "K_HIT_LAYER_PEEL",
                "DINO_FSTAR",
                "SOURCE_RESPONSE_INTERVENTIONS",
                "STAGE48_RENDERER_WITNESS"
            ],
        },
        "subjects": rows,
        "split_created": False,
        "unseen_sealed": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["corpus_manifest_identity_sha256"] = _digest(payload)
    return payload


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-raw-audit-v2", required=True)
    parser.add_argument("--historical-reproduction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.sealed_raw_audit_v2).resolve().read_text())
    court = json.loads(Path(args.historical_reproduction).resolve().read_text())
    result = build(audit, court)
    output = Path(args.output).resolve()
    _write(output, result)
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "subject_count": result["corpus"]["subject_count"],
        "historical_reproduction": result["historical_reproduction"],
        "development_or_regression_tainted_subject_count": result["corpus"]["development_or_regression_tainted_subject_count"],
        "iris_tessa_clean_full_teacher_raw_candidate_count": result["corpus"]["iris_tessa_clean_full_teacher_raw_candidate_count"],
        "materialization_candidate_counts": result["corpus"]["materialization_candidate_counts"],
        "corpus_manifest_identity_sha256": result["corpus_manifest_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
