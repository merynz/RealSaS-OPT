#!/usr/bin/env python3
"""Build capability-aware heavy-materialization work orders from the structural corpus manifest.

Jobs are capability-local: failure or absence of optional Response Evidence may not suppress
Graphics, visibility/occlusion, Relations, or TESSA work. Historical-reproduction findings are
retained in the parent manifest but are not an operational materialization stop after structural
byte/parse validity has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

INPUT_SCHEMA = "RealSaS.Truth139CorpusManifest.v2"
OUTPUT_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"

DENSE_GRAPHICS_IMPLEMENTATION = {
    "producer": "tools/corpus/build_truth139_dense_graphics_gbuffer_v1.py",
    "producer_pending_status": "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING",
    "independent_court": "tools/corpus/validate_truth139_dense_gbuffer_geometry_v1.py",
    "required_court_status": "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER",
}
VISIBILITY_OCCLUSION_IMPLEMENTATION = {
    "producer": "tools/corpus/build_truth139_visibility_occlusion_teacher_v2.py",
    "producer_revision": "V2_SILHOUETTE_NEIGHBOR_FIRST",
    "producer_pending_status": "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING",
    "independent_court": "tools/corpus/validate_truth139_visibility_occlusion_teacher_v1.py",
    "required_court_status": "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER",
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require_source(row: dict) -> dict:
    source = dict(row.get("source") or {})
    required = ("corpus_relative_path", "sha256", "byte_size", "format", "creator", "pack_id")
    missing = [name for name in required if source.get(name) in (None, "")]
    if missing:
        raise ValueError("TRUTH139_WORK_ORDER_SOURCE_BINDING_MISSING:" + ",".join(missing))
    snapshot = dict(source.get("source_page_snapshot") or {})
    if not str(snapshot.get("path") or "") or len(str(snapshot.get("sha256") or "")) != 64:
        raise ValueError("TRUTH139_WORK_ORDER_PROVENANCE_BINDING_MISSING:" + str(row.get("subject_id") or ""))
    deps = list(source.get("dependencies") or ())
    if int(source.get("dependency_count", len(deps))) != len(deps):
        raise ValueError("TRUTH139_WORK_ORDER_DEPENDENCY_COUNT_DRIFT:" + str(row.get("subject_id") or ""))
    return source


def _jobs(raw: dict) -> dict[str, dict]:
    observation = bool(raw.get("OBSERVATION_FSTAR"))
    normals = bool(raw.get("SURFACE_NORMAL_TRUTH"))
    response = bool(raw.get("STAGE47_MECHANICS"))
    renderer_witness = bool(raw.get("STAGE48_APPEARANCE_PRESENTATION"))
    topology = bool(raw.get("TESSA_TOPOLOGY_TEACHER"))
    dense_graphics = observation

    return {
        "SOURCE_IMPORT_AND_8VIEW_OBSERVATION": {
            "enabled": observation,
            "execution": "EXTERNAL_WORKER",
            "depends_on": [],
        },
        "GEOMETRY_FSTAR_DEPTH_NORMAL_TRUTH": {
            "enabled": observation and normals,
            "execution": "EXTERNAL_WORKER",
            "depends_on": ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"],
        },
        "IRIS_RESPONSE_EVIDENCE": {
            "enabled": response,
            "execution": "EXTERNAL_WORKER_CPU_HEAVY",
            "depends_on": ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"],
        },
        "DENSE_GRAPHICS_FRONT_SURFACE": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_GPU_PLUS_CPU_COURT",
            "depends_on": ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"],
            "implementation": dict(DENSE_GRAPHICS_IMPLEMENTATION),
            "admission_rule": "PRODUCER_OUTPUT_IS_DERIVED_TRUTH_PENDING_UNTIL_INDEPENDENT_COURT_REQUIRED_STATUS_MATCHES",
        },
        "VISIBILITY_OCCLUSION_CROSS_VIEW": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_GPU_PLUS_CPU_COURT",
            "depends_on": ["DENSE_GRAPHICS_FRONT_SURFACE"],
            "implementation": dict(VISIBILITY_OCCLUSION_IMPLEMENTATION),
            "admission_rule": "V2_PRODUCER_ONLY__CPU_RAY_ORACLE_COURT_MUST_PASS_BEFORE_DERIVED_TRUTH_ADMISSION",
        },
        "K_HIT_GEOMETRIC_PEEL": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_GPU_PLUS_CPU_COURT",
            "depends_on": ["DENSE_GRAPHICS_FRONT_SURFACE"],
        },
        "ANONYMOUS_COMPONENT_MASKS": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_CPU",
            "depends_on": ["K_HIT_GEOMETRIC_PEEL"],
            "conditional": "REQUIRES_EXACT_FACE_OWNER_BINDING__ABSTAIN_IF_UNAVAILABLE",
        },
        "DRAWABLE_AFFINITY_RELATIONS": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_CPU",
            "depends_on": ["VISIBILITY_OCCLUSION_CROSS_VIEW", "K_HIT_GEOMETRIC_PEEL"],
            "conditional": "SOURCE_COMPONENT_AFFINITY_SUBFAMILY_REQUIRES_ANONYMOUS_COMPONENT_MASKS",
        },
        "PRIMITIVE_RELATIONS": {
            "enabled": dense_graphics,
            "execution": "EXTERNAL_WORKER_CPU",
            "depends_on": ["VISIBILITY_OCCLUSION_CROSS_VIEW", "DENSE_GRAPHICS_FRONT_SURFACE"],
        },
        "STAGE48_RENDERER_INTERVENTION_WITNESS": {
            "enabled": renderer_witness,
            "execution": "EXTERNAL_WORKER_CPU_RENDERER_HEAVY",
            "depends_on": ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"],
        },
        "GENERIC_EXACT_TOPOLOGY_TRUTH": {
            "enabled": topology,
            "execution": "EXTERNAL_WORKER_CPU",
            "depends_on": ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"],
            "note": "CORPUS_TOPOLOGY_TRUTH_ONLY__NO_TESSA_CONDITIONING_OR_MODEL_TRAINING",
        },
    }


def build(manifest: dict) -> dict:
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("TRUTH139_WORK_ORDER_PARENT_SCHEMA")
    parent_identity = str(manifest.get("corpus_manifest_identity_sha256") or "")
    if len(parent_identity) != 64:
        raise ValueError("TRUTH139_WORK_ORDER_PARENT_IDENTITY")
    rows = list(manifest.get("subjects") or ())
    if not rows:
        raise ValueError("TRUTH139_WORK_ORDER_NO_SUBJECTS")

    counts = Counter()
    subjects = []
    for row in sorted(rows, key=lambda item: str(item.get("subject_id") or "")):
        subject_id = str(row.get("subject_id") or "")
        if not subject_id:
            raise ValueError("TRUTH139_WORK_ORDER_SUBJECT_ID")
        source = _require_source(row)
        raw = dict(row.get("raw_capabilities") or {})
        jobs = _jobs(raw)
        for name, spec in jobs.items():
            counts[name] += int(bool(spec["enabled"]))
        subjects.append({
            "subject_id": subject_id,
            "subject_identity_signature": str(row.get("subject_identity_signature") or ""),
            "source": source,
            "development_or_regression_tainted": bool(row.get("development_or_regression_tainted")),
            "raw_capabilities": raw,
            "jobs": jobs,
            "derived_truth_admitted": False,
            "training_authorized": False,
        })

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": "READY_TO_MATERIALIZE_CAPABILITY_AWARE_TRUTH__DERIVED_TRUTH_NOT_ADMITTED",
        "parent_corpus_manifest_identity_sha256": parent_identity,
        "parent_historical_reproduction": dict(manifest.get("historical_reproduction") or {}),
        "subject_count": len(subjects),
        "job_candidate_counts": dict(sorted(counts.items())),
        "canonical_implementations": {
            "DENSE_GRAPHICS_FRONT_SURFACE": dict(DENSE_GRAPHICS_IMPLEMENTATION),
            "VISIBILITY_OCCLUSION_CROSS_VIEW": dict(VISIBILITY_OCCLUSION_IMPLEMENTATION),
        },
        "policy": {
            "missing_optional_truth_does_not_delete_subject": True,
            "per_subject_jobs_follow_raw_capabilities": True,
            "graphics_does_not_depend_on_response_evidence": True,
            "response_evidence_failure_does_not_block_graphics_or_relations": True,
            "development_taint_does_not_delete_corpus_identity": True,
            "historical_reproduction_finding_does_not_block_safe_materialization": True,
            "producer_output_is_not_admitted_without_required_independent_court": True,
            "work_order_implementation_paths_are_canonical_and_machine_readable": True,
            "no_train_unseen_split_here": True,
            "no_model_training_here": True,
            "no_tessa_conditioning_here": True,
            "failed_jobs_remain_in_failure_ledger": True,
            "missing_truth_may_not_be_zero_filled": True,
            "subject_specific_code_forbidden": True,
        },
        "subjects": subjects,
        "split_created": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["work_order_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest-v2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.corpus_manifest_v2.resolve().read_text())
    result = build(manifest)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_suffix(out.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(out)
    print(json.dumps({
        "status": result["status"],
        "subject_count": result["subject_count"],
        "job_candidate_counts": result["job_candidate_counts"],
        "work_order_identity_sha256": result["work_order_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
