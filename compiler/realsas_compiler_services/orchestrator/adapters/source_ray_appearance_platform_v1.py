from __future__ import annotations

"""Platform admission adapter for sparse source-ray appearance measurements.

The input is the exact #243 source-equivalence court bundle. The adapter does
not bind teacher face indices to CanonicalMeshIR and therefore does not mint a
``RayStackEvidenceV2`` admission map. Its sole job is to reseal the already
measured, source-equivalent sparse appearance domain as immutable typed Platform
artifacts. Canonical-M address binding remains a distinct downstream consumer
court.

External import alone is never qualification. This adapter revalidates the
court, its exact subject measurement payloads and their deterministic hashes;
only the resulting stage outputs may be registered by Platform execution.
"""

import hashlib
import json
import re
from typing import Any

from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import write_json
from compiler.realsas_compiler_core.types import QualificationError

BUNDLE_SCHEMA = "RealSaS.SourceRayAppearanceCourtBundle.v1"
COURT_SCHEMA = "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1"
MEASUREMENT_SCHEMA = "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"
QUALIFICATION_SCHEMA = "RealSaS.SourceRayAppearanceQualification.v1"
EVIDENCE_SCHEMA = "RealSaS.SourceRayAppearanceEvidence.v1"
FAMILY = "RAY_COMPOSITING"
PASS_COURT = "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE"
PASS_MEASUREMENT = "PASS_MEASURED_SOURCE_RAY_APPEARANCE_CANDIDATE"
PASS_FACTORIZATION = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_hash(value: Any, code: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise QualificationError(code)
    return text


def _load_bundle(ctx: dict[str, Any]) -> dict[str, Any]:
    bundle = ctx["run_manifest"].get("source_ray_appearance_court_bundle")
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_BUNDLE_REQUIRED")
    if bundle.get("registry_qualification_minted") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_EXTERNAL_PREQUALIFICATION_FORBIDDEN")
    if bundle.get("canonical_mesh_binding_minted") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_EXTERNAL_MESH_BINDING_FORBIDDEN")
    if bundle.get("training_authorized") is not False or bundle.get("product_authority_changed") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_EXTERNAL_AUTHORITY_ESCALATION")
    return bundle


def _validate_measurement(subject_id: str, payload: dict[str, Any], expected_sha: str) -> tuple[int, int]:
    if payload.get("schema") != MEASUREMENT_SCHEMA:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_SCHEMA")
    if str(payload.get("subject_id") or "") != subject_id:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_SUBJECT_DRIFT")
    if str(payload.get("court_status") or "") != PASS_MEASUREMENT:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_NOT_PASS")
    if payload.get("qualification_minted") is not False or payload.get("qualified_sampler") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_PREMATURE_QUALIFICATION")
    if payload.get("training_allowed") is not False or payload.get("product_pass_claimed") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_AUTHORITY_ESCALATION")
    if _canonical_hash(payload) != expected_sha:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_HASH_DRIFT")

    rays = payload.get("rays")
    if not isinstance(rays, list) or not rays:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_RAYS_REQUIRED")
    hit_count = 0
    for ray in rays:
        if not isinstance(ray, dict) or ray.get("complete") is not True or ray.get("has_depth_tie") is not False:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_RAY_NOT_ADMISSIBLE")
        measurement = ray.get("measurement")
        if not isinstance(measurement, dict) or measurement.get("valid") is not True or str(measurement.get("status") or "") != PASS_FACTORIZATION:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_FACTORIZATION_NOT_PASS")
        hits = ray.get("canonical_hits")
        if not isinstance(hits, list) or len(hits) != int(ray.get("hit_count", -1)):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_HIT_COUNT_DRIFT")
        if len(measurement.get("effective_alpha") or ()) != len(hits) or len(measurement.get("straight_linear_rgb") or ()) != len(hits):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_VECTOR_DRIFT")
        hit_count += len(hits)
    return len(rays), hit_count


def _expected_measurement_parents(court: dict[str, Any]) -> dict[str, str]:
    rows = court.get("subjects")
    if not isinstance(rows, list) or not rows:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_SUBJECTS_REQUIRED")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_SUBJECT_INVALID")
        subject_id = str(row.get("subject_id") or "")
        parents = row.get("parents")
        if not subject_id or subject_id in expected or not isinstance(parents, dict):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_SUBJECT_INVALID")
        packed_a = _require_hash(parents.get("packed_a_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_PACKED_PARENT_REQUIRED")
        packed_b = _require_hash(parents.get("packed_b_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_PACKED_PARENT_REQUIRED")
        if packed_a != packed_b or row.get("packed_replay_core_exact") is not True:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_PACKED_REPLAY_NOT_EXACT")
        if row.get("source_equivalent_on_declared_hits") is not True or row.get("source_reference_replay_core_exact") is not True:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_SUBJECT_NOT_SOURCE_EQUIVALENT")
        expected[subject_id] = packed_a
    return expected


def _validate_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[str, ...], int, int]:
    court = bundle.get("court")
    if not isinstance(court, dict) or court.get("schema") != COURT_SCHEMA:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_SCHEMA")
    court_sha = _require_hash(bundle.get("court_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_COURT_HASH_REQUIRED")
    if _canonical_hash(court) != court_sha:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_HASH_DRIFT")
    if str(court.get("court_status") or "") != PASS_COURT:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_NOT_PASS")

    candidate = court.get("family_qualification_input")
    decision = court.get("local_qualification_decision")
    if not isinstance(candidate, dict) or not isinstance(decision, dict):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_DECISION_REQUIRED")
    if str(candidate.get("family") or "") != FAMILY or str(decision.get("family") or "") != FAMILY:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_FAMILY_DRIFT")
    if candidate.get("measurement_status") != "MEASURED" or candidate.get("source_equivalent") is not True or candidate.get("coverage_qualified") is not True or candidate.get("replay_deterministic") is not True:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_QUALIFICATION_INPUT_NOT_ADMISSIBLE")
    if decision.get("oracle_eligible") is not True:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_ORACLE_INELIGIBLE")
    if decision.get("deterministic_student_supervision_eligible") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_STUDENT_SCOPE_ESCALATION")
    if court.get("global_sampler_qualification_claimed") is not False or court.get("registry_admitted") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_PREEXISTING_REGISTRY_AUTHORITY")
    if court.get("training_allowed") is not False or court.get("product_pass_claimed") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_AUTHORITY_ESCALATION")

    expected_parents = _expected_measurement_parents(court)
    measurements = bundle.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENTS_REQUIRED")
    seen_subjects: set[str] = set()
    measurement_rows: list[dict[str, Any]] = []
    ray_count = hit_count = 0
    for row in measurements:
        if not isinstance(row, dict):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_BINDING_INVALID")
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in expected_parents or subject_id in seen_subjects:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_SUBJECT_DRIFT")
        seen_subjects.add(subject_id)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_PAYLOAD_REQUIRED")
        sha = _require_hash(row.get("sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_HASH_REQUIRED")
        if sha != expected_parents[subject_id]:
            raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_PARENT_HASH_DRIFT")
        rays, hits = _validate_measurement(subject_id, payload, sha)
        ray_count += rays
        hit_count += hits
        measurement_rows.append({"subject_id": subject_id, "sha256": sha, "payload": payload})

    if seen_subjects != set(expected_parents):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_MEASUREMENT_COHORT_INCOMPLETE")
    if len(seen_subjects) != int(court.get("subject_count", -1)) or ray_count != int(court.get("ray_count", -1)) or hit_count != int(court.get("hit_count", -1)):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COHORT_COUNT_DRIFT")
    if hit_count != int(candidate.get("valid_count", -1)) or hit_count != int(candidate.get("total_count", -1)):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_QUALIFICATION_COUNT_DRIFT")

    witness = _require_hash(court.get("witness_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_WITNESS_REQUIRED")
    if witness != _require_hash(candidate.get("witness_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_CANDIDATE_WITNESS_REQUIRED"):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_WITNESS_DRIFT")
    return court, measurement_rows, tuple(sorted(seen_subjects)), ray_count, hit_count


def admit_source_ray_appearance_v1(ctx: dict[str, Any]) -> dict[str, Any]:
    bundle = _load_bundle(ctx)
    court, measurements, subjects, ray_count, hit_count = _validate_bundle(bundle)
    root = ctx["run_root"] / "artifacts" / str((ctx.get("stage") or {}).get("id") or "SOURCE_RAY_APPEARANCE_PLATFORM_ADMISSION")

    measurement_set_sha256 = _canonical_hash([
        {"subject_id": row["subject_id"], "sha256": row["sha256"]} for row in measurements
    ])
    qualification = {
        "schema": QUALIFICATION_SCHEMA,
        "family": FAMILY,
        "scope": {
            "kind": "SPARSE_MEASURED_RAYS_ONLY",
            "subject_ids": list(subjects),
            "subject_count": len(subjects),
            "ray_count": ray_count,
            "hit_count": hit_count,
            "coverage_scope": str((court.get("policy") or {}).get("coverage_scope") or ""),
        },
        "court_sha256": str(bundle["court_sha256"]),
        "court_witness_sha256": str(court["witness_sha256"]),
        "measurement_set_sha256": measurement_set_sha256,
        "source_equivalent": True,
        "replay_deterministic": True,
        "coverage_qualified": True,
        "oracle_eligible": True,
        "global_sampler_qualified": False,
        "deterministic_student_supervision_eligible": False,
        "training_authorized": False,
        "product_authority_changed": False,
        "genericity_claimed": False,
        "canonical_mesh_binding_minted": False,
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "family": FAMILY,
        "scope": dict(qualification["scope"]),
        "court_sha256": str(bundle["court_sha256"]),
        "measurement_set_sha256": measurement_set_sha256,
        "measurements": measurements,
        "global_sampler_qualified": False,
        "training_authorized": False,
        "product_authority_changed": False,
        "canonical_mesh_binding_minted": False,
    }

    outputs = [
        write_json(root / "source_ray_appearance_qualification.json", qualification,
                   authority_class="QUALIFIED_SPARSE_SOURCE_RAY_APPEARANCE",
                   schema=QUALIFICATION_SCHEMA),
        write_json(root / "source_ray_appearance_evidence.json", evidence,
                   authority_class="QUALIFIED_SPARSE_SOURCE_RAY_APPEARANCE_EVIDENCE",
                   schema=EVIDENCE_SCHEMA),
    ]
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "schema": "RealSaS.SourceRayAppearancePlatformAdmissionDiagnostics.v1",
            "subject_count": len(subjects),
            "ray_count": ray_count,
            "hit_count": hit_count,
            "global_sampler_qualified": False,
            "canonical_mesh_binding_minted": False,
            "training_authorized": False,
            "product_authority_changed": False,
        },
    }
