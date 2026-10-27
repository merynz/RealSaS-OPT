from __future__ import annotations

"""Platform admission adapter for sparse source-ray appearance evidence.

This adapter does not re-measure appearance and cannot upgrade an arbitrary
external file by filename or by a free boolean.  It consumes the exact
``source_ray_appearance_court_bundle`` manifest section, verifies the sealed
court identities and candidate decision, then emits two typed outputs:

* a qualification receipt which records the admitted sparse domain and the
  exact court/source-equivalence witnesses; and
* an evidence payload containing only the exact record hashes admitted by that
  qualification.

The Platform capability/stage execution subsequently seals these bytes into CAS
and Registry.  Product activation, model promotion and genericity are never
minted here.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import write_json
from compiler.realsas_compiler_core.types import QualificationError

BUNDLE_SCHEMA = "RealSaS.SourceRayAppearanceCourtBundle.v1"
COURT_SCHEMA = "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1"
QUALIFICATION_SCHEMA = "RealSaS.SourceRayAppearanceQualification.v1"
EVIDENCE_SCHEMA = "RealSaS.SourceRayAppearanceEvidence.v1"
FAMILY = "RAY_COMPOSITING"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hash(value: Any, code: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise QualificationError(code)
    return text


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(raw)


def _load_bundle(ctx: dict[str, Any]) -> dict[str, Any]:
    manifest = ctx["run_manifest"]
    bundle = manifest.get("source_ray_appearance_court_bundle")
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_BUNDLE_REQUIRED")
    return bundle


def _validate_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    court = bundle.get("court")
    evidence = bundle.get("evidence")
    if not isinstance(court, dict) or court.get("schema") != COURT_SCHEMA:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_SCHEMA")
    if not isinstance(evidence, dict):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_EVIDENCE_REQUIRED")

    court_sha = _require_hash(bundle.get("court_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_COURT_HASH_REQUIRED")
    if _canonical_hash(court) != court_sha:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_HASH_DRIFT")

    if str(court.get("status") or "") != "PASS_SOURCE_EQUIVALENT_SPARSE_APPEARANCE_CANDIDATE":
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_COURT_NOT_PASS")

    decision = court.get("qualification_decision")
    if not isinstance(decision, dict):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_DECISION_REQUIRED")
    if str(decision.get("family") or "") != FAMILY or decision.get("oracle_eligible") is not True:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_ORACLE_INELIGIBLE")

    scope = court.get("scope")
    if not isinstance(scope, dict) or str(scope.get("domain") or "") != "SPARSE_MEASURED_RAYS_ONLY":
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_SCOPE_DRIFT")
    if scope.get("global_sampler_qualified") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_GLOBAL_SCOPE_FORBIDDEN")

    if court.get("training_authorized") is not False or court.get("product_authority_changed") is not False:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_AUTHORITY_ESCALATION")

    record_hashes_raw = evidence.get("record_hashes")
    if not isinstance(record_hashes_raw, list) or not record_hashes_raw:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_RECORD_HASHES_REQUIRED")
    record_hashes = tuple(_require_hash(v, "SOURCE_RAY_APPEARANCE_PLATFORM_RECORD_HASH_INVALID") for v in record_hashes_raw)
    if len(set(record_hashes)) != len(record_hashes):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_RECORD_HASH_DUPLICATE")

    expected_count = int(scope.get("valid_hit_count", -1))
    if expected_count != len(record_hashes):
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_RECORD_COUNT_DRIFT")

    evidence_sha = _require_hash(bundle.get("evidence_sha256"), "SOURCE_RAY_APPEARANCE_PLATFORM_EVIDENCE_HASH_REQUIRED")
    if _canonical_hash(evidence) != evidence_sha:
        raise QualificationError("SOURCE_RAY_APPEARANCE_PLATFORM_EVIDENCE_HASH_DRIFT")

    return court, evidence, record_hashes


def admit_source_ray_appearance_v1(ctx: dict[str, Any]) -> dict[str, Any]:
    bundle = _load_bundle(ctx)
    court, evidence, record_hashes = _validate_bundle(bundle)

    root = ctx["run_root"] / "artifacts" / str((ctx.get("stage") or {}).get("id") or "SOURCE_RAY_APPEARANCE_PLATFORM_ADMISSION")
    qualification = {
        "schema": QUALIFICATION_SCHEMA,
        "family": FAMILY,
        "scope": dict(court["scope"]),
        "court_sha256": str(bundle["court_sha256"]),
        "evidence_payload_sha256": str(bundle["evidence_sha256"]),
        "source_equivalent": True,
        "replay_deterministic": True,
        "coverage_qualified": True,
        "oracle_eligible": True,
        "record_count": len(record_hashes),
        "record_set_sha256": _canonical_hash(sorted(record_hashes)),
        "global_sampler_qualified": False,
        "training_authorized": False,
        "student_target_admitted": False,
        "product_authority_changed": False,
        "genericity_claimed": False,
    }
    admitted_evidence = {
        "schema": EVIDENCE_SCHEMA,
        "family": FAMILY,
        "scope": dict(court["scope"]),
        "record_hashes": list(record_hashes),
        "measurement_payload": evidence.get("measurement_payload"),
        "source_equivalence_witness_sha256": str(bundle["court_sha256"]),
        "global_sampler_qualified": False,
        "training_authorized": False,
        "product_authority_changed": False,
    }

    outputs = [
        write_json(
            root / "source_ray_appearance_qualification.json",
            qualification,
            authority_class="QUALIFIED_SPARSE_SOURCE_RAY_APPEARANCE",
            schema=QUALIFICATION_SCHEMA,
        ),
        write_json(
            root / "source_ray_appearance_evidence.json",
            admitted_evidence,
            authority_class="QUALIFIED_SPARSE_SOURCE_RAY_APPEARANCE_EVIDENCE",
            schema=EVIDENCE_SCHEMA,
        ),
    ]
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "schema": "RealSaS.SourceRayAppearancePlatformAdmissionDiagnostics.v1",
            "record_count": len(record_hashes),
            "global_sampler_qualified": False,
            "training_authorized": False,
            "product_authority_changed": False,
        },
    }
