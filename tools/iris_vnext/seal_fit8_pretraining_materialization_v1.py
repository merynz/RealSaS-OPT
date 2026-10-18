from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.IRISVNextFit8PretrainingMaterializationReceipt.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_PRETRAINING_MATERIALIZATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def seal_materialization(
    *,
    expected_source_sha: str,
    candidate_roundtrip_receipt: Path,
    relation_manifest: Path,
    boundary_manifest: Path,
    quarantine_audit_report: Path,
    migration_receipt: Path,
) -> dict[str, Any]:
    source = expected_source_sha.strip().lower()
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_SOURCE_SHA_INVALID")

    candidate = _load(candidate_roundtrip_receipt)
    relation = _load(relation_manifest)
    boundary = _load(boundary_manifest)
    quarantine = _load(quarantine_audit_report)
    migration = _load(migration_receipt)

    if candidate.get("schema") != "RealSaS.IRISVNextFit8CandidateRoundTripReceipt.v1":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_CANDIDATE_SCHEMA_INVALID")
    if candidate.get("status") != "ROUNDTRIP_VERIFIED_CANDIDATE":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_CANDIDATE_NOT_VERIFIED")
    if candidate.get("expected_source_sha") != source:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_CANDIDATE_SOURCE_DRIFT")
    if candidate.get("roundtrip_sha256_equal") is not True:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_ROUNDTRIP_REQUIRED")
    if candidate.get("candidate_authorized_for_training") is not False:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_PREMATURE_AUTHORIZATION")

    if relation.get("schema") != "RealSaS.IRISVNextFit8RelationTruthManifest.v2":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_RELATION_SCHEMA_INVALID")
    if boundary.get("schema") != "RealSaS.IRISVNextFit8BoundaryCausalityManifest.v1":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_BOUNDARY_SCHEMA_INVALID")
    if len(relation.get("subjects") or ()) != 8 or len(boundary.get("subjects") or ()) != 8:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_FIT8_CARDINALITY_INVALID")
    if boundary.get("relation_truth_v2_manifest_sha256") != _sha(relation_manifest):
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_BOUNDARY_RELATION_BINDING_DRIFT")

    if migration.get("schema") != "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_MIGRATION_SCHEMA_INVALID")
    if migration.get("status") != "PASS" or int(migration.get("training_steps_performed", -1)) != 0:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_MIGRATION_INVALID")
    if migration.get("source_v8", {}).get("checkpoint_sha256") != "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_V8_CHECKPOINT_DRIFT")

    if quarantine.get("schema") != "RealSaS.IRISVNextFit8RemotePreparationAudit.v1":
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_QUARANTINE_SCHEMA_INVALID")
    if quarantine.get("expected_current_source_sha") != source:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_QUARANTINE_SOURCE_DRIFT")
    if quarantine.get("remote_consumption_authorized") is not False:
        raise ValueError("VNEXT_PRETRAINING_MATERIALIZATION_QUARANTINE_AUTHORITY_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "MATERIALIZED_AND_BYTE_VERIFIED__TRAINING_NOT_STARTED",
        "source_commit_sha": source,
        "candidate_roundtrip_receipt_sha256": _sha(candidate_roundtrip_receipt),
        "candidate_remote": candidate["candidate_remote"],
        "relation_truth_v2_manifest_sha256": _sha(relation_manifest),
        "relation_truth_v2_manifest_value_sha256": relation.get("manifest_value_sha256"),
        "boundary_causality_v1_manifest_sha256": _sha(boundary_manifest),
        "boundary_causality_v1_manifest_value_sha256": boundary.get("manifest_value_sha256"),
        "quarantine_audit_report_sha256": _sha(quarantine_audit_report),
        "v8_relation_head_to_v2_migration_receipt_sha256": _sha(migration_receipt),
        "v8_relation_head_v2_artifact_sha256": migration.get("target", {}).get("sha256"),
        "quarantine_remote_status": quarantine.get("status"),
        "quarantined_remote_consumed": False,
        "candidate_roundtrip_verified": True,
        "truth_materialized": True,
        "teacher_inputs_in_inference_forward": False,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
        "training_authorized": False,
        "remaining_before_training": [
            "ARTIFACT_BOUND_FOCUSED_EXPERIMENT_PREREGISTRATION",
            "PRETRAINING_READINESS_AUDIT",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--candidate-roundtrip-receipt", type=Path, required=True)
    p.add_argument("--relation-manifest", type=Path, required=True)
    p.add_argument("--boundary-manifest", type=Path, required=True)
    p.add_argument("--quarantine-audit-report", type=Path, required=True)
    p.add_argument("--migration-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = seal_materialization(
        expected_source_sha=args.expected_source_sha,
        candidate_roundtrip_receipt=args.candidate_roundtrip_receipt.resolve(),
        relation_manifest=args.relation_manifest.resolve(),
        boundary_manifest=args.boundary_manifest.resolve(),
        quarantine_audit_report=args.quarantine_audit_report.resolve(),
        migration_receipt=args.migration_receipt.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_FIT8_PRETRAINING_MATERIALIZATION_SEAL_PASS",
        f"source={report['source_commit_sha']}",
        "training_authorized=false",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
