#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for the Stage48-bound graphics-family authority.

The court verifies the canonical Stage48 witness and the exact normalized adapter
contract, then delegates numerical measured-domain verification only to the already
independent V1 court.  It never imports the V2 producer or V1 producer.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.validate_presentation_family_authority_v1 import validate as validate_v1

PARENT_SCHEMA = "RealSaS.PresentationFamilyAuthority.v2"
PARENT_STATUS = "PASS_PRESENTATION_FAMILY_STAGE48_BOUND_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
WITNESS_SCHEMA = "RealSaS.Truth139Stage48RendererWitness.v1"
WITNESS_STATUS = "PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
SCHEMA = "RealSaS.PresentationFamilyAuthorityCourt.v2"
STATUS = "PASS_PRESENTATION_FAMILY_STAGE48_BOUND_INDEPENDENT_COURT__STUDENT_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    if not path.is_file() or sha256(path) != str(row.get("sha256") or ""):
        raise ValueError(code)
    return path


def _expected_normalized(witness: dict, arrays_path: Path) -> dict:
    return {
        "schema": "RealSaS.PresentationResponseTruthCandidate.v1",
        "subject_id": witness["subject_id"],
        "array_sha256": sha256(arrays_path),
        "policy": {
            "student_observable_contribution_floor": float(
                (witness.get("policy") or {}).get("student_observable_contribution_floor", 1.0e-6)
            )
        },
        "renderer_subject_qualification": witness["renderer_subject_qualification"],
        "normalization_origin": "DERIVED_EXACTLY_FROM_STAGE48_RENDERER_WITNESS_V1",
        "stage48_identity_sha256": witness.get("stage48_identity_sha256"),
    }


def validate(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("PRESENTATION_V2_COURT_PARENT")
    parents = dict(manifest.get("parents") or {})
    witness_path = bound(parents.get("stage48_witness"), "PRESENTATION_V2_COURT_WITNESS_DRIFT")
    arrays_path = bound(parents.get("stage48_arrays"), "PRESENTATION_V2_COURT_ARRAYS_DRIFT")
    normalized_path = bound(parents.get("normalized_summary"), "PRESENTATION_V2_COURT_NORMALIZED_DRIFT")
    v1_manifest_path = bound(parents.get("measured_domain_v1_manifest"), "PRESENTATION_V2_COURT_V1_MANIFEST_DRIFT")
    bound(parents.get("measured_domain_v1_arrays"), "PRESENTATION_V2_COURT_V1_ARRAYS_DRIFT")

    witness = json.loads(witness_path.read_text())
    normalized = json.loads(normalized_path.read_text())
    if witness.get("schema") != WITNESS_SCHEMA or witness.get("status") != WITNESS_STATUS:
        raise ValueError("PRESENTATION_V2_COURT_WITNESS_STATUS")
    subject = str(manifest.get("subject_id") or "")
    if not subject or witness.get("subject_id") != subject:
        raise ValueError("PRESENTATION_V2_COURT_SUBJECT")
    if witness.get("qualification_minted") is not False or witness.get("training_performed") is not False:
        raise ValueError("PRESENTATION_V2_COURT_WITNESS_ESCALATION")
    if sha256(arrays_path) != str((witness.get("arrays") or {}).get("sha256") or ""):
        raise ValueError("PRESENTATION_V2_COURT_ARRAY_WITNESS_BINDING")

    qualification = dict(witness.get("renderer_subject_qualification") or {})
    local = dict(qualification.get("local_qualification_decision") or {})
    family_input = dict(qualification.get("family_qualification_input") or {})
    if (
        qualification.get("status") != "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE"
        or qualification.get("source_equivalent") is not True
        or qualification.get("packed_replay_exact") is not True
        or qualification.get("source_reference_replay_exact") is not True
        or qualification.get("student_supervision_authorized") is not False
        or local.get("oracle_eligible") is not True
        or family_input.get("measurement_status") != "MEASURED"
        or family_input.get("replay_deterministic") is not True
        or family_input.get("source_equivalent") is not True
    ):
        raise ValueError("PRESENTATION_V2_COURT_RENDERER_QUALIFICATION")

    if normalized != _expected_normalized(witness, arrays_path):
        raise ValueError("PRESENTATION_V2_COURT_NORMALIZED_SUMMARY")

    replays = dict(witness.get("replays") or {})
    v1_manifest = json.loads(v1_manifest_path.read_text())
    v1_parents = dict(v1_manifest.get("parents") or {})
    mapping = {
        "packed_a": "packed_a",
        "packed_b": "packed_b",
        "source_reference_a": "source_reference_a",
        "source_reference_b": "source_reference_b",
    }
    for witness_key, v1_key in mapping.items():
        expected = dict(replays.get(witness_key) or {})
        actual = dict(v1_parents.get(v1_key) or {})
        if expected.get("sha256") != actual.get("sha256"):
            raise ValueError("PRESENTATION_V2_COURT_REPLAY_PARENT:" + witness_key)
        expected_path = Path(str(expected.get("path") or "")).resolve()
        actual_path = Path(str(actual.get("path") or "")).resolve()
        if expected_path != actual_path or not expected_path.is_file() or sha256(expected_path) != expected.get("sha256"):
            raise ValueError("PRESENTATION_V2_COURT_REPLAY_BYTES:" + witness_key)

    v1_court = validate_v1(v1_manifest_path)
    if v1_court.get("status") != "PASS_PRESENTATION_FAMILY_INDEPENDENT_COURT__STUDENT_ADMISSION_SEPARATE":
        raise ValueError("PRESENTATION_V2_COURT_V1_NOT_PASS")
    if (v1_court.get("authority") or {}).get("student_supervision_authorized") is not False:
        raise ValueError("PRESENTATION_V2_COURT_V1_STUDENT_ESCALATION")

    declared_stage48 = dict(manifest.get("stage48") or {})
    if declared_stage48.get("identity_sha256") != witness.get("stage48_identity_sha256"):
        raise ValueError("PRESENTATION_V2_COURT_STAGE48_IDENTITY")
    if int(declared_stage48.get("pass_ray_count", -1)) != int((witness.get("pass_domain") or {}).get("packed_a_pass_ray_count", -2)):
        raise ValueError("PRESENTATION_V2_COURT_PASS_RAY_COUNT")
    if int(declared_stage48.get("measured_hit_count", -1)) != int((witness.get("counts") or {}).get("measured_composite_hit_count", -2)):
        raise ValueError("PRESENTATION_V2_COURT_HIT_COUNT")
    if int(declared_stage48.get("measured_pair_count", -1)) != int((witness.get("counts") or {}).get("measured_composite_pair_count", -2)):
        raise ValueError("PRESENTATION_V2_COURT_PAIR_COUNT")

    if dict(manifest.get("family_candidate_outcomes") or {}) != dict(v1_court.get("family_candidate_outcomes") or {}):
        raise ValueError("PRESENTATION_V2_COURT_FAMILY_OUTCOME")
    authority = dict(manifest.get("authority") or {})
    if authority.get("student_supervision_authorized") is not False or authority.get("training_authorized") is not False:
        raise ValueError("PRESENTATION_V2_COURT_PREMATURE_TRAINING_AUTHORITY")

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject,
        "stage48_identity_sha256": witness.get("stage48_identity_sha256"),
        "counts": dict(v1_court.get("counts") or {}),
        "family_candidate_outcomes": dict(v1_court.get("family_candidate_outcomes") or {}),
        "authority": {
            "independent_court_passed": True,
            "student_supervision_authorized": False,
            "final_family_authority_minted": False,
            "training_authorized": False,
        },
        "parent_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.manifest)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
