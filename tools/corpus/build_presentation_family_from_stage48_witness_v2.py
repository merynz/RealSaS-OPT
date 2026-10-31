#!/usr/bin/env python3
from __future__ import annotations

"""Bind graphics measured-domain extraction to the canonical Stage48 witness.

The Stage48 witness is the only parent locator.  No downstream sweep is allowed to
infer PACKED/SOURCE_REFERENCE paths from directory names.  A small normalized
summary is derived from the witness solely to reuse the V1 measured-domain
extractor; V2 authority hash-binds both the witness and that extraction.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.build_presentation_family_authority_v1 import build as build_v1

SCHEMA = "RealSaS.PresentationFamilyAuthority.v2"
STATUS = "PASS_PRESENTATION_FAMILY_STAGE48_BOUND_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
WITNESS_SCHEMA = "RealSaS.Truth139Stage48RendererWitness.v1"
WITNESS_STATUS = "PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def ref(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return {"path": str(path), "sha256": sha256(path)}


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    if not path.is_file() or sha256(path) != str(row.get("sha256") or ""):
        raise ValueError(code)
    return path


def _normalized_summary(witness: dict, arrays_path: Path) -> dict:
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


def build(stage48_witness_path: Path, output_dir: Path) -> dict:
    stage48_witness_path = stage48_witness_path.resolve()
    output_dir = output_dir.resolve()
    witness = json.loads(stage48_witness_path.read_text())
    if witness.get("schema") != WITNESS_SCHEMA or witness.get("status") != WITNESS_STATUS:
        raise ValueError("PRESENTATION_V2_STAGE48_WITNESS_STATUS")
    subject_id = str(witness.get("subject_id") or "")
    if not subject_id:
        raise ValueError("PRESENTATION_V2_SUBJECT")
    if witness.get("qualification_minted") is not False or witness.get("training_performed") is not False:
        raise ValueError("PRESENTATION_V2_STAGE48_AUTHORITY_ESCALATION")

    arrays_path = bound(witness.get("arrays"), "PRESENTATION_V2_ARRAYS_DRIFT")
    replays = dict(witness.get("replays") or {})
    packed_a = bound(replays.get("packed_a"), "PRESENTATION_V2_PACKED_A_DRIFT")
    packed_b = bound(replays.get("packed_b"), "PRESENTATION_V2_PACKED_B_DRIFT")
    source_a = bound(replays.get("source_reference_a"), "PRESENTATION_V2_SOURCE_A_DRIFT")
    source_b = bound(replays.get("source_reference_b"), "PRESENTATION_V2_SOURCE_B_DRIFT")

    qualification = dict(witness.get("renderer_subject_qualification") or {})
    if (
        qualification.get("status") != "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE"
        or qualification.get("source_equivalent") is not True
        or qualification.get("packed_replay_exact") is not True
        or qualification.get("source_reference_replay_exact") is not True
        or qualification.get("student_supervision_authorized") is not False
    ):
        raise ValueError("PRESENTATION_V2_RENDERER_QUALIFICATION")
    local = dict(qualification.get("local_qualification_decision") or {})
    if local.get("oracle_eligible") is not True:
        raise ValueError("PRESENTATION_V2_ORACLE_NOT_ELIGIBLE")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "STAGE48_NORMALIZED_PRESENTATION_SUMMARY_V1.json"
    normalized = _normalized_summary(witness, arrays_path)
    normalized_path.write_text(json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n")

    v1_dir = output_dir / "measured_domain_v1"
    v1 = build_v1(
        subject_id,
        normalized_path,
        arrays_path,
        packed_a,
        packed_b,
        source_a,
        source_b,
        v1_dir,
    )
    v1_manifest_path = v1_dir / "PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V1.json"
    v1_output_path = Path(v1["output"]["path"]).resolve()

    family = dict(v1.get("family_candidate_outcomes") or {})
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "parents": {
            "stage48_witness": ref(stage48_witness_path),
            "stage48_arrays": ref(arrays_path),
            "normalized_summary": ref(normalized_path),
            "measured_domain_v1_manifest": ref(v1_manifest_path),
            "measured_domain_v1_arrays": ref(v1_output_path),
        },
        "stage48": {
            "identity_sha256": witness.get("stage48_identity_sha256"),
            "pass_ray_count": int((witness.get("pass_domain") or {}).get("packed_a_pass_ray_count", 0)),
            "measured_hit_count": int((witness.get("counts") or {}).get("measured_composite_hit_count", 0)),
            "measured_pair_count": int((witness.get("counts") or {}).get("measured_composite_pair_count", 0)),
            "source_equivalent": True,
            "replay_deterministic": True,
        },
        "family_candidate_outcomes": family,
        "observability": dict(v1.get("observability") or {}),
        "policy": {
            "path_discovery": "ONLY_HASH_BOUND_STAGE48_WITNESS_REFS",
            "measured_truth_is_not_student_authority": True,
            "compiler_oracle_preserved": True,
            "student_projection_receipt_required": True,
            "final_split_required_before_student_supervision_seal": True,
        },
        "authority": {
            "stage48_bound_measurement_candidate": True,
            "independent_court_required": True,
            "student_supervision_authorized": False,
            "final_family_authority_minted": False,
            "training_authorized": False,
        },
    }
    manifest["identity_sha256"] = digest(manifest)
    manifest_path = output_dir / "PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V2.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage48-witness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.stage48_witness, args.output_dir)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "stage48": result["stage48"]}, sort_keys=True))


if __name__ == "__main__":
    main()
