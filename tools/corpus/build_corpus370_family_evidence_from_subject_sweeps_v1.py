#!/usr/bin/env python3
from __future__ import annotations

"""Normalize current per-subject truth-family sweep courts into Corpus370 evidence.

This adapter is deliberately conservative. It accepts only exact whitelisted court
schemas/statuses and exact source SHA binding. It NEVER maps the historical/local
`SIGNED_DISTANCE_OCCUPANCY` component result to `GLOBAL_SOLID_SDF`; global solid
SDF is admitted only by the separate explicit-solid composite adapter.

Graphics measured-domain courts are also intentionally not promoted here because
the corpus-level student-observability gate remains separate.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"

COURTS = {
    "source_observation": (
        "RealSaS.SourceObservationAuthorityCourt.v1",
        "PASS_SOURCE_OBSERVATION_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
    ),
    "geometry": (
        "RealSaS.ComponentGeometryAuthorityCourt.v1",
        "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
    ),
    "mechanical_surface": (
        "RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1",
        "PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
    ),
    "rig_skin_importer": (
        "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1",
        "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE",
    ),
    "animation_replay": (
        "RealSaS.GltfAnimationCanonicalReplayCourt.v1",
        "PASS_GLTF_ANIMATION_CANONICAL_REPLAY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
    ),
    "causal_response": (
        "RealSaS.CausalResponseAuthorityCourt.v1",
        "PASS_CAUSAL_RESPONSE_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
    ),
    "tessa_v5": (
        "RealSaS.CorpusMechanicalTESSATeacherCourt.v5",
        "PASS_CORPUS_MECHANICAL_TESSA_V5_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _bound_json(ref: dict[str, Any], code: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    require(path.is_file() and len(expected) == 64 and sha256(path) == expected, code)
    value = json.loads(path.read_text())
    require(isinstance(value, dict), code + "_JSON")
    return path, value


def _court(receipt: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    path, value = _bound_json(dict((receipt.get("courts") or {}).get(key) or {}), "CORPUS370_SWEEP_COURT_" + key.upper())
    schema, status = COURTS[key]
    require(value.get("schema") == schema and value.get("status") == status, "CORPUS370_SWEEP_COURT_CONTRACT_" + key.upper())
    require((value.get("authority") or {}).get("training_authorized") is not True, "CORPUS370_SWEEP_PREMATURE_TRAINING_AUTHORITY")
    return path, value


def _claim(path: Path, court: dict[str, Any], *, status: str = "PASS", reasons: list[str] | None = None, notes: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason_codes": list(reasons or ()),
        "evidence_refs": [
            {
                "schema": str(court["schema"]),
                "sha256": sha256(path),
                "path": str(path),
            }
        ],
        "court": {
            "schema": str(court["schema"]),
            "sha256": sha256(path),
            "path": str(path),
            "status": "PASS",
        },
        "notes": notes,
    }


def _qualified(outcome: Any) -> bool:
    return str(outcome or "") in {"QUALIFIED_CORE_CANDIDATE", "QUALIFIED_MASKED_CANDIDATE", "QUALIFIED_CONSUMER_CANDIDATE"}


def _masked_if_unavailable(path: Path, court: dict[str, Any], outcome: Any, reason: str) -> dict[str, Any] | None:
    if _qualified(outcome):
        return _claim(path, court, notes="Current court qualifies this family; internal row masks remain authoritative where present.")
    if str(outcome or "") == "UNAVAILABLE_CANDIDATE":
        return _claim(path, court, status="MASKED", reasons=[reason], notes="Current independent court found no consumable target rows for this family.")
    return None


def normalize_subject(subject_id: str, source_sha: str, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("schema") == "RealSaS.SubjectTruthFamilySweepReceipt.v1", "CORPUS370_SWEEP_RECEIPT_SCHEMA")
    require(receipt.get("status") == "PASS_SUBJECT_TRUTH_FAMILY_SWEEP__FINAL_ADMISSION_PENDING", "CORPUS370_SWEEP_RECEIPT_STATUS")
    require(str(receipt.get("subject_id") or "") == subject_id, "CORPUS370_SWEEP_RECEIPT_SUBJECT")
    source_cache = dict(receipt.get("source_cache") or {})
    require(str(source_cache.get("actual_sha256") or "") == source_sha, "CORPUS370_SWEEP_SOURCE_SHA")
    require((receipt.get("authority") or {}).get("training_authorized") is not True, "CORPUS370_SWEEP_RECEIPT_PREAUTHORIZED")

    families: dict[str, Any] = {}

    so_path, so = _court(receipt, "source_observation")
    so_out = dict(so.get("family_candidate_outcomes") or {})
    require(so.get("source_sha256") == source_sha, "CORPUS370_SOURCE_OBSERVATION_SOURCE_SHA")
    for family in ("SOURCE_IMPORT", "CAMERA_OBSERVATION"):
        require(so_out.get(family) == "QUALIFIED_CORE_CANDIDATE", "CORPUS370_SOURCE_OBSERVATION_OUTCOME_" + family)
        families[family] = _claim(so_path, so)

    geo_path, geo = _court(receipt, "geometry")
    geo_out = dict(geo.get("family_candidate_outcomes") or {})
    surface = _masked_if_unavailable(geo_path, geo, geo_out.get("SURFACE_GEOMETRY"), "NO_VALID_SURFACE_GEOMETRY_ROWS")
    normals = _masked_if_unavailable(geo_path, geo, geo_out.get("ORIENTED_NORMALS"), "NO_VALID_SOURCE_ORIENTED_NORMAL_ROWS")
    if surface is not None:
        families["SURFACE_GEOMETRY_UDF"] = surface
    if normals is not None:
        families["SURFACE_NORMALS"] = normals
    # Deliberately ignore geo_out['SIGNED_DISTANCE_OCCUPANCY'] here. Component
    # sign is not global ExplicitSolid authority under the current corpus policy.

    mech_path, mech = _court(receipt, "mechanical_surface")
    mech_out = dict(mech.get("family_candidate_outcomes") or {})
    require(_qualified(mech_out.get("SURFACE_TOPOLOGY")), "CORPUS370_MECHANICAL_SURFACE_OUTCOME")
    families["MECHANICAL_SURFACE_TOPOLOGY"] = _claim(mech_path, mech)

    rig_path, rig = _court(receipt, "rig_skin_importer")
    rig_out = dict(rig.get("family_candidate_outcomes") or {})
    for family in ("RIG_HIERARCHY", "SKIN_WEIGHTS"):
        require(rig_out.get(family) == "QUALIFIED_CORE_CANDIDATE", "CORPUS370_RIG_SKIN_OUTCOME_" + family)
        families[family] = _claim(rig_path, rig)

    if "animation_replay" in (receipt.get("courts") or {}):
        anim_path, anim = _court(receipt, "animation_replay")
        anim_out = dict(anim.get("family_candidate_outcomes") or {})
        require(_qualified(anim_out.get("ANIMATION_KINEMATICS")), "CORPUS370_ANIMATION_REPLAY_OUTCOME")
        families["ANIMATION_KINEMATICS"] = _claim(anim_path, anim)

    response_path, response = _court(receipt, "causal_response")
    response_out = dict(response.get("family_candidate_outcomes") or {})
    require(_qualified(response_out.get("CAUSAL_RESPONSE")), "CORPUS370_CAUSAL_RESPONSE_OUTCOME")
    families["CAUSAL_RESPONSE"] = _claim(response_path, response)

    tessa_path, tessa = _court(receipt, "tessa_v5")
    require((tessa.get("authority") or {}).get("tessa_mechanical_surface_consumer_court_passed") is True, "CORPUS370_TESSA_V5_AUTHORITY")
    families["TESSA_TEACHER"] = _claim(
        tessa_path,
        tessa,
        notes="TESSA V5 current consumer court passed. Selected SOURCE_FALLBACK is allowed only when the current measured-all-kinds contract and all mechanical tests pass; legacy fallback is not authority.",
    )

    return {
        "subject_id": subject_id,
        "source_sha256": source_sha,
        "families": families,
        "subject_sweep_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
    }


def build(roster_path: Path, sweep_root: Path) -> dict[str, Any]:
    roster = json.loads(roster_path.read_text())
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "CORPUS370_SWEEP_ADAPTER_ROSTER_SCHEMA")
    rows = []
    missing = []
    for source in roster.get("subjects") or ():
        subject_id = str(source["subject_id"])
        receipt_path = (sweep_root / "subjects" / subject_id / "SUBJECT_TRUTH_FAMILY_SWEEP_RECEIPT_V1.json").resolve()
        if not receipt_path.is_file():
            missing.append(subject_id)
            continue
        rows.append(normalize_subject(subject_id, str(source["source_sha256"]), receipt_path))
    return {
        "schema": SCHEMA,
        "status": "CURRENT_SUBJECT_SWEEP_EVIDENCE_NORMALIZED__GLOBAL_SOLID_AND_GRAPHICS_ADMISSION_SEPARATE",
        "subject_count": len(rows),
        "candidate_subject_count": int(roster.get("subject_count", 0)),
        "missing_subject_count": len(missing),
        "missing_subject_ids": missing,
        "subjects": rows,
        "explicitly_not_mapped": [
            "COMPONENT_SIGNED_DISTANCE_OCCUPANCY_TO_GLOBAL_SOLID_SDF",
            "GRAPHICS_MEASUREMENT_TO_STUDENT_SUPERVISION_BEFORE_OBSERVABILITY_ADMISSION",
        ],
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--subject-sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.exact_source_roster, args.subject_sweep_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("schema", "status", "subject_count", "missing_subject_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
