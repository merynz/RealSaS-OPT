#!/usr/bin/env python3
"""Audit exact Stage50 352 source-geometry courts as per-head evidence candidates.

This is a strict readback of *existing independent measurements*, not a new
teacher, a final family authority court, a split, or a V3.2 training manifest.
In particular the component-local solid-angle sign is NOT global solid SDF.
Unknown/unmeasured response, drawable ownership, graphics and TESSA targets
are represented as UNKNOWN, not zero-valued targets or negative labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370V5HeadTargetReadinessReadback.v1"
V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
REST_SHA = "384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8"
GEOMETRY_SCHEMA = "RealSaS.Corpus370V5ComponentGeometryIndependentSweep.v1"
V5_SCHEMA = "RealSaS.Corpus370PreparedObservationPackets.v5"
COURT_SCHEMA = "RealSaS.ComponentGeometryAuthorityCourt.v1"
COURT_STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
GEOMETRY_FAMILIES = (
    "SURFACE_GEOMETRY", "SURFACE_TOPOLOGY", "ORIENTED_NORMALS",
    "SIGNED_DISTANCE_OCCUPANCY",
)
CANDIDATE = ("QUALIFIED_CORE_CANDIDATE", "QUALIFIED_MASKED_CANDIDATE", "UNAVAILABLE_CANDIDATE")
UNKNOWN_TARGETS = (
    "IRIS_GLOBAL_SOLID_SDF", "IRIS_GLOBAL_OCCUPANCY",
    "IRIS_RESPONSE", "IRIS_MECHANICAL_OWNERSHIP",
    "IRIS_DRAWABLE_OWNERSHIP", "IRIS_GRAPHICS_RELATIONS",
    "IRIS_VIEW_OCCLUSION", "IRIS_ALPHA_COMPOSITING",
    "TESSA_MECHANICAL_TEACHER", "TESSA_FREE_RUNNING_CONSEQUENCE",
)
HEAD_FAMILY = {
    "IRIS_SURFACE_UDF": "SURFACE_GEOMETRY",
    "IRIS_SOURCE_NORMALS": "ORIENTED_NORMALS",
    "IRIS_SOURCE_TOPOLOGY": "SURFACE_TOPOLOGY",
    "IRIS_COMPONENT_LOCAL_SIGN_ONLY": "SIGNED_DISTANCE_OCCUPANCY",
}


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ishex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _read_ref(ref: dict[str, Any], code: str) -> dict[str, Any]:
    require(isinstance(ref, dict) and bool(ref.get("path")) and ishex(ref.get("sha256")),
            code + "_REF")
    path = Path(ref["path"])
    require(path.is_file() and sha256(path) == ref["sha256"], code + "_SHA")
    return json.loads(path.read_text())


def verify_actual_court_receipts(geometry: dict[str, Any]) -> None:
    """Require exact independent per-subject court bytes on the worker filesystem."""
    for sid, row in geometry["subject_results"].items():
        require(row["status"] == "PASS_COMPONENT_GEOMETRY_COURT", "READBACK_FAIL_GEOMETRY_COURT:" + sid)
        manifest = _read_ref(row["manifest"], "READBACK_MANIFEST:" + sid)
        court = _read_ref(row["court"], "READBACK_COURT:" + sid)
        require(court.get("schema") == COURT_SCHEMA and court.get("status") == COURT_STATUS,
                "READBACK_COURT_CONTRACT:" + sid)
        authority = court.get("authority") or {}
        require(authority.get("independent_court_passed") is True
                and authority.get("final_subject_disposition_minted") is False
                and authority.get("training_authorized") is False,
                "READBACK_COURT_PROMOTION:" + sid)
        require(manifest.get("subject_id") == sid and court.get("subject_id") == sid,
                "READBACK_COURT_SUBJECT_ID:" + sid)
        require(court.get("family_candidate_outcomes") == row["family_candidate_outcomes"]
                and court.get("counts") == row["counts"],
                "READBACK_COURT_INDEX_DRIFT:" + sid)
        require(row.get("training_authorized") is False, "READBACK_COURT_TRAINING:" + sid)


def build(v5: dict[str, Any], geometry: dict[str, Any], *, v5_sha: str,
          geometry_sha: str) -> dict[str, Any]:
    require(v5_sha == V5_SHA, "READBACK_SOURCE_INDEX_SHA")
    require(v5.get("schema") == V5_SCHEMA and v5.get("subject_count") == 370
            and v5.get("prepared_subject_count") == 352
            and v5.get("failure_count") == 18
            and v5.get("training_authorized") is False,
            "READBACK_V5_COHORT")
    qualified = {r["subject_id"]: r for r in v5["subjects"]}
    excluded = {r["subject_id"] for r in v5["failures"]}
    require(len(qualified) == 352 and len(excluded) == 18
            and not excluded.intersection(qualified)
            and len(qualified) + len(excluded) == 370,
            "READBACK_V5_SUBJECT_PARTITION")
    require(all(ishex(row.get("source_sha256")) for row in qualified.values()),
            "READBACK_V5_SOURCE_SHA")
    require(geometry.get("schema") == GEOMETRY_SCHEMA
            and geometry.get("status") == "ALL_352_COURTS_ACCOUNTED__NOT_TRAINING"
            and geometry.get("universe_subject_count") == 370
            and geometry.get("observation_qualified_count") == 352
            and geometry.get("observation_excluded_count") == 18
            and geometry.get("training_authorized") is False,
            "READBACK_GEOMETRY_SCOPE")
    identity = geometry.get("identity") or {}
    require(identity.get("v5_index_sha256") == V5_SHA
            and identity.get("rest_index_sha256") == REST_SHA,
            "READBACK_GEOMETRY_PARENT_DRIFT")
    authority = geometry.get("authority") or {}
    for key in ("model_targets_issued", "global_union_sign_proven",
                "family_admission_minted", "training_authorized"):
        require(authority.get(key) is False, "READBACK_GEOMETRY_FORGED_AUTHORITY:" + key)
    require(authority.get("candidate_only") is True, "READBACK_GEOMETRY_NOT_CANDIDATE")
    require(set(geometry.get("excluded_observation_subject_ids") or []) == excluded,
            "READBACK_GEOMETRY_EXCLUSIONS")
    rows = geometry.get("subject_results") or {}
    require(set(rows) == set(qualified), "READBACK_GEOMETRY_MISSING_OR_EXTRA_SUBJECT")
    dispositions: dict[str, Counter] = {name: Counter() for name in GEOMETRY_FAMILIES}
    target_counts: dict[str, Counter] = {name: Counter() for name in HEAD_FAMILY}
    queries = total_signed = surface_faces = valid_faces = 0
    subject_rows = []
    for sid in sorted(qualified):
        row = rows[sid]
        require(row.get("status") == "PASS_COMPONENT_GEOMETRY_COURT"
                and row.get("training_authorized") is False,
                "READBACK_GEOMETRY_SUBJECT_COURT:" + sid)
        src = row.get("input") or {}
        require(src.get("source_sha256") == qualified[sid]["source_sha256"],
                "READBACK_GEOMETRY_SOURCE_IDENTITY:" + sid)
        family = row.get("family_candidate_outcomes") or {}
        require(set(family) == set(GEOMETRY_FAMILIES)
                and all(value in CANDIDATE for value in family.values()),
                "READBACK_GEOMETRY_FAMILY_ENUM:" + sid)
        counts = row.get("counts") or {}
        require(all(type(counts.get(key)) is int for key in
                    ("query_count", "signed_query_count", "face_count", "valid_face_count")),
                "READBACK_GEOMETRY_COUNT_TYPE:" + sid)
        require(counts["query_count"] == 8192
                and 0 <= counts["signed_query_count"] <= counts["query_count"]
                and 0 <= counts["valid_face_count"] <= counts["face_count"]
                and counts["face_count"] > 0,
                "READBACK_GEOMETRY_COUNT_RANGE:" + sid)
        has_sign = counts["signed_query_count"] > 0
        require((family["SIGNED_DISTANCE_OCCUPANCY"] == "UNAVAILABLE_CANDIDATE") == (not has_sign),
                "READBACK_GEOMETRY_SIGN_MASK:" + sid)
        require((family["SIGNED_DISTANCE_OCCUPANCY"] == "QUALIFIED_CORE_CANDIDATE") ==
                (counts["signed_query_count"] == counts["query_count"]),
                "READBACK_GEOMETRY_SIGN_CORE:" + sid)
        require((family["SURFACE_GEOMETRY"] == "QUALIFIED_CORE_CANDIDATE") ==
                (counts["valid_face_count"] == counts["face_count"]),
                "READBACK_GEOMETRY_FACE_CORE:" + sid)
        queries += counts["query_count"]
        total_signed += counts["signed_query_count"]
        surface_faces += counts["face_count"]
        valid_faces += counts["valid_face_count"]
        for key in GEOMETRY_FAMILIES:
            dispositions[key][family[key]] += 1
        present = {}
        for head, key in HEAD_FAMILY.items():
            value = family[key]
            present[head] = value
            target_counts[head][value] += 1
        subject_rows.append({
            "subject_id": sid,
            "source_sha256": qualified[sid]["source_sha256"],
            "observed_source_rest_court": "PASS",
            "measured_geometry_candidates_only": present,
            "global_solid_sign_authority": "UNKNOWN_NOT_COURTED_BY_COMPONENT_SIGN",
            "other_head_targets": "UNKNOWN_NOT_COURTED_BY_SOURCE_GEOMETRY",
            "counts": {
                "court_query_count": counts["query_count"],
                "component_local_valid_signed_queries": counts["signed_query_count"],
                "court_face_count": counts["face_count"],
                "court_valid_faces": counts["valid_face_count"],
            },
        })
    require(len(subject_rows) == 352, "READBACK_SUBJECT_COUNT")
    require(sum(dispositions["SURFACE_GEOMETRY"].values()) == 352, "READBACK_GEOMETRY_ROLLUP")
    require(sum(dispositions["SIGNED_DISTANCE_OCCUPANCY"].values()) == 352, "READBACK_SIGN_ROLLUP")
    return {
        "schema": SCHEMA,
        "status": "PASS_EXACT_STAGE50_COURT_READBACK__TARGET_SUFFICIENCY_NOT_ESTABLISHED",
        "source_v5_index_sha256": v5_sha, "component_geometry_index_sha256": geometry_sha,
        "candidate_subject_count": 370, "observation_pass_count": 352,
        "observation_fail_count": 18, "independent_geometry_court_pass_count": 352,
        "geometry_family_candidates": {key: dict(sorted(c.items())) for key,c in dispositions.items()},
        "per_head_geometry_candidate_counts": {key: dict(sorted(c.items())) for key,c in target_counts.items()},
        "measured_component_geometry_query_samples": queries,
        "measured_component_local_signed_query_samples": total_signed,
        "measured_valid_surface_faces": valid_faces,
        "measured_all_surface_faces": surface_faces,
        "unknown_head_targets": {head: {"status": "UNKNOWN_NOT_ESTABLISHED_BY_STAGE50_GEOMETRY",
            "qualified_subject_count": None, "training_sample_count": None} for head in UNKNOWN_TARGETS},
        "cross_family_intersection": None,
        "lineage_safe_split": None,
        "independently_qualified_mechanical_drawable_pair_counts": None,
        "materialized_numeric_sample_counts": None,
        "blocked_next_steps": [
            "CURRENT_STAGE49_51_RESPONSE_GRAPHICS_RELATION_TESSA_FAMILY_COURTS_REQUIRED",
            "EXACT_GLOBAL_SOLID_SDF_INDEPENDENT_COURT_REQUIRED",
            "POSITIVE_NEGATIVE_ABSTAIN_AND_SOURCE_LOCUS_JOIN_COUNTS_REQUIRED",
            "GLOBAL_LINEAGE_HOLDOUT_AND_HEAD_TARGET_MATERIALIZATION_REQUIRED",
        ],
        "subject_geometry_candidate_rows": subject_rows,
        "source_component_sign_is_global_sdf": False,
        "family_authority_minted": False, "model_targets_issued": False,
        "materialization_authorized": False, "training_authorized": False,
    }


def render(report: dict[str, Any]) -> str:
    rows = [
        "# Corpus370: exact Stage50 geometry vs head-target sufficiency",
        "",
        "**" + report["status"] + "**",
        "",
        "All counts below are measured candidate counts, **not** model training admission.",
        "The component-local signed-query count is never global solid SDF truth.",
        "",
        "| Geometry family | CORE candidate | MASKED candidate | UNAVAILABLE candidate |",
        "|---|---:|---:|---:|",
    ]
    for family, counts in report["geometry_family_candidates"].items():
        rows.append("| " + family + " | " + " | ".join(
            str(counts.get(key, 0)) for key in CANDIDATE) + " |")
    rows.extend([
        "",
        "Component source queries: " + str(report["measured_component_geometry_query_samples"]),
        "Component-only qualified signed queries: " + str(report["measured_component_local_signed_query_samples"]),
        "Surface faces (valid / all): " + str(report["measured_valid_surface_faces"])
            + " / " + str(report["measured_all_surface_faces"]),
        "",
        "### Not established by the input",
    ])
    rows.extend("- " + key + ": UNKNOWN" for key in report["unknown_head_targets"])
    rows.extend(["", "Joint mechanical / graphics / TESSA eligibility: **UNKNOWN, not zero**.",
                 "Lineage split / training authority: **NOT GRANTED**.", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-index", type=Path, required=True)
    parser.add_argument("--component-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(sha256(args.v5_index) == V5_SHA, "READBACK_EXACT_V5_BYTES_REQUIRED")
    geometry_sha = sha256(args.component_index)
    v5 = json.loads(args.v5_index.read_text())
    geometry = json.loads(args.component_index.read_text())
    verify_actual_court_receipts(geometry)
    report = build(v5, geometry, v5_sha=V5_SHA, geometry_sha=geometry_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CORPUS370_STAGE50_HEAD_TARGET_READBACK_V1.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "CORPUS370_STAGE50_HEAD_TARGET_READBACK_V1.md").write_text(render(report))
    print(json.dumps({key: report[key] for key in
        ("status", "candidate_subject_count", "observation_pass_count",
         "geometry_family_candidates", "measured_component_geometry_query_samples",
         "measured_component_local_signed_query_samples", "training_authorized")},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
