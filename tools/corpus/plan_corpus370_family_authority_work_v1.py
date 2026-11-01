#!/usr/bin/env python3
from __future__ import annotations

"""Plan missing Corpus370 truth-family work without granting authority.

The planner consumes the exact source roster plus the raw capability census. It
routes subjects to format/capability-specific producer/court lanes. Its output is
an execution worklist, not an admission matrix; no row can become PASS here.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "RealSaS.Corpus370FamilyAuthorityWorkPlan.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _rig_lane(source_format: str, raw: dict) -> str:
    if not (raw.get("has_rig") and raw.get("has_skin_weights")):
        return "COURT_AUTHORED_RIG_SKIN_ABSENCE__DO_NOT_INFER_FROM_PACK"
    if source_format in {".gltf", ".glb"}:
        return "GLTF2_RAW_SOURCE_RIG_SKIN_PLUS_IMPORT_EQUIVALENCE"
    if source_format == ".fbx":
        return "FBX_PINNED_BLENDER_DECODE_PLUS_INDEPENDENT_IMPORTER_REPLAY_COURT"
    return "UNSUPPORTED_SOURCE_FORMAT"


def _animation_lane(source_format: str, raw: dict) -> str:
    if not raw.get("has_animation"):
        return "COURT_AUTHORED_ANIMATION_ABSENCE__MASK_IF_CONFIRMED"
    if not raw.get("has_rig"):
        return "NODE_OR_MORPH_ANIMATION_ONLY_CANDIDATE__NOT_AXIS_RIG_KINEMATICS"
    if source_format in {".gltf", ".glb"}:
        return "GLTF2_RAW_ANIMATION_PLUS_CANONICAL_REPLAY"
    if source_format == ".fbx":
        return "FBX_PINNED_NATIVE_ANIMATION_IMPORT_PLUS_CANONICAL_REPLAY_REQUIRED"
    return "UNSUPPORTED_SOURCE_FORMAT"


def plan(roster_path: Path, census_path: Path) -> dict:
    roster = json.loads(roster_path.read_text())
    census = json.loads(census_path.read_text())
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "CORPUS370_WORKPLAN_ROSTER_SCHEMA")
    require(census.get("schema") == "RealSaS.Corpus370RawSourceCapabilityCensus.v1", "CORPUS370_WORKPLAN_CENSUS_SCHEMA")
    roster_rows = {row["subject_id"]: row for row in roster.get("subjects") or ()}
    census_rows = {row["subject_id"]: row for row in census.get("subjects") or ()}
    require(len(roster_rows) == 370 and set(roster_rows) == set(census_rows), "CORPUS370_WORKPLAN_SUBJECT_SET")
    rows = []
    for subject_id in sorted(roster_rows):
        source = roster_rows[subject_id]
        measured = census_rows[subject_id]
        require(source["source_sha256"] == measured["source_sha256"], "CORPUS370_WORKPLAN_SOURCE_SHA")
        raw = measured["raw_capabilities"]
        established = source.get("origin") == "ESTABLISHED_BASE"
        rows.append(
            {
                "subject_id": subject_id,
                "origin": source.get("origin"),
                "source_id": source.get("source_id"),
                "source_format": source.get("source_format"),
                "source_sha256": source["source_sha256"],
                "work": {
                    "SOURCE_IMPORT": "REVALIDATE_EXISTING_CURRENT_COURT" if established else "PRODUCE_AND_COURT_CONTROLLED_IMPORT",
                    "CAMERA_OBSERVATION": "REVALIDATE_EXISTING_8VIEW_BINDING" if established else "PRODUCE_8VIEW_BIND_REST_AND_COURT",
                    "SURFACE_GEOMETRY_UDF": "REUSE_AFTER_CURRENT_NUMERIC_REVALIDATION" if established else "PRODUCE_EXACT_SURFACE_UDF_AND_COURT",
                    "SURFACE_NORMALS": "REUSE_SOURCE_ORIENTED_NORMALS_AFTER_CURRENT_REVALIDATION" if established else "PRODUCE_SOURCE_ORIENTED_NORMALS_AND_COURT",
                    "GLOBAL_SOLID_SDF": "ASSESS_EXPLICIT_SOLID_ASSEMBLY__BOOLEAN_BOUNDARY_SIGN_DISTANCE_COURT__MASK_IF_NOT_QUALIFIED",
                    "MECHANICAL_SURFACE_TOPOLOGY": "PRODUCE_FROM_CURRENT_DEFORMATION_EVIDENCE_AND_INDEPENDENT_COURT",
                    "CAUSAL_RESPONSE": "PRODUCE_ON_QUALIFIED_MECHANICAL_TOPOLOGY_AND_COURT",
                    "APPEARANCE_CAA": "PRODUCE_OR_REVALIDATE_PRESENTATION_MEASUREMENT__STUDENT_OBSERVABILITY_GATE",
                    "VISIBILITY_OCCLUSION": "PRODUCE_OR_REVALIDATE_PRESENTATION_MEASUREMENT__STUDENT_OBSERVABILITY_GATE",
                    "RAY_COMPOSITING": "PRODUCE_OR_REVALIDATE_PRESENTATION_MEASUREMENT__COMPILER_ORACLE_SCOPE_SEPARATE",
                    "RIG_HIERARCHY": _rig_lane(source["source_format"], raw),
                    "SKIN_WEIGHTS": _rig_lane(source["source_format"], raw),
                    "ANIMATION_KINEMATICS": _animation_lane(source["source_format"], raw),
                    "TESSA_TEACHER": "BUILD_V5_ONLY_AFTER_MECHANICAL_TOPOLOGY_AND_RESPONSE_COURTS_PASS",
                },
                "raw_measurement": {
                    "has_rig": raw.get("has_rig"),
                    "has_skin_weights": raw.get("has_skin_weights"),
                    "has_animation": raw.get("has_animation"),
                },
                "training_authorized": False,
            }
        )
    lane_counts: dict[str, Counter] = {}
    for row in rows:
        for family, lane in row["work"].items():
            lane_counts.setdefault(family, Counter())[lane] += 1
    return {
        "schema": SCHEMA,
        "status": "EXECUTION_WORKLIST_ONLY__NO_FAMILY_AUTHORITY",
        "subject_count": len(rows),
        "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
        "raw_source_census": {"path": str(census_path), "sha256": sha256(census_path)},
        "lane_counts": {family: dict(sorted(counter.items())) for family, counter in sorted(lane_counts.items())},
        "subjects": rows,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--raw-source-census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.exact_source_roster, args.raw_source_census)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"], "lane_counts": result["lane_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
