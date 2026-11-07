#!/usr/bin/env python3
"""Read-only, exact-370, historical current-family + new three-head audit snapshot.

Loads a REAL previously courted Corpus370 whole-universe matrix (not a synthetic
roster) and projects exact family states into the proposed IRIS Geometry /
Semantic / Graphics and animation-ready TESSA target inventory. New targets
never receive authority by inference. Produces a fully enumerated *status
matrix*, not magically complete truth or training eligibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "RealSaS.Corpus370ThreeHeadTruthStatusSnapshot.v1"
INPUT_SCHEMA = "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1"
KNOWN = {"PASS", "MASKED", "FAIL", "QUARANTINED", "UNKNOWN"}
SOURCE_FAMILIES = (
    "SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF",
    "GLOBAL_SOLID_SDF", "SURFACE_NORMALS", "APPEARANCE_CAA",
    "VISIBILITY_OCCLUSION", "RAY_COMPOSITING",
    "MECHANICAL_SURFACE_TOPOLOGY", "CAUSAL_RESPONSE",
    "RIG_HIERARCHY", "SKIN_WEIGHTS", "ANIMATION_KINEMATICS",
    "TESSA_TEACHER",
)
# A direct relation only when the existing family's statement genuinely
# entails the *same* target. New semantic/controllability/animation-ready
# statements are deliberately NOT smuggled through old family PASS.
HEAD_TARGETS = {
    "IRIS_GEOMETRY": {
        "surface_udf": "SURFACE_GEOMETRY_UDF",
        "source_normals": "SURFACE_NORMALS",
        "global_solid_sdf": "GLOBAL_SOLID_SDF",
    },
    "IRIS_SEMANTIC": {
        "semantic_concept": None,
        "component_instance_support": None,
        "functional_affordance_prior": None,
        "structural_component_relations": None,
        "semantic_view_consistency": None,
    },
    "IRIS_GRAPHICS": {
        "source_appearance_caa": "APPEARANCE_CAA",
        "visibility_occlusion": "VISIBILITY_OCCLUSION",
        "ray_compositing": "RAY_COMPOSITING",
        "motion_conditioned_rgb": None,
        "graphic_edit_reveal_consequence": None,
    },
    "TESSA": {
        "mechanical_surface_topology": "MECHANICAL_SURFACE_TOPOLOGY",
        "legacy_tessa_teacher": "TESSA_TEACHER",
        "animation_ready_deformation_consequence": None,
    },
    "AXIS": {
        "source_rig_hierarchy": "RIG_HIERARCHY",
        "source_animation_kinematics": "ANIMATION_KINEMATICS",
        "typed_controllable_rig_program_teacher": None,
    },
    "MIRA": {"source_skin_weights": "SKIN_WEIGHTS"},
}
NEW_TARGET_REASONS = {
    "semantic_concept": "CURATED_SEMANTIC_CONCEPT_COURT_NOT_MINTED",
    "component_instance_support": "3D_COMPONENT_SUPPORT_REVIEW_COURT_NOT_MINTED",
    "functional_affordance_prior": "DESIGNED_PRIOR_NOT_YET_BOUND_TO_REVIEWED_SOURCE_REGION",
    "structural_component_relations": "SEMANTIC_TYPED_RELATION_COURT_NOT_MINTED",
    "semantic_view_consistency": "SEMANTIC_CROSS_VIEW_RASTER_COURT_NOT_MINTED",
    "motion_conditioned_rgb": "POSE_CONDITIONED_SOURCE_APPEARANCE_COURT_NOT_MINTED",
    "graphic_edit_reveal_consequence": "GRAPHIC_CONSEQUENCE_COURT_NOT_MINTED",
    "animation_ready_deformation_consequence": "TESSA_DEFORMATION_CONSEQUENCE_COURT_NOT_MINTED",
    "typed_controllable_rig_program_teacher": "AXIS_TYPED_RIG_PROGRAM_COURT_NOT_MINTED",
}
NEW_TARGET_OWNER = {
    "semantic_concept": "semantic_teacher_annotation",
    "component_instance_support": "semantic_source_support_and_3D_address_court",
    "functional_affordance_prior": "designed_prior_policy_and_source_region_court",
    "structural_component_relations": "semantic_region_relation_teacher_court",
    "semantic_view_consistency": "semantic_view_projector_and_renderer_court",
    "motion_conditioned_rgb": "graphics_appearance_transport_court",
    "graphic_edit_reveal_consequence": "source_layer_intervention_and_graphics_court",
    "animation_ready_deformation_consequence": "tessa_deformation_consequence_court",
    "typed_controllable_rig_program_teacher": "axis_controllability_teacher_court",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def require(value: bool, msg: str) -> None:
    if not value:
        raise ValueError("CORPUS370_THREE_HEAD_SNAPSHOT_" + msg)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def audit_legacy_matrix(doc: dict) -> list[dict]:
    require(isinstance(doc, dict) and doc.get("schema") == INPUT_SCHEMA,
            "INPUT_SCHEMA")
    require(doc.get("subject_count") == 370 and len(doc.get("subjects") or []) == 370,
            "EXACT_370_REQUIRED")
    require(doc.get("training_authorized") is False,
            "INPUT_CANNOT_AUTHORIZE_TRAINING")
    seen = set()
    rollup = {f: Counter() for f in SOURCE_FAMILIES}
    for row in doc["subjects"]:
        sid = row.get("subject_id")
        require(isinstance(sid, str) and sid and sid not in seen,
                "SUBJECT_UNIQUE")
        seen.add(sid)
        require(SHA_RE.fullmatch(str(row.get("source_sha256") or "")) is not None,
                "SOURCE_SHA")
        families = row.get("families")
        require(isinstance(families, dict) and set(families) == set(SOURCE_FAMILIES),
                "FAMILY_SET")
        for family in SOURCE_FAMILIES:
            cell = families[family]
            require(isinstance(cell, dict) and cell.get("status") in KNOWN,
                    "FAMILY_STATUS")
            status = cell["status"]
            reasons = cell.get("reason_codes")
            require(isinstance(reasons, list), "CELL_REASON_LIST")
            if status == "PASS":
                refs = cell.get("evidence_refs")
                courts = cell.get("court_refs")
                require(isinstance(refs, list) and refs and
                        isinstance(courts, list) and courts, "PASS_MISSING_COURT_REFS")
            else:
                require(reasons, "NONPASS_MISSING_REASONS")
            rollup[family][status] += 1
    expected = (doc.get("rollup") or {}).get("families")
    require(isinstance(expected, dict), "ROLLUP_REQUIRED")
    for family in SOURCE_FAMILIES:
        require(dict(rollup[family]) == expected.get(family, {}),
                "ROLLUP_DRIFT:" + family)
    return sorted(doc["subjects"], key=lambda x: x["subject_id"])


def _project_cell(row: dict, source_family: str | None, target: str) -> dict:
    if source_family is None:
        return {
            "status": "UNKNOWN",
            "reason_codes": [NEW_TARGET_REASONS[target]],
            "claim_kind": "NEW_TARGET_UNCOURTED",
            "source_family": None,
            "independent_current_court": False,
            "training_consumable": False,
        }
    src = row["families"][source_family]
    return {
        "status": src["status"],
        "reason_codes": list(src["reason_codes"]),
        "claim_kind": "INHERITED_COURTED_SOURCE_FAMILY_STATUS",
        "source_family": source_family,
        "independent_current_court": bool(src.get("court_refs")),
        "training_consumable": False,  # no new model admission from old matrix
    }


def build(*, legacy: dict, upstream_run_id: int, upstream_artifact_id: int,
          upstream_sha256: str, semantic_policy_sha256: str) -> dict:
    subjects = audit_legacy_matrix(legacy)
    require(type(upstream_run_id) is int and upstream_run_id > 0, "UPSTREAM_RUN")
    require(type(upstream_artifact_id) is int and upstream_artifact_id > 0, "UPSTREAM_ARTIFACT")
    require(SHA_RE.fullmatch(upstream_sha256) is not None, "UPSTREAM_BYTES_SHA")
    require(SHA_RE.fullmatch(semantic_policy_sha256) is not None, "PRIOR_SHA")
    projected = []
    counts = {h: {t: Counter() for t in ts} for h, ts in HEAD_TARGETS.items()}
    base_counts = {fam: Counter() for fam in SOURCE_FAMILIES}
    worklists = {f"{head}.{target}": [] for head, ts in HEAD_TARGETS.items()
                 for target in ts}
    for row in subjects:
        heads = {}
        for head, targets in HEAD_TARGETS.items():
            heads[head] = {}
            for target, source_family in targets.items():
                cell = _project_cell(row, source_family, target)
                heads[head][target] = cell
                counts[head][target][cell["status"]] += 1
                if cell["status"] != "PASS":
                    worklists[f"{head}.{target}"].append({
                        "subject_id": row["subject_id"],
                        "source_sha256": row["source_sha256"],
                        "status": cell["status"],
                        "reason_codes": cell["reason_codes"],
                        "next_producer": NEW_TARGET_OWNER.get(
                            target, "existing_" + (source_family or "unknown").lower() + "_family_court"
                        ),
                    })
        for family in SOURCE_FAMILIES:
            base_counts[family][row["families"][family]["status"]] += 1
        projected.append({
            "subject_id": row["subject_id"],
            "source_sha256": row["source_sha256"],
            "origin": row.get("origin"),
            "source_format": row.get("source_format"),
            "source_id": row.get("source_id"),
            "source_relative_path": row.get("source_relative_path"),
            "original_families": {
                family: row["families"][family] for family in SOURCE_FAMILIES
            },
            "heads": heads,
            "student_target_training_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "status": "FULL_370_ENUMERATION__TRUTH_STILL_OPEN",
        "subject_count": 370,
        "snapshot_kind": "HISTORICAL_REAL_COURTED_FAMILY_MATRIX_PLUS_UNCOURTED_NEW_TARGETS",
        "source_authority": {
            "upstream_github_run_id": upstream_run_id,
            "upstream_artifact_id": upstream_artifact_id,
            "upstream_matrix_sha256": upstream_sha256,
            "upstream_matrix_schema": INPUT_SCHEMA,
            "semantic_policy_sha256": semantic_policy_sha256,
            "fresh_revalidation_of_parent_court_files": False,
        },
        "target_definition": HEAD_TARGETS,
        "subjects": projected,
        "rollup": {
            "original_families": {k: dict(sorted(v.items())) for k, v in base_counts.items()},
            "head_targets": {
                h: {t: dict(sorted(c.items())) for t, c in ts.items()}
                for h, ts in counts.items()
            },
            "head_target_unknown_count": sum(
                c["UNKNOWN"] for ts in counts.values()
                for c in ts.values()
            ),
            "head_target_total_cells": 370 * sum(len(x) for x in HEAD_TARGETS.values()),
        },
        "worklists": worklists,
        "semantic_prior_is_source_physics": False,
        "target_admission_minted": False,
        "training_authorized": False,
    }


def write_outputs(doc: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    target_cols = [f"{head}.{target}" for head, targets in HEAD_TARGETS.items()
                   for target in targets]
    csv_path = directory / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.csv"
    columns = ["subject_id", "source_sha256", "origin", "source_format"]
    columns += list(SOURCE_FAMILIES) + target_cols
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in doc["subjects"]:
            flat = {k: row.get(k) for k in columns[:4]}
            flat.update({k: row["original_families"][k]["status"] for k in SOURCE_FAMILIES})
            flat.update({
                f"{head}.{target}": row["heads"][head][target]["status"]
                for head, targets in HEAD_TARGETS.items() for target in targets
            })
            writer.writerow(flat)
    lines = [
        "# Corpus370 — exact 370-subject truth-family status snapshot",
        "",
        f"Upstream real courted matrix: GitHub Actions run {doc['source_authority']['upstream_github_run_id']}, "
        f"artifact {doc['source_authority']['upstream_artifact_id']}.",
        "",
        "**Do not interpret this as 370 closed truths.** All 370 subject IDs and all target "
        "cells are populated with a typed status; every uncovered target is UNKNOWN and "
        "cannot be trained. Archived matrix evidence is not independently re-opened here.",
        "",
        "## Current independently courted source-family status",
        "",
        "| Family | PASS | MASKED | FAIL | QUARANTINED | UNKNOWN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, n in doc["rollup"]["original_families"].items():
        lines.append("| " + family + " | " + " | ".join(str(n.get(k, 0))
                     for k in ("PASS", "MASKED", "FAIL", "QUARANTINED", "UNKNOWN")) + " |")
    lines += ["", "## New logical head target coverage", "",
              "| Head / target | PASS | MASKED | FAIL | QUARANTINED | UNKNOWN |",
              "|---|---:|---:|---:|---:|---:|"]
    for head, targets in doc["rollup"]["head_targets"].items():
        for target, n in targets.items():
            lines.append("| " + head + "." + target + " | " + " | ".join(
                str(n.get(k, 0)) for k in
                ("PASS", "MASKED", "FAIL", "QUARANTINED", "UNKNOWN")) + " |")
    lines += ["", "## Next P0", "",
              "1. Source-face-addressed Semantic concept/instance/support + functional prior teacher; "
              "independent review and raster consistency court.",
              "2. Re-qualify existing Graphics/Ray/CAA targets with the 3-head schema; "
              "do not turn mechanical response into a visual edit label.",
              "3. Obtain animation-ready TESSA deformation consequence targets using "
              "native rig/skin/pose evidence and independent geometry gates.",
              "4. Reconcile all 370 source observation failures, source format masks, "
              "lineage collisions, training eligibility and sealed materialization.",
              "",
              "Per-subject exact status and reason codes are available in the JSON and CSV.",
              ""]
    (directory / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    work = {
        "schema": "RealSaS.Corpus370ThreeHeadTruthWorklist.v1",
        "subject_count": 370,
        "source_matrix_sha256": doc["source_authority"]["upstream_matrix_sha256"],
        "target_worklists": doc["worklists"],
        "training_authorized": False,
    }
    (directory / "CORPUS370_THREE_HEAD_TRUTH_WORKLIST_V1.json").write_text(
        json.dumps(work, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--upstream-run-id", type=int, required=True)
    parser.add_argument("--upstream-artifact-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.matrix.read_text(encoding="utf-8"))
    require(json.loads(args.policy.read_text(encoding="utf-8")).get("schema") ==
            "RealSaS.SemanticPriorPolicy.v1", "SEMANTIC_PRIOR_SCHEMA")
    doc = build(legacy=old, upstream_run_id=args.upstream_run_id,
                upstream_artifact_id=args.upstream_artifact_id,
                upstream_sha256=digest(args.matrix),
                semantic_policy_sha256=digest(args.policy))
    write_outputs(doc, args.output)
    print(json.dumps({
        "status": doc["status"],
        "subjects": doc["subject_count"],
        "head_unknown_cells": doc["rollup"]["head_target_unknown_count"],
        "original_family_rollup": doc["rollup"]["original_families"],
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
