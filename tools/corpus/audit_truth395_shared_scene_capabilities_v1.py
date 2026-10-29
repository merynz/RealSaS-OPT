#!/usr/bin/env python3
"""Audit all sealed Quaternius GLTF/GLB assets for IRIS/TESSA shared-scene truth.

Historical Truth139 intentionally selected only rigged+animated subjects because its old
"full truth" court coupled appearance/presentation to mechanical-response eligibility.
That is not the current IRIS/TESSA architecture. Geometry, Graphics, Relations and TESSA
topology are independently useful on static/rigid assets; deformation-response evidence is
optional and rig/skin authority belongs downstream to AXIS/MIRA.

This audit therefore scans every one of the sealed 395 GLTF/GLB files, preserves file-level
capabilities, deduplicates exact structural identities for subject-level courts, and reports
how many assets are usable for each truth family. It does not render, train, split or admit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus import audit_truth139_gltf_capabilities_v1 as v1
from tools.corpus import audit_truth139_gltf_capabilities_v2 as v2

SCHEMA = "RealSaS.Truth395SharedSceneCapabilityAudit.v1"
SUBJECT_SCHEMA = "RealSaS.Truth395SharedSceneSubject.v1"
FILE_SCHEMA = "RealSaS.Truth395SharedSceneFile.v1"
EXPECTED_FILES = 395


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _shared_caps(row: dict) -> dict:
    legacy = dict(row.get("capabilities") or {})
    # These legacy primitive checks are still useful, but we deliberately decouple
    # Graphics from Stage47/mechanics. Response evidence is optional for IRIS/TESSA.
    geometry = bool(legacy.get("SURFACE_NORMAL_TRUTH"))
    graphics = bool(legacy.get("OBSERVATION_FSTAR"))
    topology = bool(legacy.get("TESSA_TOPOLOGY_TEACHER"))
    response = bool(legacy.get("STAGE47_MECHANICS"))
    relations = bool(geometry and topology)
    shared_core = bool(geometry and graphics and topology)
    return {
        "GEOMETRY_TEACHER": geometry,
        "GRAPHICS_TEACHER": graphics,
        "RELATION_TEACHER_STATIC": relations,
        "TESSA_TOPOLOGY_TEACHER": topology,
        "DEFORMATION_RESPONSE_EVIDENCE": response,
        "IRIS_TESSA_SHARED_CORE": shared_core,
        "IRIS_TESSA_SHARED_CORE_WITH_RESPONSE": bool(shared_core and response),
    }


def _rank(row: dict) -> tuple:
    caps = row["shared_scene_capabilities"]
    family_count = sum(int(bool(caps[k])) for k in (
        "GEOMETRY_TEACHER", "GRAPHICS_TEACHER", "RELATION_TEACHER_STATIC",
        "TESSA_TOPOLOGY_TEACHER", "DEFORMATION_RESPONSE_EVIDENCE"))
    path = str((row.get("source") or {}).get("relative_path") or "").lower()
    textured = 0 if "textured" in path else 1
    flat = 1 if "/flat/" in path or "flat" in Path(path).stem.lower() else 0
    return (-int(caps["IRIS_TESSA_SHARED_CORE"]), -family_count, textured, flat, path)


def _union_lineage(group: list[dict]) -> dict:
    fields = (
        "source_sha256", "source_identity", "skeleton_hierarchy_signature",
        "topology_lineage_signature", "material_lineage_signature",
    )
    return {
        field: sorted({str(token) for row in group for token in ((row.get("lineage") or {}).get(field) or ())})
        for field in fields
    }


def audit(raw_root: Path, fit8_selection: Path) -> dict:
    raw_root = raw_root.resolve()
    development_shas, development_aliases = v1._development_aliases(fit8_selection.resolve())
    scan_root = raw_root / "raw" / "quaternius"
    if not scan_root.is_dir():
        scan_root = raw_root
    paths = sorted({*scan_root.rglob("*.gltf"), *scan_root.rglob("*.glb")})

    files: list[dict] = []
    failures: list[dict] = []
    for index, path in enumerate(paths, 1):
        try:
            base = v2._audit_file_v2(path, raw_root, development_shas, development_aliases)
            row = dict(base)
            row["schema"] = FILE_SCHEMA
            row["shared_scene_capabilities"] = _shared_caps(base)
            row["historical_truth139_member_candidate"] = bool(base.get("historical_rigged_animated_candidate"))
            files.append(row)
            caps = row["shared_scene_capabilities"]
            print(
                f"[TRUTH395] {index}/{len(paths)} {row['source']['relative_path']} "
                f"core={int(caps['IRIS_TESSA_SHARED_CORE'])} response={int(caps['DEFORMATION_RESPONSE_EVIDENCE'])} "
                f"hist139={int(row['historical_truth139_member_candidate'])}",
                flush=True,
            )
        except Exception as exc:
            failures.append({
                "relative_path": path.relative_to(raw_root).as_posix(),
                "error": f"{type(exc).__name__}:{exc}",
            })

    grouped: dict[str, list[dict]] = {}
    for row in files:
        grouped.setdefault(str(row["subject_identity_signature"]), []).append(row)

    subjects: list[dict] = []
    duplicate_groups: list[dict] = []
    for identity, group in sorted(grouped.items()):
        ordered = sorted(group, key=_rank)
        rep = dict(ordered[0])
        caps_union = {
            key: any(bool(item["shared_scene_capabilities"].get(key)) for item in group)
            for key in rep["shared_scene_capabilities"]
        }
        subject_id = "truth395_" + identity[:20]
        subjects.append({
            **rep,
            "schema": SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "subject_identity_signature": identity,
            "shared_scene_capabilities": caps_union,
            "lineage": _union_lineage(group),
            "source_variants": [item["source"] for item in sorted(group, key=lambda x: x["source"]["relative_path"])],
            "source_variant_count": len(group),
            "historical_truth139_member_candidate": any(bool(item.get("historical_truth139_member_candidate")) for item in group),
            "admitted": False,
            "training_authorized": False,
        })
        if len(group) > 1:
            duplicate_groups.append({
                "subject_id": subject_id,
                "identity": identity,
                "representative": rep["source"]["relative_path"],
                "source_files": [item["source"]["relative_path"] for item in sorted(group, key=lambda x: x["source"]["relative_path"])],
            })

    def count_file(cap: str) -> int:
        return sum(bool(row["shared_scene_capabilities"].get(cap)) for row in files)
    def count_subject(cap: str) -> int:
        return sum(bool(row["shared_scene_capabilities"].get(cap)) for row in subjects)

    cap_names = (
        "GEOMETRY_TEACHER", "GRAPHICS_TEACHER", "RELATION_TEACHER_STATIC",
        "TESSA_TOPOLOGY_TEACHER", "DEFORMATION_RESPONSE_EVIDENCE",
        "IRIS_TESSA_SHARED_CORE", "IRIS_TESSA_SHARED_CORE_WITH_RESPONSE",
    )
    counts = {
        "gltf_glb_files_scanned": len(paths),
        "files_audited": len(files),
        "file_failures": len(failures),
        "unique_structural_subjects": len(subjects),
        "duplicate_files_collapsed": len(files) - len(subjects),
        "historical_rigged_animated_files": sum(bool(row.get("historical_truth139_member_candidate")) for row in files),
        "historical_rigged_animated_unique_subjects": sum(bool(row.get("historical_truth139_member_candidate")) for row in subjects),
        "file_capability_counts": {name: count_file(name) for name in cap_names},
        "subject_capability_counts": {name: count_subject(name) for name in cap_names},
    }
    exact = len(paths) == EXPECTED_FILES and not failures
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_TRUTH395_SHARED_SCENE_CAPABILITY_AUDIT__ADMISSION_PENDING" if exact else "FAIL_OR_INCOMPLETE_TRUTH395_SHARED_SCENE_CAPABILITY_AUDIT",
        "policy": {
            "scan_all_395_gltf_glb": True,
            "geometry_graphics_relations_topology_do_not_require_rig_or_animation": True,
            "deformation_response_is_optional_for_iris_tessa": True,
            "axis_mira_rig_skin_authority_not_minted_here": True,
            "legacy_stage48_mechanics_coupling_rejected": True,
            "duplicates_retained_as_source_variants_but_not_independent_subjects": True,
        },
        "counts": counts,
        "files": files,
        "subjects": subjects,
        "duplicate_groups": duplicate_groups,
        "file_failures": failures,
        "split_authorized": False,
        "training_authorized": False,
        "next_gate": "IMPORT_RENDER_AND_RICH_TRUTH_MATERIALIZATION_FOR_ALL_SHARED_CORE_CANDIDATES",
    }
    payload["audit_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--fit8-selection", default="canonical/FIT8_DIVERSITY_SELECTION_V1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.raw_root), Path(args.fit8_selection))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "counts": result["counts"],
        "audit_identity_sha256": result["audit_identity_sha256"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    if not result["status"].startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
