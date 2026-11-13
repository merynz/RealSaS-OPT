#!/usr/bin/env python3
"""Truth Foundry deep per-source, source-SHA-bound audit of immutable Blender import receipts.

This does NOT substitute for actual scene viewing, independent morphology review,
source-authored control tests, signed source geometry or 2D-art review. It does
not write training data, source assets or inferred semantic labels.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "RealSaS.TruthFoundryDeepSourceAudit.v1"
RECEIPT_SCHEMA = "RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1"
MAX_DECOMPRESSED = 512 * (1 << 20)
NUMERIC_FIELDS = (
    "source_scene_node_count", "mesh_object_count", "mesh_disconnected_components",
    "mesh_boundary_edges", "mesh_nonmanifold_edges", "polygon_count", "vertex_count",
    "armature_count", "observed_bone_count", "bone_parented_mesh_candidates",
    "skin_rows_sampled", "skin_rows_bad_sum", "skin_rows_zero_weights",
    "material_count", "image_count", "image_dependency_missing_count",
    "observed_action_count", "observed_action_curve_count",
    "observed_driver_channel_count", "observed_ik_constraint_count",
    "observed_nla_strip_count", "observed_nondeform_bone_count"
)
MORPH_CLASS = (
    "HUMAN", "HUMANOID_NONHUMAN", "QUADRUPED_MAMMAL", "QUADRUPED_OTHER",
    "REPTILE", "DINOSAUR", "BIRD", "WINGED_FANTASY", "SERPENTINE",
    "ARTHROPOD_MULTILEGGED", "AQUATIC", "MONSTER_OTHER", "ROBOT_MECH",
    "TANK_GROUND_VEHICLE", "OTHER_GROUND_VEHICLE", "AIRCRAFT", "WATERCRAFT",
    "PROP_TOOL_WEAPON", "BUILDING_ENVIRONMENT", "PLANT", "OTHER", "UNKNOWN"
)
STAGES = (
    "source_identity_rights", "scene_import_and_layout", "source_author_pose_and_front_up",
    "character_type_and_morphology", "mesh_modularity_and_attachments",
    "rig_skin_bind_and_controls", "materials_textures_and_uv",
    "semantic_source_face_regions", "source_bound_multiview_render",
    "2d_style_and_identity_review", "layer_edit_reveal_and_draw_order",
    "mechanical_intervention", "head_target_independent_courts",
)


def fail(code: str):
    raise ValueError("DEEP_SOURCE_AUDIT_" + code)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        fail("EMPTY_OUTPUT:" + path.name)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def hash_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_gzip_json(path: Path):
    with gzip.open(path, "rb") as f:
        data = f.read(MAX_DECOMPRESSED + 1)
    if len(data) > MAX_DECOMPRESSED:
        fail("GZIP_UNCOMPRESSED_SIZE_LIMIT:" + path.name)
    return json.loads(data)


def _node_index(nodes, ty):
    return [n for n in nodes if n.get("object_type") == ty]


def _count_animated_bone_names(actions):
    bones = set()
    keyframes = 0
    curve_paths = 0
    for a in actions:
        for curve in a.get("curves", []):
            dp = curve.get("data_path_teacher_only") or ""
            match = re.search(r'pose\.bones\["([^"\\]+)"\]', dp)
            if match:
                bones.add(match.group(1))
            keyframes += int(curve.get("keyframe_count", 0))
            curve_paths += 1
    return len(bones), keyframes, curve_paths


def _mesh_modularity(mesh_count, island_count):
    if not mesh_count:
        return "NO_IMPORTED_MESH"
    if island_count < mesh_count:
        return "INCONSISTENT_COMPONENT_COUNT_REVIEW"
    if mesh_count == 1 and island_count == 1:
        return "ONE_OBJECT_ONE_CONNECTED_SURFACE__NOT_ANATOMY"
    if mesh_count == 1 and island_count > 1:
        return "ONE_OBJECT_MULTI_DISCONNECTED_ISLANDS__NOT_DRAWABLES"
    if mesh_count > 1 and island_count == mesh_count:
        return "MULTI_OBJECT_COMPONENT_BALANCED__NOT_ANATOMY"
    return "MULTI_OBJECT_MULTI_ISLANDS__NOT_DRAWABLES"


def analyze_one(row: dict, summary: dict, full: dict, gz_sha: str,
                summary_sha: str) -> tuple[dict, list[dict]]:
    aid = row["asset_id"]
    source_sha = row["source_sha256"]
    if summary.get("schema") != RECEIPT_SCHEMA or full.get("schema") != RECEIPT_SCHEMA:
        fail("SCHEMA:" + aid)
    for k in ("asset_id", "source_sha256", "candidate_id", "source_format"):
        if summary.get(k) != full.get(k):
            fail("SUMMARY_FULL_MISMATCH:" + aid + ":" + k)
    if source_sha != full.get("source_sha256") or row["candidate_id"] != full["candidate_id"]:
        fail("SOURCE_IDENTITY:" + aid)
    if summary.get("receipt_gz_sha256") != gz_sha:
        fail("GZIP_SHA:" + aid)
    if str(row["census_summary_sha256"]) != summary_sha:
        fail("SUMMARY_SHA:" + aid)
    if any(summary.get(x) is not False for x in ("training_authorized", "commercial_ml_rights_courted")):
        fail("FORGED_SUMMARY_AUTHORITY:" + aid)
    if full.get("training_authorized") is not False or full.get("semantic_region_truth_courted") is not False:
        fail("FORGED_FULL_AUTHORITY:" + aid)
    counts = full.get("numeric_census")
    if not isinstance(counts, dict) or counts != summary.get("numeric_census"):
        fail("CENSUS_SOURCE_MISMATCH:" + aid)
    for k in NUMERIC_FIELDS:
        if type(counts.get(k)) is not int or counts[k] < 0 or str(counts[k]) != str(row.get(k)):
            fail("COUNT_PARITY:" + aid + ":" + k)
    scene = full.get("original_source_scene") or {}
    imported = scene.get("imported_scene") or {}
    nodes = imported.get("imported_nodes") or []
    meshes = imported.get("meshes") or []
    armatures = imported.get("armatures") or []
    materials = scene.get("material_and_texture_inventory") or {}
    rig = full.get("imported_rig_and_animation_control") or {}
    actions = rig.get("actions") or []
    armature_mods = sum("ARMATURE" in (n.get("modifiers") or []) for n in nodes if n.get("object_type") == "MESH")
    mesh_parents_to_bone = sum(n.get("parent_type") == "BONE" for n in _node_index(nodes, "MESH"))
    mesh_parents_to_armature = sum(n.get("parent_type") == "OBJECT" and n.get("parent_node_index") is not None
                                   for n in _node_index(nodes, "MESH"))
    mesh_stats = [
        {"scene_node_index": m.get("scene_node_index"),
         "vertex_count": m.get("vertex_count"), "polygon_count": m.get("polygon_count"),
         "material_slots": m.get("material_slots"),
         "connected_islands": (m.get("index_topology") or {}).get("source_index_component_count"),
         "boundary_edges": (m.get("index_topology") or {}).get("boundary_edge_count"),
         "nonmanifold_edges": (m.get("index_topology") or {}).get("nonmanifold_edge_count"),
         "skin_row_count": m.get("skin_rows_sampled"), "unweighted_row_count": m.get("skin_rows_without_any_groups"),
         "vertex_group_count": m.get("vertex_groups")}
        for m in meshes
    ]
    animated_bones, keyframes, curve_count = _count_animated_bone_names(actions)
    if len(nodes) != counts["source_scene_node_count"] or len(meshes) != counts["mesh_object_count"]:
        fail("SCENE_GRAPH_COUNT_MISMATCH:" + aid)
    if len(armatures) != counts["armature_count"] or len(actions) != counts["observed_action_count"]:
        fail("RIG_OR_ACTION_COUNT_MISMATCH:" + aid)
    if curve_count != counts["observed_action_curve_count"]:
        fail("ACTION_CURVE_COUNT_MISMATCH:" + aid)
    if sum(int(m.get("vertex_count", 0)) for m in meshes) != counts["vertex_count"]:
        fail("MESH_VERTEX_PARITY:" + aid)
    if sum(int(m.get("polygon_count", 0)) for m in meshes) != counts["polygon_count"]:
        fail("MESH_POLYGON_PARITY:" + aid)
    # Imported source layout does not prove authoring controls, anatomical rest or independent 2D drawable.
    source_rec = {
        "schema": SCHEMA, "asset_id": aid, "source_sha256": source_sha,
        "candidate_id": row["candidate_id"], "original_import_summary_sha256": summary_sha,
        "full_receipt_gz_sha256": gz_sha, "source_format": row["source_format"],
        "historical_split": row["historical_split"],
        "counts": {k: counts[k] for k in NUMERIC_FIELDS},
        "structural_class": _mesh_modularity(counts["mesh_object_count"], counts["mesh_disconnected_components"]),
        "mesh_records": mesh_stats,
        "armature_modifier_mesh_object_count": armature_mods,
        "bone_parented_mesh_object_count": mesh_parents_to_bone,
        "object_parented_mesh_object_count": mesh_parents_to_armature,
        "observed_animated_bone_name_count_teacher_only": animated_bones,
        "observed_action_keyframe_count": keyframes,
        "imported_constraint_type_counts": rig.get("imported_constraint_type_counts") or {},
        "imported_camera_count": rig.get("imported_camera_count", 0),
        "full_scene_per_asset_details_at": "sha256:" + gz_sha,
        "privileged_source_metadata_teacher_only": {
            "scene_node_names": [n.get("name_teacher_only") for n in nodes],
            "action_names": [a.get("action_name_teacher_only") for a in actions],
            "material_or_image_names": [i.get("name_teacher_only") for i in materials.get("images_teacher_only", [])]
        },
        "character_vs_prop": "UNKNOWN_NEEDS_VISUAL_SOURCE_COURT",
        "morphology": "UNKNOWN_NEEDS_VISUAL_SOURCE_COURT",
        "anatomical_orientation": "UNKNOWN_NEEDS_SOURCE_ANATOMICAL_COURT",
        "authored_rest_pose_class": "UNKNOWN_NEEDS_POSE_COURT",
        "native_control_functionality": "UNKNOWN_NEEDS_NATIVE_SOURCE_INTERVENTION",
        "source_texture_rgba_parity": "UNKNOWN_NEEDS_MATERIAL_PIXEL_COURT",
        "2d_art_suitability": "UNKNOWN_NEEDS_RENDER_AND_VISUAL_COURT",
        "source_semantic_regions": "UNKNOWN_NEEDS_SOURCE_FACE_REVIEW",
        "drawable_layerability": "UNKNOWN_NEEDS_GEOMETRY_SEMANTICS_EDIT_CONSEQUENCE",
        "drawable_count": None,
        "layer_view_occlusion": "UNKNOWN_NEEDS_SOURCE_BOUND_MULTI_VIEW_REVEAL",
        "source_license_training_rights": "UNKNOWN_UNQUALIFIED",
        "training_authorized": False,
        "materialization_authorized": False,
    }
    review = []
    for stage in STAGES:
        # Scene/import is an observation, not original authoring parity. No stage silently passes.
        known = stage in ("scene_import_and_layout", "mesh_modularity_and_attachments")
        review.append({
            "asset_id": aid, "source_sha256": source_sha, "stage": stage,
            "observation_state": "IMPORTED_NUMERIC_AND_GRAPH_OBSERVATION" if known else "NEEDS_INDEPENDENT_COURT",
            "independent_court_state": "UNKNOWN", "review_receipt_sha256": "",
            "training_authorized": False,
            "reason": ("SOURCE_IMPORT_IS_NOT_NATIVE_PARITY" if known else "NOT_YET_INDEPENDENTLY_VALIDATED"),
        })
    if counts["mesh_disconnected_components"] > 1:
        review.append({"asset_id": aid, "source_sha256": source_sha,
                       "stage": "connected_islands_semantic_versus_drawable_test",
                       "observation_state": "MULTI_ISLANDS_NUMERIC_ONLY", "independent_court_state": "UNKNOWN",
                       "review_receipt_sha256": "", "training_authorized": False,
                       "reason": "DISCONNECTED_GEOMETRY_IS_NOT_SEMANTIC_LAYER"})
    if counts["image_dependency_missing_count"]:
        review.append({"asset_id": aid, "source_sha256": source_sha,
                       "stage": "texture_relink_and_material_parity",
                       "observation_state": "MISSING_IMPORTED_EXTERNAL_IMAGE_REFERENCE", "independent_court_state": "UNKNOWN",
                       "review_receipt_sha256": "", "training_authorized": False,
                       "reason": "SOURCE_TEXTURE_PIXELS_MUST_BE_RECOVERED"})
    return source_rec, review


def run(physical: Path, wave_roots: tuple[Path, ...], out: Path, *, require_total: int = 2466, core: Path | None = None):
    physical_rows = load_csv(physical)
    if len(physical_rows) != 3993:
        fail("MASTER_3993_PHYSICAL_ROSTER")
    physical_map = {r["asset_id"]: r for r in physical_rows}
    if len(physical_map) != len(physical_rows):
        fail("DUPLICATE_ASSET_ROSTER")
    success_ids = {aid for aid, row in physical_map.items() if row["census_state"] == "IMPORTED_NUMERIC_OBSERVATION_ONLY"}
    if len(success_ids) != require_total:
        fail("EXPECTED_IMPORTED_SUBJECT_COUNT:" + str(len(success_ids)))
    detail: dict[str, dict] = {}
    reviews: list[dict] = []
    source_gz_hashes = set()
    for root in wave_roots:
        if not root.is_dir():
            fail("MISSING_WAVE_DIR:" + root.name)
        for summary_path in sorted(root.rglob("asset_*.summary.json")):
            summary = json.loads(summary_path.read_text("utf-8"))
            aid = summary.get("asset_id")
            if aid not in success_ids or aid in detail:
                fail("UNKNOWN_OR_DUPLICATE_RECEIPT:" + str(aid))
            gz = summary_path.with_name(aid + ".full.source.json.gz")
            if not gz.is_file():
                fail("MISSING_FULL_GZ:" + aid)
            digest = sha(gz)
            if digest in source_gz_hashes:
                fail("REPEATED_FULL_RECEIPT_BYTES:" + aid)
            source_gz_hashes.add(digest)
            src, todo = analyze_one(physical_map[aid], summary, read_gzip_json(gz), digest, sha(summary_path))
            detail[aid] = src
            reviews.extend(todo)
    if set(detail) != success_ids:
        fail("INCOMPLETE_FULL_RECEIPT_COHORT:missing=" + str(len(success_ids - set(detail))))
    out.mkdir(parents=True, exist_ok=True)
    deep_path = out / "MASTER2466_FULL_SOURCE_DEEP_AUDIT_V1.jsonl"
    with deep_path.open("w", encoding="utf-8") as f:
        for aid in sorted(detail):
            f.write(json.dumps(detail[aid], ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    review_path = out / "MASTER3993_ALL_STAGE_REVIEW_TASKS_V1.csv"
    summary_rows = []
    shape_counts = collections.Counter()
    for row in physical_rows:
        aid = row["asset_id"]
        a = detail.get(aid)
        structural = a["structural_class"] if a else "SOURCE_SCENE_NOT_OBSERVED"
        shape_counts[structural] += 1
        all_counts = a["counts"] if a else {}
        summary_rows.append({
            "corpus": "MASTER3993", "asset_id": aid, "source_sha256": row["source_sha256"],
            "provider": row["provider"], "historical_split": row["historical_split"],
            "imported_scene_deep_checked": bool(a),
            "imported_structure_class": structural,
            "mesh_objects": all_counts.get("mesh_object_count", ""),
            "geometric_islands": all_counts.get("mesh_disconnected_components", ""),
            "mesh_boundary_edges": all_counts.get("mesh_boundary_edges", ""),
            "nonmanifold_edges": all_counts.get("mesh_nonmanifold_edges", ""),
            "armatures": all_counts.get("armature_count", ""),
            "bones": all_counts.get("observed_bone_count", ""),
            "skin_rows": all_counts.get("skin_rows_sampled", ""),
            "actions": all_counts.get("observed_action_count", ""),
            "animated_bone_name_count_teacher_only": a["observed_animated_bone_name_count_teacher_only"] if a else "",
            "constraints_count_imported": a["counts"]["observed_ik_constraint_count"] if a else "",
            "driver_channels_imported": all_counts.get("observed_driver_channel_count", ""),
            "image_references": all_counts.get("image_count", ""),
            "missing_texture_refs": all_counts.get("image_dependency_missing_count", ""),
            "character_vs_prop": "UNKNOWN", "morphology": "UNKNOWN",
            "authored_pose_front_up": "UNKNOWN", "semantic_source_face_truth": "UNKNOWN",
            "drawable_layerability": "UNKNOWN", "2d_style_suitability": "UNKNOWN",
            "independent_source_court_passed": False,
            "training_authorized": False,
        })
        if not a:
            for stage in STAGES:
                reviews.append({"asset_id": aid, "source_sha256": row["source_sha256"],
                                "stage": stage, "observation_state": "IMPORT_NOT_OBSERVED",
                                "independent_court_state": "UNKNOWN", "review_receipt_sha256": "",
                                "training_authorized": False,
                                "reason": "SOURCE_RECOVERY_OR_IMPORT_REQUIRES_REVIEW"})
    write_csv(out / "MASTER3993_DEEP_AUDIT_COVERAGE_V1.csv", summary_rows)
    write_csv(review_path, reviews)
    total_master_review_tasks = len(reviews)
    core_count = 0
    if core is not None:
        cohort = load_csv(core)
        if len(cohort) != 370 or len({r["subject_id"] for r in cohort}) != 370:
            fail("CORE370_ROSTER_COUNT_OR_IDENTITY")
        if any(r["training_authorized"] != "False" for r in cohort):
            fail("CORE370_PREAUTHORIZED")
        core_count = 370
        for r in cohort:
            aid = r["subject_id"]
            row = {col: "" for col in summary_rows[0]}
            row.update({
                "corpus": "CORE370", "asset_id": aid, "source_sha256": r["source_sha256"],
                "provider": "CORE370_HISTORICAL_SOURCE_ORIGIN_" + r.get("source_origin", "UNKNOWN"),
                "historical_split": "SEPARATE_CORE370_SPLIT_TO_RECONCILE",
                "imported_scene_deep_checked": False,
                "imported_structure_class": "CORE370_HISTORICAL_EVIDENCE_NOT_JOINED_TO_THIS_IMPORT_AUDIT",
                "character_vs_prop": "UNKNOWN", "morphology": "UNKNOWN",
                "authored_pose_front_up": "UNKNOWN", "semantic_source_face_truth": "UNKNOWN",
                "drawable_layerability": "UNKNOWN", "2d_style_suitability": "UNKNOWN",
                "independent_source_court_passed": False, "training_authorized": False,
            })
            summary_rows.append(row)
            for stage in STAGES:
                reviews.append({
                    "asset_id": aid, "source_sha256": r["source_sha256"],
                    "stage": stage, "observation_state": "CORE370_HISTORICAL_EVIDENCE_NOT_BOUND",
                    "independent_court_state": "UNKNOWN", "review_receipt_sha256": "",
                    "training_authorized": False, "reason": "CORE370_PR327_REVALIDATION_AND_53_SHA_LINEAGE_OVERLAY_PENDING",
                })
        write_csv(out / "ALL_4363_COHORT_DEEP_AUDIT_COVERAGE_V1.csv", summary_rows)
        write_csv(out / "ALL_4363_COHORT_REVIEW_TASKS_V1.csv", reviews)
    manifest = {
        "schema": SCHEMA, "asset_records": len(physical_rows), "core370_historical_records": core_count,
        "all_corpus_source_records": len(physical_rows) + core_count,
        "full_import_evidence_verified": len(detail),
        "not_yet_imported": len(physical_rows) - len(detail),
        "source_structure_categories": dict(sorted(shape_counts.items())),
        "asset_type_morphology_court_pass": 0, "source_authored_pose_court_pass": 0,
        "semantic_source_face_court_pass": 0, "drawable_consequence_court_pass": 0,
        "styled_2d_game_art_court_pass": 0, "native_ik_control_functionality_court_pass": 0,
        "review_task_count": total_master_review_tasks, "all_corpus_review_task_count": len(reviews),
        "review_stages_per_asset": list(STAGES),
        "recommended_visual_review_taxonomy": list(MORPH_CLASS),
        "native_source_original_bytes_not_read": True,
        "imported_full_scene_receipts_sha_verified": True,
        "no_visual_review_falsely_claimed": True,
        "output_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()},
        "training_authorized": False, "materialization_authorized": False,
    }
    (out / "MASTER3993_DEEP_AUDIT_SUMMARY_V1.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--core", type=Path)
    for i in range(1, 5):
        p.add_argument(f"--wave{i}", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.physical, tuple(getattr(args, f"wave{i}") for i in range(1, 5)), args.out, core=args.core), indent=2))

if __name__ == "__main__":
    main()