#!/usr/bin/env python3
"""Source-only, typed graphics edit-consequence ledger from independent courts.

Extends the already existing sparse V18j Graphics Evidence IR v1 with *different*
source-rest, camera, Cycles and operation evidence. Does NOT join observations
at pixel/pose level, infer art drawables, or mint per-head training targets.
This is a family-candidate *bridge*, NOT final Corpus370 authority matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.Corpus370SourceGraphicsEditConsequenceLedger.v1"
STATUS = "SOURCE_GRAPHICS_CONSEQUENCE_COURTED__FINAL_FAMILY_AUTHORITY_PENDING"
PARENT_PALETTE = "RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1"
PARENT_RENDER = "RealSaS.Corpus370OriginalThreePaletteCommonCyclesSampleEditEvidence.v1"
PARENT_COURT = "RealSaS.Corpus370IndependentOriginalThreePaletteCyclesEditCourt.v1"
PARENT_NATIVE = "RealSaS.Corpus370IndependentOperationalNativeSupportCourt.v1"
REQUIRED_FAMILIES = (
    "APPEARANCE_CAA", "VISIBILITY_OCCLUSION", "RAY_COMPOSITING", "CAUSAL_RESPONSE"
)
FORBIDDEN_FLAGS = (
    "graphics_owner_authorized", "iris_training_authorized",
    "tessa_training_authorized", "materialization_authorized", "training_authorized"
)
UNKNOWN_DOMAINS = (
    "SOURCE_ARTIST_DRAWABLE_OWNERSHIP",
    "OPERATIONAL_SAME_DIFFERENT_DRAWABLE_EDGES",
    "ATTACHMENT_AND_DETACHMENT",
    "SECONDARY_MOTION_AND_SKIN_DEFORMATION",
    "POSE_PERSISTENT_GRAPHICS_RELATIONS",
    "TEXTURE_UV_LOCAL_RECOLOR",
    "NONBINARY_ALPHA_AND_SHADOW_INTERACTION",
    "ONE_VIEW_STUDENT_INFERENCE",
)

def require(ok, code):
    if not ok:
        raise ValueError("GRAPHICS_CONSEQUENCE_LEDGER_" + code)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def serial(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

def compose(palette, renderer, court, native, contract, *, parents, subject_id):
    require(palette.get("schema") == PARENT_PALETTE and
            renderer.get("schema") == PARENT_RENDER and
            court.get("schema") == PARENT_COURT and
            native.get("schema") == PARENT_NATIVE,
            "SOURCE_PARENT_SCHEMAS")
    require(court.get("status") ==
            "PASS_ORIGINAL_PHYSICAL_K3_SAME_CYCLES_SAMPLE_EDIT_COURT__OWNER_UNKNOWN"
            and court.get("failure_reasons") == [] and
            renderer.get("status") ==
            "SOURCE_ORIGINAL_K3_CYCLES_EDIT_SAMPLES_RENDERED__COURT_REQUIRED",
            "INDEPENDENT_RENDER_COURT_NOT_PASS")
    for obj in (palette, renderer, court, native):
        require(all(obj.get(flag) is False for flag in FORBIDDEN_FLAGS),
                "SOURCE_PARENT_AUTHORITY_FORGED")
    source = palette.get("original_source_sha256")
    require(isinstance(source, str) and len(source) == 64 and
            source == renderer.get("original_source_sha256") ==
            court.get("original_source_sha256") ==
            native.get("original_source_sha256") ==
            parents["original_source_sha256"] and
            subject_id == native.get("subject_id_private"),
            "ORIGINAL_SOURCE_SUBJECT_SHA_BIND")
    require(court.get("original_source_renderer_manifest_sha256") ==
            parents["renderer_manifest_sha256"] and
            court.get("original_source_palette_facts_sha256") ==
            parents["palette_facts_sha256"] ==
            renderer.get("source_original_material_palette_sha256") and
            court.get("original_source_renderer_rgba_npz_sha256") ==
            renderer.get("source_original_edited_sample_npz_sha256"),
            "INDEPENDENT_RENDER_SAMPLE_SHA_BIND")
    k = palette.get("palette_operation_count")
    faces = palette.get("source_face_count")
    labels = palette.get("face_palette_color_key_index_or_null")
    validity = palette.get("face_palette_validity")
    counts = renderer.get("source_original_triangles_per_color")
    colors = palette.get("constant_palette_color_keys_hex_rgba")
    require(type(k) is int and k > 0 and type(faces) is int and faces > 0 and
            type(labels) is list and type(validity) is list and
            len(labels) == len(validity) == faces and
            all(type(x) is bool for x in validity) and
            type(counts) is list and type(colors) is list and
            len(colors) == len(counts) == k and
            len(set(tuple(x) for x in colors)) == k and
            all(type(x) is int and 0 <= x < k for x in labels) and
            all(validity) and
            [labels.count(i) for i in range(k)] == counts and
            sum(counts) == faces == native.get("source_face_count") and
            palette.get("qualified_constant_face_count") == faces and
            palette.get("unsupported_face_count") == 0 and
            court.get("source_group_count_K") == k,
            "SOURCE_FACE_PALETTE_AND_PHYSICAL_ISOLATION_CENSUS")
    views = renderer.get("source_cycle_camera_view_indices")
    offsets = renderer.get("source_cycle_sample_indices")
    rows = court.get("source_per_view_results")
    resolution = renderer.get("source_cycle_resolution")
    require(type(views) is list and len(views) > 0 and
            len(set(views)) == len(views) and
            all(type(v) is int for v in views) and
            type(offsets) is list and len(offsets) > 0 and
            len(set(offsets)) == len(offsets) and
            all(type(v) is int for v in offsets) and
            type(resolution) is int and resolution > 0 and
            renderer.get("source_original_rest_pose_only") is True and
            renderer.get("source_original_artwork_heldout_pose_test") is False and
            renderer.get("physical_nonselected_faces_removed") is True and
            renderer.get("source_full_and_edited_full_are_court_targets_only") is True and
            court.get("producer_court_source_full_not_used_to_choose_depth") is True and
            court.get("original_camera_views") == views and
            court.get("cycles_indices") == offsets and
            court.get("source_original_rest_only") is True and
            type(rows) is list and len(rows) == len(views),
            "SOURCE_REST_SAMPLE_CONTEXT_AND_HOLDOUT")
    require(contract.get("schema") == "RealSaS.CorpusTruthAuthorityContract.v1" and
            all(f in contract.get("family_order", []) for f in REQUIRED_FAMILIES) and
            "PENDING" in contract.get("family_authority_states", []) and
            contract.get("authority_boundary", {}).get("this_contract_mints_training_readiness") is False,
            "FINAL_MATRIX_CONTRACT_INCOMPATIBLE")

    view_rows = []
    known_total = unknown_total = overlap_total = 0
    per_color = [0 for _ in range(k)]
    expected_arms = ["SOURCE_FULL"] + [f"SOURCE_EDITED_FULL_{i}" for i in range(k)]
    for v, row in zip(views, rows):
        require(row.get("source_view_index") == v and
                row.get("source_full_alpha_intermediate_single_source_sample_count") == 0 and
                row.get("source_isolate_nonbinary_alpha_count") == 0 and
                row.get("source_invalid_depth_sample_count") == 0,
                "PER_VIEW_SOURCE_OR_ALPHA_DOMAIN")
        metrics = row.get("metric_rows")
        require(type(metrics) is list and
                [m.get("arm") for m in metrics] == expected_arms,
                "SOURCE_EDIT_ARMS_NOT_EXACT")
        expected_count = len(offsets) * resolution * resolution
        qualified = metrics[0].get("qualified_source_sample_pixels")
        unknown = metrics[0].get("unknown_source_samples")
        overlap = row.get("source_multi_group_overlapping_sample_count")
        require(all(type(x) is int and x >= 0 for x in (qualified, unknown, overlap)) and
                qualified + unknown == expected_count and
                overlap <= qualified and
                row.get("source_unknown_depth_tie_sample_count") == unknown,
                "SOURCE_SAMPLE_MASK_CENSUS")
        q8 = metrics[0].get("fully_qualified_eight_sample_pixels")
        require(type(q8) is int and 0 <= q8 <= resolution * resolution,
                "SOURCE_EIGHT_SAMPLE_CENSUS")
        effects = []
        for i, item in enumerate(metrics):
            changed = item.get("physical_full_edit_changed_sample_pixel_count")
            require(item.get("qualified_source_sample_pixels") == qualified and
                    item.get("unknown_source_samples") == unknown and
                    item.get("fully_qualified_eight_sample_pixels") == q8 and
                    item.get("sample_rgba_mismatch_count") == 0 and
                    item.get("eight_sample_mean_rgba_mismatch_count") == 0 and
                    type(changed) is int and 0 <= changed <= qualified and
                    (i != 0 or changed == 0),
                    "SAMPLE_EDIT_CONSEQUENCE_NOT_INDEPENDENTLY_COURTED")
            if i:
                per_color[i-1] += changed
                effects.append({
                    "source_palette_local_index": i - 1,
                    "source_color_hex_rgba": colors[i-1],
                    "observed_changed_source_sample_pixels": changed,
                    "full_edit_reconstruction_mismatches_at_court_tolerance": 0,
                    "relation_to_art_drawable_owner": "UNKNOWN",
                })
        view_rows.append({
            "source_pose_scope": "ORIGINAL_REST_ONLY",
            "source_camera_view": v,
            "sample_index_policy": "SAME_CYCLES_SOURCE_SAMPLE_INDICES",
            "original_cycles_sample_indices": offsets,
            "spatial_granularity": "VIEW_AGGREGATE__NOT_PER_PIXEL_OWNER_OR_PER_RAY_MASK",
            "source_depth_and_visibility": {
                "qualified_sample_count": qualified,
                "unknown_sample_count": unknown,
                "multi_group_source_geometry_overlap_samples": overlap,
                "individual_source_depth_stacks_in_this_ledger": "NOT_INCLUDED",
            },
            "source_rgba_recomposition": {
                "qualified_sample_count": qualified,
                "mismatch_at_original_court_tolerance": 0,
                "sample_mean_mismatch_at_original_court_tolerance": 0,
                "tolerance_source_parent": "INDEPENDENT_K3_COURT__0P025_MAX_PREMULTIPLIED_RGBA",
            },
            "source_palette_edits": effects,
        })
        known_total += qualified
        unknown_total += unknown
        overlap_total += overlap
    require(known_total == court.get("qualified_sample_total") and
            overlap_total == court.get("original_physical_overlapping_source_sample_count") and
            unknown_total == len(views) * len(offsets) * resolution * resolution - known_total and
            all(x >= 8 for x in per_color) and overlap_total >= 8,
            "GLOBAL_SOURCE_COVERAGE_OR_EDIT_NONVACUITY")
    source_families = {
        "APPEARANCE_CAA": ("SOURCE_CONSTANT_SHADER_PALETTE_FACE_OBSERVATION", faces),
        "VISIBILITY_OCCLUSION": ("SOURCE_SAMPLE_MULTI_GROUP_GEOMETRIC_OVERLAP", overlap_total),
        "RAY_COMPOSITING": ("SOURCE_ORIGINAL_K_DEPTH_ORDERED_RGBA_RECOMPOSITION", known_total),
        "CAUSAL_RESPONSE": ("SOURCE_RGB_PALETTE_INTERVENTION_RENDER_CONSEQUENCE", sum(per_color)),
    }
    return {
        "schema": SCHEMA, "status": STATUS,
        "subject_id": subject_id, "original_source_sha256": source,
        "original_source_teacher_private": True,
        "parent_receipts": parents,
        "original_source_group_count_K": k,
        "palette_group_kind": "EXACT_SOURCE_CONSTANT_COLOR_EQUIVALENCE_NOT_ART_PART",
        "original_source_palette_face_membership": {
            "qualified_source_faces": faces, "unknown_source_faces": 0,
            "source_triangles_per_palette_color": counts,
            "sample_local_color_groups_not_persistent_art_drawables": True,
        },
        "original_source_view_edit_consequence": view_rows,
        "original_source_sample_totals": {
            "qualified": known_total, "unknown": unknown_total, "overlap": overlap_total,
            "independent_color_operations": k,
            "per_color_changed_sample_incidents": per_color,
        },
        "family_matrix_bridge": {
            family: {
                "candidate_source_observation": kind,
                "source_qualified_observation_count": count,
                "final_family_authority_state": "PENDING",
                "final_training_validity_mask_sha256": None,
                "student_access_to_original_teacher_source": False,
                "training_target_minted": False,
            } for family, (kind, count) in source_families.items()
        },
        "unmeasured_or_unqualified_domains": {
            kind: "UNKNOWN_NOT_EVALUATED_BY_REST_POSE_CONSTANT_COLOR_COURT"
            for kind in UNKNOWN_DOMAINS
        },
        "source_pose2_tested": False, "nonbinary_alpha_tested": False,
        "student_model_target_count": 0,
        "same_drawable_training_pairs_qualified": 0,
        "different_drawable_training_pairs_qualified": 0,
        "source_artist_drawable_owner": "UNKNOWN",
        "source_graphics_compiler_draw_order_global": "UNKNOWN",
        **{x: False for x in FORBIDDEN_FLAGS},
    }

def compile_from_paths(palette_path, render_path, court_path, native_path,
                       contract_path, *, subject_id, expected_court_sha):
    require(isinstance(expected_court_sha, str) and len(expected_court_sha) == 64 and
            sha(court_path) == expected_court_sha, "EXACT_INDEPENDENT_COURT_RECEIPT_SHA")
    p, r, c, n, t = (read(path) for path in (
        palette_path, render_path, court_path, native_path, contract_path))
    parents = {
        "original_source_sha256": p["original_source_sha256"],
        "palette_facts_sha256": sha(palette_path),
        "renderer_manifest_sha256": sha(render_path),
        "independent_render_court_sha256": sha(court_path),
        "independent_native_support_court_sha256": sha(native_path),
        "truth_authority_contract_sha256": sha(contract_path),
        "source_sample_npz_sha256_teacher_only":
            r["source_original_edited_sample_npz_sha256"],
        "original_rest_geometry_npz_sha256_teacher_only":
            r["source_original_rest_geometry_sha256"],
        "original_camera_bundle_sha256_teacher_only":
            r["source_original_camera_bundle_sha256"],
        "original_render_policy_sha256_teacher_only":
            r["source_original_render_policy_sha256"],
    }
    return compose(p, r, c, n, t, parents=parents, subject_id=subject_id)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("palette", "render-manifest", "independent-court",
                 "native-court", "truth-contract", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--expected-independent-court-sha256", required=True)
    a = ap.parse_args()
    evidence = compile_from_paths(a.palette, a.render_manifest,
        a.independent_court, a.native_court, a.truth_contract,
        subject_id=a.subject_id,
        expected_court_sha=a.expected_independent_court_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(serial(evidence))
    print(json.dumps({"status": evidence["status"],
        "source_qualified_samples": evidence["original_source_sample_totals"]["qualified"],
        "graphics_owner_authorized": False, "training_authorized": False}, sort_keys=True))

if __name__ == "__main__":
    main()
