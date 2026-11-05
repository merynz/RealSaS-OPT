#!/usr/bin/env python3
"""V15: source-native generic control -> directly visible / isolated RGBA effect.

SOURCE INPUT: SHA-locked real V14 original native render/source vs soup,
original private independent court, and already-sealed public aggregate.
This is a source TEACHER diagnostic, never model input or drawable truth.

Output: per-pose/view/sign binary image-effect signatures for two ANONYMOUS
source-native controls, with original-source and independently isolated face
responses. No segmentation K, author attachment/bone names or semantic tags.
The private NPZ remains in Drive; the public report records non-owner counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "RealSaS.Corpus370V15AnonymousNativeCausalVisualSupportProposal.v1"
PARENT = "RealSaS.Corpus370V14OriginalNativeMechanicalRGBAIndependentCourtRealResult.v1"
RENDER = "RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14"
BATCH = "RealSaS.Corpus370MechanicalRasterJointPrivateBatch.v14"
COURT = "RealSaS.Corpus370SameNativeControlGeometricRasterCourt.v14"
V14_PASS = "V14_REAL_CAUSAL_SOURCE_MECHANICAL_TO_RGBA_JOIN_QUALIFIED__OWNER_UNKNOWN"
EFFECT_TOL = .025
ALPHA_TOL = .025


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()


def read(path):
    with Path(path).open("r", encoding="utf-8") as fd:
        return json.load(fd)


def write(path, data):
    Path(path).write_text(json.dumps(data, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")


def verified_parent(args):
    public = read(args.parent_public)
    require(public.get("schema") == PARENT
            and public.get("status") ==
            "REAL_ORIGINAL_SOURCE_NATIVE_CONTROL_TO_RGBA_CAUSAL_REPAIR_3_OF_3_COURTS_PASS__DRAWABLE_OWNER_UNKNOWN"
            and public.get("corrected_real_blender_run_id") == 42075029264
            and public.get("graphics_owner_authorized") is False
            and public.get("iris_training_authorized") is False
            and public.get("tessa_training_authorized") is False
            and public.get("materialization_authorized") is False
            and public.get("independent_artist_same_drawable_pairs") == 0
            and public.get("independent_artist_different_drawable_pairs") == 0,
            "V15_FORGED_PUBLIC_V14_CAUSAL_AUTHORITY")
    rows = public["source_results"]
    require(len(rows) == 3 and [r["ordinal"] for r in rows] == [0, 1, 2]
            and type(args.ordinal) is int and 0 <= args.ordinal < 3,
            "V15_SOURCE_ORDINAL_AND_ROSTER")
    row = rows[args.ordinal]
    batch = read(args.parent_batch)
    render = read(args.parent_manifest)
    court = read(args.parent_court)
    require(sha(args.parent_batch) == row["private_batch_sha256"]
            and sha(args.parent_court) == row["private_independent_court_sha256"]
            and sha(args.parent_rgba) == row["private_native_rgba_npz_sha256"]
            and sha(args.parent_manifest) == batch["source_native_render_manifest_sha256"]
            and batch["schema"] == BATCH
            and batch["status"] ==
               "MEASURED_REAL_SAME_CAUSAL_NATIVE_CONTROL_RGBA_V14__OWNER_UNKNOWN"
            and court["schema"] == COURT and court["status"] == V14_PASS
            and court["all_original_source_camera_cases_passed"] is True
            and render["schema"] == RENDER
            and render["source_native_control_raster_animated_action_frozen_then_restored"] is True
            and render["source_native_perturbed_vertices_verified_before_and_after_render"] is True
            and render["original_perturbed_pose_rgba_npz_sha256"] ==
                row["private_native_rgba_npz_sha256"]
            and render["subject_id"] == batch["subject_id"] == court["subject_id"] ==
                row["subject_id"]
            and render["source_sha256"] == batch["source_sha256"] ==
                court["source_sha256"] == row["source_sha256"]
            and render["graphics_owner_authorized"] is False
            and render["training_authorized"] is False
            and batch["graphics_owner_authorized"] is False
            and batch["training_authorized"] is False
            and court["graphics_owner_authorized"] is False
            and court["training_authorized"] is False
            and court["artist_same_drawable_pairs"] ==
                court["artist_different_drawable_pairs"] == 0
            and court["baseline_source_vs_soup_and_root_noop_qualified_pose_views"] == 12
            and court["original_native_perturbation_vs_source_soup_qualified_camera_cases"] == 48
            and court["original_native_perturbation_vs_source_soup_total_camera_cases"] == 48
            and court["selected_source_control_count"] == 2
            and court["source_original_camera_views"] == [0, 2, 4, 6]
            and render["rendered_source_camera_indices"] == [0, 2, 4, 6]
            and render["rendered_pose_indices"] ==
                render["selected_source_controls"]["native_pose_indices_teacher_only"]
            and render["selected_source_controls"]["selected_control_count"] == 2
            and batch["independent_court_sha256"] == sha(args.parent_court)
            and batch["source_native_rgba_npz_sha256"] == sha(args.parent_rgba),
            "V15_V14_REAL_NATIVE_EFFECT_PARENT_SHA_OR_AUTHORITY")
    from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan
    arms, _ = arms_for_plan(render["selected_source_controls"])
    require(render["rendered_arms"] == arms and len(arms) == 19
            and render["source_active_faces_are_not_artist_drawables"] is True
            and render["artist_same_drawable_pairs"] ==
                render["artist_different_drawable_pairs"] == 0,
            "V15_ORIGINAL_NATIVE_ACTION_ARMS_OR_DRAWABLE_LEAK")
    with np.load(args.parent_rgba, allow_pickle=False) as d:
        require(set(d.files) == {"original_native_actuator_rgba"},
                "V15_PRIVATE_SOURCE_V14_ARRAY_UNEXPECTED_MEMBERS")
        x = np.asarray(d["original_native_actuator_rgba"], np.float32)
    require(x.shape == (3, 4, 19, 192, 192, 4)
            and np.isfinite(x).all()
            and float(x[..., 3].min()) >= -1e-5
            and float(x[..., 3].max()) <= 1.00001,
            "V15_PRIVATE_ORIGINAL_RGBA_DIMENSIONS")
    return public, row, batch, render, court, arms, x


def predict(x, arms):
    """Only the V14 image tensor; no K, author source control names or labels."""
    ix = {name: k for k, name in enumerate(arms)}
    require(x.shape == (3, 4, len(arms), 192, 192, 4)
            and len(arms) == 19, "V15_IMAGE_ONLY_EFFECT_PROBE_SHAPE")
    whole = np.zeros((3, 4, 2, 2, 192, 192), dtype=np.uint8)
    isolated = whole.copy()
    alpha = whole.copy()
    rows = []
    pair_rows = []
    for p in range(3):
        for v, view in enumerate((0, 2, 4, 6)):
            base = x[p, v, ix["ORIGINAL_BASE"]]
            for ci in range(2):
                ab = x[p, v, ix[f"ACTIVE_C{ci:02d}_BASE"]]
                for si, sign in enumerate(("PLUS", "MINUS")):
                    rendered = x[p, v, ix[f"SOURCE_C{ci:02d}_{sign}"]]
                    active = x[p, v, ix[f"ACTIVE_C{ci:02d}_{sign}"]]
                    whole[p, v, ci, si] = (
                        np.max(np.abs(rendered - base), axis=-1) > EFFECT_TOL
                    ).astype(np.uint8)
                    isolated[p, v, ci, si] = (
                        np.max(np.abs(active - ab), axis=-1) > EFFECT_TOL
                    ).astype(np.uint8)
                    alpha[p, v, ci, si] = (
                        np.abs(rendered[..., 3] - base[..., 3]) > ALPHA_TOL
                    ).astype(np.uint8)
                    w = whole[p, v, ci, si].astype(bool)
                    iso = isolated[p, v, ci, si].astype(bool)
                    aa = alpha[p, v, ci, si].astype(bool)
                    require(np.all(~aa | w), "V15_ALPHA_RESPONSE_NOT_SUBSET_OF_RGBA")
                    rows.append({
                        "pose_ordinal": p, "source_camera_view": view,
                        "anonymous_control_ordinal": ci, "sign": sign,
                        "visible_whole_source_change_pixel_instances": int(w.sum()),
                        "isolated_active_source_face_change_pixel_instances": int(iso.sum()),
                        "isolation_only_changed_pixel_instances":
                            int(np.count_nonzero(iso & ~w)),
                        "alpha_changed_pixel_instances": int(aa.sum()),
                        "source_effect_visibility": (
                            "OBSERVED_IN_ORIGINAL_RGBA" if w.any()
                            else "NO_VISIBLE_EFFECT_IN_THIS_POSE_AND_VIEW"
                        ),
                        "artist_drawable_owner": "UNKNOWN"
                    })
            for sign_a, aa in enumerate(("PLUS", "MINUS")):
                for sign_b, bb in enumerate(("PLUS", "MINUS")):
                    a = whole[p, v, 0, sign_a].astype(bool)
                    b = whole[p, v, 1, sign_b].astype(bool)
                    both = int(np.count_nonzero(a & b))
                    only_a = int(np.count_nonzero(a & ~b))
                    only_b = int(np.count_nonzero(b & ~a))
                    total = both + only_a + only_b
                    pair_rows.append({
                        "pose_ordinal": p, "source_camera_view": view,
                        "anonymous_control_a": 0, "anonymous_control_b": 1,
                        "sign_a": aa, "sign_b": bb,
                        "shared_effect_pixel_instances": both,
                        "a_exclusive_pixel_instances": only_a,
                        "b_exclusive_pixel_instances": only_b,
                        "union_pixel_instances": total,
                        "support_iou": float(both / total) if total else None,
                        "support_relation": (
                            "DISTINGUISHABLE_VISIBLE_EFFECT_SUPPORT"
                            if only_a + only_b > 0 else
                            "INDISTINGUISHABLE_IN_THIS_OBSERVATION"
                            if total else "NO_OBSERVABLE_EFFECT"
                        ),
                        "artist_same_drawable_label": "UNKNOWN",
                        "artist_different_drawable_label": "UNKNOWN"
                    })
    require(len(rows) == 48 and len(pair_rows) == 48,
            "V15_POSE_CAMERA_SIGN_SOURCE_RESPONSE_COVERAGE")
    totals = lambda a: [int(a[:, :, i].sum()) for i in range(2)]
    aggregates = {
        "whole_effect_pixel_instances_by_anonymous_control": totals(whole),
        "active_isolated_effect_pixel_instances_by_anonymous_control": totals(isolated),
        "alpha_response_pixel_instances_by_anonymous_control": totals(alpha),
        "isolation_only_pixel_instances": int(np.count_nonzero((isolated != 0) & (whole == 0))),
        "controls_with_positive_direct_effect": int(sum(z > 0 for z in totals(whole))),
        "pair_conditions_with_distinguishable_visual_support":
            int(sum(z["support_relation"] == "DISTINGUISHABLE_VISIBLE_EFFECT_SUPPORT"
                    for z in pair_rows)),
        "pair_conditions_total": len(pair_rows),
        "poses": 3, "views": [0, 2, 4, 6], "source_control_ordinals": [0, 1],
        "signs": ["PLUS", "MINUS"]
    }
    return (whole, isolated, alpha), rows, pair_rows, aggregates


def run(args):
    public, row, batch, render, court, arms, x = verified_parent(args)
    masks, per_case, pair_cases, totals = predict(x, arms)
    require(totals["whole_effect_pixel_instances_by_anonymous_control"] ==
                row["source_perturbed_control_visible_pixel_instances"] ==
                court["whole_source_perturbation_effective_pixel_instances_by_selected_control"]
            and totals["active_isolated_effect_pixel_instances_by_anonymous_control"] ==
                row["causal_active_face_isolation_pixel_instances"] ==
                court["causal_active_face_isolation_changed_pixel_instances_by_selected_control"],
            "V15_RECOMPUTED_RAW_RGBA_MUST_EQUAL_INDEPENDENT_V14_COURT")
    for item, receipt in zip(per_case, court["selected_source_native_control_paired_rgba"]):
        require(item["pose_ordinal"] == receipt["native_pose_ordinal"]
                and item["source_camera_view"] == receipt["source_camera_view"]
                and item["anonymous_control_ordinal"] ==
                    receipt["precommitted_source_control_ordinal"]
                and item["sign"] == receipt["perturbation_sign"]
                and item["visible_whole_source_change_pixel_instances"] ==
                    receipt["whole_source_perturbation_changed_pixel_instances"]
                and item["isolated_active_source_face_change_pixel_instances"] ==
                    receipt["causal_source_face_active_isolation_changed_pixel_instances"]
                and receipt["source_perturbation_soup_parity_qualified"] is True
                and receipt["artist_drawable_label"] == "UNKNOWN",
                "V15_V14_INDEPENDENT_48_CAUSAL_IMAGE_CASE_REPLAY_MISMATCH")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_arrays = args.output_dir / "PRIVATE_V15_ANONYMOUS_NATIVE_VISUAL_SUPPORT.npz"
    np.savez_compressed(private_arrays, whole_effect=masks[0],
                        isolated_effect=masks[1], alpha_effect=masks[2])
    proposal = {
        "schema": SCHEMA,
        "status": "V15_REAL_ORIGINAL_NATIVE_CAUSAL_RGBA_SUPPORT_PROPOSED__INDEPENDENT_COURT_REQUIRED",
        "source": row["source"],
        "subject_id": row["subject_id"],
        "source_sha256": row["source_sha256"],
        "source_ordinal": args.ordinal,
        "original_V14_run_id": 42075029264,
        "parents": {
            "public_v14_real_result_sha256": sha(args.parent_public),
            "private_v14_batch_sha256": sha(args.parent_batch),
            "private_v14_manifest_sha256": sha(args.parent_manifest),
            "private_v14_court_sha256": sha(args.parent_court),
            "private_v14_original_rgba_npz_sha256": sha(args.parent_rgba)
        },
        "private_prediction_npz_sha256": sha(private_arrays),
        "evidence_threshold_rgba": EFFECT_TOL,
        "evidence_threshold_alpha": ALPHA_TOL,
        "anonymous_control_count": 2,
        "model_source_control_type_or_name_seen": False,
        "source_editor_art_slots_seen": False,
        "fixed_layer_slot_count": False,
        "full_resolution_caa_proven": False,
        "depth_draw_order_proven": False,
        "cross_control_joint_AB_intervention_measured": False,
        "geometry_source_control_is_artist_drawable": False,
        "original_author_same_drawable_labels": 0,
        "original_author_different_drawable_labels": 0,
        "model_training_authorized": False,
        "graphics_owner_authorized": False,
        "materialization_authorized": False,
        "case_rows": per_case,
        "pair_response_support_rows": pair_cases,
        "aggregates": totals
    }
    p = args.output_dir / "PRIVATE_V15_NATIVE_VISUAL_SUPPORT_PROPOSAL.json"
    write(p, proposal)
    print(json.dumps({
        "subject_id": row["subject_id"], "case_rows": len(per_case),
        "pair_cases": len(pair_cases),
        "original_V14_renderer_qualified": True,
        "whole_control_effects": totals["whole_effect_pixel_instances_by_anonymous_control"],
        "isolated_only": totals["isolation_only_pixel_instances"],
        "owner_minted": False
    }, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ordinal", type=int, required=True)
    for name in ("parent_public", "parent_batch", "parent_manifest",
                 "parent_court", "parent_rgba"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
