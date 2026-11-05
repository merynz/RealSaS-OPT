"""Adversarial contracts for an entirely image-only generic split/merge proposal.

Synthetic tests never constitute proof of recovering human-authored drawables.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tools.corpus.artist_image_only_edit_baseline_v1 import (
    INPUT_SCHEMA, digest, write_prediction, court as court_v1,
)
from tools.corpus.artist_image_only_split_merge_v2 import (
    METHODS, PARAMS, PRED_SCHEMA, COURT_SCHEMA,
    build_seed, court_candidates, merge_regions,
    predict_candidates, supported_labels, write_candidate,
    score_edit_support,
)


def fixtures(tmp_path):
    model = tmp_path / "model_input"
    private = tmp_path / "private_artist_court"
    model.mkdir()
    private.mkdir()
    img = np.zeros((96, 96, 4), np.uint8)
    # Two disconnected opaque author-art zones, plus visible authored edits.
    img[10:48, 12:55] = (213, 72, 58, 255)
    img[18:33, 18:40] = (220, 88, 62, 255)  # soft shading, NOT extra author slot
    img[57:88, 61:91] = (35, 70, 222, 255)
    raw = img.tobytes()
    im_path = model / "rgba.bin"
    im_path.write_bytes(raw)
    pub = {
        "schema": INPUT_SCHEMA,
        "format": "RGBA8888_UNPREMULTIPLIED",
        "width": 96, "height": 96,
        "image_RGBA_SHA256": digest(raw),
        "consumer_image_only_source_input": True,
        "source_mesh_face_incidence_present": False,
        "original_artist_slot_names_present": False,
        "original_artist_layer_count_present": False,
        "source_bone_rig_or_attachment_names_present": False,
        "training_authorized": False,
    }
    pub_path = model / "manifest.json"
    pub_path.write_text(json.dumps(pub))
    refs = []
    for i in range(3):
        hidden = img.copy()
        isolated = np.zeros_like(img)
        if i == 0:
            hidden[10:48, 12:55] = 0
            isolated[10:48, 12:55] = img[10:48, 12:55]
        elif i == 1:
            hidden[57:88, 61:91] = 0
            isolated[57:88, 61:91] = img[57:88, 61:91]
        else:
            isolated[27:30, 27:30] = (220, 255, 20, 255)
        h, iso = hidden.tobytes(), isolated.tobytes()
        (private / f"slot{i}_hidden.rgba").write_bytes(h)
        (private / f"slot{i}_isolated.rgba").write_bytes(iso)
        refs.append({
            "original_slot_index_teacher_only": i,
            "original_hidden_source_RGBA_sha256": digest(h),
            "original_isolated_source_RGBA_sha256": digest(iso),
            "original_hide_changes_composite_RGBA_pixel_count":
                int(np.any(hidden != img, axis=-1).sum()),
            "original_isolated_alpha_pixel_count":
                int((isolated[..., 3] > 0).sum()),
        })
    private_json = private / "reference.json"
    private_json.write_text(json.dumps({
        "schema": "RealSaS.IndependentOriginalArtistSlotRGBAHideIsolateCourtOnly.v1",
        "original_model_input_RGBA_SHA256": digest(raw),
        "official_spine_runtime_reference": True,
        "original_hide_isolate_restoration_exact": True,
        "training_authorized": False,
        "native_editable_source_artist_project_file_verified": False,
        "active_original_author_slot_count_teacher_only": 3,
        "source_author_slot_hide_isolate_references_teacher_only": refs,
    }))
    b_npz, b_manifest, b_receipt = (
        model / "v1.npz", model / "v1.json", model / "v1_court.json"
    )
    c_npz, c_manifest = model / "v2.npz", model / "v2.json"
    write_prediction(im_path, pub_path, b_npz, b_manifest)
    b = court_v1(im_path, pub_path, b_npz, b_manifest, private_json, private)
    b_receipt.write_text(json.dumps(b))
    return {
        "image": im_path, "public": pub_path,
        "base_npz": b_npz, "base_manifest": b_manifest,
        "base_receipt": b_receipt, "candidate_npz": c_npz,
        "candidate_manifest": c_manifest, "private_json": private_json,
        "private": private, "rgba": img
    }


def produce(x):
    return write_candidate(x["image"], x["public"], x["base_npz"],
                           x["base_manifest"], x["candidate_npz"],
                           x["candidate_manifest"])


def court(x):
    return court_candidates(
        x["image"], x["public"], x["base_npz"],
        x["base_manifest"], x["base_receipt"],
        x["candidate_npz"], x["candidate_manifest"],
        x["private_json"], x["private"]
    )


def test_generic_split_merge_candidate_exact_image_only_and_court(tmp_path):
    x = fixtures(tmp_path)
    pred = produce(x)
    result = court(x)
    assert pred["schema"] == PRED_SCHEMA
    assert result["schema"] == COURT_SCHEMA
    assert result["observed_original_author_hide_effects"] == 2
    assert result["masked_original_author_hide_effects"] == 1
    assert result["candidate_methods_precommitted"] == list(METHODS)
    assert pred["source_artist_slot_name_or_count_read"] is False
    assert pred["native_source_controls_as_model_prior"] is False
    assert pred["original_artist_edit_reference_read_during_prediction"] is False
    assert result["training_authorized"] is False
    assert result["graphics_owner_authorized"] is False
    assert result["new_same_drawable_pairs_minted"] == 0
    assert result["new_different_drawable_pairs_minted"] == 0
    assert all(
        m["predicted_region_count"] >= 2
        for m in result["v2_image_only_candidate_results"].values()
    )


def test_source_image_alpha_background_never_merges_disconnected_islands(tmp_path):
    img = fixtures(tmp_path)["rgba"]
    mask = img[..., 3] > 0
    graph = build_seed(img, PARAMS["fine_graph"])
    regions, info = merge_regions(graph, img, PARAMS["merge_adaptive"])
    assert regions.shape == mask.shape
    assert np.all(regions[~mask] == -1)
    assert set(np.unique(regions[10:48, 12:55])).isdisjoint(
        set(np.unique(regions[57:88, 61:91]))
    )
    assert info["output_regions"] == info["input_regions"] - info[
        "accepted_source_image_only_merges"
    ]


def test_transparency_bridge_raw_graph_label_cut_to_true_foreground_components():
    mask = np.zeros((16, 16), bool)
    mask[1:4, 1:4] = True
    mask[12:15, 12:15] = True
    bogus_raw_segment = np.zeros((16, 16), np.int32)
    regions, n = supported_labels(bogus_raw_segment, mask)
    assert n == 2
    assert regions[2, 2] != regions[13, 13]
    assert np.all(regions[~mask] == -1)


def test_no_fixed_artist_part_count_and_deterministic_visual_optimization(tmp_path):
    img = fixtures(tmp_path)["rgba"]
    a, report_a = predict_candidates(img)
    b, report_b = predict_candidates(img)
    for m in METHODS:
        assert np.array_equal(a[m], b[m])
        assert report_a[m] == report_b[m]
        assert report_a[m]["output_regions"] <= report_a[m]["input_regions"]
        assert "part_count" not in json.dumps(PARAMS)
        assert "artist" not in json.dumps(PARAMS)


def test_independent_court_scoring_identical_on_parent_baselines(tmp_path):
    x = fixtures(tmp_path)
    from tools.corpus.artist_image_only_split_merge_v2 import (
        read_private_edit_effects
    )
    from tools.corpus.artist_image_only_edit_baseline_v1 import read_image
    rgba, pub = read_image(x["image"], x["public"])
    supports, masked, _ = read_private_edit_effects(
        rgba, pub, x["private_json"], x["private"]
    )
    with np.load(x["base_npz"], allow_pickle=False) as pred:
        for method in ("alpha_connected", "image_graph"):
            current = score_edit_support(
                supports, np.asarray(pred[method]), rgba[..., 3] > 0
            )
            sealed = json.loads(x["base_receipt"].read_text())[
                "metrics_mean_edit_support_iou_not_artist_membership_accuracy"
            ][method]
            assert abs(
                current["mean_one_to_one_edit_support_iou_over_all_observable_edits"] -
                sealed["mean_one_to_one_edit_support_iou_over_all_observable_edits"]
            ) < 1e-10
            assert abs(
                current["matched_edit_effect_pixel_recall_over_all_edit_effect_instances"] -
                sealed["matched_edit_effect_pixel_recall_over_all_edit_effect_instances"]
            ) < 1e-10


@pytest.mark.parametrize("attack", [
    "parent_baseline_corrupt", "parent_manifest_artist_leak",
    "candidate_npz_corrupt", "candidate_manifest_training",
    "candidate_manifest_owner", "candidate_manifest_edit_reference_used",
    "candidate_manifest_mechanics_prior", "candidate_manifest_slot_count",
    "candidate_manifest_forged_sha", "candidate_manifest_forged_parameters",
    "private_hide_byte_forgery", "private_author_count_forgery",
    "private_source_sha_forgery", "private_editor_claim",
    "parent_court_fake_owner", "parent_court_observable_drift",
])
def test_false_authority_source_leak_and_art_edit_court_fail_closed(tmp_path, attack):
    x = fixtures(tmp_path)
    produce(x)
    if attack == "parent_baseline_corrupt":
        with x["base_npz"].open("ab") as fd:
            fd.write(b"fake")
    elif attack == "candidate_npz_corrupt":
        with x["candidate_npz"].open("ab") as fd:
            fd.write(b"fake")
    elif attack == "private_hide_byte_forgery":
        with (x["private"] / "slot0_hidden.rgba").open("ab") as fd:
            fd.write(b"fake")
    else:
        key = {
            "parent_manifest_artist_leak": "base_manifest",
            "candidate_manifest_training": "candidate_manifest",
            "candidate_manifest_owner": "candidate_manifest",
            "candidate_manifest_edit_reference_used": "candidate_manifest",
            "candidate_manifest_mechanics_prior": "candidate_manifest",
            "candidate_manifest_slot_count": "candidate_manifest",
            "candidate_manifest_forged_sha": "candidate_manifest",
            "candidate_manifest_forged_parameters": "candidate_manifest",
            "private_author_count_forgery": "private_json",
            "private_source_sha_forgery": "private_json",
            "private_editor_claim": "private_json",
            "parent_court_fake_owner": "base_receipt",
            "parent_court_observable_drift": "base_receipt",
        }[attack]
        path = x[key]
        payload = json.loads(path.read_text())
        if attack == "parent_manifest_artist_leak":
            payload["training_authorized"] = True
        elif attack == "candidate_manifest_training":
            payload["training_authorized"] = True
        elif attack == "candidate_manifest_owner":
            payload["graphics_owner_authorized"] = True
        elif attack == "candidate_manifest_edit_reference_used":
            payload["original_artist_edit_reference_read_during_prediction"] = True
        elif attack == "candidate_manifest_mechanics_prior":
            payload["native_source_controls_as_model_prior"] = True
        elif attack == "candidate_manifest_slot_count":
            payload["source_artist_slot_name_or_count_read"] = True
        elif attack == "candidate_manifest_forged_sha":
            payload["v2_candidate_arrays_sha256"] = "a" * 64
        elif attack == "candidate_manifest_forged_parameters":
            payload["precommitted_algorithm_parameters"]["merge_adaptive"][
                "max_score"
            ] = 1.0
        elif attack == "private_author_count_forgery":
            payload["active_original_author_slot_count_teacher_only"] = 21
        elif attack == "private_source_sha_forgery":
            payload["original_model_input_RGBA_SHA256"] = "0" * 64
        elif attack == "private_editor_claim":
            payload["native_editable_source_artist_project_file_verified"] = True
        elif attack == "parent_court_fake_owner":
            payload["graphics_owner_authorized"] = True
        elif attack == "parent_court_observable_drift":
            payload["author_hide_observable_cases"] = 12
        path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, AssertionError, KeyError, OSError)):
        court(x)
