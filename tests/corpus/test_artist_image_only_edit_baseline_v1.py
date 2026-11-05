"""Adversarial original-artist image-only baseline and private edit court contracts."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.artist_image_only_edit_baseline_v1 import (
    COURT_SCHEMA, INPUT_SCHEMA, PRED_SCHEMA, court, digest, write_prediction,
)


def fixture(tmp_path):
    image = np.zeros((24, 24, 4), dtype=np.uint8)
    image[2:9, 3:10] = (240, 60, 50, 255)
    image[13:21, 14:22] = (60, 60, 225, 255)
    baseline = image.tobytes()
    modeldir = tmp_path / "model_input"
    court_dir = tmp_path / "private_artist_court"
    modeldir.mkdir()
    court_dir.mkdir()
    raw_path = modeldir / "anonymous_flattened_rgba.bin"
    raw_path.write_bytes(baseline)
    public = {
        "schema": INPUT_SCHEMA,
        "width": 24, "height": 24,
        "format": "RGBA8888_UNPREMULTIPLIED",
        "image_RGBA_SHA256": digest(baseline),
        "original_artist_slot_names_present": False,
        "original_artist_layer_count_present": False,
        "source_mesh_face_incidence_present": False,
        "source_bone_rig_or_attachment_names_present": False,
        "consumer_image_only_source_input": True,
        "training_authorized": False,
    }
    public_path = modeldir / "manifest.json"
    public_path.write_text(json.dumps(public))
    author_rows = []
    for i in range(3):
        hidden = image.copy()
        isolated = np.zeros_like(image)
        if i == 0:
            hidden[2:9, 3:10] = 0
            isolated[2:9, 3:10] = image[2:9, 3:10]
        elif i == 1:
            hidden[13:21, 14:22] = 0
            isolated[13:21, 14:22] = image[13:21, 14:22]
        else:
            # An artwork layer wholly obscured in the composite.
            isolated[5:7, 5:7] = (50, 240, 70, 255)
        h_raw, i_raw = hidden.tobytes(), isolated.tobytes()
        (court_dir / f"slot{i}_hidden.rgba").write_bytes(h_raw)
        (court_dir / f"slot{i}_isolated.rgba").write_bytes(i_raw)
        author_rows.append({
            "original_slot_index_teacher_only": i,
            "original_slot_name_teacher_only": f"hidden-art-name-{i}",
            "original_isolated_source_RGBA_sha256": digest(i_raw),
            "original_hidden_source_RGBA_sha256": digest(h_raw),
            "original_isolated_alpha_pixel_count": int((isolated[..., 3] > 0).sum()),
            "original_hide_changes_composite_RGBA_pixel_count":
                int(np.any(hidden != image, axis=2).sum()),
        })
    reference = {
        "schema": "RealSaS.IndependentOriginalArtistSlotRGBAHideIsolateCourtOnly.v1",
        "original_model_input_RGBA_SHA256": digest(baseline),
        "official_spine_runtime_reference": True,
        "original_hide_isolate_restoration_exact": True,
        "native_editable_source_artist_project_file_verified": False,
        "training_authorized": False,
        "active_original_author_slot_count_teacher_only": 3,
        "source_author_slot_hide_isolate_references_teacher_only": author_rows,
    }
    private_path = court_dir / "reference.json"
    private_path.write_text(json.dumps(reference))
    pred_npz = modeldir / "predicted_regions.npz"
    pred_json = modeldir / "predicted_regions.json"
    return raw_path, public_path, pred_npz, pred_json, private_path, court_dir


def getargs(tmp_path):
    inputs = fixture(tmp_path)
    write_prediction(*inputs[:4])
    return inputs


def test_unmodified_image_only_baseline_matches_two_visible_edit_islands(tmp_path):
    inputs = getargs(tmp_path)
    result = court(*inputs)
    assert result["schema"] == COURT_SCHEMA
    assert result["author_hide_observable_cases"] == 2
    assert result["author_hide_unobservable_masked_cases"] == 1
    alpha = result["metrics_mean_edit_support_iou_not_artist_membership_accuracy"]["alpha_connected"]
    assert alpha["predicted_region_count"] == 2
    assert alpha["matched_observable_source_edits"] == 2
    assert alpha["mean_one_to_one_edit_support_iou_over_all_observable_edits"] == 1.0
    assert alpha["matched_edit_effect_pixel_recall_over_all_edit_effect_instances"] == 1.0
    assert result["graphics_owner_authorized"] is False
    assert result["training_authorized"] is False


def test_predict_has_no_artist_reference_parameter_or_author_count(tmp_path):
    image, public, npz, manifest, reference, private_dir = getargs(tmp_path)
    pred = json.loads(manifest.read_text())
    assert pred["schema"] == PRED_SCHEMA
    assert "slot" not in json.dumps(pred).lower()
    assert pred["reference_used_during_prediction"] is False
    assert pred["artist_slot_names_or_layer_count_exposed"] is False
    assert pred["drawable_ownership_authorized"] is False
    assert pred["training_authorized"] is False


@pytest.mark.parametrize("attack", [
    "image_bytes_drift", "public_identity_leak", "prediction_tamper",
    "prediction_used_teacher", "fake_training", "reference_wrong_image",
    "false_edit_restoration", "slot_blob_corruption", "duplicated_slot",
    "author_case_count_forgery", "unsupported_spine_project_claim",
])
def test_image_only_and_independent_artist_court_fail_closed(tmp_path, attack):
    image, public, npz, pred, ref, root = getargs(tmp_path)
    if attack == "image_bytes_drift":
        data = bytearray(image.read_bytes())
        data[20] ^= 3
        image.write_bytes(data)
    elif attack == "public_identity_leak":
        d = json.loads(public.read_text())
        d["original_artist_layer_count_present"] = True
        public.write_text(json.dumps(d))
    elif attack == "prediction_tamper":
        with npz.open("ab") as f:
            f.write(b"faked region answer")
    elif attack in ("prediction_used_teacher", "fake_training"):
        d = json.loads(pred.read_text())
        d["reference_used_during_prediction" if attack == "prediction_used_teacher"
          else "training_authorized"] = True
        pred.write_text(json.dumps(d))
    elif attack == "slot_blob_corruption":
        source = root / "slot0_hidden.rgba"
        source.write_bytes(source.read_bytes() + b"forged")
    else:
        d = json.loads(ref.read_text())
        if attack == "reference_wrong_image":
            d["original_model_input_RGBA_SHA256"] = "0" * 64
        elif attack == "false_edit_restoration":
            d["original_hide_isolate_restoration_exact"] = False
        elif attack == "duplicated_slot":
            d["source_author_slot_hide_isolate_references_teacher_only"][1][
                "original_slot_index_teacher_only"] = 0
        elif attack == "author_case_count_forgery":
            d["active_original_author_slot_count_teacher_only"] = 2
        elif attack == "unsupported_spine_project_claim":
            d["native_editable_source_artist_project_file_verified"] = True
        ref.write_text(json.dumps(d))
    with pytest.raises((ValueError, OSError, KeyError)):
        court(image, public, npz, pred, ref, root)


def test_artist_slot_count_not_an_optimal_model_k(tmp_path):
    image, public, npz, pred, ref, root = getargs(tmp_path)
    data = json.loads(pred.read_text())
    # Three real author slots, including one image-invisible, do not
    # turn into three supervised model regions or three known drawables.
    assert data["predicted_region_counts"]["alpha_connected"] == 2
    r = court(image, public, npz, pred, ref, root)
    assert r["exact_artist_drawable_identity_proven"] is False
    assert r["artist_project_original_editor_file_verified"] is False
