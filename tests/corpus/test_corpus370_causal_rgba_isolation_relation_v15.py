"""V15 alpha-aware source mechanical->RGBA contribution court falsification.

Synthetic pixel fixtures test consequences only; they do NOT mint artist
drawables, approved owner negatives or a model-ready source field.
"""
from __future__ import annotations
import copy
import json

import numpy as np
import pytest

from tools.corpus.court_corpus370_causal_rgba_isolation_relation_v15 import (
    SCHEMA, TOL, contribution_masks, signal, verify_parents
)


def image(alpha=1,red=.1):
    return np.array([[[red,0,0,alpha]]],np.float32)


def test_same_source_edit_visible_in_full_and_isolate():
    b=image(); edit=image(red=.8)
    m=contribution_masks(b,edit,b,edit)
    assert m["whole_scene_effect_pixel_instances"]==1
    assert m["isolated_active_surface_effect_pixel_instances"]==1
    assert m["same_pixel_observed_in_both_effects"]==1
    assert m["isolation_only_effect_pixel_instances__visibility_or_compositing_candidate"]==0
    assert m["whole_scene_only_effect_pixel_instances__contextual_compositing_candidate"]==0
    assert m["occlusion_order_authorized"] is False
    assert m["drawable_owner_authorized"] is False


def test_source_invisible_but_active_isolation_responds_not_author_negative():
    full=image(red=.2)
    m=contribution_masks(full,full,full,image(red=.8))
    assert m["whole_scene_effect_pixel_instances"]==0
    assert m["isolation_only_effect_pixel_instances__visibility_or_compositing_candidate"]==1
    assert m["whole_effect_explained_by_isolated_active_surface_fraction"] is None
    assert m["isolated_effect_visible_in_whole_scene_fraction"]==0
    assert m["occlusion_order_authorized"] is False


def test_compositor_context_change_outside_isolated_face_is_not_drawable():
    b=image(red=.1);f=image(red=.8)
    m=contribution_masks(b,f,b,b)
    assert m["whole_scene_only_effect_pixel_instances__contextual_compositing_candidate"]==1
    assert m["isolated_active_surface_effect_pixel_instances"]==0
    assert m["drawable_owner_authorized"] is False


def test_transparent_rgb_change_cannot_create_fake_rgba_effect():
    b=image(alpha=0.,red=.1);e=image(alpha=0.,red=.9)
    assert signal(b,e)[0,0]==0
    assert contribution_masks(b,e,b,e)["whole_scene_effect_pixel_instances"]==0


def test_true_alpha_change_still_has_weight_with_transparent_rgb():
    b=image(alpha=0.,red=.5);e=image(alpha=.8,red=.5)
    assert signal(b,e)[0,0]>.5
    assert contribution_masks(b,e,b,e)["same_pixel_observed_in_both_effects"]==1


@pytest.mark.parametrize("threshold",[-.1,0,.025000001,.2])
def test_no_promotion_by_changing_precommitted_alpha_threshold(threshold):
    b=image()
    with pytest.raises(ValueError,match="V15_STRICT_RGBA_TOLERANCE"):
        contribution_masks(b,b,b,b,tolerance=threshold)


@pytest.mark.parametrize("bad",[
    np.array([[[np.nan,0,0,1]]],np.float32),
    np.ones((2,2,3),np.float32),
])
def test_rgba_nonfinite_and_unsupported_shape_rejected(bad):
    with pytest.raises(ValueError,match="V15_RGBA_IMAGE_SHAPE"):
        signal(bad,bad)


def test_canonical_parent_absence_never_creates_training_pass(tmp_path):
    fake={"schema":"RealSaS.Corpus370V14OriginalNativeMechanicalRGBAIndependentCourtRealResult.v1",
          "status":"REAL_ORIGINAL_SOURCE_NATIVE_CONTROL_TO_RGBA_CAUSAL_REPAIR_3_OF_3_COURTS_PASS__DRAWABLE_OWNER_UNKNOWN",
          "original_perturbed_source_vs_soup_qualified_camera_cases":144,
          "baseline_parity_qualified_pose_views":36,
          "graphics_owner_authorized":False,
          "iris_training_authorized":False,
          "tessa_training_authorized":False,
          "materialization_authorized":False,
          "source_results":[{"source":"CAT","subject_id":"cat","source_sha256":"a"*64,
              "private_independent_court_sha256":"b"*64,
              "private_batch_sha256":"c"*64,
              "private_native_rgba_npz_sha256":"d"*64}]}
    with pytest.raises(ValueError,match="V15_REQUIRED_PRIVATE_SOURCE_BYTES_ABSENT"):
        verify_parents(tmp_path,fake,"CAT")
    for mutation in ("training","owner","materialize","camera_count","source"):
        d=copy.deepcopy(fake)
        if mutation=="training":d["iris_training_authorized"]=True
        elif mutation=="owner":d["graphics_owner_authorized"]=True
        elif mutation=="materialize":d["materialization_authorized"]=True
        elif mutation=="camera_count":d["original_perturbed_source_vs_soup_qualified_camera_cases"]=143
        else:d["source_results"]=[]
        with pytest.raises(ValueError):
            verify_parents(tmp_path,d,"CAT")


def test_v15_licensing_and_source_control_ids_remain_independent_from_student():
    assert SCHEMA=="RealSaS.Corpus370IndependentCausalRGBAOverlapCourt.v15"
    assert TOL==.025
    m=contribution_masks(image(),image(red=.8),image(),image(red=.8))
    assert "artist_drawable_labels" not in m
    assert all("bone_name" not in key for key in m)


def test_v15_adversarial_pixel_condition_noop_stays_zero():
    b=np.zeros((16,16,4),np.float32);b[3:8,4:9,:]=[.2,.6,.4,1.]
    m=contribution_masks(b,b,b,b)
    assert all(m[key]==0 for key in (
        "whole_scene_effect_pixel_instances",
        "isolated_active_surface_effect_pixel_instances",
        "same_pixel_observed_in_both_effects",
        "isolation_only_effect_pixel_instances__visibility_or_compositing_candidate",
        "whole_scene_only_effect_pixel_instances__contextual_compositing_candidate"
    ))
