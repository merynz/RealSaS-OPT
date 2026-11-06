"""V18o redteams: graphics candidate complexity != independent source edit proof."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18o_anonymous_partition_admissibility import (
  SCHEMA,STATUS,partition_profile,edit_consequences,TOL,
)

def profile(counts):
    labels=np.repeat(np.arange(len(counts),dtype=np.int64),counts)
    # Source faces only need three vertices for candidate census; no
    # semantic material, source rig or artist part labels are used.
    faces=np.tile([0,1,2],(len(labels),1))
    return partition_profile(labels,faces)


def test_nearly_whole_character_is_not_drawable_success():
    p=profile([1298]+[1]*36)
    assert p["source_face_count"]==1334
    assert p["largest_component_faces"]==1298
    assert p["largest_component_has_at_least_90pct_source_faces"]
    assert p["candidate_component_count"]==37
    assert p["effective_component_count_by_simpson_inverse"]<1.1
    assert p["source_candidate_display_part_names_or_artist_gold"] is False


def test_many_fragments_are_not_necessarily_288_art_drawables():
    p=profile([22]*20+[1]*244)
    assert p["source_face_count"]==684
    assert p["candidate_component_count"]==264
    assert p["partition_has_at_least_64_components"]
    assert p["candidate_component_count_over_25pct_of_face_count"]
    assert p["source_face_count_is_not_screen_area"]


def test_animal_single_component_is_not_wrong_artist_gold():
    p=profile([200])
    assert p["source_candidate_trivially_k1"]
    assert p["largest_component_has_at_least_90pct_source_faces"]
    assert p["effective_component_count_by_simpson_inverse"]==1.0
    # This is a diagnostics flag, not a ban on single-piece puppet subjects.


def fixture(source_change=True):
    a=np.zeros((4,8,4,192,192,4),dtype=np.float32)
    # SOURCE_NATIVE and SOUP_FULL are source replay identical.
    a[:,:,0,20,30]=(.20,.50,.90,1.)
    a[:,:,1,20,30]=(.20,.50,.90,1.)
    if source_change:
        # Removing frozen group exposes a different visible original pixel.
        a[:,:,3,20,30]=(.85,.20,.10,1.)
    else:
        a[:,:,3,20,30]=(.20,.50,.90,1.)
    a[:,:,2,20,30]=(.20,.50,.90,1.)
    return a


def test_visible_group_edit_only_is_local_original_proof():
    out=edit_consequences(fixture(),9,30)
    assert out["source_selected_candidate_group_size_faces"]==9
    assert out["source_selected_candidate_fraction_of_all_faces"]==.3
    assert out["source_selected_candidate_observable_removal_across_any_view"]
    assert out["all_original_source_camera_native_soup_sample_failures"]==0
    assert out["source_selected_group_removal_sample_effects_total"]==4*8
    assert all(q["original_source_group_removal_observable_screen_pixels"]==1
               for q in out["per_original_source_camera"])
    assert out["remaining_component_heldout_edit_independence"]=="UNKNOWN"
    assert out["source_selected_component_artist_edit_intent"]=="UNKNOWN"


def test_zero_effect_cannot_be_treated_as_art_reveal_positive():
    out=edit_consequences(fixture(False),1,30)
    assert not out["source_selected_candidate_observable_removal_across_any_view"]
    assert out["source_selected_group_removal_sample_effects_total"]==0


def test_full_native_soup_source_parity_failure_is_counted():
    a=fixture()
    a[:,:,0,20,30]=(.0,.0,.0,0.)
    out=edit_consequences(a,9,30)
    assert out["all_original_source_camera_native_soup_sample_failures"]==4*8
    # Source parity problem must not disappear into successful G/R composition.


@pytest.mark.parametrize("broken",("nan","bad_alpha","bad_shape","k1","whole_source"))
def test_invalid_source_edit_fails_closed(broken):
    x=fixture()
    selected=9
    if broken=="nan":x[0,0,0,1,1,0]=np.nan
    if broken=="bad_alpha":x[0,0,0,1,1,3]=1.8
    if broken=="bad_shape":x=x[:2]
    if broken=="k1":selected=0
    if broken=="whole_source":selected=30
    with pytest.raises(ValueError,match="V18O_"):
        edit_consequences(x,selected,30)


def test_nonnegative_source_face_labels_are_mandatory():
    f=np.tile([0,1,2],(4,1))
    with pytest.raises(ValueError,match="V18O_"):
        partition_profile(np.asarray([0,1,-1,2]),f)


def test_no_graphics_owner_authority_declared():
    assert SCHEMA.endswith(".v18o")
    assert STATUS.endswith("OWNER_UNKNOWN")
    assert TOL==.025
