"""Synthetic consumer consequence gates; never a Corpus370 subject authority."""
from __future__ import annotations

import copy
from pathlib import Path
import runpy

import pytest

from tools.corpus.court_corpus370_source_locus_visibility_v1 import qualify
from tools.corpus.court_corpus370_visibility_oracle_consumer_v1 import consume


def sample():
    return runpy.run_path(str(Path(__file__).with_name(
        "test_corpus370_source_locus_visibility_v1.py")))["fixture"]()


def test_oracle_retains_revealed_occluded_locus():
    source = sample()
    vis = qualify(source)
    vis["parents"] = {"source_manifest_sha256": "b"*64}
    result = consume(source, vis)
    assert result["occluded_edit_sources_recoverable"] == 1
    assert result["abstain_pixel_count"] == 0
    assert result["max_full_rgba_replay_error"] <= 0.025
    assert result["rows"][0]["render_decision"] == "SOURCE_HIT_A_FRONT__RETAIN_HIDDEN_SOURCE_B"
    assert result["rows"][0]["hidden_recolor_effect_given_front_removal"] > 0.025
    assert result["rows"][0]["first_hit_only_baseline_would_discard_hidden_source"]
    assert result["qualified_same_drawable_positive_pairs"] == 0
    assert result["qualified_different_drawable_negative_pairs"] == 0
    assert result["graphics_owner_truth_minted"] is False
    assert result["training_authorized"] is False


def test_vis_label_forgery_rejected():
    source = sample()
    vis = qualify(source)
    vis["rows"][0]["targets"][1]["visibility_label"] = "POSITIVE_VISIBLE_CONTRIBUTOR"
    with pytest.raises(ValueError, match="VISIBILITY_COURT_REPLAY_MISMATCH"):
        consume(source, vis)


def test_fake_owner_label_rejected():
    source = sample()
    vis = qualify(source)
    vis["qualified_same_drawable_positives"] = 1
    with pytest.raises(ValueError, match="VISIBILITY_COURT_REPLAY_MISMATCH"):
        consume(source, vis)


def test_fake_training_authority_rejected():
    source = sample()
    vis = qualify(source)
    vis["training_authorized"] = True
    with pytest.raises(ValueError, match="VISIBILITY_COURT_REPLAY_MISMATCH|FALSE_TRAINING_AUTHORITY"):
        consume(source, vis)


def test_unproven_reveal_abstains_not_negative_drawable():
    source = sample()
    row = source["rows"][0]["generic_counterfactual_v2"]["rgba"]
    row["recolor_b_given_remove_a"] = row["remove_a"].copy()
    result = consume(source)
    assert result["occluded_edit_sources_recoverable"] == 0
    assert result["abstain_pixel_count"] == 1
    assert result["qualified_different_drawable_negative_pairs"] == 0


def test_broken_render_replay_rejected():
    source = sample()
    source["rows"][0]["standalone_rgba"] = [0,0,0,0]
    with pytest.raises(ValueError):
        consume(source)


def test_two_opposite_views_remain_distinct_oracle_pixels():
    a = sample()
    b = copy.deepcopy(a)
    for row in b["rows"]:
        row["view_index"] = 4
        row["generic_counterfactual_v2"]["view_index"] = 4
        row["generic_counterfactual"]["view_index"] = 4
    aa = consume(a)
    bb = consume(b)
    assert aa["rows"][0]["view_index"] == 1
    assert bb["rows"][0]["view_index"] == 4
    assert aa["occluded_edit_sources_recoverable"] == 1
    assert bb["occluded_edit_sources_recoverable"] == 1
