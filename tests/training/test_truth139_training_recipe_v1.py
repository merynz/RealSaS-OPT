from __future__ import annotations

from tools.training.build_truth139_training_recipe_v1 import build_recipe


def _split(train_count=111, unseen_count=28):
    return {
        "schema": "RealSaS.LineageSafeSubjectSplit.v1",
        "split_identity_sha256": "a" * 64,
        "train_subjects": [f"train_{i:03d}" for i in range(train_count)],
        "unseen_subjects": [f"unseen_{i:03d}" for i in range(unseen_count)],
    }


def test_recipe_preserves_per_subject_exposure_and_global_update_semantics():
    recipe = build_recipe(_split())
    assert recipe["train_subject_count"] == 111
    assert recipe["optimizer"]["subjects_per_optimizer_step"] == 111
    assert recipe["optimizer"]["logical_update_semantics"] == (
        "MEAN_GRADIENT_OVER_ALL_TRAIN_SUBJECTS_PER_GLOBAL_STEP"
    )
    assert recipe["schedule"][0]["epochs"] == 1000
    assert recipe["schedule"][0]["subject_exposures_per_subject"] == 1000
    assert recipe["schedule"][1]["epochs"] == 5000
    assert recipe["schedule"][1]["subject_exposures_per_subject"] == 5000
    assert recipe["optimizer"]["cohort_size_changes_adam_update_count"] is False


def test_recipe_keeps_dense_carrier_and_repairs_without_unseen_access():
    recipe = build_recipe(_split(115, 24))
    config = recipe["model_config"]
    assert config["carrier_resolution"] == 40
    assert config["carrier_channels"] == 160
    assert recipe["geometry"]["primary_authority"] == "DIRECT_FSTAR"
    assert recipe["geometry"]["sign_weight"] > 0
    assert 0 < recipe["geometry"]["normal_weight"] < 0.1
    assert recipe["geometry"]["normal_forward_input"] is False
    assert recipe["mechanics"]["learned_per_query_variance"] is False
    assert recipe["appearance"]["objective"] == "UNCHANGED_SOURCE_EQUIVALENT_RGBA"
    assert recipe["firewall"]["train_manifest_may_contain_unseen_ids"] is False
    assert recipe["firewall"]["unseen_used_for_checkpoint_selection"] is False
