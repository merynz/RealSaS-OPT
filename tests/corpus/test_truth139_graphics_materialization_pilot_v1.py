from __future__ import annotations

from tools.corpus.build_truth139_graphics_materialization_pilot_v1 import build


def _subject(subject_id, *, dev=False, normal=True, graphics=True):
    return {
        "subject_id": subject_id,
        "subject_identity_signature": "sig-" + subject_id,
        "source": {
            "pack_id": "pack-" + subject_id,
            "corpus_relative_path": "raw/" + subject_id + ".gltf",
            "sha256": (subject_id[-1] if subject_id[-1].isdigit() else "a") * 64,
        },
        "development_or_regression_tainted": dev,
        "jobs": {
            "DENSE_GRAPHICS_FRONT_SURFACE": {"enabled": graphics},
            "VISIBILITY_OCCLUSION_CROSS_VIEW": {"enabled": graphics},
            "GEOMETRY_FSTAR_DEPTH_NORMAL_TRUTH": {"enabled": normal},
        },
    }


def _work(subjects):
    return {
        "schema": "RealSaS.Truth139MaterializationWorkOrders.v1",
        "work_order_identity_sha256": "b" * 64,
        "subject_count": len(subjects),
        "subjects": subjects,
    }


def _census(components):
    return {
        "schema": "RealSaS.Truth139StructuralDiversityCensus.v1",
        "census_identity_sha256": "c" * 64,
        "components": components,
    }


def test_pilot_uses_largest_components_and_prefers_clean_then_normal_truth():
    subjects = [
        _subject("s1", dev=True, normal=True),
        _subject("s2", dev=False, normal=False),
        _subject("s3", dev=False, normal=True),
        _subject("s4", dev=False, normal=True),
        _subject("s5", dev=False, normal=False),
    ]
    components = [
        {"component_ordinal": 1, "size": 10, "cross_pack": True, "pack_counts": {"a": 5, "b": 5}, "subject_ids": ["s1", "s2", "s3"]},
        {"component_ordinal": 2, "size": 5, "cross_pack": False, "pack_counts": {"c": 5}, "subject_ids": ["s4"]},
        {"component_ordinal": 3, "size": 2, "cross_pack": False, "pack_counts": {"d": 2}, "subject_ids": ["s5"]},
    ]
    result = build(_work(subjects), _census(components), pilot_size=2)
    assert [row["component_ordinal"] for row in result["selected"]] == [1, 2]
    # s3 beats dev-tainted s1 and clean-but-no-normal s2.
    assert result["selected"][0]["subject_id"] == "s3"
    assert result["selected"][1]["subject_id"] == "s4"
    assert result["coverage"]["development_or_regression_tainted_count"] == 0
    assert result["coverage"]["optional_geometry_normal_truth_available_count"] == 2
    assert result["training_authorized"] is False
    assert result["split_created"] is False


def test_missing_optional_normal_truth_does_not_remove_graphics_candidate():
    subjects = [_subject("s1", normal=False)]
    components = [
        {"component_ordinal": 1, "size": 1, "cross_pack": False, "pack_counts": {"a": 1}, "subject_ids": ["s1"]}
    ]
    result = build(_work(subjects), _census(components), pilot_size=1)
    assert result["selected"][0]["subject_id"] == "s1"
    assert result["coverage"]["optional_geometry_normal_truth_missing_count"] == 1


def test_graphics_ineligible_subject_is_not_silently_selected():
    subjects = [_subject("s1", graphics=False)]
    components = [
        {"component_ordinal": 1, "size": 1, "cross_pack": False, "pack_counts": {"a": 1}, "subject_ids": ["s1"]}
    ]
    try:
        build(_work(subjects), _census(components), pilot_size=1)
    except ValueError as exc:
        assert "COMPONENT_HAS_NO_GRAPHICS_CANDIDATE" in str(exc)
    else:
        raise AssertionError("ineligible component must fail closed")
