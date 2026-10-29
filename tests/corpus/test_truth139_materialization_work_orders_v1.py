from __future__ import annotations

from tools.corpus.build_truth139_materialization_work_orders_v1 import build


def _subject(index: int, *, observation=True, normals=True, response=True, stage48=True, topology=True, dev=False):
    sid = f"truth139_{index:03d}"
    return {
        "subject_id": sid,
        "subject_identity_signature": f"identity-{index:03d}",
        "source": {
            "corpus_relative_path": f"raw/quaternius/pack/glTF/{sid}.gltf",
            "sha256": f"{index % 10}" * 64,
            "byte_size": 1234,
            "format": "GLTF",
            "creator": "quaternius",
            "pack_id": "quaternius-pack",
            "dependencies": [],
            "dependency_count": 0,
            "source_page_snapshot": {"path": "provenance/source_pages/pack.html", "sha256": "a" * 64},
            "official_url": "https://example.invalid",
            "drive_folder_id": "folder",
            "source_family_signature": [f"SRCF:{index:03d}"],
        },
        "raw_capabilities": {
            "OBSERVATION_FSTAR": observation,
            "SURFACE_NORMAL_TRUTH": normals,
            "STAGE47_MECHANICS": response,
            "STAGE48_APPEARANCE_PRESENTATION": stage48,
            "TESSA_TOPOLOGY_TEACHER": topology,
        },
        "development_or_regression_tainted": dev,
    }


def _manifest(subjects):
    return {
        "schema": "RealSaS.Truth139CorpusManifest.v2",
        "corpus_manifest_identity_sha256": "b" * 64,
        "subjects": subjects,
    }


def test_work_orders_retain_all_139_and_schedule_capabilities_independently():
    subjects = [_subject(i) for i in range(139)]
    subjects[3] = _subject(3, response=False, stage48=False)
    subjects[7] = _subject(7, topology=False, dev=True)
    result = build(_manifest(subjects))

    assert result["subject_count"] == 139
    assert len(result["subjects"]) == 139
    assert result["policy"]["all_historical_139_subjects_retained"] is True
    assert result["policy"]["missing_optional_truth_does_not_delete_subject"] is True
    assert result["policy"]["no_tessa_conditioning_here"] is True

    partial = next(row for row in result["subjects"] if row["subject_id"] == "truth139_003")
    assert partial["jobs"]["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"]["enabled"] is True
    assert partial["jobs"]["GEOMETRY_FSTAR_DEPTH_NORMAL_TRUTH"]["enabled"] is True
    assert partial["jobs"]["IRIS_RESPONSE_EVIDENCE"]["enabled"] is False
    assert partial["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["enabled"] is False
    assert partial["jobs"]["STAGE48_RENDERER_INTERVENTION_WITNESS"]["enabled"] is False
    assert partial["jobs"]["GENERIC_EXACT_TOPOLOGY_TRUTH"]["enabled"] is True

    dev = next(row for row in result["subjects"] if row["subject_id"] == "truth139_007")
    assert dev["development_or_regression_tainted"] is True
    assert dev["jobs"]["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"]["enabled"] is True
    assert dev["jobs"]["GENERIC_EXACT_TOPOLOGY_TRUTH"]["enabled"] is False


def test_dense_graphics_requires_observation_and_response_rest_geometry_support():
    subjects = [_subject(i) for i in range(139)]
    subjects[10] = _subject(10, observation=False, normals=False, response=True)
    result = build(_manifest(subjects))
    row = next(item for item in result["subjects"] if item["subject_id"] == "truth139_010")
    assert row["jobs"]["IRIS_RESPONSE_EVIDENCE"]["enabled"] is True
    assert row["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["enabled"] is False
    assert row["jobs"]["VISIBILITY_OCCLUSION_CROSS_VIEW"]["enabled"] is False


def test_work_orders_do_not_create_split_or_training_authority():
    result = build(_manifest([_subject(i) for i in range(139)]))
    assert result["split_created"] is False
    assert result["training_authorized"] is False
    assert result["qualification_minted"] is False
