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


def _manifest(subjects, *, historical_pass=True):
    return {
        "schema": "RealSaS.Truth139CorpusManifest.v2",
        "corpus_manifest_identity_sha256": "b" * 64,
        "historical_reproduction": {
            "passed": historical_pass,
            "operational_blocker": False,
        },
        "subjects": subjects,
    }


def test_work_orders_retain_subjects_and_schedule_capabilities_independently():
    subjects = [_subject(i) for i in range(139)]
    subjects[3] = _subject(3, response=False, stage48=False)
    subjects[7] = _subject(7, topology=False, dev=True)
    result = build(_manifest(subjects))

    assert result["subject_count"] == 139
    assert len(result["subjects"]) == 139
    assert result["policy"]["missing_optional_truth_does_not_delete_subject"] is True
    assert result["policy"]["graphics_does_not_depend_on_response_evidence"] is True
    assert result["policy"]["no_tessa_conditioning_here"] is True
    assert result["policy"]["producer_output_is_not_admitted_without_required_independent_court"] is True

    partial = next(row for row in result["subjects"] if row["subject_id"] == "truth139_003")
    assert partial["jobs"]["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"]["enabled"] is True
    assert partial["jobs"]["GEOMETRY_FSTAR_DEPTH_NORMAL_TRUTH"]["enabled"] is True
    assert partial["jobs"]["IRIS_RESPONSE_EVIDENCE"]["enabled"] is False
    assert partial["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["enabled"] is True
    assert partial["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["depends_on"] == ["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"]
    assert partial["jobs"]["VISIBILITY_OCCLUSION_CROSS_VIEW"]["enabled"] is True
    assert partial["jobs"]["STAGE48_RENDERER_INTERVENTION_WITNESS"]["enabled"] is False
    assert partial["jobs"]["GENERIC_EXACT_TOPOLOGY_TRUTH"]["enabled"] is True

    dev = next(row for row in result["subjects"] if row["subject_id"] == "truth139_007")
    assert dev["development_or_regression_tainted"] is True
    assert dev["jobs"]["SOURCE_IMPORT_AND_8VIEW_OBSERVATION"]["enabled"] is True
    assert dev["jobs"]["GENERIC_EXACT_TOPOLOGY_TRUTH"]["enabled"] is False


def test_dense_graphics_requires_observation_but_not_response_evidence():
    subjects = [_subject(i) for i in range(139)]
    subjects[10] = _subject(10, observation=False, normals=False, response=True)
    subjects[11] = _subject(11, observation=True, normals=True, response=False)
    result = build(_manifest(subjects))
    no_observation = next(item for item in result["subjects"] if item["subject_id"] == "truth139_010")
    assert no_observation["jobs"]["IRIS_RESPONSE_EVIDENCE"]["enabled"] is True
    assert no_observation["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["enabled"] is False
    assert no_observation["jobs"]["VISIBILITY_OCCLUSION_CROSS_VIEW"]["enabled"] is False

    no_response = next(item for item in result["subjects"] if item["subject_id"] == "truth139_011")
    assert no_response["jobs"]["IRIS_RESPONSE_EVIDENCE"]["enabled"] is False
    assert no_response["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["enabled"] is True
    assert no_response["jobs"]["VISIBILITY_OCCLUSION_CROSS_VIEW"]["enabled"] is True


def test_graphics_work_orders_bind_canonical_producers_and_independent_courts():
    result = build(_manifest([_subject(i) for i in range(139)]))
    canonical = result["canonical_implementations"]
    dense = canonical["DENSE_GRAPHICS_FRONT_SURFACE"]
    visibility = canonical["VISIBILITY_OCCLUSION_CROSS_VIEW"]

    assert dense["producer"] == "tools/corpus/build_truth139_dense_graphics_gbuffer_v1.py"
    assert dense["independent_court"] == "tools/corpus/validate_truth139_dense_graphics_geometry_v1.py"
    assert dense["required_court_status"] == "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER"

    assert visibility["producer"] == "tools/corpus/build_truth139_visibility_occlusion_teacher_v2.py"
    assert visibility["producer_revision"] == "V2_SILHOUETTE_NEIGHBOR_FIRST"
    assert visibility["independent_court"] == "tools/corpus/validate_truth139_visibility_occlusion_teacher_v1.py"
    assert visibility["required_court_status"] == "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER"

    subject = result["subjects"][0]
    assert subject["jobs"]["DENSE_GRAPHICS_FRONT_SURFACE"]["implementation"] == dense
    assert subject["jobs"]["VISIBILITY_OCCLUSION_CROSS_VIEW"]["implementation"] == visibility


def test_historical_reproduction_finding_does_not_block_work_orders():
    result = build(_manifest([_subject(i) for i in range(138)], historical_pass=False))
    assert result["subject_count"] == 138
    assert result["parent_historical_reproduction"]["passed"] is False
    assert result["policy"]["historical_reproduction_finding_does_not_block_safe_materialization"] is True


def test_work_orders_do_not_create_split_or_training_authority():
    result = build(_manifest([_subject(i) for i in range(139)]))
    assert result["split_created"] is False
    assert result["training_authorized"] is False
    assert result["qualification_minted"] is False
