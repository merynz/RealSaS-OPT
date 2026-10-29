from __future__ import annotations

import pytest

from tools.corpus.build_truth139_graphics_apparatus_manifest_v1 import build


def _source(subject_id: str) -> dict:
    return {
        "corpus_relative_path": f"raw/{subject_id}.gltf",
        "sha256": (subject_id[-1] if subject_id[-1].isdigit() else "a") * 64,
        "byte_size": 123,
        "format": "GLTF",
        "creator": "quaternius",
        "pack_id": "pack",
        "dependencies": [],
        "dependency_count": 0,
        "source_page_snapshot": {"path": "snapshot.html", "sha256": "f" * 64},
    }


def _work() -> dict:
    subjects = []
    for index in range(2):
        sid = f"s{index}"
        subjects.append({
            "subject_id": sid,
            "subject_identity_signature": (str(index + 1) * 64),
            "source": _source(sid),
            "jobs": {
                "SOURCE_IMPORT_AND_8VIEW_OBSERVATION": {"enabled": True},
                "DENSE_GRAPHICS_FRONT_SURFACE": {"enabled": True},
            },
        })
    return {
        "schema": "RealSaS.Truth139MaterializationWorkOrders.v1",
        "work_order_identity_sha256": "b" * 64,
        "subjects": subjects,
    }


def _pilot(work: dict) -> dict:
    return {
        "schema": "RealSaS.Truth139GraphicsMaterializationPilot.v1",
        "status": "READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY",
        "parent_work_order_identity_sha256": work["work_order_identity_sha256"],
        "pilot_identity_sha256": "c" * 64,
        "pilot_size": 2,
        "selected": [
            {
                "pilot_index": index,
                "component_ordinal": index + 1,
                "subject_id": row["subject_id"],
                "source_sha256": row["source"]["sha256"],
                "corpus_relative_path": row["source"]["corpus_relative_path"],
            }
            for index, row in enumerate(work["subjects"])
        ],
    }


def test_apparatus_bridge_preserves_exact_source_rows_and_no_training_authority():
    work = _work()
    result = build(_pilot(work), work)
    assert result["schema"] == "RealSaS.Truth139ApparatusManifest.v1"
    assert result["status"] == "READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED"
    assert result["subject_count"] == 2
    assert [row["subject_key"] for row in result["selected"]] == ["s0", "s1"]
    assert all(row["source"] == work["subjects"][i]["source"] for i, row in enumerate(result["selected"]))
    assert all(row["surface_geometry_export"] == "REQUIRED_RESPONSE_INDEPENDENT_REST_GEOMETRY" for row in result["selected"])
    assert result["training_authorized"] is False
    assert result["split_created"] is False
    assert result["qualification_minted"] is False


def test_apparatus_bridge_rejects_source_drift():
    work = _work()
    pilot = _pilot(work)
    assert pilot["selected"][0]["source_sha256"] == "0" * 64
    pilot["selected"][0]["source_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="SOURCE_DRIFT"):
        build(pilot, work)
