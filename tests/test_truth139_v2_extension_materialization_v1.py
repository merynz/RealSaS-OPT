from __future__ import annotations

import json
from pathlib import Path

from tools.corpus import build_truth139_graphics_layer_peel_v2 as peel
from tools.corpus import build_truth139_relations_teacher_v2 as relations
from tools.corpus import run_truth139_v2_extension_subject_v1 as runner
from tools.corpus import validate_truth139_graphics_layer_peel_v2 as peel_court
from tools.corpus import validate_truth139_relations_teacher_v2 as relations_court


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "canonical/TRUTH139_V2_EXTENSION_MATERIALIZATION_V1.json"


def test_truth139_v2_extension_contract_paths_and_statuses_are_bound() -> None:
    payload = json.loads(CONTRACT.read_text())
    assert payload["schema"] == "RealSaS.Truth139V2ExtensionMaterializationContract.v1"
    assert payload["policy"]["legacy_dense_graphics_teacher_authority_forbidden"] is True
    assert payload["policy"]["cuda_barycentric_truth_authority"] is False
    assert payload["policy"]["raw_k_capacity_saturation_preserves_overflow_unknown"] is True

    jobs = payload["jobs"]
    assert jobs["K_HIT_GEOMETRIC_PEEL"]["producer_status"] == peel.STATUS
    assert jobs["K_HIT_GEOMETRIC_PEEL"]["court_status"] == peel_court.PASS_STATUS
    assert jobs["PRIMITIVE_RELATIONS"]["producer_status"] == relations.STATUS
    assert jobs["PRIMITIVE_RELATIONS"]["court_status"] == relations_court.PASS_STATUS

    for spec in jobs.values():
        assert (ROOT / spec["producer"]).is_file()
        assert (ROOT / spec["court"]).is_file()
    assert (ROOT / payload["runner"]).is_file()


def test_truth139_v2_extension_runner_status_vocabulary_matches_contract() -> None:
    payload = json.loads(CONTRACT.read_text())
    assert runner.EXPECTED["layer_manifest"] == payload["jobs"]["K_HIT_GEOMETRIC_PEEL"]["producer_status"]
    assert runner.EXPECTED["layer_court"] == payload["jobs"]["K_HIT_GEOMETRIC_PEEL"]["court_status"]
    assert runner.EXPECTED["component_manifest"] == payload["jobs"]["ANONYMOUS_COMPONENT_MASKS"]["producer_status"]
    assert runner.EXPECTED["component_court"] == payload["jobs"]["ANONYMOUS_COMPONENT_MASKS"]["court_status"]
    assert runner.EXPECTED["affinity_manifest"] == payload["jobs"]["DRAWABLE_AFFINITY_RELATIONS"]["producer_status"]
    assert runner.EXPECTED["affinity_court"] == payload["jobs"]["DRAWABLE_AFFINITY_RELATIONS"]["court_status"]
    assert runner.EXPECTED["relations_manifest"] == payload["jobs"]["PRIMITIVE_RELATIONS"]["producer_status"]
    assert runner.EXPECTED["relations_court"] == payload["jobs"]["PRIMITIVE_RELATIONS"]["court_status"]


def test_relations_v2_does_not_redeclare_legacy_graphics_parent() -> None:
    assert relations.PRODUCER_REVISION == "V2_DENSE_GBUFFER_VISIBILITY_V2_REBIND"
    assert relations.GRAPHICS_SCHEMA == "RealSaS.Truth139DenseGraphicsGBuffer.v1"
    assert "DenseGraphicsTeacher" not in relations.__doc__ or "never recreates" in relations.__doc__
