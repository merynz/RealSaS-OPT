"""Synthetic contract tests only: NO real Corpus370 semantic labels or admission."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.iris.truth.semantic_source_groups_v1 import (
    build_semantic_source_groups,
    validate_hint_policy,
)
from models.iris.truth.semantic_prior_contract_v1 import (
    build_corpus370_candidate_cohort,
    build_semantic_candidate,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / "canonical/REALSAS_SEMANTIC_PRIOR_POLICY_V1.json").read_text())
HINTS = json.loads((ROOT / "canonical/REALSAS_SEMANTIC_SOURCE_HINT_POLICY_V1.json").read_text())
SHA = "a" * 64


def _snapshot():
    # Two connected hair faces, one disjoint island in the same domain,
    # one sword under another owner. Teacher source mesh only.
    return {
        "vertices": np.asarray([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
            [4., 0., 0.], [5., 0., 0.], [4., 1., 0.],
            [0., 0., 2.], [1., 0., 2.], [0., 1., 2.],
        ]),
        "faces": np.asarray([
            [0, 1, 2], [1, 3, 2], [4, 5, 6], [7, 8, 9],
        ]),
        "face_owner": np.asarray([0, 0, 0, 1]),
        "face_material": np.asarray([0, 0, 0, 0]),
        "owners": [
            {"anonymous_owner_index": 0, "source_import_name": "HairBack",
             "material_slot_names": ["Hair_01"]},
            {"anonymous_owner_index": 1, "source_import_name": "KnightSword",
             "material_slot_names": ["Steel"]},
        ],
    }


def test_designed_prior_is_valid_and_unknown_abstains():
    validate_policy(POLICY)
    validate_hint_policy(HINTS, concepts=set(POLICY["concepts"]))
    assert POLICY["concepts"]["UNKNOWN"]["prior"] == {}
    assert POLICY["concepts"]["HAIR_LIKE"]["prior"]["SECONDARY_MOTION_ELIGIBLE"] is True


def test_exact_source_domains_are_covered_and_connected():
    out = build_semantic_source_groups(
        subject_id="synthetic_test_subject", source_sha256=SHA,
        snapshot=_snapshot(), hint_policy=HINTS, semantic_prior_policy=POLICY,
    )
    assert out["source_face_count"] == 4
    assert out["group_count"] == 3
    assert [x["source_face_ids"] for x in out["groups"]] == [[0, 1], [2], [3]]
    assert [x["suggested_concept"] for x in out["groups"]] == [
        "HAIR_LIKE", "HAIR_LIKE", "HANDHELD_PROP_LIKE"]
    assert all(x["training_authorized"] is False for x in out["groups"])
    assert all(x["annotation_required"] is True for x in out["groups"])
    assert out["training_authorized"] is False
    assert out["class_labels_admitted"] is False


def test_conflicting_names_abstain_instead_of_forcing_class():
    scene = _snapshot()
    scene["owners"][0]["source_import_name"] = "ClothCape"
    out = build_semantic_source_groups(
        subject_id="synthetic_test_subject", source_sha256=SHA,
        snapshot=scene, hint_policy=HINTS, semantic_prior_policy=POLICY,
    )
    assert out["groups"][0]["hint_status"] == "CONFLICTING_HINTS"
    assert out["groups"][0]["suggested_concept"] == "UNKNOWN"


def test_missing_material_slot_refuses_forged_source_binding():
    scene = _snapshot()
    scene["face_material"][0] = 3
    with pytest.raises(ValueError, match="MATERIAL_SLOT_OUT_OF_RANGE"):
        build_semantic_source_groups(
            subject_id="synthetic_test_subject", source_sha256=SHA,
            snapshot=scene, hint_policy=HINTS, semantic_prior_policy=POLICY,
        )


def test_reviewed_prior_does_not_promote_unnamed_faces_or_training():
    annotation = {
        "subject_id": "synthetic_test_subject",
        "source_sha256": SHA,
        "source_face_count": 4,
        "components": [
            {"teacher_component_ref": "private_ref_01", "concept": "CLOTH_LIKE",
             "source_face_ids": [0, 1], "review": {
                 "kind": "HUMAN_CURATED", "receipt_sha256": SHA,
             }},
        ],
        "training_authorized": False,
    }
    out = build_semantic_candidate(policy=POLICY, annotation=annotation, source_sha256=SHA)
    assert out["covered_face_count"] == 2
    assert out["unlabeled_face_count"] == 2
    assert out["unlabeled_faces_are_negative"] is False
    assert out["component_candidates"][0]["functional_prior_targets"][
        "SECONDARY_MOTION_ELIGIBLE"] is True
    assert out["semantic_family_independent_court_passed"] is False
    assert out["teacher_component_ref_exposed_to_student"] is False


def test_unsupported_auto_label_cannot_mint_reviewed_semantic_truth():
    annotation = {
        "subject_id": "synthetic_test_subject",
        "source_sha256": SHA, "source_face_count": 4,
        "components": [{
            "teacher_component_ref": "hint_only", "concept": "HAIR_LIKE",
            "source_face_ids": [0], "review": {
                "kind": "AUTOMATIC_NAME_HINT", "receipt_sha256": SHA,
            },
        }],
    }
    with pytest.raises(ValueError, match="INDEPENDENT_REVIEW_REQUIRED"):
        build_semantic_candidate(policy=POLICY, annotation=annotation, source_sha256=SHA)


def test_subject_census_preserves_352_unknown_and_18_excluded():
    roster = {
        "schema": "RealSaS.Corpus370PreparedObservationPackets.v5",
        "subject_count": 370, "prepared_subject_count": 352, "failure_count": 18,
        "subjects": [{"subject_id": f"subject_{i:03d}", "source_sha256": SHA}
                     for i in range(352)],
        "failures": [{"subject_id": f"subject_{i:03d}"} for i in range(352, 370)],
        "training_authorized": False,
    }
    annotations = {"schema": "RealSaS.Corpus370SemanticAnnotationInput.v1", "subjects": []}
    out = build_corpus370_candidate_cohort(
        policy=POLICY, roster=roster, annotations=annotations,
    )
    assert len(out["subjects"]) == 370
    assert out["counts"] == {"OBSERVATION_EXCLUDED": 18, "SEMANTIC_UNKNOWN": 352}
    assert out["semantic_training_authorized"] is False
    assert out["family_states_minted"] is False


def test_conflicting_prior_policy_is_rejected():
    bad = json.loads(json.dumps(POLICY))
    bad["concepts"]["RIGID_PLATE_LIKE"]["prior"]["FLEXIBLE_DEFORMATION"] = True
    with pytest.raises(ValueError, match="CONFLICTING_RIGID_FLEXIBLE"):
        validate_policy(bad)


def test_source_input_is_unchanged():
    scene = _snapshot()
    before = scene["faces"].copy()
    build_semantic_source_groups(
        subject_id="synthetic_test_subject", source_sha256=SHA,
        snapshot=scene, hint_policy=HINTS, semantic_prior_policy=POLICY,
    )
    assert np.array_equal(before, scene["faces"])
