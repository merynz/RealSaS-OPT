"""Semantic REGION identity tests: islands are not semantic components."""
import json
from pathlib import Path

import pytest

from models.iris.truth.semantic_region_hierarchy_v1 import build


ROOT=Path(__file__).resolve().parents[2]
POLICY=json.loads((ROOT/"canonical/REALSAS_SEMANTIC_PRIOR_POLICY_V1.json").read_text())
SHA="a"*64


def _source():
    return {
        "schema":"RealSaS.Corpus370SemanticSourceGroupCandidates.v1",
        "subject_id":"synthetic_source",
        "source_sha256":SHA,
        "source_face_count":4,
        "training_authorized":False,
        "class_labels_admitted":False,
        "semantic_review_required":True,
        "groups":[
            {"teacher_group_id":"island1","source_owner_slot":0,
             "source_material_slot":0,"privileged_object_name":"HairBack",
             "privileged_material_name":None,"source_face_ids":[0,1],
             "suggested_concept":"HAIR_LIKE","hint_status":"ONE_UNVERIFIED_HINT"},
            {"teacher_group_id":"island2","source_owner_slot":0,
             "source_material_slot":0,"privileged_object_name":"HairBack",
             "privileged_material_name":None,"source_face_ids":[2],
             "suggested_concept":"HAIR_LIKE","hint_status":"ONE_UNVERIFIED_HINT"},
            {"teacher_group_id":"island3","source_owner_slot":1,
             "source_material_slot":0,"privileged_object_name":"Sword",
             "privileged_material_name":None,"source_face_ids":[3],
             "suggested_concept":"HANDHELD_PROP_LIKE","hint_status":"ONE_UNVERIFIED_HINT"},
        ]
    }


def test_two_connected_islands_one_semantic_owner_material_region_candidate():
    r=build(source_groups=_source(),semantic_policy=POLICY)
    assert r["source_face_count"]==4
    assert r["child_island_count"]==3
    assert r["parent_region_count"]==2
    assert r["regions"][0]["source_face_ids"]==[0,1,2]
    assert r["regions"][0]["child_island_count"]==2
    assert r["regions"][0]["functional_affordance_prior_hint"][
        "SECONDARY_MOTION_ELIGIBLE"] is True
    assert r["regions"][0]["region_kind"]=="SOURCE_OWNER_MATERIAL_DOMAIN_NOT_SEMANTIC_INSTANCE"
    assert r["regions"][0]["training_authorized"] is False
    assert r["semantic_class_court_passed"] is False


def test_child_faces_cannot_overlap_or_be_omitted():
    d=_source()
    d["groups"][1]["source_face_ids"]=[1,2]
    with pytest.raises(ValueError,match="ISLAND_FACE_DISJOINT"):
        build(source_groups=d,semantic_policy=POLICY)
    d=_source()
    d["groups"][0]["source_face_ids"]=[0]
    with pytest.raises(ValueError,match="ALL_SOURCE_FACES"):
        build(source_groups=d,semantic_policy=POLICY)


def test_single_object_conflicting_hints_abstains():
    d=_source()
    d["groups"][1]["suggested_concept"]="CLOTH_LIKE"
    r=build(source_groups=d,semantic_policy=POLICY)
    assert r["regions"][0]["semantic_concept_hint"]=="UNKNOWN"
    assert r["regions"][0]["hint_status"]=="CONFLICTING_HINTS"
    assert r["regions"][0]["functional_affordance_prior_hint"]=={}


def test_prior_never_mints_authority():
    d=_source()
    d["training_authorized"]=True
    with pytest.raises(ValueError,match="INVALID_V1_PARENT"):
        build(source_groups=d,semantic_policy=POLICY)
    d=_source()
    d["groups"][0]["source_owner_slot"]=1
    with pytest.raises(ValueError,match="OWNER_NAME_DRIFT"):
        build(source_groups=d,semantic_policy=POLICY)
