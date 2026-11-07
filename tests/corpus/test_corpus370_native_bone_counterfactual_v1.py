"""Adversarial independent-court tests; synthetic, not real Blender runs."""
from __future__ import annotations
import copy
import hashlib
import json

import numpy as np
import pytest
from tools.corpus.court_corpus370_native_bone_counterfactual_v1 import (
    court, measure, STATES, SCHEMA,
)


def geometry():
    v=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]],np.float64)
    f=np.asarray([[0,1,2],[2,1,3]],np.int32)
    poses=np.broadcast_to(v[None],(len(STATES),4,3)).copy()
    for key,ix,sign in [("A_POS",2,1),("A_NEG",2,-1),
                        ("B_POS",3,1),("B_NEG",3,-1)]:
        poses[STATES.index(key),ix,2]=sign*0.1
    poses[STATES.index("AB_POS"),2,2]=0.1
    poses[STATES.index("AB_POS"),3,2]=0.1
    poses[STATES.index("COMMON_ROOT_SHAM"),:,0]+=0.06
    return v,f,poses


def manifest(sha):
    return {
        "schema":"RealSaS.Corpus370NativeBoneCounterfactual.v1",
        "status":"MEASURED_ISOLATED_RIG_CONTROL_CANDIDATE__DRAWABLE_AND_CAUSAL_COURTS_SEPARATE",
        "source_sha256":"a"*64,
        "control_selection_uses_render_or_motion_outcome":False,
        "labels_from_source_names":False,
        "independent_replay_courted":False,
        "same_drawable":"UNKNOWN",
        "different_drawable":"UNKNOWN",
        "mechanical_owner":"UNKNOWN",
        "causal_response_authorized":False,
        "teacher_only":True,
        "training_authorized":False,
        "control_states":list(STATES),"pose_count":len(STATES),
        "root_sham_available":True,
        "amplitude_radians":0.10471975511965978,
        "rest_face_triangulation_frozen":True,
        "source_polygon_connectivity_stable":True,
        "output":{"sha256":sha},
    }


def checksum(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files(tmp_path):
    v,f,p=geometry()
    rest=tmp_path/"rest.npz"
    controls=tmp_path/"command.npz"
    np.savez_compressed(rest,rest_vertices_source_world=v,faces=f)
    np.savez_compressed(controls,rest_vertices_source_world=v,faces=f,
        face_owner=np.zeros(len(f),np.int32),face_material=np.zeros(len(f),np.int32),
        intervention_positions_source_world=p)
    return rest,controls,manifest(checksum(controls))


def test_true_distinct_positive_and_reverse_controls_are_observations_only():
    v,f,p=geometry()
    d=measure(v,f,p,v,f,root_sham_available=True)
    assert d["status"].startswith("PASS_TWO_INDEPENDENT_COMMAND")
    assert d["controls"]["A_POS"]["affected_source_vertices"]==1
    assert d["controls"]["B_POS"]["affected_source_vertices"]==1
    assert d["inverse_controls"]["A"]["opposition_cosine"]==pytest.approx(-1)
    assert d["inverse_controls"]["B"]["opposition_cosine"]==pytest.approx(-1)
    assert d["common_root_sham"]["measured_effect"] is True
    assert d["same_drawable"]=="UNKNOWN"
    assert d["different_drawable"]=="UNKNOWN"
    assert d["training_authorized"] is False


def test_noop_forgery_fails():
    v,f,p=geometry()
    p[1,0,1]+=0.02
    with pytest.raises(ValueError,match="ZERO_PARITY_FAIL"):
        measure(v,f,p,v,f,root_sham_available=True)


def test_source_rest_face_order_drift_fails_even_when_numbers_same():
    v,f,p=geometry()
    with pytest.raises(ValueError,match="FACE_ORDER_DRIFT"):
        measure(v,f,p,v,f[::-1],root_sham_available=True)


def test_source_geometry_forgery_fails():
    v,f,p=geometry()
    source=v.copy();source[2,2]+=.1
    with pytest.raises(ValueError,match="INDEPENDENT_REST_GEOMETRY_DRIFT"):
        measure(v,f,p,source,f,root_sham_available=True)


def test_b_only_zero_effect_not_falsely_different_drawable():
    v,f,p=geometry()
    p[STATES.index("B_POS")]=v
    p[STATES.index("B_NEG")]=v
    d=measure(v,f,p,v,f,root_sham_available=True)
    assert d["status"].startswith("MASKED_")
    assert d["same_drawable"]=="UNKNOWN"
    assert d["two_measurable_control_effects"] is False


def test_common_root_sham_unavailable_not_a_zero_effect_negative():
    v,f,p=geometry()
    p[STATES.index("COMMON_ROOT_SHAM")]=v
    d=measure(v,f,p,v,f,root_sham_available=False)
    assert d["common_root_sham"]["status"]=="MASKED_NO_DISTINCT_COMMON_ANCESTOR"
    assert d["common_root_sham"]["measured_effect"] is None


def test_non_finite_pose_rejected():
    v,f,p=geometry()
    p[3,1,1]=float("nan")
    with pytest.raises(ValueError,match="POSED_ARRAY_INVALID"):
        measure(v,f,p,v,f,root_sham_available=True)


def test_reversal_not_proven_when_both_perturbations_same_direction():
    v,f,p=geometry()
    p[STATES.index("A_NEG")]=p[STATES.index("A_POS")]
    d=measure(v,f,p,v,f,root_sham_available=True)
    assert d["status"].startswith("MASKED_")
    assert d["inverse_controls"]["A"]["status"].startswith("ABSTAIN_")


def test_court_valid_sha_cross_source_equivalence(tmp_path):
    rest,controls,doc=files(tmp_path)
    d=court(doc,source_sha256="a"*64,source_rest_sha256=checksum(rest),
            npz_path=controls,source_rest_path=rest)
    assert d["status"].startswith("PASS_TWO_INDEPENDENT_COMMAND")
    assert d["independent_blender_replay_required"] is True
    assert d["causal_response_family_authorized"] is False


def test_manifest_forbidden_ownership_label_fails_closed(tmp_path):
    rest,controls,doc=files(tmp_path)
    doc["same_drawable"]="POSITIVE"
    with pytest.raises(ValueError,match="SOURCE_OR_AUTHORITY"):
        court(doc,source_sha256="a"*64,source_rest_sha256=checksum(rest),
              npz_path=controls,source_rest_path=rest)


def test_manifest_sham_or_control_schema_tampering_fails(tmp_path):
    rest,controls,doc=files(tmp_path)
    doc["control_states"][0]="FAKE"
    with pytest.raises(ValueError,match="EXACT_COMMAND_PROTOCOL"):
        court(doc,source_sha256="a"*64,source_rest_sha256=checksum(rest),
              npz_path=controls,source_rest_path=rest)


def test_control_payload_byte_corruption_fails(tmp_path):
    rest,controls,doc=files(tmp_path)
    controls.write_bytes(controls.read_bytes()+b"tampered")
    with pytest.raises(ValueError,match="PARENT_ARRAY_BYTES_SHA"):
        court(doc,source_sha256="a"*64,source_rest_sha256=checksum(rest),
              npz_path=controls,source_rest_path=rest)


def test_masked_absent_bone_pair_is_not_zero_negative():
    doc=manifest("b"*64)
    doc["status"]="MASKED_NO_TWO_WEIGHTED_PARENT_CHILD_COMMANDS"
    del doc["output"]
    result=court(doc,source_sha256="a"*64,source_rest_sha256="c"*64)
    assert result["status"]=="MASKED_NO_SOURCE_RIG_CONTROL_PAIR__NOT_ZERO"
    assert result["different_drawable"]=="UNKNOWN"


def test_masked_source_with_fake_array_rejected(tmp_path):
    _,controls,doc=files(tmp_path)
    doc["status"]="MASKED_NO_TWO_WEIGHTED_PARENT_CHILD_COMMANDS"
    with pytest.raises(ValueError,match="FALSE_MASKED"):
        court(doc,source_sha256="a"*64,source_rest_sha256="c"*64,npz_path=controls)


def test_wrong_subject_source_digest_rejected(tmp_path):
    rest,controls,doc=files(tmp_path)
    with pytest.raises(ValueError,match="SOURCE_OR_AUTHORITY"):
        court(doc,source_sha256="c"*64,source_rest_sha256=checksum(rest),
              npz_path=controls,source_rest_path=rest)
