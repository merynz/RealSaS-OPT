"""Source-only derived control hypothesis: no oracle, no compiler repair."""
import copy
import pytest

from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import (
    discover,oracle_evaluate,sha_payload,
)


def sample():
    # Generic non-humanoid chain; names do not encode limb roles.
    return [
        {"bone_id":jid,"parent_bone_id":pid,
         "head_source":h,"tail_source":t,
         "use_deform":True,"skin_weight_mass":m,
         "skin_influenced_vertices":v,
         "source_motion_curve_count":c}
        for jid,pid,h,t,m,v,c in (
            ("G0",None,[0.,0.,0.],[1.,0.,0.],8.,40,1),
            ("G1","G0",[1.,0.,0.],[2.,0.,0.],5.,80,4),
            ("G2","G1",[2.,0.,0.],[3.,0.,0.],4.,60,4),
            ("H1","G0",[1.,0.,0.],[1.,1.,0.],2.,20,1),
            ("H2","H1",[1.,1.,0.],[1.,2.,0.],2.,20,1),
        )
    ]


def test_generic_joint_skin_motion_creates_control_chain_candidates():
    r=discover(sample(),max_depth=4,max_candidates=12)
    assert r["source_joint_count"]==5
    assert r["source_deform_weighted_distal_count"]==2
    assert r["discovered_candidate_count"]>=3
    assert r["source_IK_FK_controller_truth"]=="UNKNOWN"
    assert r["validated_working_derived_IK_count"]==0
    assert all(c["ik_control_kind_proposal"]=="IK_EFFECTOR"
               and not c["source_mechanical_functionality_courted"]
               for c in r["proposals"])


def test_oracle_fields_cannot_enter_blind_source_snapshot():
    contaminated=sample()
    contaminated[2]["original_native_ik_constraint"]={
        "chain_count":3,"native_target":"target-pole"}
    with pytest.raises(ValueError,match="BLIND_INPUT_ALLOWLIST"):
        discover(contaminated)


def test_oracle_cannot_change_frozen_prediction_and_hits_measured_afterward():
    prediction=discover(sample(),max_depth=4,max_candidates=12)
    before=prediction["blind_proposal_sha256"]
    out=oracle_evaluate(prediction,[{
        "constraint_id":"oracle-hidden-1",
        "owner_bone_id":"G2",
        "source_native_chain_bone_ids":["G0","G1","G2"],
    }],top_k=12)
    assert out["blind_candidate_sha256"]==before
    assert out["oracle_ik_constraint_count"]==1
    assert out["exact_native_owner_end_effector_recall"]==1
    assert prediction["blind_proposal_sha256"]==before
    assert not out["source_native_ik_label_recovered"]


def test_oracle_mutated_prediction_is_fatal():
    r=discover(sample())
    r["proposals"][0]["end_bone_id"]="ORACLE_INJECTED"
    with pytest.raises(ValueError,match="CANDIDATE_SHA_MUST_BE_FROZEN"):
        oracle_evaluate(r,[])


def test_unknown_ik_is_not_negative_label():
    r=discover(sample())
    report=oracle_evaluate(r,[],top_k=10)
    assert report["oracle_ik_constraint_count"]==0
    assert report["exact_native_owner_end_effector_recall"]==0
    assert r["source_IK_FK_controller_truth"]=="UNKNOWN"
    assert r["source_training_admitted"] is False


def test_cycle_or_forged_skin_support_is_rejected():
    v=sample()
    v[0]["parent_bone_id"]="G2"
    with pytest.raises(ValueError,match="CYCLE"):
        discover(v)
    z=sample()
    z[1]["skin_weight_mass"]=float("nan")
    with pytest.raises(ValueError,match="SKIN_WEIGHT_MASS"):
        discover(z)


def test_morphology_and_bone_name_do_not_define_rank_score():
    a=sample()
    b=copy.deepcopy(a)
    swap={v["bone_id"]:"opaque_%03d"%i for i,v in enumerate(a)}
    for x in b:
        x["bone_id"]=swap[x["bone_id"]]
        if x["parent_bone_id"] is not None:
            x["parent_bone_id"]=swap[x["parent_bone_id"]]
    pa=discover(a,max_depth=4,max_candidates=16)
    pb=discover(b,max_depth=4,max_candidates=16)
    assert sorted(x["candidate_score"] for x in pa["proposals"])==sorted(
        x["candidate_score"] for x in pb["proposals"])
    assert pa["discovered_candidate_count"]==pb["discovered_candidate_count"]
