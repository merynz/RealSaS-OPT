"""Generic internal articulation frontier ablation, no oracle input."""
from __future__ import annotations

import copy
import pytest

from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import (
    discover,discover_articulation_frontiers,oracle_evaluate,
)


def skeleton():
    return [{
        "bone_id":str(i),
        "parent_bone_id":None if i==0 else str(i-1),
        "head_source":[float(i),0.,0.],
        "tail_source":[float(i+1),0.,0.],
        "use_deform":True,
        "skin_weight_mass":float(i+1),
        "skin_influenced_vertices":int(2+i),
        "source_motion_curve_count":0,
    } for i in range(5)]


def test_internal_owner_is_retrievable_from_source_only_frontier():
    bones=skeleton()
    baseline=discover(bones,max_depth=5,max_candidates=128)
    expanded=discover_articulation_frontiers(bones,max_depth=5,max_candidates=512)
    native=[{
        "constraint_id":"HIDDEN_ORACLE_ONLY",
        "owner_bone_id":"3",
        "source_native_chain_bone_ids":["1","2","3"],
    }]
    b=oracle_evaluate(baseline,native,top_k=128)
    e=oracle_evaluate(expanded,native,top_k=128)
    assert b["exact_native_owner_end_effector_recall"]==0
    assert e["exact_native_owner_end_effector_recall"]==1
    assert expanded["blind_proposal_sha256"]==e["blind_candidate_sha256"]
    assert expanded["source_IK_FK_controller_truth"]=="UNKNOWN"
    assert expanded["validated_working_derived_IK_count"]==0
    assert all(not row["native_ik_owner_or_chain_used_in_proposal"]
               for row in expanded["proposals"])
    assert expanded["source_internal_frontier_candidate_joint_count"]>0


def test_no_skin_evidence_mints_no_ik_teacher():
    empty=skeleton()
    for b in empty:
        b["skin_weight_mass"]=0.
        b["skin_influenced_vertices"]=0
    result=discover_articulation_frontiers(empty)
    assert result["discovered_candidate_count"]==0
    assert result["source_training_admitted"] is False


def test_oracle_or_native_constraint_in_input_rejected():
    poisoned=skeleton()
    poisoned[3]["ik_constraint_target"]="HIDDEN"
    with pytest.raises(ValueError,match="BLIND_INPUT_ALLOWLIST"):
        discover_articulation_frontiers(poisoned)


def test_name_rename_stability_ignores_anatomy_semantics():
    a=skeleton()
    b=copy.deepcopy(a)
    ids={str(i):"opaque_bone_%02d"%i for i in range(5)}
    for row in b:
        row["bone_id"]=ids[row["bone_id"]]
        if row["parent_bone_id"] is not None:
            row["parent_bone_id"]=ids[row["parent_bone_id"]]
    x=discover_articulation_frontiers(a)
    y=discover_articulation_frontiers(b)
    assert sorted(z["candidate_score"] for z in x["proposals"]) == sorted(
        z["candidate_score"] for z in y["proposals"])
    assert x["source_internal_frontier_candidate_joint_count"] == y[
        "source_internal_frontier_candidate_joint_count"]
