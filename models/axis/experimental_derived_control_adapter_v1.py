"""AXIS-side experimental adapter: BLIND source prior -> typed RigProgram parts.

This is NOT a Compiler repair feature. It is not connected to the shipping
AXIS decoder, training data writer, or runtime. Research-only candidate rows
can be contrasted with AXIS V6 model-generated controls and behavior courts.

The source-derived IK suggestion is an *INFERRED* control hypothesis. Valid
RigProgram references do NOT mean it is an approved source IK teacher,
a working constraint solver, or a materialized puppet control.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping,Sequence

from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramChainProposalIR,
    RigProgramControlProposalIR,
    RigProgramConstraintProposalIR,
)
from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import (
    SCHEMA as BLIND_SCHEMA,sha_payload,
)


def require(ok,code):
    if not ok:
        raise ValueError("AXIS_DERIVED_CONTROL_"+code)


def adapt_blind_prior_to_rigprogram_parts(
    source_only_prior:dict,
    *,
    source_to_proposal_joint_id:Mapping[str,str],
    source_joint_parent_id:Mapping[str,str|None],
    source_joint_tail_P:Mapping[str,Sequence[float]],
    max_controls:int=12,
):
    require(1<=max_controls<=64,"CONTROL_LIMIT")
    p=source_only_prior
    require(p.get("schema")==BLIND_SCHEMA
            and p.get("source_rig_ik_constraint_visibility")=="BLINDED"
            and p.get("validated_working_derived_IK_count")==0
            and p.get("source_training_admitted") is False
            and p.get("AXIS_RIGPROGRAM_control_proposal_not_compiler_repair") is True,
            "BLIND_PRIOR_PROVENANCE")
    assert_hash=sha_payload({
        k:v for k,v in p.items() if k!="blind_proposal_sha256"})
    require(p.get("blind_proposal_sha256")==assert_hash,"BLIND_PRIOR_SHA")
    require(len(set(source_to_proposal_joint_id.values()))==
            len(source_to_proposal_joint_id),"ONE_TO_ONE_JOINT_MAPPING")
    chains=[]
    controls=[]
    constraints=[]
    ends=set()
    for suggestion in p["proposals"]:
        if len(controls)>=max_controls:break
        end=suggestion["end_bone_id"]
        if end in ends:continue
        ids=suggestion["chain_bone_ids"]
        require(len(ids)>=2 and len(ids)==len(set(ids))
                and ids[-1]==end,"CHAIN_CONTIGUITY_SHAPE")
        require(all(x in source_to_proposal_joint_id and
                    x in source_joint_parent_id
                    for x in ids),"CHAIN_JOINT_MAPPING")
        require(all(source_joint_parent_id[b]==a
                    for a,b in zip(ids,ids[1:])),
                "CHAIN_NOT_CAUSAL_CONTIGUOUS")
        require(end in source_joint_tail_P,"MISSING_EFFECTOR_REST_POINT")
        tail=tuple(map(float,source_joint_tail_P[end]))
        require(len(tail)==3 and all(math.isfinite(x) for x in tail),
                "EFFECTOR_TAIL_NONFINITE")
        sids=tuple(source_to_proposal_joint_id[x] for x in ids)
        idx=len(controls)
        chain_id=f"DERIVED_PRIOR:CHAIN:{idx:04d}"
        control_id=f"DERIVED_PRIOR:CONTROL:{idx:04d}"
        constraint_id=f"DERIVED_PRIOR:CONSTRAINT:{idx:04d}"
        proof={
            "authority":"INFERRED_IK_PRIOR",
            "source_only_prior_sha256":p["blind_proposal_sha256"],
            "original_native_IK_not_available_to_producer":True,
            "native_ik_controller_proven":False,
            "solver_functionality_proven":False,
            "source_skin_deformation_proven":False,
            "license_and_training_admission":False,
            "source_contact_or_pole_truth":"UNKNOWN",
            "source_angle_limits_truth":"UNKNOWN",
            "research_adapter_not_shipping_decoder":True,
        }
        chains.append(RigProgramChainProposalIR(
            proposal_chain_id=chain_id,
            proposal_joint_ids=sids,
            chain_kind="GENERIC_KINEMATIC_CHAIN",
            metadata=dict(proof),
        ))
        controls.append(RigProgramControlProposalIR(
            proposal_control_id=control_id,
            control_kind="IK_EFFECTOR",
            target_proposal_joint_ids=(sids[-1],),
            rest_P=tail,
            metadata=dict(proof),
        ))
        constraints.append(RigProgramConstraintProposalIR(
            proposal_constraint_id=constraint_id,
            operator_id="IK",
            input_proposal_control_ids=(control_id,),
            output_proposal_joint_ids=sids,
            proposal_chain_id=chain_id,
            parameters={},
            metadata=dict(proof),
        ))
        ends.add(end)
    return {
        "source_blind_prediction_sha256":p["blind_proposal_sha256"],
        "source_blind_candidate_count":len(p["proposals"]),
        "provider_control_proposal_count":len(controls),
        "chains":tuple(chains),
        "controls":tuple(controls),
        "constraints":tuple(constraints),
        "original_native_ik_constraints_copied":False,
        "native_ik_solver_functionality":"UNKNOWN",
        "training_authorized":False,
        "materialization_authorized":False,
    }
