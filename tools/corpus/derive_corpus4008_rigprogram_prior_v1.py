#!/usr/bin/env python3
"""Source-only generic AXIS RigProgram IK-chain *prior*, never native IK truth.

The function receives ONLY an explicit allowlisted joint forest, deform skin
influence coverage and action curve counts. Original IK constraints, pole
targets, control collections, constraint chain_count, owner and target IDs
are NOT readable here. The native oracle is opened only after candidate
predictions are frozen to a SHA.

Joint IDs are opaque identities. Names and strings are NEVER parsed to guess
body parts or IK controllers. No assumption of humanoid/feet/hands/up axis.
This is an exploratory provider-side hypothesis, NOT Compiler autocompletion.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "RealSaS.Corpus4008BlindDerivedRigProgramPrior.v1"
BONE_KEYS = frozenset((
    "bone_id", "parent_bone_id", "head_source", "tail_source",
    "use_deform", "skin_weight_mass", "skin_influenced_vertices",
    "source_motion_curve_count",
))


def require(ok, code):
    if not ok:
        raise ValueError("C4008_DERIVED_IK_" + code)


def finite3(value):
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(
        type(x) in (float, int) and math.isfinite(x) for x in value)


def sha_payload(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()


def validate_bones(bones):
    require(isinstance(bones, list) and 1 <= len(bones) <= 20_000,
            "JOINT_CENSUS_BOUNDS")
    parent = {}
    for b in bones:
        require(type(b) is dict and set(b) == BONE_KEYS,
                "BLIND_INPUT_ALLOWLIST_FORBIDS_CONSTRAINT_OR_ORACLE")
        jid=b["bone_id"]
        pid=b["parent_bone_id"]
        require(isinstance(jid,str) and jid and len(jid)<=512
                and jid not in parent, "JOINT_ID_UNIQUE")
        require(pid is None or isinstance(pid,str), "JOINT_PARENT_SHAPE")
        require(finite3(b["head_source"]) and finite3(b["tail_source"]),
                "JOINT_POSITION_NONFINITE")
        require(type(b["use_deform"]) is bool,
                "DEFORM_FLAG")
        require(type(b["skin_weight_mass"]) in (int,float)
                and math.isfinite(b["skin_weight_mass"])
                and b["skin_weight_mass"]>=0,
                "SKIN_WEIGHT_MASS")
        require(type(b["skin_influenced_vertices"]) is int
                and b["skin_influenced_vertices"]>=0,
                "SKIN_VERTEX_SUPPORT")
        require(type(b["source_motion_curve_count"]) is int
                and b["source_motion_curve_count"]>=0,
                "MOTION_SOURCE_HINT")
        parent[jid]=pid
    # Reject cycles even when source Blender hierarchy should be a forest.
    for jid,pid in parent.items():
        require(pid is None or pid in parent and pid!=jid,
                "PARENT_OUTSIDE_SOURCE_GRAPH")
    visited=set()
    active=set()
    for jid in parent:
        if jid in visited:continue
        path=[]
        cur=jid
        while cur is not None and cur not in visited:
            require(cur not in active,"CYCLE")
            active.add(cur)
            path.append(cur)
            cur=parent[cur]
        visited.update(path)
        for key in path:active.discard(key)
    return parent


def discover(blind_bones, *, max_depth=4, max_candidates=192):
    require(type(max_depth) is int and 2<=max_depth<=6,
            "DEPTH_LIMIT")
    require(type(max_candidates) is int and 1<=max_candidates<=512,
            "CANDIDATE_BUDGET")
    parents=validate_bones(blind_bones)
    by_id={b["bone_id"]:b for b in blind_bones}
    # Weight-backed deform leaf discovery is independent of oracle IK
    # owners/targets. Joint orientation and contact semantics stay UNKNOWN.
    nchildren={b["bone_id"]:0 for b in blind_bones}
    for pid in parents.values():
        if pid is not None:nchildren[pid]+=1
    has_weighted_child={jid:False for jid in parents}
    # Traverse parent forest upward; a weighted deform descendant makes a
    # candidate less distal. No bone name string parsing here.
    for b in blind_bones:
        if not b["use_deform"] or b["skin_weight_mass"]<=1e-8:
            continue
        cur=parents[b["bone_id"]]
        while cur is not None:
            has_weighted_child[cur]=True
            cur=parents[cur]
    leaves=[
        b for b in blind_bones
        if b["use_deform"] and b["skin_weight_mass"]>1e-8
        and b["skin_influenced_vertices"]>0
        and not has_weighted_child[b["bone_id"]]
    ]
    proposal=[]
    for end in leaves:
        chain_end_to_root=[]
        cur=end["bone_id"]
        while cur is not None and len(chain_end_to_root)<max_depth:
            current=by_id[cur]
            length=math.dist(current["head_source"],current["tail_source"])
            if length<=1e-9:break
            chain_end_to_root.append(cur)
            cur=parents[cur]
        if len(chain_end_to_root)<2:continue
        for depth in range(2,len(chain_end_to_root)+1):
            path=list(reversed(chain_end_to_root[:depth]))
            # Source-observed mass and motion are only ranking priors.
            mass=sum(by_id[x]["skin_weight_mass"] for x in path)
            motion=sum(by_id[x]["source_motion_curve_count"] for x in path)
            reach=sum(math.dist(by_id[x]["head_source"],
                                by_id[x]["tail_source"]) for x in path)
            score=(1.5*math.log1p(end["skin_influenced_vertices"])
                   +0.25*math.log1p(mass)
                   +0.15*math.log1p(motion)
                   +0.40*math.log1p(reach)
                   -0.07*(depth-2))
            require(math.isfinite(score),"PRIOR_SCORE_NONFINITE")
            proposal.append({
                "end_bone_id":end["bone_id"],
                "chain_bone_ids":path,
                "path_length_bones":len(path),
                "ik_control_kind_proposal":"IK_EFFECTOR",
                "operator_proposal":"IK",
                "candidate_score":round(score,9),
                "candidate_input_families":[
                    "JOINT_PARENT_FOREST","SOURCE_REST_GEOMETRY",
                    "IMPORTED_SPARSE_SKIN_INFLUENCE_HINT",
                    "SOURCE_ACTION_CURVE_COUNT_HINT"],
                "contact_semantics":"UNKNOWN",
                "axis_limits":"UNKNOWN",
                "native_ik_owner_or_chain_used_in_proposal":False,
                "source_mechanical_functionality_courted":False,
                "representation_policy_admitted":False,
            })
    # Do not derive rank from native constraint identity or body names.
    proposal.sort(key=lambda x:(-x["candidate_score"],
                                 x["path_length_bones"],
                                 sha_payload(x["chain_bone_ids"])))
    proposal=proposal[:max_candidates]
    result={
        "schema":SCHEMA,
        "status":"SOURCE_GRAPH_SKIN_MOTION_IK_PRIORS_ONLY",
        "source_joint_count":len(blind_bones),
        "source_deform_weighted_distal_count":len(leaves),
        "discovered_candidate_count":len(proposal),
        "max_chain_depth_bones":max_depth,
        "proposals":proposal,
        "source_rig_ik_constraint_visibility":"BLINDED",
        "source_IK_FK_controller_truth":"UNKNOWN",
        "validated_working_derived_IK_count":0,
        "AXIS_RIGPROGRAM_control_proposal_not_compiler_repair":True,
        "MIRA_skin_deformation_consequence":"NOT_EVALUATED",
        "commercial_training_rights":"UNKNOWN",
        "source_training_admitted":False,
    }
    result["blind_proposal_sha256"]=sha_payload(result)
    return result


def discover_articulation_frontiers(
    blind_bones, *, max_depth=5, max_candidates=512,
):
    """Ablation: internal joints of *source-only* weighted distal chains.

    A joint on the path to a distal deform region may itself be a candidate
    animator effector (e.g. a limb hinge or spine branch); native IK owners
    and constraints are NOT consulted. This is an ablation inspired by a
    weakness in the development rigs. It requires fresh, held-out native rigs
    to evaluate generalization.
    """
    require(2<=max_depth<=6 and 1<=max_candidates<=512,
            "FRONTIER_LIMITS")
    parents=validate_bones(blind_bones)
    by_id={b["bone_id"]:b for b in blind_bones}
    initial=discover(blind_bones,max_depth=max_depth,max_candidates=512)
    proposals=list(initial["proposals"])
    source_by_chain={}
    for p in proposals:
        key=(p["end_bone_id"],tuple(p["chain_bone_ids"]))
        source_by_chain[key]=p
    # Candidate ends are strictly ancestors of a weighted distal leaf.
    # Do NOT read body-part names, IK chain count, original native
    # constraints, targets, poles, drivers or Controller collections.
    frontier_ends={}
    for base in initial["proposals"]:
        distal=base["end_bone_id"]
        dist=0
        cur=parents[distal]
        while cur is not None and dist<max_depth-1:
            dist+=1
            if cur not in frontier_ends or dist<frontier_ends[cur]:
                frontier_ends[cur]=dist
            cur=parents[cur]
    for end_id,dist in frontier_ends.items():
        ancestor_list=[]
        cur=end_id
        while cur is not None and len(ancestor_list)<max_depth:
            b=by_id[cur]
            if math.dist(b["head_source"],b["tail_source"])<=1e-9:
                break
            ancestor_list.append(cur)
            cur=parents[cur]
        if len(ancestor_list)<2:continue
        for depth in range(2,len(ancestor_list)+1):
            path=tuple(reversed(ancestor_list[:depth]))
            key=(end_id,path)
            if key in source_by_chain:continue
            mass=sum(by_id[x]["skin_weight_mass"] for x in path)
            motion=sum(by_id[x]["source_motion_curve_count"] for x in path)
            reach=sum(math.dist(by_id[x]["head_source"],
                                by_id[x]["tail_source"]) for x in path)
            downstream_support=max(
                (by_id[x["end_bone_id"]]["skin_influenced_vertices"]
                 for x in initial["proposals"]
                 if end_id in x["chain_bone_ids"]),
                default=0)
            score=(1.5*math.log1p(downstream_support)
                   +.25*math.log1p(mass)
                   +.15*math.log1p(motion)
                   +.4*math.log1p(reach)
                   -.07*(depth-2)-.32*dist)
            require(math.isfinite(score),"FRONTIER_SCORE_NONFINITE")
            item={
                "end_bone_id":end_id,
                "chain_bone_ids":list(path),
                "path_length_bones":len(path),
                "ik_control_kind_proposal":"IK_EFFECTOR",
                "operator_proposal":"IK",
                "candidate_score":round(score,9),
                "candidate_family":"STRUCTURAL_ANCESTOR_FRONTIER",
                "distance_from_weighted_distal_evidence":dist,
                "candidate_input_families":[
                    "JOINT_PARENT_FOREST","SOURCE_REST_GEOMETRY",
                    "IMPORTED_SPARSE_SKIN_INFLUENCE_HINT",
                    "SOURCE_ACTION_CURVE_COUNT_HINT"],
                "contact_semantics":"UNKNOWN",
                "axis_limits":"UNKNOWN",
                "native_ik_owner_or_chain_used_in_proposal":False,
                "source_mechanical_functionality_courted":False,
                "representation_policy_admitted":False,
            }
            source_by_chain[key]=item
    for item in proposals:
        item["candidate_family"]="WEIGHTED_DEFORM_DISTAL"
        item["distance_from_weighted_distal_evidence"]=0
    proposals=list(source_by_chain.values())
    proposals.sort(key=lambda x:(-x["candidate_score"],
                                 x["path_length_bones"],
                                 sha_payload(x["chain_bone_ids"])))
    result={
        "schema":SCHEMA,
        "status":"SOURCE_GRAPH_SKIN_MOTION_IK_PRIORS_ONLY",
        "source_joint_count":len(blind_bones),
        "source_deform_weighted_distal_count":initial[
            "source_deform_weighted_distal_count"],
        "source_internal_frontier_candidate_joint_count":
            len(frontier_ends),
        "discovered_candidate_count":min(max_candidates,len(proposals)),
        "max_chain_depth_bones":max_depth,
        "candidate_strategy":
            "SOURCE_WEIGHTED_DISTAL_PLUS_STRUCTURAL_ANCESTOR_FRONTIERS",
        "previous_baseline_candidate_sha256":
            initial["blind_proposal_sha256"],
        "proposals":proposals[:max_candidates],
        "source_rig_ik_constraint_visibility":"BLINDED",
        "source_IK_FK_controller_truth":"UNKNOWN",
        "validated_working_derived_IK_count":0,
        "AXIS_RIGPROGRAM_control_proposal_not_compiler_repair":True,
        "MIRA_skin_deformation_consequence":"NOT_EVALUATED",
        "commercial_training_rights":"UNKNOWN",
        "source_training_admitted":False,
    }
    result["blind_proposal_sha256"]=sha_payload(result)
    return result


def oracle_evaluate(blind_proposals, truth_rows, *, top_k=32):
    """Post-freeze research scoring. Oracle is forbidden in discover()."""
    require(blind_proposals["schema"]==SCHEMA
            and blind_proposals.get("blind_proposal_sha256")==sha_payload({
                k:v for k,v in blind_proposals.items()
                if k!="blind_proposal_sha256"}),
            "CANDIDATE_SHA_MUST_BE_FROZEN_BEFORE_ORACLE")
    require(1<=top_k<=512,"TOPK")
    results=[]
    candidates=blind_proposals["proposals"][:top_k]
    for row in truth_rows:
        owner=row["owner_bone_id"]
        native_chain=row["source_native_chain_bone_ids"]
        require(isinstance(owner,str) and
                isinstance(native_chain,list) and native_chain
                and len(native_chain)==len(set(native_chain)),
                "ORACLE_REFERENCE")
        # Distal end matching is deliberately separate from chain overlap.
        best=0.0
        overlap_rank=None
        effector_rank=None
        for idx,c in enumerate(candidates,1):
            cand=set(c["chain_bone_ids"])
            oracle=set(native_chain)
            jaccard=len(cand & oracle)/len(cand | oracle)
            best=max(best,jaccard)
            if overlap_rank is None and jaccard>=.5:
                overlap_rank=idx
            if effector_rank is None and c["end_bone_id"]==owner:
                effector_rank=idx
        results.append({
            "oracle_constraint_id_evaluation_only":row["constraint_id"],
            "native_owner_bone_id_evaluation_only":owner,
            "native_chain_length_bones":len(native_chain),
            "max_chain_jaccard_topk":best,
            "first_at_least_half_chain_overlap_rank":overlap_rank,
            "exact_distal_owner_effector_rank":effector_rank,
            "candidate_prediction_mutated_using_oracle":False,
        })
    return {
        "oracle_ik_constraint_count":len(truth_rows),
        "blind_top_k":top_k,
        "chain_overlap_at_least_half":sum(
            x["first_at_least_half_chain_overlap_rank"] is not None
            for x in results),
        "exact_native_owner_end_effector_recall":sum(
            x["exact_distal_owner_effector_rank"] is not None
            for x in results),
        "source_native_ik_label_recovered":False,
        "blind_candidate_sha256":blind_proposals["blind_proposal_sha256"],
        "oracle_evaluation":results,
        "training_authorized":False,
    }
