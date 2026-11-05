#!/usr/bin/env python3
"""V11 geometry-only heldout source intervention query design.

All source-face/group addresses are privileged diagnostic probes, never model
training labels or authored drawable ownership. Prior V9/V10 render outcomes
may prioritize still-underdetermined partition comparisons, but selected
probes must target previously UNTESTED finest source geometric groups.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import numpy as np
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan

SCHEMA="RealSaS.Corpus370GeometryOnlyHeldoutBranchProbePlan.v11"
V9="RealSaS.Corpus370NativeAnimatedPartitionSourceRender.v9"
V10="RealSaS.Corpus370NativeAnimatedEditResponseEquivalenceCourt.v10"

def require(ok,why):
    if not ok:raise ValueError(why)

def build_adaptive_plan(proposal,pairs,prior_v9_manifest,v10_report,
                        source_rest,*,source_rest_sha256,max_queries=12):
    plan=build_plan(proposal,pairs)
    sizes=plan["source_geometry_candidate_sizes"]
    source=plan["source_sha256"]
    require(type(max_queries) is int and 4<=max_queries<=32
            and prior_v9_manifest.get("schema")==V9
            and prior_v9_manifest.get("source_sha256")==source
            and prior_v9_manifest.get("plan")==plan
            and prior_v9_manifest.get("teacher_only") is True
            and prior_v9_manifest.get("graphics_owner_authorized") is False
            and prior_v9_manifest.get("training_authorized") is False
            and v10_report.get("schema")==V10
            and v10_report.get("status")==
                "REAL_NATIVE_EDIT_RESPONSE_PAIRWISE_EQUIVALENCE_MEASURED__ARTIST_OWNER_UNKNOWN"
            and v10_report.get("subject_id")==prior_v9_manifest.get("subject_id")
            and v10_report.get("source_sha256")==source
            and v10_report.get("candidate_sizes")==sizes
            and v10_report.get("artist_same_drawable_positive_pairs")==0
            and v10_report.get("artist_different_drawable_negative_pairs")==0
            and v10_report.get("graphics_owner_authorized") is False
            and v10_report.get("training_authorized") is False
            and v10_report.get("finest_partition_is_artist_truth") is False
            and pairs.get("parents",{}).get("rest_geometry_sha256")==source_rest_sha256,
            "V11_ORIGINAL_SOURCE_V9_V10_REST_AND_NO_OWNER_LINEAGE")
    vertices=np.asarray(source_rest["rest_vertices_source_world"],np.float64)
    faces=np.asarray(source_rest["faces"])
    require(vertices.ndim==2 and vertices.shape[1]==3
            and faces.ndim==2 and faces.shape==(plan["source_face_count"],3)
            and faces.dtype.kind in "iu"
            and len(vertices)>=3 and np.all(faces>=0)
            and np.all(faces<len(vertices))
            and np.isfinite(vertices).all(),
            "V11_EXACT_SOURCE_TRIANGLE_DOMAIN")
    by_size={c["slot_count"]:np.asarray(c["face_slot_teacher_only"],np.int64)
             for c in proposal["candidates"]
             if c["slot_count"] in sizes}
    require(set(by_size)==set(sizes),"V11_MISSING_SOURCE_GEOMETRIC_GROUPS")
    finest=sizes[-1]
    previous_faces=[t["source_face_teacher_only"] for t in plan["tasks"]]
    require(len(previous_faces)==len(set(previous_faces)),
            "V11_OLD_SOURCE_TASK_DUPLICATION")
    excluded={int(by_size[finest][f]) for f in previous_faces}
    require(len(excluded)<finest,"V11_NO_HELDOUT_FINEST_SOURCE_LEAVES")

    pairwise={}
    for row in v10_report.get("pairwise_equivalence_classes_scoped_to_tested_source_edits",[]):
        key=(row.get("candidate_a"),row.get("candidate_b"))
        require(key not in pairwise and key[0] in sizes and key[1] in sizes
                and key[0]<key[1]
                and row.get("total_source_edit_task_count")==len(previous_faces)
                and type(row.get("indistinguishable_source_edit_task_count")) is int
                and row.get("graphics_owner_authorized") is False
                and row.get("training_authorized") is False,
                "V11_PRIOR_EXACT_PARTITION_PAIRWISE_CENSUS")
        pairwise[key]=row
    require(set(pairwise)==set(combinations(sizes,2)),
            "V11_PRIOR_SOURCE_PAIR_CENSUS_MISSING")
    unresolved={(a,b) for (a,b),p in pairwise.items()
                if p["indistinguishable_source_edit_task_count"]==
                    p["total_source_edit_task_count"]}

    cross=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                   vertices[faces[:,2]]-vertices[faces[:,0]])
    area=np.linalg.norm(cross,axis=1)*.5
    center=vertices[faces].mean(axis=1)
    extent=max(1.,float(np.linalg.norm(np.ptp(vertices,axis=0))))
    valid=area>max(1e-15,extent*extent*1e-12)
    require(np.count_nonzero(valid)>finest//2,
            "V11_SOURCE_GEOMETRY_ALL_DEGENERATE")
    event_children=defaultdict(set)
    for a,b in zip(sizes[:-1],sizes[1:]):
        for child in range(b):
            parents=set(by_size[a][by_size[b]==child].tolist())
            require(len(parents)==1,"V11_NONNESTED_SOURCE_GEOMETRY")
            parent=next(iter(parents))
            event_children[(a,b,parent)].add(child)
    splits={key for key,children in event_children.items() if len(children)>=2}
    require(splits,"V11_NO_GEOMETRIC_BRANCHING")
    candidates={}
    for leaf in range(finest):
        if leaf in excluded:continue
        idx=np.flatnonzero((by_size[finest]==leaf)&valid)
        if len(idx)==0:continue
        centroid=np.average(center[idx],axis=0,weights=area[idx])
        face=int(idx[np.argmin(np.sum((center[idx]-centroid)**2,axis=1))])
        ancestors={(size,int(by_size[size][face])) for size in sizes}
        events={(a,b,int(by_size[a][face])) for a,b in zip(sizes[:-1],sizes[1:])}
        events&=splits
        require(events,"V11_SOURCE_FINEST_LEAF_NO_SPLIT_BRANCH")
        candidates[leaf]={"face":face,"area":float(area[idx].sum()),
                          "events":events,"ancestors":ancestors}
    require(candidates,"V11_NO_VALID_HELDOUT_GEOMETRY_PATCHES")
    selected=[];used_events=set();used_ancestors=set()
    while candidates and len(selected)<max_queries:
        def priority(c):
            events=c["events"]-used_events
            ancestors=c["ancestors"]-used_ancestors
            bonus=sum(6 for a,b,_ in events if (a,b) in unresolved)
            return (bonus+len(events)*5+len(ancestors)*2,
                    len(events),len(ancestors),c["area"],-c["face"])
        leaf=max(candidates,key=lambda x:priority(candidates[x]))
        target=candidates.pop(leaf)
        used_events|=target["events"];used_ancestors|=target["ancestors"]
        face=target["face"]
        selected.append({
            "test_index":len(selected),
            "source_face_teacher_only":face,
            "finest_group_id_teacher_only":leaf,
            "source_groups_per_candidate_size_teacher_only":{
                str(size):int(by_size[size][face]) for size in sizes},
            "source_triangle_area_teacher_only":float(area[face]),
            "source_leaf_surface_area_teacher_only":target["area"],
            "new_candidate_split_branch_events_teacher_only":[
                {"coarse_size":a,"fine_size":b,"coarse_group_id":p}
                for a,b,p in sorted(target["events"])],
            "same_drawable":"UNKNOWN",
            "different_drawable":"UNKNOWN",
            "training_authorized":False,
        })
    leaves={x["finest_group_id_teacher_only"] for x in selected}
    require(len(selected)==len(leaves)
            and not(leaves&excluded)
            and all(x["same_drawable"]=="UNKNOWN" for x in selected),
            "V11_PREEXISTING_SOURCE_EDIT_TARGET_LEAKAGE")
    return {
        "schema":SCHEMA,
        "status":"SOURCE_EXACT_NONSEMANTIC_HELDOUT_SPLIT_QUERY_PLAN__NOT_DRAWABLE_TRUTH",
        "subject_id":prior_v9_manifest["subject_id"],
        "source_sha256":source,
        "source_rest_geometry_sha256":source_rest_sha256,
        "candidate_group_sizes_are_experimental_not_production":sizes,
        "source_face_count":len(faces),
        "source_rest_degenerate_triangle_count":int(np.count_nonzero(~valid)),
        "prior_privileged_source_edit_target_count":len(previous_faces),
        "prior_finest_group_ids_excluded_count":len(excluded),
        "source_finest_groups_count":finest,
        "source_finest_heldout_valid_leaf_count":len(selected)+len(candidates),
        "new_source_face_intervention_query_count":len(selected),
        "query_budget":max_queries,
        "potential_splittable_parent_events":len(splits),
        "selected_splittable_parent_events":len(used_events),
        "prior_edit_indistinguishable_candidate_size_pairs":[
            {"coarse_size":a,"fine_size":b} for a,b in sorted(unresolved)],
        "query_targets_source_teacher_only":selected,
        "real_rgba_consequences_measured_for_heldout_queries":False,
        "group_owner_labels_minted":False,
        "source_group_ids_student_targets":False,
        "source_mesh_bone_action_names_as_labels":False,
        "source_native_resolution_caa_authorized":False,
        "graphics_owner_authorized":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "optimal_group_count_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
