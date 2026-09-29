from __future__ import annotations

import copy

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _fixture():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("a",(0.0,0.0,0.0),(0,),("p",),("oa",)),
            SurfaceNode("b",(1.0,0.0,0.0),(0,),("p",),("ob",)),
        ),
        (SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),),
        "surface-repartition",
    )
    parent=build_structural_partition(surface)
    directive={
        "schema":"RealSaS.MechanicalRepartitionDirective.v2",
        "status":"REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY",
        "source_candidate_lineage_hash":"candidate",
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent.partition_lineage_hash,
        "source_skeleton_lineage_hash":"skeleton-parent",
        "source_skin_lineage_hash":"skin-parent",
        "compatibility_report_hash":"compat",
        "unsafe_face_count":1,
        "candidate_separate_pair_count":1,
        "unresolved_unsafe_face_count":0,
        "proposed_boundary_overrides":(
            {
                "constraint_id":"dyn-ab",
                "a_surface_id":"a",
                "b_surface_id":"b",
                "decision":"SEPARATE",
                "evidence_refs":("compat:FACE:0",),
                "max_pairwise_skin_l1":2.0,
                "unsafe_face_indices":(0,),
                "confidence":1.0,
                "metadata":{
                    "evidence_class":"STAGE35_DYNAMIC_SKIN_TOPOLOGY_MECHANICAL",
                    "automatic":True,
                    "manual_authoring_used":False,
                },
            },
        ),
        "repair_operation":"STAGE17_REPARTITION_THEN_STAGE18_HOLELESS_DENSE_SUBDIVISION",
        "face_deletion_count":0,
        "weight_mutation":False,
        "vertex_position_mutation_at_stage35":False,
        "restart_from":"17_MECHANICAL_PARTITION_QUALIFIED",
        "mandatory_requalification_through":"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "auto_apply_allowed":False,
        "requires_trustworthy_skin_reliability_authority":True,
        "fail_closed_if_not_repartitioned":True,
        "directive_hash":"",
    }
    directive["directive_hash"]=content_sha256(
        {k:v for k,v in directive.items() if k!="directive_hash"}
    )
    auth={
        "schema":"RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status":"PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash":directive["directive_hash"],
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent.partition_lineage_hash,
        "source_skeleton_lineage_hash":"skeleton-parent",
        "source_skin_lineage_hash":"skin-parent",
        "teacher_inputs_used_by_predictor":False,
        "weight_reliability_closure_passed":True,
        "weight_reliability_evidence_hash":"e"*64,
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)
    return surface,parent,directive,auth


def test_authorized_repartition_builds_distinct_two_component_child():
    surface,parent,directive,auth=_fixture()
    child=build_repartitioned_partition_v2(
        surface=surface,parent_partition=parent,directive=directive,authorization=auth
    )
    assert len(parent.components)==1
    assert len(child.components)==2
    assert child.partition_lineage_hash!=parent.partition_lineage_hash
    assert sum(x.decision=="SEPARATE" for x in child.boundary_constraints)==1
    assert child.metadata["repair_child_attempt"] is True
    assert child.metadata["parent_partition_lineage_hash"]==parent.partition_lineage_hash
    assert child.metadata["repair_directive_hash"]==directive["directive_hash"]
    assert child.metadata["repartition_authorization_hash"]==auth["authorization_hash"]


def test_repartition_rejects_stale_skin_authorization():
    surface,parent,directive,auth=_fixture()
    bad=copy.deepcopy(auth)
    bad["source_skin_lineage_hash"]="other-skin"
    bad["authorization_hash"]=repartition_authorization_hash_v1(bad)
    with pytest.raises(QualificationError,match="REPARTITION_AUTHORIZATION_SKIN_LINEAGE_DRIFT"):
        build_repartitioned_partition_v2(
            surface=surface,parent_partition=parent,directive=directive,authorization=bad
        )


def test_repartition_rejects_teacher_input_predictor_authorization():
    surface,parent,directive,auth=_fixture()
    bad=copy.deepcopy(auth)
    bad["teacher_inputs_used_by_predictor"]=True
    bad["authorization_hash"]=repartition_authorization_hash_v1(bad)
    with pytest.raises(QualificationError,match="TEACHER_PREDICTOR_INPUT_FORBIDDEN"):
        build_repartitioned_partition_v2(
            surface=surface,parent_partition=parent,directive=directive,authorization=bad
        )


def test_repartition_rejects_unclosed_weight_reliability():
    surface,parent,directive,auth=_fixture()
    bad=copy.deepcopy(auth)
    bad["weight_reliability_closure_passed"]=False
    bad["authorization_hash"]=repartition_authorization_hash_v1(bad)
    with pytest.raises(QualificationError,match="WEIGHT_CLOSURE_NOT_PASS"):
        build_repartitioned_partition_v2(
            surface=surface,parent_partition=parent,directive=directive,authorization=bad
        )


def test_repartition_closes_noncut_seed_into_induced_graph_cut():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("a",(0.0,0.0,0.0),(0,),("p",),("oa",)),
            SurfaceNode("b",(1.0,0.0,0.0),(0,),("p",),("ob",)),
            SurfaceNode("c",(0.5,1.0,0.0),(0,),("p",),("oc",)),
        ),
        (
            SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("bc","b","c","LOCAL_NEIGHBOR",0.9),
            SurfaceRelation("ca","c","a","LOCAL_NEIGHBOR",0.8),
        ),
        "surface-cycle-repartition",
    )
    parent=build_structural_partition(surface)
    directive={
        "schema":"RealSaS.MechanicalRepartitionDirective.v2",
        "status":"REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY",
        "source_candidate_lineage_hash":"candidate",
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent.partition_lineage_hash,
        "source_skeleton_lineage_hash":"skeleton-parent",
        "source_skin_lineage_hash":"skin-parent",
        "compatibility_report_hash":"compat",
        "unsafe_face_count":1,
        "candidate_separate_pair_count":1,
        "unresolved_unsafe_face_count":0,
        "proposed_boundary_overrides":(
            {
                "constraint_id":"dyn-ab",
                "a_surface_id":"a",
                "b_surface_id":"b",
                "decision":"SEPARATE",
                "evidence_refs":("compat:FACE:0",),
                "max_pairwise_skin_l1":2.0,
                "unsafe_face_indices":(0,),
                "confidence":1.0,
                "metadata":{"evidence_class":"STAGE35_DYNAMIC_SKIN_TOPOLOGY_MECHANICAL"},
            },
        ),
        "repair_operation":"STAGE17_REPARTITION_THEN_STAGE18_HOLELESS_DENSE_SUBDIVISION",
        "face_deletion_count":0,
        "weight_mutation":False,
        "vertex_position_mutation_at_stage35":False,
        "restart_from":"17_MECHANICAL_PARTITION_QUALIFIED",
        "mandatory_requalification_through":"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "auto_apply_allowed":False,
        "requires_trustworthy_skin_reliability_authority":True,
        "fail_closed_if_not_repartitioned":True,
        "directive_hash":"",
    }
    directive["directive_hash"]=content_sha256(
        {k:v for k,v in directive.items() if k!="directive_hash"}
    )
    auth={
        "schema":"RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status":"PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash":directive["directive_hash"],
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent.partition_lineage_hash,
        "source_skeleton_lineage_hash":"skeleton-parent",
        "source_skin_lineage_hash":"skin-parent",
        "teacher_inputs_used_by_predictor":False,
        "weight_reliability_closure_passed":True,
        "weight_reliability_evidence_hash":"e"*64,
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)

    child=build_repartitioned_partition_v2(
        surface=surface,parent_partition=parent,directive=directive,authorization=auth
    )

    assert len(child.components)==2
    separate={
        tuple(sorted((x.a_surface_id,x.b_surface_id)))
        for x in child.boundary_constraints if x.decision=="SEPARATE"
    }
    assert ("a","b") in separate
    # The direct seed a-b is not a bridge in the triangle.  Stage17 must add
    # the induced crossing edge needed to make the seed a real component cut.
    assert len(separate)==2
    audit=child.metadata["partition_cut_closure"]
    assert audit["direct_seed_count"]==1
    assert audit["closure_added_separate_count"]==1
    assert audit["seed_constraint_violation_count"]==0
