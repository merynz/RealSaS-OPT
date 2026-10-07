from types import SimpleNamespace
import pytest

from compiler.realsas_compiler_core.required_tree_v1 import qualify_required_skeleton_tree_v1
from compiler.realsas_compiler_core.types import SkeletonProposalIR,SkeletonProposalJoint,SkeletonProposalEdge,QualificationError


def _joint(i,pos):
    return SkeletonProposalJoint(
        f"P{i}",pos,1.0 if i==0 else 0.0,1.0,("S0",),
        metadata={
            "sequence_index":i,
            "hard_causal_parent_authority":True,
            "discrete_xyz_authority":True,
            "coord_bins_per_axis":256,
        },
    )


def _proposal(parent_for_2=1):
    joints=(_joint(0,(0,0,0)),_joint(1,(1,0,0)),_joint(2,(2,0,0)))
    edges=(
        SkeletonProposalEdge("E01","P0","P1",1.0,1.0,True,False,"hard",{}),
        SkeletonProposalEdge("E12",f"P{parent_for_2}","P2",1.0,1.0,True,False,"hard",{}),
    )
    return SkeletonProposalIR(
        joints,edges,"SURFACE","AXIS",
        metadata={
            "hard_causal_parent_authority":True,
            "compiler_role":"VALIDATE_AND_MATERIALIZE",
            "compiler_parent_reselection_allowed":False,
            "tree_serialization":"BFS_PARENT_BEFORE_CHILD",
            "geometry_authority":"ORDERED_256_BIN_XYZ_ARGMAX",
            "conditional_residual_diffusion":False,
            "diffusion_runtime_authority":False,
        },
    )


def test_axis_required_tree_is_materialized_without_optimizer_reselection():
    surface=SimpleNamespace(geometry_lineage_hash="SURFACE",surface_nodes=(SimpleNamespace(surface_id="S0"),))
    value=qualify_required_skeleton_tree_v1(surface,_proposal())
    q=value.qualification_report
    assert q["solver"]=="AXIS_REQUIRED_TREE_VALIDATOR_V1"
    assert q["selection_search_performed"] is False
    assert q["parent_reselection_performed"] is False
    assert len(value.joints)==3


def test_parent_before_child_is_a_hard_contract():
    surface=SimpleNamespace(geometry_lineage_hash="SURFACE",surface_nodes=(SimpleNamespace(surface_id="S0"),))
    p=_proposal(parent_for_2=1)
    bad=list(p.joints)
    m1=dict(bad[1].metadata); m1["sequence_index"]=2
    m2=dict(bad[2].metadata); m2["sequence_index"]=1
    bad[1]=SkeletonProposalJoint("P1",(1,0,0),0.0,1.0,("S0",),metadata=m1)
    bad[2]=SkeletonProposalJoint("P2",(2,0,0),0.0,1.0,("S0",),metadata=m2)
    p=SkeletonProposalIR(tuple(bad),p.edges,p.surface_binding_hash,p.model_provenance,p.schema_version,p.metadata)
    with pytest.raises(QualificationError,match="PARENT_NOT_BEFORE_CHILD"):
        qualify_required_skeleton_tree_v1(surface,p)


def test_required_tree_rejects_diffusion_or_non_discrete_geometry_authority():
    surface=SimpleNamespace(geometry_lineage_hash="SURFACE",surface_nodes=(SimpleNamespace(surface_id="S0"),))
    p=_proposal()
    meta=dict(p.metadata); meta["diffusion_runtime_authority"]=True
    bad=SkeletonProposalIR(p.joints,p.edges,p.surface_binding_hash,p.model_provenance,p.schema_version,meta)
    with pytest.raises(QualificationError,match="DIFFUSION_AUTHORITY_FORBIDDEN"):
        qualify_required_skeleton_tree_v1(surface,bad)
