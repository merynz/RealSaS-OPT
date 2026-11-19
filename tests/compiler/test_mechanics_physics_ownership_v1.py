from __future__ import annotations

"""Physics single-owner court on actual canonical joint/skin addresses.

Tiny manually authored fixtures only validate execution ownership. They do
not prove learned wing, cloth, material physics or source animation quality.
"""

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, CanonicalSkinRowIRV2,
)
from compiler.realsas_compiler_core.mechanics_secondary_motion_v1 import (
    SecondaryChainPolicyV1,SecondaryMotionPolicyV1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_domain_v1 import (
    XPBDChainDomainV1,XPBDSecondaryDomainV1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfaceDomainV1,
)
from compiler.realsas_compiler_core.mechanics_physics_ownership_v1 import (
    qualify_mechanics_physics_domain_ownership_v1 as audit,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_secondary_runtime_contract_v1 import _rig


def _skin() -> CanonicalSkinIRV2:
    return CanonicalSkinIRV2(
        rows=(
            CanonicalSkinRowIRV2("V_FREE",(("T1",.7),("BASE",.3)),"source-free"),
            CanonicalSkinRowIRV2("V_PIN",(("T1",1.),),"source-pin"),
            CanonicalSkinRowIRV2("V_OTHER",(("BASE",1.),),"source-other"),
        ),
        mesh_lineage_hash="fixture-mesh",
        deform_structure_hash="fixture-deform",
        mesh_carrier_evidence_hash="e"*64,
        mesh_carrier_topology_hash="t"*64,
        mesh_carrier_geometry_hash="g"*64,
        carrier_skin_qualification_hash="q"*64,
        skin_code_proposal_hash="p"*64,
        skin_codec_architecture_id="fixture-codec",
        skin_codec_checkpoint_hash="c"*64,
        skin_lineage_hash="fixture-skin",
    )


def _surface(dynamic=("V_FREE",)) -> XPBDSurfaceDomainV1:
    return XPBDSurfaceDomainV1(
        mesh_lineage_hash="fixture-mesh",rig_lineage_hash="fixture-rig",
        skin_lineage_hash="fixture-skin",
        mechanics_lineage_hash="fixture-mechanics",
        deform_structure_hash="fixture-deform",
        control_program_hash="fixture-control",
        selected_faces=(("V_FREE","V_PIN","V_OTHER"),),
        vertex_ids=("V_FREE","V_PIN","V_OTHER"),
        pinned_vertex_ids=tuple(v for v in ("V_PIN","V_FREE","V_OTHER")
                                if v not in dynamic),
        dynamic_vertex_ids=dynamic,
        stretch_edges=(),bend_edges=(),triangle_rest_areas=(.5,),
        domain_hash="fixture-surface",
    )


def _oscillator() -> SecondaryMotionPolicyV1:
    return SecondaryMotionPolicyV1(
        rig_lineage_hash="fixture-rig",
        control_program_hash="fixture-control",
        chains=(SecondaryChainPolicyV1("TRAIL"),),
    )


def _xpbd_joints() -> XPBDSecondaryDomainV1:
    return XPBDSecondaryDomainV1(
        mesh_lineage_hash="fixture-mesh",rig_lineage_hash="fixture-rig",
        skin_lineage_hash="fixture-skin",mechanics_lineage_hash="fixture-mechanics",
        deform_structure_hash="fixture-deform",control_program_hash="fixture-control",
        regions=(XPBDChainDomainV1(
            canonical_chain_id="TRAIL",pinned_root_joint_id="BASE",
            dynamic_joint_ids=("T1","T2"),segment_rest_lengths=(1.,1.),
            skinned_vertex_ids=("V_FREE","V_PIN"),skinned_edge_ids=(),
            skin_support_threshold=.1,
        ),),
        domain_hash="fixture-secondary",
    )


def test_oscillator_and_xpbd_surface_cannot_move_same_skin_vertex():
    with pytest.raises(QualificationError,match="LIVE_SECONDARY_JOINT_SURFACE_DOUBLE_OWNERSHIP"):
        audit(_rig(),_skin(),
              secondary_motion_policy=_oscillator(),
              xpbd_surface_domain=_surface())


def test_pinned_boundary_may_follow_secondary_bone_but_dynamic_part_is_separate():
    report=audit(
        _rig(),_skin(),secondary_motion_policy=_oscillator(),
        xpbd_surface_domain=_surface(("V_OTHER",)),
    )
    assert report.disjoint
    assert report.secondary_joint_mode=="DECLARED_OSCILLATOR"
    assert report.surface_mode=="DECLARED_XPBD_SURFACE"
    assert report.secondary_joint_count==2
    assert report.surface_dynamic_vertex_count==1
    assert report.source_material_truth_claimed is False


def test_xpbd_joint_and_surface_legacy_ownership_is_still_rejected():
    with pytest.raises(QualificationError,match="LIVE_XPBD_JOINT_SURFACE_DOUBLE_OWNERSHIP"):
        audit(_rig(),_skin(),xpbd_secondary_domain=_xpbd_joints(),
              xpbd_surface_domain=_surface())
    with pytest.raises(QualificationError,match="SECONDARY_SOLVER_DOUBLE_OWNERSHIP"):
        audit(_rig(),_skin(),secondary_motion_policy=_oscillator(),
              xpbd_secondary_domain=_xpbd_joints())


def test_unbound_physics_does_not_claim_truth_or_reassign_vertices():
    report=audit(_rig(),_skin())
    assert report.secondary_joint_mode=="NONE"
    assert report.surface_mode=="NONE"
    assert report.secondary_skinned_vertex_count==0
    assert report.surface_dynamic_vertex_count==0
    with pytest.raises(QualificationError,match="SECONDARY_POLICY_AUTHORITY_LINEAGE_DRIFT"):
        audit(_rig(),_skin(),secondary_motion_policy=replace(
            _oscillator(),rig_lineage_hash="forged",
        ))
    with pytest.raises(QualificationError,match="SURFACE_VERTEX_DUPLICATE"):
        audit(_rig(),_skin(),xpbd_surface_domain=_surface(("V_OTHER","V_OTHER")))
