from __future__ import annotations

"""XPBD ownership admission; no unqualified legacy numerics or source labels."""

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
)
from compiler.realsas_compiler_core.mechanics_xpbd_domain_v1 import (
    HISTORICAL_XPBD_SOURCE_SHA256,
    DeclaredXPBDSecondaryRequestV1,
    prepare_declared_xpbd_secondary_domain_v1,
    require_promoted_xpbd_solver_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_control_solver_v1 import _mesh, _rig, _skin


def _fixture(*, primary_constraint: bool = False, ik_control: bool = False):
    mesh = _mesh()
    rig = _rig(mesh)
    rig = replace(
        rig,
        constraints=rig.constraints if primary_constraint else (),
        controls=rig.controls if ik_control else (rig.controls[0],),
    )
    skin = _skin(mesh, rig)
    mechanics = CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=tuple(c.canonical_constraint_id for c in rig.constraints),
        executable_operator_ids=("IK",) if primary_constraint else (),
        rest_pose_hash="1" * 64,
        qualification_hash="2" * 64,
        mechanics_lineage_hash="3" * 64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )
    return mesh, rig, skin, mechanics


def _prepare(obj, request=None):
    return prepare_declared_xpbd_secondary_domain_v1(
        *obj,
        request if request is not None else
        DeclaredXPBDSecondaryRequestV1(("K0",)),
    )


def test_explicit_chain_binds_canonical_joints_skin_surface_edges_without_physics_claim():
    result = _prepare(_fixture())
    assert result.schema_version == "RealSaS.XPBD.SecondaryDomain.v1"
    assert len(result.domain_hash) == 64
    assert result.regions[0].canonical_chain_id == "K0"
    assert result.regions[0].pinned_root_joint_id == "J0"
    assert result.regions[0].dynamic_joint_ids == ("J1", "J2")
    assert result.regions[0].segment_rest_lengths == pytest.approx((1., 1.))
    assert result.regions[0].skinned_vertex_ids == ("MV:1", "MV:2")
    assert result.regions[0].skinned_edge_ids == (("MV:1", "MV:2"),)
    assert result.metadata["partverse_semantics_authorized_for_activation"] is False
    assert result.metadata["physical_parameters_source_qualified"] is False
    assert result.metadata["solver_executed"] is False
    assert result == _prepare(_fixture())


def test_source_label_or_heuristic_is_not_xpbd_activation():
    fixture = _fixture()
    with pytest.raises(QualificationError, match="EXPLICIT_SELECTION_REQUIRED"):
        _prepare(fixture, DeclaredXPBDSecondaryRequestV1(
            ("K0",), activation_authority="PARTVERSE_TAIL_LABEL",
        ))
    with pytest.raises(QualificationError, match="EXPLICIT_SELECTION_REQUIRED"):
        _prepare(fixture, DeclaredXPBDSecondaryRequestV1(()))
    with pytest.raises(QualificationError, match="EXPLICIT_SELECTION_REQUIRED"):
        _prepare(fixture, DeclaredXPBDSecondaryRequestV1(("K0", "K0")))
    with pytest.raises(QualificationError, match="CHAIN_UNDECLARED"):
        _prepare(fixture, DeclaredXPBDSecondaryRequestV1(("UNKNOWN",)))


def test_primary_ik_control_and_attachment_cannot_be_secondary_owned():
    fixture = _fixture(primary_constraint=True, ik_control=True)
    with pytest.raises(QualificationError, match="PRIMARY_CONSTRAINT_CONFLICT"):
        _prepare(fixture)
    fixture = _fixture(ik_control=True)
    with pytest.raises(QualificationError, match="PRIMARY_CONTROL_CONFLICT"):
        _prepare(fixture)
    mesh, rig, skin, mechanics = _fixture()
    anchor = CanonicalAttachmentAnchorIRV2(
        "SWORD", parent_canonical_joint_id="J2",
        source_proposal_anchor_id="fixture:weapon",
    )
    attached = replace(rig, attachment_anchors=(anchor,))
    with pytest.raises(QualificationError, match="ATTACHMENT_OWNER_CONFLICT"):
        _prepare((mesh, attached, skin, mechanics))


def test_fails_closed_on_hash_drift_and_resource_policy():
    mesh, rig, skin, mechanics = _fixture()
    with pytest.raises(QualificationError, match="CANONICAL_LINEAGE_DRIFT"):
        _prepare((mesh, rig, replace(skin, mesh_lineage_hash="foreign"), mechanics))
    with pytest.raises(QualificationError, match="RESOURCE_POLICY_INVALID"):
        _prepare((mesh, rig, skin, mechanics), DeclaredXPBDSecondaryRequestV1(
            ("K0",), skin_support_threshold=0.,
        ))
    with pytest.raises(QualificationError, match="SURFACE_BUDGET"):
        _prepare((mesh, rig, skin, mechanics), DeclaredXPBDSecondaryRequestV1(
            ("K0",), max_surface_vertices=1,
        ))


def test_old_xpbd_bytes_are_pinned_but_not_executable_in_current_main():
    domain = _prepare(_fixture())
    with pytest.raises(QualificationError, match="HISTORICAL_SOURCE_HASH_MISMATCH"):
        require_promoted_xpbd_solver_v1(domain, source_sha256="0" * 64)
    with pytest.raises(QualificationError, match="SOLVER_NOT_PROMOTED"):
        require_promoted_xpbd_solver_v1(
            domain, source_sha256=HISTORICAL_XPBD_SOURCE_SHA256,
        )
