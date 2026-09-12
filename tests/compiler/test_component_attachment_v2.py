from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.component_attachment import ComponentEvidenceIR
from compiler.realsas_compiler_core.component_attachment_v2 import (
    qualify_component_set_against_mechanical_v2,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedJoint,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
)
from compiler.realsas_compiler_core.v4 import qualified_skeleton_v2_lineage_hash
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


def _fixture(owner_weight: float = 1.0):
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("S_BODY", (0.0, 0.0, 0.0), tuple(range(8)), ("p",), ("o",)),
            SurfaceNode("S_HAT", (0.0, 1.0, 0.0), tuple(range(8)), ("p",), ("o",)),
        ),
        geometry_lineage_hash="SURFACE",
    )
    sk0 = QualifiedSkeletonIRV2(
        joints=(
            QualifiedJoint("J_ROOT", (0.0, 0.0, 0.0), None, ("S_BODY",), "P0"),
            QualifiedJoint("J_HEAD", (0.0, 1.0, 0.0), "J_ROOT", ("S_HAT",), "P1"),
        ),
        deform_root_ids=("J_ROOT",),
        assembly_root_binding={},
        qualification_report={"passed": True},
        skeleton_lineage_hash="",
    )
    skeleton = replace(sk0, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk0))
    skin = QualifiedSkinIR(
        rows=(
            QualifiedSkinRow("S_BODY", (("J_ROOT", 1.0),), 0.0, 0.0),
            QualifiedSkinRow(
                "S_HAT",
                (("J_HEAD", owner_weight), ("J_ROOT", 1.0 - owner_weight)),
                0.0,
                0.0,
            ),
        ),
        surface_binding_hash="SURFACE",
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        qualification_report={"passed": True},
        skin_lineage_hash="SKIN",
    )
    evidence = (
        ComponentEvidenceIR(
            component_id="body",
            source_component_refs=("src:body",),
            observation_views=tuple(range(8)),
            surface_ids=("S_BODY",),
            proposed_mechanical_class="DEFORMABLE_COMPONENT",
            detachability_class="FIXED_COMPONENT",
            source_geometry_hash="GEO_BODY",
            skin_or_deformer_lineage_hash="SKIN",
        ),
        ComponentEvidenceIR(
            component_id="hat",
            source_component_refs=("src:hat",),
            observation_views=tuple(range(8)),
            surface_ids=("S_HAT",),
            proposed_mechanical_class="RIGID_SKINNED_COMPONENT",
            proposed_parent_joint_id="J_HEAD",
            detachability_class="DETACHABILITY_UNKNOWN",
            bind_state_authority_hash="BIND_HAT",
            source_geometry_hash="GEO_HAT",
            skin_or_deformer_lineage_hash="SKIN",
        ),
    )
    return surface, skeleton, skin, evidence


def test_v2_verifies_actual_rigid_skin_and_visible_surface_accounting():
    surface, skeleton, skin, evidence = _fixture(1.0)
    result = qualify_component_set_against_mechanical_v2(
        evidence,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    assert result.qualification_report["status"] == "PASS_MECHANICAL_EVIDENCE_VERIFIED"
    assert result.qualification_report["visible_surface_accounting_fraction"] == 1.0
    hat = next(c for c in result.components if c.component_id == "hat")
    assert hat.qualification_report["rigid_skin_evidence_verified"] is True
    assert hat.qualification_report["min_rigid_owner_weight"] == 1.0


def test_v2_rejects_rigid_label_when_exact_skin_is_not_rigid_enough():
    surface, skeleton, skin, evidence = _fixture(0.8)
    with pytest.raises(QualificationError, match="RIGID_SKIN_EVIDENCE_FAIL"):
        qualify_component_set_against_mechanical_v2(
            evidence,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
        )


def test_v2_rejects_unaccounted_visible_surface():
    surface, skeleton, skin, evidence = _fixture(1.0)
    with pytest.raises(QualificationError, match="VISIBLE_SURFACE_ACCOUNTING_INCOMPLETE"):
        qualify_component_set_against_mechanical_v2(
            evidence[:1],
            surface=surface,
            skeleton=skeleton,
            skin=skin,
        )
