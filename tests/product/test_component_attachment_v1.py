from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.component_attachment import (
    ComponentAttachmentEvidenceIR,
    bind_component_assembly_to_directional_renderable_set,
    qualify_component_assembly,
    qualify_component_attachment,
    validate_component_assembly,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.v4 import directional_visual_state_hash
from compiler.realsas_compiler_core.v4_types import DirectionalRenderableIR, DirectionalRenderableSetIR


@dataclass(frozen=True)
class _DummyRenderable:
    component_id: str


def _skeleton():
    ids = ("j_root", "j_left_hand", "j_right_hand", "j_head", "j_chest")
    return SimpleNamespace(joints=tuple(SimpleNamespace(canonical_joint_id=value) for value in ids))


def _renderables(component_id="mage_full_subject"):
    directions = tuple(
        DirectionalRenderableIR(
            view_index=view,
            camera_binding_hash=f"camera:{view}",
            components=(_DummyRenderable(component_id),),
            direction_state_hash=f"direction:{view}",
        )
        for view in range(8)
    )
    value = DirectionalRenderableSetIR(directions=directions, directional_visual_state_hash="")
    return replace(value, directional_visual_state_hash=directional_visual_state_hash(value))


def _evidence(
    component_id,
    mechanical_class,
    *,
    parent="",
    detachability="FIXED_COMPONENT",
    one_hot=False,
    one_hot_joint="",
    skin_lineage="skin:v5",
    render_component_id="mage_full_subject",
):
    return ComponentAttachmentEvidenceIR(
        component_id=component_id,
        source_provenance_refs=(f"source:{component_id}",),
        mechanical_class=mechanical_class,
        directional_render_membership=tuple((view, render_component_id) for view in range(8)),
        geometry_membership_refs=(f"P1:membership:{component_id}",),
        geometry_lineage_hash="mesh:P1_B2_G10",
        skin_deformer_lineage_hash=skin_lineage,
        canonical_parent_joint_id=parent,
        socket_id="",
        bind_state_authority_hash=f"bind:{component_id}",
        detachability_class=detachability,
        visible_required=True,
        one_hot_carry_verified=one_hot,
        one_hot_carry_joint_id=one_hot_joint,
        visual_only_qualification_hash="",
        exclusion_reason="",
        qualification_evidence_hash=f"evidence:{component_id}",
        metadata={"fixture": True},
    )


def _report(**extra):
    return {
        "passed": True,
        "source_authority_verified": True,
        "classification_inferred_from_filename": False,
        **extra,
    }


def test_full_subject_component_assembly_is_typed_and_product_hash_bound():
    skeleton = _skeleton()
    renderables = _renderables()
    evidence = (
        (_evidence("body", "DEFORMABLE_COMPONENT"), _report(deformable_multi_joint_support=True)),
        (_evidence("book", "RIGID_SKINNED_COMPONENT", parent="j_left_hand", detachability="SWAPPABLE_SLOT_COMPONENT", one_hot=True, one_hot_joint="j_left_hand"), _report()),
        (_evidence("staff", "RIGID_SKINNED_COMPONENT", parent="j_right_hand", detachability="SWAPPABLE_SLOT_COMPONENT", one_hot=True, one_hot_joint="j_right_hand"), _report()),
        (_evidence("hat", "RIGID_SKINNED_COMPONENT", parent="j_head", one_hot=True, one_hot_joint="j_head"), _report()),
        (_evidence("cape", "RIGID_SKINNED_COMPONENT", parent="j_chest", one_hot=True, one_hot_joint="j_chest"), _report()),
    )
    qualified = tuple(
        qualify_component_attachment(item, skeleton, qualification_report=report)
        for item, report in evidence
    )
    assembly = qualify_component_assembly(
        qualified,
        skeleton,
        required_visible_component_ids=("body", "book", "staff", "hat", "cape"),
        directional_renderables=renderables,
        metadata={"witness": "Mage"},
    )
    assert assembly.qualification_report["passed"] is True
    assert assembly.qualification_report["visible_required_accounting"] == 1.0
    assert len(assembly.components) == 5
    assert assembly.component_assembly_hash

    rebound = bind_component_assembly_to_directional_renderable_set(renderables, assembly, skeleton)
    assert rebound.directional_visual_state_hash != renderables.directional_visual_state_hash
    assert rebound.metadata["component_assembly_hash"] == assembly.component_assembly_hash
    assert rebound.metadata["qualified_component_assembly"]["component_assembly_hash"] == assembly.component_assembly_hash
    validate_component_assembly(assembly, skeleton, directional_renderables=rebound)


def test_rigid_skinned_component_requires_verified_one_hot_carry():
    skeleton = _skeleton()
    bad = _evidence(
        "book",
        "RIGID_SKINNED_COMPONENT",
        parent="j_left_hand",
        one_hot=False,
        one_hot_joint="j_left_hand",
    )
    with pytest.raises(QualificationError, match="VERIFIED_ONE_HOT_CARRY"):
        qualify_component_attachment(bad, skeleton, qualification_report=_report())


def test_filename_inference_is_forbidden():
    skeleton = _skeleton()
    item = _evidence(
        "hat",
        "RIGID_SKINNED_COMPONENT",
        parent="j_head",
        one_hot=True,
        one_hot_joint="j_head",
    )
    with pytest.raises(QualificationError, match="FILENAME_CLASSIFICATION_FORBIDDEN"):
        qualify_component_attachment(
            item,
            skeleton,
            qualification_report=_report(classification_inferred_from_filename=True),
        )


def test_visible_required_detachability_must_be_resolved():
    skeleton = _skeleton()
    item = _evidence(
        "staff",
        "RIGID_SKINNED_COMPONENT",
        parent="j_right_hand",
        detachability="DETACHABILITY_UNKNOWN",
        one_hot=True,
        one_hot_joint="j_right_hand",
    )
    with pytest.raises(QualificationError, match="DETACHABILITY_UNRESOLVED"):
        qualify_component_attachment(item, skeleton, qualification_report=_report())


def test_render_membership_must_exist_in_current_directional_set():
    skeleton = _skeleton()
    item = _evidence(
        "book",
        "RIGID_SKINNED_COMPONENT",
        parent="j_left_hand",
        one_hot=True,
        one_hot_joint="j_left_hand",
        render_component_id="missing_component",
    )
    qualified = qualify_component_attachment(item, skeleton, qualification_report=_report())
    with pytest.raises(QualificationError, match="RENDER_MEMBERSHIP_NOT_PRESENT"):
        qualify_component_assembly(
            (qualified,),
            skeleton,
            required_visible_component_ids=("book",),
            directional_renderables=_renderables(),
        )
