from __future__ import annotations

"""Product assembly for exact render meshes qualified on an external support substrate.

This is a narrow compatibility bridge between canonical scientific mechanics (S/G/W)
and an independently qualified render/deformation mesh such as Mage P1_B2_G10.  It
preserves CanonicalPuppetGraph.v3 and all proof/runtime identity semantics while
refusing to relabel the scientific MechanicalStateIR surface.
"""

from dataclasses import replace
from typing import Any, Mapping

from .hashing import content_sha256
from .render_support import (
    ExternalRenderSupportQualificationIR,
    qualify_external_render_support,
    validate_external_render_support_qualification,
)
from .types import QualificationError
from .v4 import (
    appearance_lineage_hash,
    capability_contract_hash,
    component_state_hash,
    direction_state_hash,
    directional_visual_state_hash,
    motion_state_hash,
    validate_appearance_binding,
    validate_capability_contract,
    validate_mechanical_state,
    validate_motion_against_mechanical,
    validate_motion_state,
    validate_product_ontology,
)
from .v4_types import (
    AppearanceBindingIR,
    CanonicalPuppetGraphV3,
    DirectionalRenderableIR,
    DirectionalRenderableSetIR,
    MotionStateIR,
    RenderableComponentIR,
)

_REQUIRED_VIEWS = tuple(range(8))
_PRODUCT_REPRESENTATION_CLASS = "DIRECTIONAL_2D_2P5D_PUPPET"
_MECHANICAL_EQUIVALENCE_CLASS = "THREE_D_EQUIVALENT_MECHANICS"
_RENDERABLE_REPRESENTATION_CLASS = "DIRECTIONAL_2D_2P5D_RENDERABLE_SET"
_EXTERNAL_KEY = "external_render_support_qualification"


def _qualification_from_component(component: RenderableComponentIR) -> Mapping[str, Any]:
    raw = component.metadata.get(_EXTERNAL_KEY)
    if not isinstance(raw, Mapping):
        raise QualificationError("EXTERNAL_RENDERABLE_REQUIRES_RENDER_SUPPORT_QUALIFICATION")
    return raw


def build_external_renderable_component(
    *,
    component_id: str,
    view_index: int,
    mesh,
    mesh_skin,
    mechanical,
    appearance: AppearanceBindingIR,
    setup_order: int,
    coverage_classification: str,
    materialization_manifest_sha256: str,
    direct_binding_manifest_sha256: str,
    completions=(),
    default_visible: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> RenderableComponentIR:
    qualification = qualify_external_render_support(
        mesh,
        mesh_skin,
        mechanical,
        materialization_manifest_sha256=materialization_manifest_sha256,
        direct_binding_manifest_sha256=direct_binding_manifest_sha256,
        metadata={"component_id": str(component_id)},
    )
    merged = {
        "external_render_support": True,
        "scientific_mechanical_surface_relabelled": False,
        "runtime_deformation_authority": True,
        "full_silhouette_substrate": True,
        _EXTERNAL_KEY: qualification.to_dict(),
        **dict(metadata or {}),
    }
    value = RenderableComponentIR(
        str(component_id), int(view_index), mesh, mesh_skin, appearance,
        int(setup_order), str(coverage_classification), "",
        tuple(completions), bool(default_visible), metadata=merged,
    )
    value = replace(value, component_state_hash=component_state_hash(value))
    validate_external_renderable_component(value, mechanical)
    return value


def validate_external_renderable_component(component: RenderableComponentIR, mechanical) -> None:
    if int(component.view_index) not in _REQUIRED_VIEWS:
        raise QualificationError("EXTERNAL_RENDERABLE_INVALID_VIEW")
    qualification = _qualification_from_component(component)
    validate_external_render_support_qualification(qualification, component.mesh, component.mesh_skin, mechanical)
    if component.mesh.view_index != component.view_index:
        raise QualificationError("EXTERNAL_RENDERABLE_MESH_VIEW_MISMATCH")
    validate_appearance_binding(component.appearance)
    if component.appearance.target_view_index != component.view_index:
        raise QualificationError("EXTERNAL_RENDERABLE_APPEARANCE_VIEW_MISMATCH")
    if component.appearance.mesh_binding_hash != component.mesh.mesh_lineage_hash:
        raise QualificationError("EXTERNAL_RENDERABLE_APPEARANCE_MESH_MISMATCH")
    if component.appearance.camera_binding_hash != component.mesh.camera_binding_hash:
        raise QualificationError("EXTERNAL_RENDERABLE_APPEARANCE_CAMERA_MISMATCH")
    expected = {(fi, ci) for fi, face in enumerate(component.mesh.faces) for ci in range(len(face))}
    actual = {(c.face_index, c.corner_index) for c in component.appearance.corner_bindings}
    if actual != expected:
        raise QualificationError("EXTERNAL_RENDERABLE_APPEARANCE_COVERAGE_INCOMPLETE")
    completion_by_id = {c.completion_id: c for c in component.completions}
    if len(completion_by_id) != len(component.completions):
        raise QualificationError("EXTERNAL_RENDERABLE_DUPLICATE_COMPLETION_ID")
    referenced = {c.completion_id for c in component.appearance.corner_bindings if c.authority_class == "QUALIFIED_COMPLETION"}
    if not referenced.issubset(completion_by_id):
        raise QualificationError("EXTERNAL_RENDERABLE_UNQUALIFIED_COMPLETION_REFERENCE")
    required = {c.completion_id for c in component.completions if c.required}
    if not required.issubset(referenced):
        raise QualificationError("EXTERNAL_RENDERABLE_REQUIRED_COMPLETION_NOT_BOUND")
    if component.component_state_hash != component_state_hash(component):
        raise QualificationError("EXTERNAL_RENDERABLE_COMPONENT_HASH_MISMATCH")


def build_external_directional_renderable(*, view_index: int, camera_binding_hash: str, components, mechanical, metadata=None):
    value = DirectionalRenderableIR(int(view_index), str(camera_binding_hash), tuple(components), "", metadata=dict(metadata or {}))
    value = replace(value, direction_state_hash=direction_state_hash(value))
    validate_external_directional_renderable(value, mechanical)
    return value


def validate_external_directional_renderable(direction: DirectionalRenderableIR, mechanical) -> None:
    if direction.view_index not in _REQUIRED_VIEWS or not direction.components:
        raise QualificationError("EXTERNAL_DIRECTION_INVALID_OR_EMPTY")
    ids = [c.component_id for c in direction.components]
    orders = [c.setup_order for c in direction.components]
    if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
        raise QualificationError("EXTERNAL_DIRECTION_DUPLICATE_COMPONENT_OR_ORDER")
    for component in direction.components:
        validate_external_renderable_component(component, mechanical)
        if component.view_index != direction.view_index:
            raise QualificationError("EXTERNAL_DIRECTION_COMPONENT_VIEW_MISMATCH")
        if component.mesh.camera_binding_hash != direction.camera_binding_hash:
            raise QualificationError("EXTERNAL_DIRECTION_CAMERA_BINDING_MISMATCH")
    if direction.direction_state_hash != direction_state_hash(direction):
        raise QualificationError("EXTERNAL_DIRECTION_STATE_HASH_MISMATCH")


def build_external_directional_renderable_set(directions, mechanical, *, metadata=None):
    value = DirectionalRenderableSetIR(
        tuple(sorted(directions, key=lambda d: d.view_index)),
        "",
        metadata={
            "external_render_support": True,
            "scientific_mechanical_surface_relabelled": False,
            **dict(metadata or {}),
        },
    )
    value = replace(value, directional_visual_state_hash=directional_visual_state_hash(value))
    validate_external_directional_renderable_set(value, mechanical)
    return value


def validate_external_directional_renderable_set(value: DirectionalRenderableSetIR, mechanical) -> None:
    if value.representation_class != _RENDERABLE_REPRESENTATION_CLASS:
        raise QualificationError("EXTERNAL_DIRECTIONAL_RENDERABLE_REPRESENTATION_CLASS_MISMATCH")
    if value.exact_cardinality != 8 or len(value.directions) != 8:
        raise QualificationError("EXTERNAL_DIRECTIONAL_RENDERABLE_SET_REQUIRES_EXACTLY_8")
    if tuple(d.view_index for d in value.directions) != _REQUIRED_VIEWS:
        raise QualificationError("EXTERNAL_DIRECTIONAL_RENDERABLE_SET_VIEW_ORDER_MUST_BE_0_TO_7")
    for direction in value.directions:
        validate_external_directional_renderable(direction, mechanical)
    if value.directional_visual_state_hash != directional_visual_state_hash(value):
        raise QualificationError("EXTERNAL_DIRECTIONAL_RENDERABLE_SET_HASH_MISMATCH")


def _effective_motion(track) -> bool:
    if len(track.keys) < 2:
        return False
    first = track.keys[0]
    baseline = (first.translation_xy, first.rotation_deg, first.scale_xy, first.depth_offset)
    return any((k.translation_xy, k.rotation_deg, k.scale_xy, k.depth_offset) != baseline for k in track.keys[1:])


def assemble_product_v3_with_external_render_support(
    mechanical,
    directional_renderables: DirectionalRenderableSetIR,
    capability_contract,
    motion_state: MotionStateIR,
    *,
    parent_state_hash: str | None = None,
    editable_metadata: Mapping[str, Any] | None = None,
    runtime_policy: Mapping[str, Any] | None = None,
) -> CanonicalPuppetGraphV3:
    """Assemble v3 without pretending the P1 render support substrate is canonical S."""
    validate_mechanical_state(mechanical)
    validate_external_directional_renderable_set(directional_renderables, mechanical)
    validate_capability_contract(capability_contract)
    validate_motion_state(motion_state)
    validate_motion_against_mechanical(motion_state, mechanical)
    required = {r.capability_id for r in capability_contract.requirements if r.activation == "REQUIRED"}
    if "VISUAL_8_DIRECTION" in required and len(directional_renderables.directions) != 8:
        raise QualificationError("REQUIRED_VISUAL_8_DIRECTION_NOT_SATISFIED")
    if "PRESET_MOTION" in required:
        preset_ids = {c.clip_id for c in motion_state.clips if c.clip_kind == "PRESET"}
        if not preset_ids:
            raise QualificationError("REQUIRED_PRESET_MOTION_NOT_SATISFIED")
        if not any(t.clip_id in preset_ids and _effective_motion(t) for t in motion_state.joint_tracks):
            raise QualificationError("REQUIRED_PRESET_MOTION_HAS_NO_EFFECTIVE_JOINT_MOTION")
    ledger = (
        {"stage": "MECHANICAL_STATE", "hash": mechanical.mechanical_state_hash},
        {"stage": "EXTERNAL_RENDER_SUPPORT", "hash": directional_renderables.directional_visual_state_hash},
        {"stage": "CAPABILITY_CONTRACT", "hash": capability_contract.capability_contract_hash},
        {"stage": "MOTION_STATE", "hash": motion_state.motion_state_hash},
    )
    policy = dict(runtime_policy or {})
    if policy.get("full_3d_reconstruction_authority") is True:
        raise QualificationError("FULL_3D_RECONSTRUCTION_RUNTIME_POLICY_FORBIDDEN")
    policy.update({
        "representation_class": _PRODUCT_REPRESENTATION_CLASS,
        "mechanical_equivalence_class": _MECHANICAL_EQUIVALENCE_CLASS,
        "full_3d_reconstruction_authority": False,
        "external_render_support_allowed": True,
        "scientific_mechanical_surface_relabelled": False,
    })
    editable = dict(editable_metadata or {})
    payload = {
        "schema": "RealSaS.CanonicalPuppetGraph.v3",
        "representation_class": _PRODUCT_REPRESENTATION_CLASS,
        "mechanical_equivalence_class": _MECHANICAL_EQUIVALENCE_CLASS,
        "full_3d_reconstruction_authority": False,
        "parent": parent_state_hash,
        "mechanical": mechanical.mechanical_state_hash,
        "directional_visual": directional_renderables.directional_visual_state_hash,
        "capability_contract": capability_contract.capability_contract_hash,
        "motion": motion_state.motion_state_hash,
        "ledger": ledger,
        "editable": editable,
        "runtime_policy": policy,
    }
    state = content_sha256(payload)
    lineage = "PUPPETV3:" + content_sha256({"genesis": state, "parent": parent_state_hash})[:24]
    product = CanonicalPuppetGraphV3(
        lineage,
        state,
        parent_state_hash,
        mechanical,
        directional_renderables,
        capability_contract,
        motion_state,
        mechanical.mechanical_state_hash,
        directional_renderables.directional_visual_state_hash,
        motion_state.motion_state_hash,
        capability_contract.capability_contract_hash,
        ledger,
        editable,
        policy,
    )
    validate_product_ontology(product)
    return product


__all__ = [
    "ExternalRenderSupportQualificationIR",
    "build_external_renderable_component",
    "validate_external_renderable_component",
    "build_external_directional_renderable",
    "validate_external_directional_renderable",
    "build_external_directional_renderable_set",
    "validate_external_directional_renderable_set",
    "assemble_product_v3_with_external_render_support",
]
