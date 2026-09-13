from __future__ import annotations

"""Typed product authority for mechanical continuity underlay composition.

A continuity underlay is not new art, a second mesh, a second skin, or a repair of
learned semantics.  It is a render-composition contract over the exact qualified
external render/deformation substrate: the same P1/P1Q mesh, the same direct-model
skin rows and the same artist appearance are drawn beneath source-qualified typed
component partitions.  The purpose is to prevent exposure seams at rigid/deformable
boundaries without inventing pixels or mechanics.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .component_attachment import (
    QualifiedComponentAssemblyIR,
    bind_component_assembly_to_directional_renderable_set,
    component_assembly_hash,
    validate_component_assembly,
)
from .hashing import content_sha256
from .product_external_render import (
    assemble_product_v3_with_external_render_support,
    validate_external_directional_renderable_set,
    validate_external_renderable_component,
)
from .types import QualificationError
from .v4 import appearance_lineage_hash, component_state_hash, directional_visual_state_hash
from .v4_types import DirectionalRenderableSetIR, RenderableComponentIR

_REQUIRED_VIEWS = tuple(range(8))
_EXTERNAL_KEY = "external_render_support_qualification"
_CONTINUITY_KEY = "mechanical_continuity_underlay"
_ALLOWED_FOREGROUND_CLASSES = {
    "RIGID_SKINNED_COMPONENT",
    "RIGID_BONE_ATTACHMENT",
    "NON_MECHANICAL_VISUAL_COMPONENT",
}


def _required_hash(name: str, value: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() in {"latest", "current", "newest", "pending", "todo"}:
        raise QualificationError(f"CONTINUITY_UNDERLAY_INVALID_{name}")
    return value


@dataclass(frozen=True)
class QualifiedContinuityUnderlayIR:
    view_index: int
    substrate_component_id: str
    substrate_component_state_hash: str
    mesh_lineage_hash: str
    mesh_skin_lineage_hash: str
    appearance_lineage_hash: str
    external_render_support_qualification_hash: str
    materialization_manifest_sha256: str
    direct_binding_manifest_sha256: str
    component_assembly_hash: str
    partition_authority_sha256: str
    deformable_component_ids: tuple[str, ...]
    foreground_component_ids: tuple[str, ...]
    partition_membership_hash: str
    qualification_hash: str
    schema_version: str = "RealSaS.QualifiedContinuityUnderlayIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedContinuityUnderlaySetIR:
    views: tuple[QualifiedContinuityUnderlayIR, ...]
    component_assembly_hash: str
    qualification_hash: str
    schema_version: str = "RealSaS.QualifiedContinuityUnderlaySetIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuityRasterMeasurementIR:
    view_index: int
    underlay_qualification_hash: str
    boundary_sample_set_sha256: str
    composed_raster_sha256: str
    exposed_seam_pixel_count: int
    evaluated_boundary_pixel_count: int
    exposed_seam_fraction: float
    max_allowed_exposed_seam_fraction: float
    passed: bool
    measurement_hash: str
    schema_version: str = "RealSaS.ContinuityRasterMeasurementIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuityUnderlayProofIR:
    directional_visual_state_hash: str
    continuity_underlay_set_hash: str
    component_assembly_hash: str
    measurements: tuple[ContinuityRasterMeasurementIR, ...]
    proof_hash: str
    schema_version: str = "RealSaS.ContinuityUnderlayProofIR.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def continuity_underlay_hash(value: QualifiedContinuityUnderlayIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def continuity_underlay_set_hash(value: QualifiedContinuityUnderlaySetIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def continuity_raster_measurement_hash(value: ContinuityRasterMeasurementIR) -> str:
    payload = value.to_dict()
    payload.pop("measurement_hash", None)
    return content_sha256(payload)


def continuity_underlay_proof_hash(value: ContinuityUnderlayProofIR) -> str:
    payload = value.to_dict()
    payload.pop("proof_hash", None)
    return content_sha256(payload)


def _active_components(component_assembly: QualifiedComponentAssemblyIR, view_index: int):
    return tuple(
        component
        for component in component_assembly.components
        if int(view_index) in set(component.product_view_membership)
        and component.mechanical_class != "EXCLUDED_SOURCE_COMPONENT"
    )


def _partition_membership_hash(active_components) -> str:
    return content_sha256(
        {
            "components": [
                {
                    "component_id": str(component.component_id),
                    "mechanical_class": str(component.mechanical_class),
                    "geometry_lineage_hash": str(component.geometry_lineage_hash),
                    "geometry_membership_refs": tuple(str(x) for x in component.geometry_membership_refs),
                    "canonical_parent_joint_id": str(component.canonical_parent_joint_id),
                    "component_lineage_hash": str(component.component_lineage_hash),
                }
                for component in sorted(active_components, key=lambda row: row.component_id)
            ]
        }
    )


def qualify_continuity_underlay(
    *,
    view_index: int,
    substrate_component: RenderableComponentIR,
    mechanical,
    component_assembly: QualifiedComponentAssemblyIR,
    partition_authority_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedContinuityUnderlayIR:
    """Bind continuity composition to exact external mesh/skin/art and typed ownership."""
    view_index = int(view_index)
    if view_index not in _REQUIRED_VIEWS:
        raise QualificationError("CONTINUITY_UNDERLAY_INVALID_VIEW")
    if int(substrate_component.view_index) != view_index:
        raise QualificationError("CONTINUITY_UNDERLAY_COMPONENT_VIEW_MISMATCH")

    validate_external_renderable_component(substrate_component, mechanical)
    validate_component_assembly(component_assembly, mechanical.skeleton)
    if component_assembly.component_assembly_hash != component_assembly_hash(component_assembly):
        raise QualificationError("CONTINUITY_UNDERLAY_COMPONENT_ASSEMBLY_HASH_MISMATCH")

    external = substrate_component.metadata.get(_EXTERNAL_KEY)
    if not isinstance(external, Mapping):
        raise QualificationError("CONTINUITY_UNDERLAY_EXTERNAL_QUALIFICATION_REQUIRED")
    external_hash = _required_hash("EXTERNAL_QUALIFICATION_HASH", external.get("qualification_hash", ""))
    materialization_hash = _required_hash(
        "MATERIALIZATION_MANIFEST_SHA256", external.get("materialization_manifest_sha256", "")
    )
    direct_binding_hash = _required_hash(
        "DIRECT_BINDING_MANIFEST_SHA256", external.get("direct_binding_manifest_sha256", "")
    )
    partition_authority_sha256 = _required_hash("PARTITION_AUTHORITY_SHA256", partition_authority_sha256)

    active = _active_components(component_assembly, view_index)
    if not active:
        raise QualificationError("CONTINUITY_UNDERLAY_NO_ACTIVE_COMPONENTS")
    required_visible = set(component_assembly.required_visible_component_ids)
    active_ids = {component.component_id for component in active}
    missing_required = required_visible - active_ids
    if missing_required:
        raise QualificationError(
            f"CONTINUITY_UNDERLAY_REQUIRED_COMPONENT_MISSING_IN_VIEW:{sorted(missing_required)}"
        )

    deformable_ids = tuple(sorted(
        component.component_id for component in active
        if component.mechanical_class == "DEFORMABLE_COMPONENT"
    ))
    foreground_ids = tuple(sorted(
        component.component_id for component in active
        if component.mechanical_class in _ALLOWED_FOREGROUND_CLASSES
    ))
    if not deformable_ids:
        raise QualificationError("CONTINUITY_UNDERLAY_REQUIRES_DEFORMABLE_SUBSTRATE_OWNER")
    if not foreground_ids:
        raise QualificationError("CONTINUITY_UNDERLAY_REQUIRES_TYPED_FOREGROUND_COMPONENT")

    forbidden = {
        "new_pixels_generated": False,
        "topology_mutated": False,
        "weights_mutated": False,
        "scientific_mechanical_surface_relabelled": False,
    }
    supplied_metadata = dict(metadata or {})
    for key, required_value in forbidden.items():
        if key in supplied_metadata and bool(supplied_metadata[key]) != required_value:
            raise QualificationError(f"CONTINUITY_UNDERLAY_FORBIDDEN_MUTATION:{key}")

    value = QualifiedContinuityUnderlayIR(
        view_index=view_index,
        substrate_component_id=str(substrate_component.component_id),
        substrate_component_state_hash=str(substrate_component.component_state_hash),
        mesh_lineage_hash=str(substrate_component.mesh.mesh_lineage_hash),
        mesh_skin_lineage_hash=str(substrate_component.mesh_skin.mesh_skin_lineage_hash),
        appearance_lineage_hash=str(substrate_component.appearance.appearance_lineage_hash),
        external_render_support_qualification_hash=external_hash,
        materialization_manifest_sha256=materialization_hash,
        direct_binding_manifest_sha256=direct_binding_hash,
        component_assembly_hash=str(component_assembly.component_assembly_hash),
        partition_authority_sha256=partition_authority_sha256,
        deformable_component_ids=deformable_ids,
        foreground_component_ids=foreground_ids,
        partition_membership_hash=_partition_membership_hash(active),
        qualification_hash="",
        metadata={
            "composition_order": ("EXACT_LBS_CONTINUITY_UNDERLAY", "TYPED_COMPONENT_FOREGROUND"),
            "pixel_authority": "EXACT_SUBSTRATE_APPEARANCE_REUSE_ONLY",
            "topology_authority": "EXACT_SUBSTRATE_MESH_ONLY",
            "skin_authority": "EXACT_DIRECT_MODEL_MESH_SKIN_ONLY",
            "component_semantics_authority": "QUALIFIED_COMPONENT_ASSEMBLY_ONLY",
            **forbidden,
            **supplied_metadata,
        },
    )
    value = replace(value, qualification_hash=continuity_underlay_hash(value))
    validate_continuity_underlay(value, substrate_component, mechanical, component_assembly)
    return value


def validate_continuity_underlay(
    value: QualifiedContinuityUnderlayIR,
    substrate_component: RenderableComponentIR,
    mechanical,
    component_assembly: QualifiedComponentAssemblyIR,
) -> None:
    validate_external_renderable_component(substrate_component, mechanical)
    validate_component_assembly(component_assembly, mechanical.skeleton)
    if int(value.view_index) != int(substrate_component.view_index):
        raise QualificationError("CONTINUITY_UNDERLAY_VIEW_DRIFT")
    if value.substrate_component_id != substrate_component.component_id:
        raise QualificationError("CONTINUITY_UNDERLAY_COMPONENT_ID_DRIFT")
    if substrate_component.component_state_hash != component_state_hash(substrate_component):
        raise QualificationError("CONTINUITY_UNDERLAY_SUBSTRATE_COMPONENT_HASH_INVALID")
    if value.substrate_component_state_hash != substrate_component.component_state_hash:
        raise QualificationError("CONTINUITY_UNDERLAY_SUBSTRATE_COMPONENT_HASH_DRIFT")
    if value.mesh_lineage_hash != substrate_component.mesh.mesh_lineage_hash:
        raise QualificationError("CONTINUITY_UNDERLAY_MESH_HASH_DRIFT")
    if value.mesh_skin_lineage_hash != substrate_component.mesh_skin.mesh_skin_lineage_hash:
        raise QualificationError("CONTINUITY_UNDERLAY_MESH_SKIN_HASH_DRIFT")
    if value.appearance_lineage_hash != appearance_lineage_hash(substrate_component.appearance):
        raise QualificationError("CONTINUITY_UNDERLAY_APPEARANCE_HASH_DRIFT")
    if value.component_assembly_hash != component_assembly.component_assembly_hash:
        raise QualificationError("CONTINUITY_UNDERLAY_COMPONENT_ASSEMBLY_DRIFT")
    _required_hash("PARTITION_AUTHORITY_SHA256", value.partition_authority_sha256)

    external = substrate_component.metadata.get(_EXTERNAL_KEY)
    if not isinstance(external, Mapping):
        raise QualificationError("CONTINUITY_UNDERLAY_EXTERNAL_QUALIFICATION_REQUIRED")
    if value.external_render_support_qualification_hash != external.get("qualification_hash"):
        raise QualificationError("CONTINUITY_UNDERLAY_EXTERNAL_QUALIFICATION_DRIFT")
    if value.materialization_manifest_sha256 != external.get("materialization_manifest_sha256"):
        raise QualificationError("CONTINUITY_UNDERLAY_MATERIALIZATION_MANIFEST_DRIFT")
    if value.direct_binding_manifest_sha256 != external.get("direct_binding_manifest_sha256"):
        raise QualificationError("CONTINUITY_UNDERLAY_DIRECT_BINDING_MANIFEST_DRIFT")

    active = _active_components(component_assembly, value.view_index)
    expected_deformable = tuple(sorted(
        component.component_id for component in active
        if component.mechanical_class == "DEFORMABLE_COMPONENT"
    ))
    expected_foreground = tuple(sorted(
        component.component_id for component in active
        if component.mechanical_class in _ALLOWED_FOREGROUND_CLASSES
    ))
    if value.deformable_component_ids != expected_deformable:
        raise QualificationError("CONTINUITY_UNDERLAY_DEFORMABLE_PARTITION_DRIFT")
    if value.foreground_component_ids != expected_foreground:
        raise QualificationError("CONTINUITY_UNDERLAY_FOREGROUND_PARTITION_DRIFT")
    if value.partition_membership_hash != _partition_membership_hash(active):
        raise QualificationError("CONTINUITY_UNDERLAY_PARTITION_MEMBERSHIP_HASH_MISMATCH")

    for key in (
        "new_pixels_generated",
        "topology_mutated",
        "weights_mutated",
        "scientific_mechanical_surface_relabelled",
    ):
        if bool(value.metadata.get(key, True)):
            raise QualificationError(f"CONTINUITY_UNDERLAY_FORBIDDEN_MUTATION:{key}")
    if value.metadata.get("pixel_authority") != "EXACT_SUBSTRATE_APPEARANCE_REUSE_ONLY":
        raise QualificationError("CONTINUITY_UNDERLAY_PIXEL_AUTHORITY_DRIFT")
    if value.metadata.get("topology_authority") != "EXACT_SUBSTRATE_MESH_ONLY":
        raise QualificationError("CONTINUITY_UNDERLAY_TOPOLOGY_AUTHORITY_DRIFT")
    if value.metadata.get("skin_authority") != "EXACT_DIRECT_MODEL_MESH_SKIN_ONLY":
        raise QualificationError("CONTINUITY_UNDERLAY_SKIN_AUTHORITY_DRIFT")
    if value.qualification_hash != continuity_underlay_hash(value):
        raise QualificationError("CONTINUITY_UNDERLAY_QUALIFICATION_HASH_MISMATCH")


def build_continuity_underlay_set(
    views: tuple[QualifiedContinuityUnderlayIR, ...],
    *,
    component_assembly: QualifiedComponentAssemblyIR,
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedContinuityUnderlaySetIR:
    value = QualifiedContinuityUnderlaySetIR(
        views=tuple(sorted(views, key=lambda row: row.view_index)),
        component_assembly_hash=str(component_assembly.component_assembly_hash),
        qualification_hash="",
        metadata={
            "requires_exact_8_views": True,
            "continuity_underlay_is_render_composition_not_new_geometry": True,
            **dict(metadata or {}),
        },
    )
    value = replace(value, qualification_hash=continuity_underlay_set_hash(value))
    validate_continuity_underlay_set(value, component_assembly)
    return value


def validate_continuity_underlay_set(
    value: QualifiedContinuityUnderlaySetIR,
    component_assembly: QualifiedComponentAssemblyIR,
) -> None:
    if len(value.views) != 8 or tuple(row.view_index for row in value.views) != _REQUIRED_VIEWS:
        raise QualificationError("CONTINUITY_UNDERLAY_SET_REQUIRES_EXACTLY_8_ORDERED_VIEWS")
    if value.component_assembly_hash != component_assembly.component_assembly_hash:
        raise QualificationError("CONTINUITY_UNDERLAY_SET_COMPONENT_ASSEMBLY_DRIFT")
    if any(row.component_assembly_hash != value.component_assembly_hash for row in value.views):
        raise QualificationError("CONTINUITY_UNDERLAY_SET_VIEW_ASSEMBLY_DRIFT")
    if len({row.qualification_hash for row in value.views}) != 8:
        raise QualificationError("CONTINUITY_UNDERLAY_SET_DUPLICATE_VIEW_QUALIFICATION")
    if value.qualification_hash != continuity_underlay_set_hash(value):
        raise QualificationError("CONTINUITY_UNDERLAY_SET_HASH_MISMATCH")


def bind_continuity_underlay_set_to_directional_renderables(
    directional_renderables: DirectionalRenderableSetIR,
    continuity_underlays: QualifiedContinuityUnderlaySetIR,
    component_assembly: QualifiedComponentAssemblyIR,
    mechanical,
) -> DirectionalRenderableSetIR:
    validate_external_directional_renderable_set(directional_renderables, mechanical)
    validate_continuity_underlay_set(continuity_underlays, component_assembly)

    bound = bind_component_assembly_to_directional_renderable_set(
        directional_renderables,
        component_assembly,
        mechanical.skeleton,
    )
    by_view = {row.view_index: row for row in continuity_underlays.views}
    for direction in bound.directions:
        underlay = by_view[int(direction.view_index)]
        by_component_id = {component.component_id: component for component in direction.components}
        substrate = by_component_id.get(underlay.substrate_component_id)
        if substrate is None:
            raise QualificationError("CONTINUITY_UNDERLAY_SUBSTRATE_COMPONENT_NOT_RENDERED")
        validate_continuity_underlay(underlay, substrate, mechanical, component_assembly)

    updated = replace(
        bound,
        directional_visual_state_hash="",
        metadata={
            **dict(bound.metadata or {}),
            "continuity_underlay_set_hash": continuity_underlays.qualification_hash,
            _CONTINUITY_KEY: continuity_underlays.to_dict(),
            "continuity_underlay_qualified": True,
        },
    )
    updated = replace(updated, directional_visual_state_hash=directional_visual_state_hash(updated))
    validate_external_directional_renderable_set(updated, mechanical)
    validate_component_assembly(component_assembly, mechanical.skeleton, directional_renderables=updated)
    return updated


def assemble_product_v3_with_continuity_underlay(
    mechanical,
    directional_renderables: DirectionalRenderableSetIR,
    component_assembly: QualifiedComponentAssemblyIR,
    continuity_underlays: QualifiedContinuityUnderlaySetIR,
    capability_contract,
    motion_state,
    *,
    parent_state_hash: str | None = None,
    editable_metadata: Mapping[str, Any] | None = None,
    runtime_policy: Mapping[str, Any] | None = None,
):
    bound = bind_continuity_underlay_set_to_directional_renderables(
        directional_renderables,
        continuity_underlays,
        component_assembly,
        mechanical,
    )
    policy = {
        **dict(runtime_policy or {}),
        "mechanical_continuity_underlay_required": True,
        "continuity_underlay_set_hash": continuity_underlays.qualification_hash,
        "component_assembly_hash": component_assembly.component_assembly_hash,
        "continuity_underlay_may_generate_new_art": False,
        "continuity_underlay_may_mutate_topology": False,
        "continuity_underlay_may_mutate_skin_weights": False,
    }
    return assemble_product_v3_with_external_render_support(
        mechanical,
        bound,
        capability_contract,
        motion_state,
        parent_state_hash=parent_state_hash,
        editable_metadata=editable_metadata,
        runtime_policy=policy,
    )


def qualify_continuity_raster_measurement(
    *,
    view_index: int,
    underlay: QualifiedContinuityUnderlayIR,
    boundary_sample_set_sha256: str,
    composed_raster_sha256: str,
    exposed_seam_pixel_count: int,
    evaluated_boundary_pixel_count: int,
    max_allowed_exposed_seam_fraction: float,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuityRasterMeasurementIR:
    view_index = int(view_index)
    if view_index != int(underlay.view_index):
        raise QualificationError("CONTINUITY_MEASUREMENT_VIEW_MISMATCH")
    boundary_sample_set_sha256 = _required_hash("BOUNDARY_SAMPLE_SET_SHA256", boundary_sample_set_sha256)
    composed_raster_sha256 = _required_hash("COMPOSED_RASTER_SHA256", composed_raster_sha256)
    exposed = int(exposed_seam_pixel_count)
    evaluated = int(evaluated_boundary_pixel_count)
    limit = float(max_allowed_exposed_seam_fraction)
    if exposed < 0 or evaluated <= 0 or exposed > evaluated:
        raise QualificationError("CONTINUITY_MEASUREMENT_INVALID_PIXEL_COUNTS")
    if not (0.0 <= limit <= 1.0):
        raise QualificationError("CONTINUITY_MEASUREMENT_INVALID_POLICY_LIMIT")
    fraction = float(exposed) / float(evaluated)
    value = ContinuityRasterMeasurementIR(
        view_index=view_index,
        underlay_qualification_hash=underlay.qualification_hash,
        boundary_sample_set_sha256=boundary_sample_set_sha256,
        composed_raster_sha256=composed_raster_sha256,
        exposed_seam_pixel_count=exposed,
        evaluated_boundary_pixel_count=evaluated,
        exposed_seam_fraction=fraction,
        max_allowed_exposed_seam_fraction=limit,
        passed=bool(fraction <= limit),
        measurement_hash="",
        metadata={
            "measurement_is_external_observation_not_renderer_self_attestation": True,
            **dict(metadata or {}),
        },
    )
    value = replace(value, measurement_hash=continuity_raster_measurement_hash(value))
    validate_continuity_raster_measurement(value, underlay)
    return value


def validate_continuity_raster_measurement(
    value: ContinuityRasterMeasurementIR,
    underlay: QualifiedContinuityUnderlayIR,
) -> None:
    if value.view_index != underlay.view_index:
        raise QualificationError("CONTINUITY_MEASUREMENT_VIEW_DRIFT")
    if value.underlay_qualification_hash != underlay.qualification_hash:
        raise QualificationError("CONTINUITY_MEASUREMENT_UNDERLAY_HASH_DRIFT")
    if value.evaluated_boundary_pixel_count <= 0:
        raise QualificationError("CONTINUITY_MEASUREMENT_EMPTY_BOUNDARY_SAMPLE")
    expected_fraction = float(value.exposed_seam_pixel_count) / float(value.evaluated_boundary_pixel_count)
    if abs(float(value.exposed_seam_fraction) - expected_fraction) > 1.0e-15:
        raise QualificationError("CONTINUITY_MEASUREMENT_FRACTION_DRIFT")
    expected_pass = expected_fraction <= float(value.max_allowed_exposed_seam_fraction)
    if bool(value.passed) != bool(expected_pass):
        raise QualificationError("CONTINUITY_MEASUREMENT_PASS_FLAG_DRIFT")
    if value.measurement_hash != continuity_raster_measurement_hash(value):
        raise QualificationError("CONTINUITY_MEASUREMENT_HASH_MISMATCH")


def build_continuity_underlay_proof(
    *,
    directional_renderables: DirectionalRenderableSetIR,
    continuity_underlays: QualifiedContinuityUnderlaySetIR,
    component_assembly: QualifiedComponentAssemblyIR,
    measurements: tuple[ContinuityRasterMeasurementIR, ...],
    metadata: Mapping[str, Any] | None = None,
) -> ContinuityUnderlayProofIR:
    if len(measurements) != 8:
        raise QualificationError("CONTINUITY_PROOF_REQUIRES_EXACTLY_8_VIEW_MEASUREMENTS")
    by_view = {row.view_index: row for row in continuity_underlays.views}
    ordered = tuple(sorted(measurements, key=lambda row: row.view_index))
    if tuple(row.view_index for row in ordered) != _REQUIRED_VIEWS:
        raise QualificationError("CONTINUITY_PROOF_MEASUREMENT_VIEW_ORDER_MUST_BE_0_TO_7")
    for measurement in ordered:
        validate_continuity_raster_measurement(measurement, by_view[measurement.view_index])
        if not measurement.passed:
            raise QualificationError(f"CONTINUITY_PROOF_VIEW_FAILED:{measurement.view_index}")

    value = ContinuityUnderlayProofIR(
        directional_visual_state_hash=str(directional_renderables.directional_visual_state_hash),
        continuity_underlay_set_hash=str(continuity_underlays.qualification_hash),
        component_assembly_hash=str(component_assembly.component_assembly_hash),
        measurements=ordered,
        proof_hash="",
        metadata={
            "status": "PASS",
            "all_eight_views_measured": True,
            "product_pass_implied": False,
            **dict(metadata or {}),
        },
    )
    value = replace(value, proof_hash=continuity_underlay_proof_hash(value))
    validate_continuity_underlay_proof(
        value,
        directional_renderables=directional_renderables,
        continuity_underlays=continuity_underlays,
        component_assembly=component_assembly,
    )
    return value


def validate_continuity_underlay_proof(
    value: ContinuityUnderlayProofIR,
    *,
    directional_renderables: DirectionalRenderableSetIR,
    continuity_underlays: QualifiedContinuityUnderlaySetIR,
    component_assembly: QualifiedComponentAssemblyIR,
) -> None:
    if value.directional_visual_state_hash != directional_renderables.directional_visual_state_hash:
        raise QualificationError("CONTINUITY_PROOF_DIRECTIONAL_VISUAL_STATE_DRIFT")
    if value.continuity_underlay_set_hash != continuity_underlays.qualification_hash:
        raise QualificationError("CONTINUITY_PROOF_UNDERLAY_SET_DRIFT")
    if value.component_assembly_hash != component_assembly.component_assembly_hash:
        raise QualificationError("CONTINUITY_PROOF_COMPONENT_ASSEMBLY_DRIFT")
    if len(value.measurements) != 8 or tuple(row.view_index for row in value.measurements) != _REQUIRED_VIEWS:
        raise QualificationError("CONTINUITY_PROOF_REQUIRES_EXACT_8_VIEW_ORDER")
    by_view = {row.view_index: row for row in continuity_underlays.views}
    for measurement in value.measurements:
        validate_continuity_raster_measurement(measurement, by_view[measurement.view_index])
        if not measurement.passed:
            raise QualificationError(f"CONTINUITY_PROOF_VIEW_FAILED:{measurement.view_index}")
    if value.metadata.get("status") != "PASS":
        raise QualificationError("CONTINUITY_PROOF_STATUS_NOT_PASS")
    if bool(value.metadata.get("product_pass_implied", True)):
        raise QualificationError("CONTINUITY_PROOF_MUST_NOT_IMPLY_PRODUCT_PASS")
    if value.proof_hash != continuity_underlay_proof_hash(value):
        raise QualificationError("CONTINUITY_PROOF_HASH_MISMATCH")


__all__ = [
    "QualifiedContinuityUnderlayIR",
    "QualifiedContinuityUnderlaySetIR",
    "ContinuityRasterMeasurementIR",
    "ContinuityUnderlayProofIR",
    "continuity_underlay_hash",
    "continuity_underlay_set_hash",
    "continuity_raster_measurement_hash",
    "continuity_underlay_proof_hash",
    "qualify_continuity_underlay",
    "validate_continuity_underlay",
    "build_continuity_underlay_set",
    "validate_continuity_underlay_set",
    "bind_continuity_underlay_set_to_directional_renderables",
    "assemble_product_v3_with_continuity_underlay",
    "qualify_continuity_raster_measurement",
    "validate_continuity_raster_measurement",
    "build_continuity_underlay_proof",
    "validate_continuity_underlay_proof",
]
