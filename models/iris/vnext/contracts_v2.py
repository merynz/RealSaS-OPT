from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .contracts_v1 import (
    ATTACHMENT_RELATIONS_V1,
    CANONICAL_VIEW_COUNT_V1,
    CANONICAL_VIEW_PROVENANCE_V1,
    DEFORMATION_MODES_V1,
    CanonicalViewEvidenceSetV1,
    validate_canonical_view_evidence_set_v1,
)

SCHEMA_VERSION = "RealSaS.UnifiedIRISContracts.v2"

GEOMETRY_FIELD_CHANNELS_V2 = (
    "unsigned_distance",
    "oriented_level_set",
    "orientation_validity",
    "geometry_log_uncertainty",
)

EXTRACTION_SUPPORT_STATES_V2 = (
    "NONE",
    "POINT_TOUCH",
    "LINE_SUPPORT",
    "AREA_SUPPORT",
)

TOPOLOGY_RESIDUAL_TYPES_V2 = (
    "EXTRA_EDGE_CROSSING",
    "FACE_CLOSED_LOOP",
    "FACE_OPEN_INTERIOR_ENDPOINT",
    "FACE_BRANCH_OR_NONMANIFOLD",
    "AUTHORED_OPEN_BOUNDARY",
    "AUTHORED_NONMANIFOLD_INCIDENCE",
    "INTERIOR_TOPOLOGY_UNKNOWN",
)


@dataclass(frozen=True)
class GeometryFieldContractV2:
    """Representation-neutral continuous geometry evidence.

    FIT8 contains closed, open and non-manifold authored surfaces. A globally
    meaningful metric sign is therefore not a corpus invariant. The continuous
    geometry field keeps unsigned proximity universally available while exposing
    an oriented/local level-set only together with explicit validity.
    """

    require_unsigned_distance: bool = True
    require_oriented_level_set: bool = True
    require_orientation_validity: bool = True
    require_uncertainty: bool = True
    global_metric_sdf_required: bool = False

    def validate(self) -> None:
        if not self.require_unsigned_distance:
            raise ValueError("UNIFIED_IRIS_V2_UNSIGNED_DISTANCE_REQUIRED")
        if not self.require_oriented_level_set:
            raise ValueError("UNIFIED_IRIS_V2_ORIENTED_LEVEL_SET_REQUIRED")
        if not self.require_orientation_validity:
            raise ValueError("UNIFIED_IRIS_V2_ORIENTATION_VALIDITY_REQUIRED")
        if not self.require_uncertainty:
            raise ValueError("UNIFIED_IRIS_V2_GEOMETRY_UNCERTAINTY_REQUIRED")
        if self.global_metric_sdf_required:
            raise ValueError("UNIFIED_IRIS_V2_GLOBAL_METRIC_SDF_FORBIDDEN_AS_CORPUS_INVARIANT")


@dataclass(frozen=True)
class FExtractionSemanticContractV2:
    """Non-parametric semantics required of any F implementation.

    This contract intentionally does not choose autoregressive, graph, template,
    sparse-convolution, or other decoder families. It defines what an implementation
    is forbidden to collapse after the FIT8 representation courts.
    """

    scale_context_required: bool = True
    support_dimension_required: bool = True
    cell_supported_sheet_identity_required: bool = True
    variable_crossing_cardinality_allowed: bool = True
    variable_face_strand_cardinality_allowed: bool = True
    source_address_identity_required_for_assembly: bool = True
    fixed_local_patch_cardinality_required: bool = False
    semantic_owner_equals_local_sheet: bool = False
    material_equals_local_sheet: bool = False
    component_equals_local_sheet: bool = False
    pointwise_binary_surface_existence_is_authority: bool = False

    def validate(self) -> None:
        required = (
            self.scale_context_required,
            self.support_dimension_required,
            self.cell_supported_sheet_identity_required,
            self.variable_crossing_cardinality_allowed,
            self.variable_face_strand_cardinality_allowed,
            self.source_address_identity_required_for_assembly,
        )
        if not all(required):
            raise ValueError("UNIFIED_IRIS_V2_F_EXTRACTION_REQUIRED_SEMANTIC_MISSING")
        if self.fixed_local_patch_cardinality_required:
            raise ValueError("UNIFIED_IRIS_V2_FIXED_LOCAL_PATCH_CARDINALITY_FORBIDDEN")
        if self.semantic_owner_equals_local_sheet:
            raise ValueError("UNIFIED_IRIS_V2_OWNER_SHEET_IDENTITY_COLLAPSE_FORBIDDEN")
        if self.material_equals_local_sheet:
            raise ValueError("UNIFIED_IRIS_V2_MATERIAL_SHEET_IDENTITY_COLLAPSE_FORBIDDEN")
        if self.component_equals_local_sheet:
            raise ValueError("UNIFIED_IRIS_V2_COMPONENT_SHEET_IDENTITY_COLLAPSE_FORBIDDEN")
        if self.pointwise_binary_surface_existence_is_authority:
            raise ValueError("UNIFIED_IRIS_V2_POINTWISE_SURFACE_EXISTENCE_AUTHORITY_FORBIDDEN")


@dataclass(frozen=True)
class UnifiedFieldContractV2:
    owner_embedding_dim: int = 64
    material_embedding_dim: int = 32
    scene_dim_floor: int = 1024
    scene_layers_floor: int = 16
    scene_heads_floor: int = 16
    triplane_resolution_floor: int = 64
    triplane_channels_floor: int = 80
    geometry: GeometryFieldContractV2 = GeometryFieldContractV2()
    extraction: FExtractionSemanticContractV2 = FExtractionSemanticContractV2()

    def validate(self) -> None:
        values = (
            self.owner_embedding_dim,
            self.material_embedding_dim,
            self.scene_dim_floor,
            self.scene_layers_floor,
            self.scene_heads_floor,
            self.triplane_resolution_floor,
            self.triplane_channels_floor,
        )
        if min(values) <= 0:
            raise ValueError("UNIFIED_IRIS_V2_FIELD_CONTRACT_DIMENSION_INVALID")
        if self.scene_dim_floor % self.scene_heads_floor:
            raise ValueError("UNIFIED_IRIS_V2_SCENE_DIM_NOT_DIVISIBLE_BY_HEADS")
        self.geometry.validate()
        self.extraction.validate()


def validate_geometry_field_outputs_v2(outputs: Mapping[str, torch.Tensor]) -> None:
    missing = [name for name in GEOMETRY_FIELD_CHANNELS_V2 if name not in outputs]
    if missing:
        raise ValueError(f"UNIFIED_IRIS_V2_GEOMETRY_OUTPUT_MISSING:{missing}")

    shapes = {tuple(outputs[name].shape) for name in GEOMETRY_FIELD_CHANNELS_V2}
    if len(shapes) != 1:
        raise ValueError("UNIFIED_IRIS_V2_GEOMETRY_OUTPUT_SHAPE_MISMATCH")
    for name in GEOMETRY_FIELD_CHANNELS_V2:
        if not torch.isfinite(outputs[name]).all():
            raise ValueError(f"UNIFIED_IRIS_V2_GEOMETRY_OUTPUT_NONFINITE:{name}")
    if torch.any(outputs["unsigned_distance"] < 0.0):
        raise ValueError("UNIFIED_IRIS_V2_UNSIGNED_DISTANCE_NEGATIVE")
    validity = outputs["orientation_validity"]
    if torch.any(validity < 0.0) or torch.any(validity > 1.0):
        raise ValueError("UNIFIED_IRIS_V2_ORIENTATION_VALIDITY_OUT_OF_RANGE")


UnifiedFieldContractV2().validate()


__all__ = [
    "ATTACHMENT_RELATIONS_V1",
    "CANONICAL_VIEW_COUNT_V1",
    "CANONICAL_VIEW_PROVENANCE_V1",
    "DEFORMATION_MODES_V1",
    "EXTRACTION_SUPPORT_STATES_V2",
    "FExtractionSemanticContractV2",
    "GEOMETRY_FIELD_CHANNELS_V2",
    "GeometryFieldContractV2",
    "SCHEMA_VERSION",
    "TOPOLOGY_RESIDUAL_TYPES_V2",
    "UnifiedFieldContractV2",
    "CanonicalViewEvidenceSetV1",
    "validate_canonical_view_evidence_set_v1",
    "validate_geometry_field_outputs_v2",
]
