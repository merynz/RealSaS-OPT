"""Unified IRIS vNext public contract surface.

Keep this package intentionally narrow until implementation choices are promoted
through separate reviewed changes. Research-only decoders and corpus probes must
not become canonical merely by being importable here.
"""

from .contracts_v1 import (
    ATTACHMENT_RELATIONS_V1,
    CANONICAL_VIEW_COUNT_V1,
    CANONICAL_VIEW_PROVENANCE_V1,
    DEFORMATION_MODES_V1,
    CanonicalViewEvidenceSetV1,
    UnifiedFieldContractV1,
    validate_canonical_view_evidence_set_v1,
)
from .contracts_v2 import (
    EXTRACTION_SUPPORT_STATES_V2,
    GEOMETRY_FIELD_CHANNELS_V2,
    TOPOLOGY_RESIDUAL_TYPES_V2,
    FExtractionSemanticContractV2,
    GeometryFieldContractV2,
    UnifiedFieldContractV2,
    validate_geometry_field_outputs_v2,
)
from .semantic_partition_contract_v1 import SemanticPartitionContractV1

__all__ = [
    "ATTACHMENT_RELATIONS_V1",
    "CANONICAL_VIEW_COUNT_V1",
    "CANONICAL_VIEW_PROVENANCE_V1",
    "DEFORMATION_MODES_V1",
    "CanonicalViewEvidenceSetV1",
    "UnifiedFieldContractV1",
    "validate_canonical_view_evidence_set_v1",
    "EXTRACTION_SUPPORT_STATES_V2",
    "GEOMETRY_FIELD_CHANNELS_V2",
    "TOPOLOGY_RESIDUAL_TYPES_V2",
    "FExtractionSemanticContractV2",
    "GeometryFieldContractV2",
    "UnifiedFieldContractV2",
    "validate_geometry_field_outputs_v2",
    "SemanticPartitionContractV1",
]
