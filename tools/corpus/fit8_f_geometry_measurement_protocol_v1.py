from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

SCHEMA_VERSION = "RealSaS.FGeometryMeasurementProtocol.v1"
NORMAL_POLICIES = {
    "ORIENTED_WHERE_VALID_ELSE_ABSOLUTE",
    "ABSOLUTE_ONLY_DIAGNOSTIC",
}
DISTANCE_NORMALIZATIONS = {"EXTRACTION_PITCH"}


@dataclass(frozen=True)
class FGeometryMeasurementProtocolV1:
    protocol_id: str
    canonical_camera_contract_sha256: str
    extraction_scale_manifest_sha256: str
    thin_structure_manifest_sha256: str
    source_surface_sampling: str
    candidate_surface_sampling: str
    source_sample_count: int
    candidate_sample_count: int
    deterministic_seed: int
    symmetric_surface_distance_evaluator: str
    normal_error_evaluator: str
    normal_orientation_policy: str
    canonical_depth_evaluator: str
    silhouette_evaluator: str
    false_bridge_missing_sheet_evaluator: str
    self_intersection_evaluator: str
    thin_structure_evaluator: str
    distance_normalization: str
    spatial_tolerance: float
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_GEOMETRY_PROTOCOL_SCHEMA_INVALID")
        if not self.protocol_id:
            raise ValueError("F_GEOMETRY_PROTOCOL_ID_MISSING")
        for name in (
            "canonical_camera_contract_sha256",
            "extraction_scale_manifest_sha256",
            "thin_structure_manifest_sha256",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"F_GEOMETRY_PROTOCOL_SHA_INVALID:{name}")
        for name in (
            "source_surface_sampling",
            "candidate_surface_sampling",
            "symmetric_surface_distance_evaluator",
            "normal_error_evaluator",
            "canonical_depth_evaluator",
            "silhouette_evaluator",
            "false_bridge_missing_sheet_evaluator",
            "self_intersection_evaluator",
            "thin_structure_evaluator",
        ):
            if not getattr(self, name):
                raise ValueError(f"F_GEOMETRY_PROTOCOL_FIELD_MISSING:{name}")
        if self.source_sample_count <= 0 or self.candidate_sample_count <= 0:
            raise ValueError("F_GEOMETRY_PROTOCOL_SAMPLE_COUNT_INVALID")
        if self.deterministic_seed < 0:
            raise ValueError("F_GEOMETRY_PROTOCOL_SEED_INVALID")
        if self.normal_orientation_policy not in NORMAL_POLICIES:
            raise ValueError("F_GEOMETRY_PROTOCOL_NORMAL_POLICY_INVALID")
        if self.distance_normalization not in DISTANCE_NORMALIZATIONS:
            raise ValueError("F_GEOMETRY_PROTOCOL_DISTANCE_NORMALIZATION_INVALID")
        if self.spatial_tolerance <= 0.0:
            raise ValueError("F_GEOMETRY_PROTOCOL_SPATIAL_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("F_GEOMETRY_PROTOCOL_PROVENANCE_MISSING")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DISTANCE_NORMALIZATIONS",
    "FGeometryMeasurementProtocolV1",
    "NORMAL_POLICIES",
    "SCHEMA_VERSION",
]
