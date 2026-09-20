from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class GeometrySubstrateViewIR:
    view_index: int
    silhouette_recall: float
    silhouette_precision: float
    largest_coherent_hole_fraction: float
    interior_uncovered_fraction: float
    source_foreground_pixel_count: int
    predicted_foreground_pixel_count: int
    component_recall: float
    silhouette_edge_p95_px: float
    passed: bool
    schema_version: str = "RealSaS.GeometrySubstrateViewIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GeometrySubstrateQualificationIR:
    zero_surface_binding_hash: str
    observation_set_binding_hash: str
    camera_set_binding_hash: str
    normalization_binding_hash: str
    policy: Json
    views: tuple[GeometrySubstrateViewIR, ...]
    qualification_report: Json
    substrate_hash: str
    schema_version: str = "RealSaS.GeometrySubstrateQualificationIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def geometry_substrate_hash(value: GeometrySubstrateQualificationIR) -> str:
    payload = value.to_dict()
    payload.pop("substrate_hash", None)
    return content_sha256(payload)


def validate_geometry_substrate(value: GeometrySubstrateQualificationIR) -> None:
    rows = tuple(sorted(value.views, key=lambda row: row.view_index))
    if len(rows) != 8 or tuple(row.view_index for row in rows) != tuple(range(8)):
        raise QualificationError("GEOMETRY_SUBSTRATE_REQUIRES_EXACT_8_FIRST_WITNESS_VIEWS")
    if not all(row.passed for row in rows):
        raise QualificationError("GEOMETRY_SUBSTRATE_HAS_FAILED_VIEW")
    if value.qualification_report.get("status") != "PASS_GEOMETRY_SUBSTRATE":
        raise QualificationError("GEOMETRY_SUBSTRATE_STATUS_INVALID")
    if value.qualification_report.get("appearance_authority_used") is not False:
        raise QualificationError("GEOMETRY_SUBSTRATE_MAY_NOT_USE_APPEARANCE_AUTHORITY")
    if value.qualification_report.get("teacher_truth_used") is not False:
        raise QualificationError("GEOMETRY_SUBSTRATE_TEACHER_TRUTH_FORBIDDEN")
    if value.substrate_hash != geometry_substrate_hash(value):
        raise QualificationError("GEOMETRY_SUBSTRATE_HASH_MISMATCH")


def build_geometry_substrate_qualification(
    *,
    zero_surface_binding_hash: str,
    observation_set_binding_hash: str,
    camera_set_binding_hash: str,
    normalization_binding_hash: str,
    policy: Mapping[str, Any],
    views,
    metadata: Mapping[str, Any] | None = None,
) -> GeometrySubstrateQualificationIR:
    rows = tuple(sorted(tuple(views), key=lambda row: row.view_index))
    all_pass = bool(rows) and all(row.passed for row in rows)
    value = GeometrySubstrateQualificationIR(
        zero_surface_binding_hash=str(zero_surface_binding_hash),
        observation_set_binding_hash=str(observation_set_binding_hash),
        camera_set_binding_hash=str(camera_set_binding_hash),
        normalization_binding_hash=str(normalization_binding_hash),
        policy=dict(policy),
        views=rows,
        qualification_report={
            "status": "PASS_GEOMETRY_SUBSTRATE" if all_pass else "FAIL_GEOMETRY_SUBSTRATE",
            "every_view_passed": all_pass,
            "view_count": len(rows),
            "appearance_authority_used": False,
            "teacher_truth_used": False,
            "claim": "GEOMETRY_CARRIER_COMPATIBILITY_ONLY",
            "does_not_claim": [
                "RGB_OR_LINE_ART_FIDELITY",
                "COMPLETE_APPEARANCE",
                "RUNTIME_VISIBILITY_CORRECTNESS",
                "TEXTURE_SAMPLING_CORRECTNESS",
            ],
        },
        substrate_hash="",
        metadata=dict(metadata or {}),
    )
    value = replace(value, substrate_hash=geometry_substrate_hash(value))
    if all_pass:
        validate_geometry_substrate(value)
    return value


def geometry_substrate_from_dict(payload: Mapping[str, Any]) -> GeometrySubstrateQualificationIR:
    rows = tuple(
        GeometrySubstrateViewIR(
            view_index=int(row["view_index"]),
            silhouette_recall=float(row["silhouette_recall"]),
            silhouette_precision=float(row["silhouette_precision"]),
            largest_coherent_hole_fraction=float(row["largest_coherent_hole_fraction"]),
            interior_uncovered_fraction=float(row["interior_uncovered_fraction"]),
            source_foreground_pixel_count=int(row["source_foreground_pixel_count"]),
            predicted_foreground_pixel_count=int(row["predicted_foreground_pixel_count"]),
            component_recall=float(row["component_recall"]),
            silhouette_edge_p95_px=float(row["silhouette_edge_p95_px"]),
            passed=bool(row["passed"]),
            schema_version=str(row.get("schema_version") or "RealSaS.GeometrySubstrateViewIR.v2"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("views") or ()
    )
    value = GeometrySubstrateQualificationIR(
        zero_surface_binding_hash=str(payload["zero_surface_binding_hash"]),
        observation_set_binding_hash=str(payload["observation_set_binding_hash"]),
        camera_set_binding_hash=str(payload["camera_set_binding_hash"]),
        normalization_binding_hash=str(payload["normalization_binding_hash"]),
        policy=dict(payload.get("policy") or {}),
        views=rows,
        qualification_report=dict(payload.get("qualification_report") or {}),
        substrate_hash=str(payload["substrate_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.GeometrySubstrateQualificationIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_geometry_substrate(value)
    return value
