from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .continuity_underlay import (
    ContinuityRasterMeasurementIR,
    QualifiedContinuityUnderlayIR,
    qualify_continuity_raster_measurement,
)
from .hashing import content_sha256
from .types import QualificationError


ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS = "DEFORMED_TYPED_ATTACHMENT_SEAM_BACKGROUND_CRACK_V2"
ENDPOINT_BINDING_AUTHORITY = "EXACT_TRIANGLE_BARYCENTRIC_FROM_REST_RASTER"
BODY_PARENT_SUPPORT_AUTHORITY = "INTERPOLATED_QUALIFIED_MESH_SKIN_PARENT_WEIGHT"
REST_SEAM_AUTHORITY = "REST_COMPOSED_RASTER_FULLY_CLOSED_ONLY"
FOREGROUND_OCCUPANCY_AUTHORITY = "EXACT_SOURCE_OWNER_MASK_ATLAS_ALPHA_GE_8"
FOREGROUND_CARRIER_ROLE = "RIGID_TRANSFORM_COORDINATE_CARRIER_ONLY"
REQUIRED_FOREGROUND_ALPHA_THRESHOLD = 8


@dataclass(frozen=True)
class AttachmentSeamPolicyV2:
    min_body_parent_weight: float = 0.50
    min_qualified_samples_per_view: int = 32
    min_rest_closed_fraction_of_parent_supported: float = 0.90
    max_new_exposed_fraction: float = 0.02
    schema_version: str = "RealSaS.AttachmentSeamPolicy.v2"

    def validate(self) -> None:
        if not (0.0 < float(self.min_body_parent_weight) <= 1.0):
            raise QualificationError("ATTACHMENT_SEAM_INVALID_PARENT_WEIGHT_POLICY")
        if int(self.min_qualified_samples_per_view) < 8:
            raise QualificationError("ATTACHMENT_SEAM_TOO_FEW_MINIMUM_SAMPLES")
        if not (0.0 < float(self.min_rest_closed_fraction_of_parent_supported) <= 1.0):
            raise QualificationError("ATTACHMENT_SEAM_INVALID_REST_CLOSED_POLICY")
        if not (0.0 <= float(self.max_new_exposed_fraction) < 1.0):
            raise QualificationError("ATTACHMENT_SEAM_INVALID_DYNAMIC_EXPOSURE_POLICY")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class QualifiedAttachmentSeamSampleSetIR:
    view_index: int
    component_assembly_hash: str
    underlay_qualification_hash: str
    mechanical_candidate_count: int
    parent_supported_candidate_count: int
    rest_closed_sample_count: int
    parent_support_evidence_sha256: str
    rest_closure_evidence_sha256: str
    sample_set_sha256: str
    policy_hash: str
    qualification_hash: str
    schema_version: str = "RealSaS.QualifiedAttachmentSeamSampleSetIR.v2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attachment_seam_sample_set_hash(value: QualifiedAttachmentSeamSampleSetIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def qualify_attachment_seam_sample_set(
    *,
    view_index: int,
    underlay: QualifiedContinuityUnderlayIR,
    component_assembly_hash: str,
    mechanical_candidate_count: int,
    parent_supported_candidate_count: int,
    rest_closed_sample_count: int,
    parent_support_evidence_sha256: str,
    rest_closure_evidence_sha256: str,
    sample_set_sha256: str,
    policy: AttachmentSeamPolicyV2 = AttachmentSeamPolicyV2(),
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedAttachmentSeamSampleSetIR:
    policy.validate()
    view = int(view_index)
    if view != int(underlay.view_index):
        raise QualificationError("ATTACHMENT_SEAM_VIEW_UNDERLAY_MISMATCH")
    assembly_hash = str(component_assembly_hash or "")
    if not assembly_hash or assembly_hash != str(underlay.component_assembly_hash):
        raise QualificationError("ATTACHMENT_SEAM_COMPONENT_ASSEMBLY_DRIFT")

    mechanical = int(mechanical_candidate_count)
    parent_supported = int(parent_supported_candidate_count)
    rest_closed = int(rest_closed_sample_count)
    if mechanical <= 0 or parent_supported < 0 or rest_closed < 0:
        raise QualificationError("ATTACHMENT_SEAM_INVALID_SAMPLE_COUNTS")
    if parent_supported > mechanical or rest_closed > parent_supported:
        raise QualificationError("ATTACHMENT_SEAM_SAMPLE_COUNT_ORDER_INVALID")
    if parent_supported < int(policy.min_qualified_samples_per_view):
        raise QualificationError(
            f"ATTACHMENT_SEAM_PARENT_SUPPORTED_SAMPLE_COUNT_TOO_LOW:{parent_supported}:"
            f"{policy.min_qualified_samples_per_view}"
        )
    if rest_closed < int(policy.min_qualified_samples_per_view):
        raise QualificationError(
            f"ATTACHMENT_SEAM_REST_CLOSED_SAMPLE_COUNT_TOO_LOW:{rest_closed}:"
            f"{policy.min_qualified_samples_per_view}"
        )
    closed_fraction = rest_closed / max(parent_supported, 1)
    if closed_fraction + 1e-12 < float(policy.min_rest_closed_fraction_of_parent_supported):
        raise QualificationError(
            f"ATTACHMENT_SEAM_REST_CLOSED_FRACTION_TOO_LOW:{closed_fraction}:"
            f"{policy.min_rest_closed_fraction_of_parent_supported}"
        )

    for name, value in (
        ("PARENT_SUPPORT_EVIDENCE_SHA256", parent_support_evidence_sha256),
        ("REST_CLOSURE_EVIDENCE_SHA256", rest_closure_evidence_sha256),
        ("SAMPLE_SET_SHA256", sample_set_sha256),
    ):
        if not str(value or ""):
            raise QualificationError(f"ATTACHMENT_SEAM_{name}_REQUIRED")

    supplied = dict(metadata or {})
    reserved = {
        "measurement_semantics",
        "body_parent_support_authority",
        "rest_seam_authority",
        "min_body_parent_weight",
        "min_rest_closed_fraction_of_parent_supported",
        "mechanical_candidate_count",
        "parent_supported_candidate_count",
        "rest_closed_sample_count",
        "rest_closed_fraction_of_parent_supported",
    }
    if reserved.intersection(supplied):
        raise QualificationError("ATTACHMENT_SEAM_RESERVED_METADATA_OVERRIDE")

    value = QualifiedAttachmentSeamSampleSetIR(
        view_index=view,
        component_assembly_hash=assembly_hash,
        underlay_qualification_hash=str(underlay.qualification_hash),
        mechanical_candidate_count=mechanical,
        parent_supported_candidate_count=parent_supported,
        rest_closed_sample_count=rest_closed,
        parent_support_evidence_sha256=str(parent_support_evidence_sha256),
        rest_closure_evidence_sha256=str(rest_closure_evidence_sha256),
        sample_set_sha256=str(sample_set_sha256),
        policy_hash=policy.policy_hash,
        qualification_hash="",
        metadata={
            "measurement_semantics": ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS,
            "body_parent_support_authority": BODY_PARENT_SUPPORT_AUTHORITY,
            "rest_seam_authority": REST_SEAM_AUTHORITY,
            "min_body_parent_weight": float(policy.min_body_parent_weight),
            "min_rest_closed_fraction_of_parent_supported": float(policy.min_rest_closed_fraction_of_parent_supported),
            "mechanical_candidate_count": mechanical,
            "parent_supported_candidate_count": parent_supported,
            "rest_closed_sample_count": rest_closed,
            "rest_closed_fraction_of_parent_supported": closed_fraction,
            **supplied,
        },
    )
    value = replace(value, qualification_hash=attachment_seam_sample_set_hash(value))
    return value


def assert_attachment_seam_sample_set(
    value: QualifiedAttachmentSeamSampleSetIR,
    *,
    underlay: QualifiedContinuityUnderlayIR,
    policy: AttachmentSeamPolicyV2 = AttachmentSeamPolicyV2(),
) -> None:
    expected = qualify_attachment_seam_sample_set(
        view_index=value.view_index,
        underlay=underlay,
        component_assembly_hash=value.component_assembly_hash,
        mechanical_candidate_count=value.mechanical_candidate_count,
        parent_supported_candidate_count=value.parent_supported_candidate_count,
        rest_closed_sample_count=value.rest_closed_sample_count,
        parent_support_evidence_sha256=value.parent_support_evidence_sha256,
        rest_closure_evidence_sha256=value.rest_closure_evidence_sha256,
        sample_set_sha256=value.sample_set_sha256,
        policy=policy,
        metadata={
            k: v for k, v in value.metadata.items()
            if k not in {
                "measurement_semantics",
                "body_parent_support_authority",
                "rest_seam_authority",
                "min_body_parent_weight",
                "min_rest_closed_fraction_of_parent_supported",
                "mechanical_candidate_count",
                "parent_supported_candidate_count",
                "rest_closed_sample_count",
                "rest_closed_fraction_of_parent_supported",
            }
        },
    )
    if expected.qualification_hash != value.qualification_hash:
        raise QualificationError("ATTACHMENT_SEAM_SAMPLE_SET_HASH_MISMATCH")


def qualify_attachment_seam_continuity_measurement(
    *,
    sample_set: QualifiedAttachmentSeamSampleSetIR,
    underlay: QualifiedContinuityUnderlayIR,
    composed_raster_sha256: str,
    newly_exposed_pixel_count: int,
    evaluated_closed_seam_pixel_count: int,
    foreground_alpha_authority_sha256: str,
    foreground_alpha_threshold: int = REQUIRED_FOREGROUND_ALPHA_THRESHOLD,
    foreground_carrier_mesh_used_as_occupancy: bool = False,
    policy: AttachmentSeamPolicyV2 = AttachmentSeamPolicyV2(),
    metadata: Mapping[str, Any] | None = None,
) -> ContinuityRasterMeasurementIR:
    assert_attachment_seam_sample_set(sample_set, underlay=underlay, policy=policy)
    if not str(composed_raster_sha256 or ""):
        raise QualificationError("ATTACHMENT_SEAM_COMPOSED_RASTER_SHA_REQUIRED")
    if not str(foreground_alpha_authority_sha256 or ""):
        raise QualificationError("ATTACHMENT_SEAM_FOREGROUND_ALPHA_AUTHORITY_REQUIRED")
    if int(foreground_alpha_threshold) != REQUIRED_FOREGROUND_ALPHA_THRESHOLD:
        raise QualificationError("ATTACHMENT_SEAM_FOREGROUND_ALPHA_THRESHOLD_DRIFT")
    if bool(foreground_carrier_mesh_used_as_occupancy):
        raise QualificationError("ATTACHMENT_SEAM_CARRIER_MESH_OCCUPANCY_FORBIDDEN")

    supplied = dict(metadata or {})
    reserved = {
        "measurement_semantics",
        "endpoint_binding_authority",
        "body_parent_support_authority",
        "rest_seam_authority",
        "foreground_occupancy_authority",
        "foreground_carrier_role",
        "foreground_carrier_mesh_used_as_occupancy",
        "foreground_alpha_threshold",
        "foreground_alpha_authority_sha256",
        "sample_set_qualification_hash",
        "rest_baseline_exposure_is_subtracted",
    }
    if reserved.intersection(supplied):
        raise QualificationError("ATTACHMENT_SEAM_MEASUREMENT_RESERVED_METADATA_OVERRIDE")

    measurement = qualify_continuity_raster_measurement(
        view_index=int(sample_set.view_index),
        underlay=underlay,
        boundary_sample_set_sha256=str(sample_set.sample_set_sha256),
        composed_raster_sha256=str(composed_raster_sha256),
        exposed_seam_pixel_count=int(newly_exposed_pixel_count),
        evaluated_boundary_pixel_count=int(evaluated_closed_seam_pixel_count),
        max_allowed_exposed_seam_fraction=float(policy.max_new_exposed_fraction),
        metadata={
            "measurement_semantics": ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS,
            "endpoint_binding_authority": ENDPOINT_BINDING_AUTHORITY,
            "body_parent_support_authority": BODY_PARENT_SUPPORT_AUTHORITY,
            "rest_seam_authority": REST_SEAM_AUTHORITY,
            "foreground_occupancy_authority": FOREGROUND_OCCUPANCY_AUTHORITY,
            "foreground_carrier_role": FOREGROUND_CARRIER_ROLE,
            "foreground_carrier_mesh_used_as_occupancy": False,
            "foreground_alpha_threshold": REQUIRED_FOREGROUND_ALPHA_THRESHOLD,
            "foreground_alpha_authority_sha256": str(foreground_alpha_authority_sha256),
            "sample_set_qualification_hash": str(sample_set.qualification_hash),
            "rest_baseline_exposure_is_subtracted": False,
            "rest_open_pairs_are_not_seams": True,
            **supplied,
        },
    )
    return measurement


def assert_attachment_seam_measurement(
    value: ContinuityRasterMeasurementIR,
    *,
    sample_set: QualifiedAttachmentSeamSampleSetIR,
    underlay: QualifiedContinuityUnderlayIR,
    policy: AttachmentSeamPolicyV2 = AttachmentSeamPolicyV2(),
) -> None:
    assert_attachment_seam_sample_set(sample_set, underlay=underlay, policy=policy)
    if int(value.view_index) != int(sample_set.view_index):
        raise QualificationError("ATTACHMENT_SEAM_MEASUREMENT_VIEW_DRIFT")
    md = dict(value.metadata or {})
    if md.get("measurement_semantics") != ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS:
        raise QualificationError("ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS_DRIFT")
    if md.get("endpoint_binding_authority") != ENDPOINT_BINDING_AUTHORITY:
        raise QualificationError("ATTACHMENT_SEAM_ENDPOINT_BINDING_DRIFT")
    if md.get("body_parent_support_authority") != BODY_PARENT_SUPPORT_AUTHORITY:
        raise QualificationError("ATTACHMENT_SEAM_PARENT_SUPPORT_AUTHORITY_DRIFT")
    if md.get("rest_seam_authority") != REST_SEAM_AUTHORITY:
        raise QualificationError("ATTACHMENT_SEAM_REST_SEAM_AUTHORITY_DRIFT")
    if md.get("foreground_occupancy_authority") != FOREGROUND_OCCUPANCY_AUTHORITY:
        raise QualificationError("ATTACHMENT_SEAM_FOREGROUND_OCCUPANCY_AUTHORITY_DRIFT")
    if md.get("foreground_carrier_role") != FOREGROUND_CARRIER_ROLE:
        raise QualificationError("ATTACHMENT_SEAM_FOREGROUND_CARRIER_ROLE_DRIFT")
    if bool(md.get("foreground_carrier_mesh_used_as_occupancy", True)):
        raise QualificationError("ATTACHMENT_SEAM_CARRIER_MESH_OCCUPANCY_DRIFT")
    if int(md.get("foreground_alpha_threshold", -1)) != REQUIRED_FOREGROUND_ALPHA_THRESHOLD:
        raise QualificationError("ATTACHMENT_SEAM_ALPHA_THRESHOLD_DRIFT")
    if str(md.get("sample_set_qualification_hash") or "") != sample_set.qualification_hash:
        raise QualificationError("ATTACHMENT_SEAM_SAMPLE_SET_BINDING_DRIFT")
    if bool(md.get("rest_baseline_exposure_is_subtracted", True)):
        raise QualificationError("ATTACHMENT_SEAM_BASELINE_SUBTRACTION_FORBIDDEN")
    if not bool(md.get("rest_open_pairs_are_not_seams", False)):
        raise QualificationError("ATTACHMENT_SEAM_REST_OPEN_PAIR_POLICY_DRIFT")


__all__ = [
    "ATTACHMENT_SEAM_MEASUREMENT_SEMANTICS",
    "ENDPOINT_BINDING_AUTHORITY",
    "BODY_PARENT_SUPPORT_AUTHORITY",
    "REST_SEAM_AUTHORITY",
    "FOREGROUND_OCCUPANCY_AUTHORITY",
    "FOREGROUND_CARRIER_ROLE",
    "REQUIRED_FOREGROUND_ALPHA_THRESHOLD",
    "AttachmentSeamPolicyV2",
    "QualifiedAttachmentSeamSampleSetIR",
    "attachment_seam_sample_set_hash",
    "qualify_attachment_seam_sample_set",
    "assert_attachment_seam_sample_set",
    "qualify_attachment_seam_continuity_measurement",
    "assert_attachment_seam_measurement",
]
