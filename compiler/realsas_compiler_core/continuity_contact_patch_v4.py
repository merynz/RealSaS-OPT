from __future__ import annotations

"""Typed continuity authority for source-visible rigid/deformable contact patches.

A visible owner boundary is not automatically a mechanical seam, and a canonical
socket is not automatically raster-visible.  The qualified continuity obligation is
therefore the exact source-visible BODY<->rigid contact component nearest the
component's canonical parent socket.  Only BODY-underlay pixels in a small rest
contact neighborhood are sampled.  During motion that rigid-local patch is carried
with the exact foreground affine and is required to remain covered by either the
qualified BODY underlay or the exact source-alpha foreground.  BODY is evaluated as
a local set of qualified triangles, so sliding within the contact neighborhood is
allowed; fixed endpoint pairing is explicitly forbidden.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

CONTACT_PATCH_SEMANTICS = "SOURCE_OWNER_CONTACT_COMPONENT_NEAREST_CANONICAL_SOCKET_MOVING_BODY_UNDERLAY_PATCH_V1"
CONTACT_AUTHORITY = "EXACT_SOURCE_OWNER_4_NEIGHBOR_CONTACT_COMPONENT"
UNDERLAY_AUTHORITY = "QUALIFIED_DENSE_BODY_LOCAL_TRIANGLE_SET"
FOREGROUND_ALPHA_AUTHORITY = "EXACT_SOURCE_OWNER_MASK_ATLAS_ALPHA_GE_8"
FOREGROUND_CARRIER_ROLE = "RIGID_TRANSFORM_COORDINATE_CARRIER_ONLY"


@dataclass(frozen=True)
class ContactPatchPolicyV4:
    foreground_alpha_threshold: int = 8
    patch_dilation_px: int = 2
    raster_neighborhood_px: int = 1
    local_body_face_rings: int = 2
    min_rest_body_samples: int = 8
    max_new_background_fraction: float = 0.02
    schema_version: str = "RealSaS.ContactPatchPolicy.v4"

    def validate(self) -> None:
        if self.foreground_alpha_threshold != 8:
            raise QualificationError("CONTACT_PATCH_ALPHA_THRESHOLD_DRIFT")
        if self.patch_dilation_px != 2:
            raise QualificationError("CONTACT_PATCH_DILATION_DRIFT")
        if self.raster_neighborhood_px != 1:
            raise QualificationError("CONTACT_PATCH_RASTER_NEIGHBORHOOD_DRIFT")
        if self.local_body_face_rings != 2:
            raise QualificationError("CONTACT_PATCH_BODY_FACE_RING_DRIFT")
        if self.min_rest_body_samples < 1:
            raise QualificationError("CONTACT_PATCH_MIN_SAMPLE_INVALID")
        if not (0.0 <= self.max_new_background_fraction <= 0.02):
            raise QualificationError("CONTACT_PATCH_THRESHOLD_RELAXATION_FORBIDDEN")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class QualifiedContactPatchIR:
    view_index: int
    component_id: str
    canonical_parent_joint_id: str
    source_contact_pair_count: int
    selected_contact_pair_count: int
    rest_body_sample_count: int
    local_body_face_count: int
    selected_contact_min_socket_distance_px: float
    source_owner_partition_sha256: str
    foreground_alpha_authority_sha256: str
    component_lineage_hash: str
    contact_component_sha256: str
    rest_body_sample_set_sha256: str
    local_body_face_set_sha256: str
    patch_hash: str
    schema_version: str = "RealSaS.QualifiedContactPatchIR.v4"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedContactPatchSetIR:
    view_index: int
    patches: tuple[QualifiedContactPatchIR, ...]
    policy_hash: str
    patch_set_hash: str
    schema_version: str = "RealSaS.QualifiedContactPatchSetIR.v4"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_without(value, name: str) -> str:
    payload = value.to_dict(); payload.pop(name, None)
    return content_sha256(payload)


def qualify_contact_patch(
    *,
    view_index: int,
    component_id: str,
    canonical_parent_joint_id: str,
    source_contact_pair_count: int,
    selected_contact_pair_count: int,
    rest_body_sample_count: int,
    local_body_face_count: int,
    selected_contact_min_socket_distance_px: float,
    source_owner_partition_sha256: str,
    foreground_alpha_authority_sha256: str,
    component_lineage_hash: str,
    contact_component_sha256: str,
    rest_body_sample_set_sha256: str,
    local_body_face_set_sha256: str,
    foreground_carrier_mesh_used_as_occupancy: bool = False,
    fixed_endpoint_pairing_used: bool = False,
    policy: ContactPatchPolicyV4 = ContactPatchPolicyV4(),
    metadata: Mapping[str, Any] | None = None,
) -> QualifiedContactPatchIR:
    policy.validate()
    if int(view_index) not in range(8):
        raise QualificationError("CONTACT_PATCH_INVALID_VIEW")
    if not component_id or not canonical_parent_joint_id:
        raise QualificationError("CONTACT_PATCH_TYPED_COMPONENT_AND_PARENT_REQUIRED")
    if foreground_carrier_mesh_used_as_occupancy:
        raise QualificationError("CONTACT_PATCH_CARRIER_QUAD_OCCUPANCY_FORBIDDEN")
    if fixed_endpoint_pairing_used:
        raise QualificationError("CONTACT_PATCH_FIXED_ENDPOINT_PAIRING_FORBIDDEN")
    if int(source_contact_pair_count) < int(selected_contact_pair_count) or int(selected_contact_pair_count) <= 0:
        raise QualificationError("CONTACT_PATCH_CONTACT_PAIR_COUNTS_INVALID")
    if int(rest_body_sample_count) < policy.min_rest_body_samples:
        raise QualificationError(
            f"CONTACT_PATCH_REST_BODY_SAMPLE_COUNT_TOO_LOW:{rest_body_sample_count}:{policy.min_rest_body_samples}"
        )
    if int(local_body_face_count) <= 0:
        raise QualificationError("CONTACT_PATCH_LOCAL_BODY_FACE_SET_EMPTY")
    if float(selected_contact_min_socket_distance_px) < 0.0:
        raise QualificationError("CONTACT_PATCH_SOCKET_DISTANCE_INVALID")
    for name, value in (
        ("SOURCE_OWNER_PARTITION", source_owner_partition_sha256),
        ("FOREGROUND_ALPHA_AUTHORITY", foreground_alpha_authority_sha256),
        ("COMPONENT_LINEAGE", component_lineage_hash),
        ("CONTACT_COMPONENT", contact_component_sha256),
        ("REST_BODY_SAMPLE_SET", rest_body_sample_set_sha256),
        ("LOCAL_BODY_FACE_SET", local_body_face_set_sha256),
    ):
        if not str(value or ""):
            raise QualificationError(f"CONTACT_PATCH_{name}_HASH_REQUIRED")

    value = QualifiedContactPatchIR(
        view_index=int(view_index), component_id=str(component_id),
        canonical_parent_joint_id=str(canonical_parent_joint_id),
        source_contact_pair_count=int(source_contact_pair_count),
        selected_contact_pair_count=int(selected_contact_pair_count),
        rest_body_sample_count=int(rest_body_sample_count),
        local_body_face_count=int(local_body_face_count),
        selected_contact_min_socket_distance_px=float(selected_contact_min_socket_distance_px),
        source_owner_partition_sha256=str(source_owner_partition_sha256),
        foreground_alpha_authority_sha256=str(foreground_alpha_authority_sha256),
        component_lineage_hash=str(component_lineage_hash),
        contact_component_sha256=str(contact_component_sha256),
        rest_body_sample_set_sha256=str(rest_body_sample_set_sha256),
        local_body_face_set_sha256=str(local_body_face_set_sha256),
        patch_hash="",
        metadata={
            "measurement_semantics": CONTACT_PATCH_SEMANTICS,
            "contact_authority": CONTACT_AUTHORITY,
            "underlay_authority": UNDERLAY_AUTHORITY,
            "foreground_alpha_authority": FOREGROUND_ALPHA_AUTHORITY,
            "foreground_carrier_role": FOREGROUND_CARRIER_ROLE,
            "owner_boundary_wholesale_promoted_to_mechanical_seam": False,
            "canonical_socket_promoted_to_raster_contact": False,
            "fixed_endpoint_pairing_used": False,
            "rest_background_required_zero": False,
            **dict(metadata or {}),
        },
    )
    value = replace(value, patch_hash=_hash_without(value, "patch_hash"))
    validate_contact_patch(value, policy=policy)
    return value


def validate_contact_patch(value: QualifiedContactPatchIR, *, policy: ContactPatchPolicyV4 = ContactPatchPolicyV4()) -> None:
    policy.validate()
    if value.rest_body_sample_count < policy.min_rest_body_samples or value.local_body_face_count <= 0:
        raise QualificationError("CONTACT_PATCH_QUALIFICATION_DRIFT")
    if value.metadata.get("measurement_semantics") != CONTACT_PATCH_SEMANTICS:
        raise QualificationError("CONTACT_PATCH_SEMANTICS_DRIFT")
    if value.metadata.get("contact_authority") != CONTACT_AUTHORITY:
        raise QualificationError("CONTACT_PATCH_CONTACT_AUTHORITY_DRIFT")
    if value.metadata.get("underlay_authority") != UNDERLAY_AUTHORITY:
        raise QualificationError("CONTACT_PATCH_UNDERLAY_AUTHORITY_DRIFT")
    if value.metadata.get("foreground_alpha_authority") != FOREGROUND_ALPHA_AUTHORITY:
        raise QualificationError("CONTACT_PATCH_FOREGROUND_AUTHORITY_DRIFT")
    if value.metadata.get("foreground_carrier_role") != FOREGROUND_CARRIER_ROLE:
        raise QualificationError("CONTACT_PATCH_CARRIER_ROLE_DRIFT")
    if bool(value.metadata.get("owner_boundary_wholesale_promoted_to_mechanical_seam", True)):
        raise QualificationError("CONTACT_PATCH_WHOLE_OWNER_BOUNDARY_FORBIDDEN")
    if bool(value.metadata.get("canonical_socket_promoted_to_raster_contact", True)):
        raise QualificationError("CONTACT_PATCH_SOCKET_AS_RASTER_CONTACT_FORBIDDEN")
    if bool(value.metadata.get("fixed_endpoint_pairing_used", True)):
        raise QualificationError("CONTACT_PATCH_FIXED_ENDPOINT_PAIRING_FORBIDDEN")
    if bool(value.metadata.get("rest_background_required_zero", True)):
        raise QualificationError("CONTACT_PATCH_ABSOLUTE_REST_BACKGROUND_GATE_FORBIDDEN")
    if value.patch_hash != _hash_without(value, "patch_hash"):
        raise QualificationError("CONTACT_PATCH_HASH_MISMATCH")


def build_contact_patch_set(*, view_index: int, patches, policy: ContactPatchPolicyV4 = ContactPatchPolicyV4(),
                            metadata: Mapping[str, Any] | None = None) -> QualifiedContactPatchSetIR:
    policy.validate()
    rows = tuple(sorted(tuple(patches), key=lambda row: row.component_id))
    if not rows:
        raise QualificationError("CONTACT_PATCH_SET_EMPTY")
    if any(int(row.view_index) != int(view_index) for row in rows):
        raise QualificationError("CONTACT_PATCH_SET_VIEW_DRIFT")
    if len({row.component_id for row in rows}) != len(rows):
        raise QualificationError("CONTACT_PATCH_SET_DUPLICATE_COMPONENT")
    for row in rows:
        validate_contact_patch(row, policy=policy)
    value = QualifiedContactPatchSetIR(
        view_index=int(view_index), patches=rows, policy_hash=policy.policy_hash, patch_set_hash="",
        metadata={
            "only_selected_source_contact_component_is_obligated": True,
            "body_sliding_within_local_triangle_set_allowed": True,
            "new_background_is_the_failure_observable": True,
            **dict(metadata or {}),
        },
    )
    value = replace(value, patch_set_hash=_hash_without(value, "patch_set_hash"))
    return value


def assert_contact_patch_measurement(*, patch_set: QualifiedContactPatchSetIR,
                                     newly_exposed_sample_count: int,
                                     evaluated_rest_body_sample_count: int,
                                     policy: ContactPatchPolicyV4 = ContactPatchPolicyV4()) -> float:
    policy.validate()
    if patch_set.policy_hash != policy.policy_hash:
        raise QualificationError("CONTACT_PATCH_MEASUREMENT_POLICY_DRIFT")
    exposed = int(newly_exposed_sample_count); evaluated = int(evaluated_rest_body_sample_count)
    if exposed < 0 or evaluated <= 0 or exposed > evaluated:
        raise QualificationError("CONTACT_PATCH_MEASUREMENT_COUNTS_INVALID")
    fraction = float(exposed) / float(evaluated)
    if fraction > policy.max_new_background_fraction:
        raise QualificationError(
            f"CONTACT_PATCH_DYNAMIC_NEW_BACKGROUND:{fraction}:{policy.max_new_background_fraction}"
        )
    return fraction


__all__ = [
    "CONTACT_PATCH_SEMANTICS", "CONTACT_AUTHORITY", "UNDERLAY_AUTHORITY",
    "FOREGROUND_ALPHA_AUTHORITY", "FOREGROUND_CARRIER_ROLE", "ContactPatchPolicyV4",
    "QualifiedContactPatchIR", "QualifiedContactPatchSetIR", "qualify_contact_patch",
    "validate_contact_patch", "build_contact_patch_set", "assert_contact_patch_measurement",
]
