from __future__ import annotations

"""Typed continuity authority for rigid attachment sockets.

Owner-raster adjacency is a visibility/occlusion relation, not a mechanical seam.
For a rigid attachment the only mechanically qualified local contact is its canonical
parent socket.  This module therefore qualifies a rest-visible bridge from the exact
canonical parent pivot to the nearest exact foreground-alpha support, and measures
whether (a) BODY underlay still covers the moving socket and (b) the rest-qualified
socket bridge remains free of background during motion.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

SOCKET_BRIDGE_SEMANTICS = "CANONICAL_PARENT_SOCKET_TO_NEAREST_EXACT_FOREGROUND_ALPHA_V1"
UNDERLAY_SOCKET_AUTHORITY = "EXACT_BODY_MESH_OCCUPANCY_AT_MOVING_CANONICAL_PARENT_SOCKET"
FOREGROUND_ALPHA_AUTHORITY = "EXACT_SOURCE_OWNER_MASK_ATLAS_ALPHA_GE_8"
FOREGROUND_CARRIER_ROLE = "RIGID_TRANSFORM_COORDINATE_CARRIER_ONLY"


@dataclass(frozen=True)
class SocketBridgePolicyV3:
    foreground_alpha_threshold: int = 8
    raster_neighborhood_px: int = 1
    max_new_background_fraction: float = 0.02
    min_qualified_bridges_per_view: int = 1
    schema_version: str = "RealSaS.SocketBridgePolicy.v3"

    def validate(self) -> None:
        if self.foreground_alpha_threshold != 8:
            raise QualificationError("SOCKET_BRIDGE_ALPHA_THRESHOLD_DRIFT")
        if self.raster_neighborhood_px != 1:
            raise QualificationError("SOCKET_BRIDGE_RASTER_NEIGHBORHOOD_DRIFT")
        if not (0.0 <= self.max_new_background_fraction <= 0.02):
            raise QualificationError("SOCKET_BRIDGE_THRESHOLD_RELAXATION_FORBIDDEN")
        if self.min_qualified_bridges_per_view < 1:
            raise QualificationError("SOCKET_BRIDGE_REQUIRES_QUALIFIED_BRIDGE")

    @property
    def policy_hash(self) -> str:
        self.validate()
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class QualifiedSocketBridgeIR:
    view_index: int
    component_id: str
    canonical_parent_joint_id: str
    pivot_rest_xy: tuple[float, float]
    nearest_foreground_alpha_rest_xy: tuple[float, float]
    rest_bridge_pixel_count: int
    rest_bridge_background_pixel_count: int
    rest_body_socket_covered: bool
    foreground_alpha_authority_sha256: str
    source_component_lineage_hash: str
    bridge_hash: str
    schema_version: str = "RealSaS.QualifiedSocketBridgeIR.v3"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedSocketBridgeSetIR:
    view_index: int
    bridges: tuple[QualifiedSocketBridgeIR, ...]
    policy_hash: str
    bridge_set_hash: str
    schema_version: str = "RealSaS.QualifiedSocketBridgeSetIR.v3"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_without(value, name: str) -> str:
    payload = value.to_dict(); payload.pop(name, None)
    return content_sha256(payload)


def qualify_socket_bridge(*, view_index: int, component_id: str, canonical_parent_joint_id: str,
                          pivot_rest_xy, nearest_foreground_alpha_rest_xy,
                          rest_bridge_pixel_count: int, rest_bridge_background_pixel_count: int,
                          rest_body_socket_covered: bool, foreground_alpha_authority_sha256: str,
                          source_component_lineage_hash: str,
                          foreground_carrier_mesh_used_as_occupancy: bool = False,
                          metadata: Mapping[str, Any] | None = None) -> QualifiedSocketBridgeIR:
    if int(view_index) not in range(8):
        raise QualificationError("SOCKET_BRIDGE_INVALID_VIEW")
    if not component_id or not canonical_parent_joint_id:
        raise QualificationError("SOCKET_BRIDGE_REQUIRES_TYPED_COMPONENT_AND_PARENT")
    if foreground_carrier_mesh_used_as_occupancy:
        raise QualificationError("SOCKET_BRIDGE_CARRIER_QUAD_OCCUPANCY_FORBIDDEN")
    if int(rest_bridge_pixel_count) <= 0:
        raise QualificationError("SOCKET_BRIDGE_EMPTY_REST_BRIDGE")
    if int(rest_bridge_background_pixel_count) != 0:
        raise QualificationError("SOCKET_BRIDGE_REST_NOT_CLOSED")
    if not bool(rest_body_socket_covered):
        raise QualificationError("SOCKET_BRIDGE_REST_BODY_SOCKET_NOT_COVERED")
    if not foreground_alpha_authority_sha256 or not source_component_lineage_hash:
        raise QualificationError("SOCKET_BRIDGE_AUTHORITY_HASH_REQUIRED")
    p = tuple(float(x) for x in pivot_rest_xy)
    q = tuple(float(x) for x in nearest_foreground_alpha_rest_xy)
    if len(p) != 2 or len(q) != 2:
        raise QualificationError("SOCKET_BRIDGE_POINT_SHAPE_INVALID")
    value = QualifiedSocketBridgeIR(
        view_index=int(view_index), component_id=str(component_id),
        canonical_parent_joint_id=str(canonical_parent_joint_id),
        pivot_rest_xy=p, nearest_foreground_alpha_rest_xy=q,
        rest_bridge_pixel_count=int(rest_bridge_pixel_count),
        rest_bridge_background_pixel_count=0, rest_body_socket_covered=True,
        foreground_alpha_authority_sha256=str(foreground_alpha_authority_sha256),
        source_component_lineage_hash=str(source_component_lineage_hash), bridge_hash="",
        metadata={
            "measurement_semantics": SOCKET_BRIDGE_SEMANTICS,
            "underlay_socket_authority": UNDERLAY_SOCKET_AUTHORITY,
            "foreground_alpha_authority": FOREGROUND_ALPHA_AUTHORITY,
            "foreground_carrier_role": FOREGROUND_CARRIER_ROLE,
            "owner_raster_adjacency_used_as_mechanical_seam": False,
            "source_side_label_used": False,
            **dict(metadata or {}),
        },
    )
    value = replace(value, bridge_hash=_hash_without(value, "bridge_hash"))
    validate_socket_bridge(value)
    return value


def validate_socket_bridge(value: QualifiedSocketBridgeIR) -> None:
    if value.rest_bridge_pixel_count <= 0 or value.rest_bridge_background_pixel_count != 0:
        raise QualificationError("SOCKET_BRIDGE_REST_QUALIFICATION_DRIFT")
    if not value.rest_body_socket_covered:
        raise QualificationError("SOCKET_BRIDGE_BODY_SOCKET_COVERAGE_DRIFT")
    if value.metadata.get("measurement_semantics") != SOCKET_BRIDGE_SEMANTICS:
        raise QualificationError("SOCKET_BRIDGE_SEMANTICS_DRIFT")
    if value.metadata.get("underlay_socket_authority") != UNDERLAY_SOCKET_AUTHORITY:
        raise QualificationError("SOCKET_BRIDGE_UNDERLAY_AUTHORITY_DRIFT")
    if value.metadata.get("foreground_alpha_authority") != FOREGROUND_ALPHA_AUTHORITY:
        raise QualificationError("SOCKET_BRIDGE_FOREGROUND_AUTHORITY_DRIFT")
    if value.metadata.get("foreground_carrier_role") != FOREGROUND_CARRIER_ROLE:
        raise QualificationError("SOCKET_BRIDGE_CARRIER_ROLE_DRIFT")
    if bool(value.metadata.get("owner_raster_adjacency_used_as_mechanical_seam", True)):
        raise QualificationError("SOCKET_BRIDGE_OWNER_ADJACENCY_FORBIDDEN")
    if value.bridge_hash != _hash_without(value, "bridge_hash"):
        raise QualificationError("SOCKET_BRIDGE_HASH_MISMATCH")


def build_socket_bridge_set(*, view_index: int, bridges, policy: SocketBridgePolicyV3 = SocketBridgePolicyV3(),
                            metadata: Mapping[str, Any] | None = None) -> QualifiedSocketBridgeSetIR:
    policy.validate()
    rows = tuple(sorted(tuple(bridges), key=lambda x: x.component_id))
    if len(rows) < policy.min_qualified_bridges_per_view:
        raise QualificationError("SOCKET_BRIDGE_SET_INSUFFICIENT_QUALIFIED_BRIDGES")
    if any(int(row.view_index) != int(view_index) for row in rows):
        raise QualificationError("SOCKET_BRIDGE_SET_VIEW_DRIFT")
    if len({row.component_id for row in rows}) != len(rows):
        raise QualificationError("SOCKET_BRIDGE_SET_DUPLICATE_COMPONENT")
    for row in rows: validate_socket_bridge(row)
    value = QualifiedSocketBridgeSetIR(
        view_index=int(view_index), bridges=rows, policy_hash=policy.policy_hash,
        bridge_set_hash="", metadata={
            "qualified_rest_visible_socket_bridges_only": True,
            "non_bridge_visibility_boundaries_excluded": True,
            **dict(metadata or {}),
        },
    )
    value = replace(value, bridge_set_hash=_hash_without(value, "bridge_set_hash"))
    if value.bridge_set_hash != _hash_without(value, "bridge_set_hash"):
        raise QualificationError("SOCKET_BRIDGE_SET_HASH_MISMATCH")
    return value


def assert_socket_bridge_measurement(*, bridge_set: QualifiedSocketBridgeSetIR,
                                     newly_exposed_pixel_count: int,
                                     evaluated_bridge_pixel_count: int,
                                     policy: SocketBridgePolicyV3 = SocketBridgePolicyV3()) -> float:
    policy.validate()
    if bridge_set.policy_hash != policy.policy_hash:
        raise QualificationError("SOCKET_BRIDGE_MEASUREMENT_POLICY_DRIFT")
    exposed = int(newly_exposed_pixel_count); evaluated = int(evaluated_bridge_pixel_count)
    if exposed < 0 or evaluated <= 0 or exposed > evaluated:
        raise QualificationError("SOCKET_BRIDGE_MEASUREMENT_COUNTS_INVALID")
    fraction = float(exposed) / float(evaluated)
    if fraction > policy.max_new_background_fraction:
        raise QualificationError(
            f"SOCKET_BRIDGE_DYNAMIC_BACKGROUND_EXPOSURE:{fraction}:{policy.max_new_background_fraction}"
        )
    return fraction


__all__ = [
    "SOCKET_BRIDGE_SEMANTICS", "UNDERLAY_SOCKET_AUTHORITY", "FOREGROUND_ALPHA_AUTHORITY",
    "FOREGROUND_CARRIER_ROLE", "SocketBridgePolicyV3", "QualifiedSocketBridgeIR",
    "QualifiedSocketBridgeSetIR", "qualify_socket_bridge", "validate_socket_bridge",
    "build_socket_bridge_set", "assert_socket_bridge_measurement",
]
