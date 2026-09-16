from __future__ import annotations

"""Generic RealSaS playback/runtime-v3 contracts.

No subject-specific policy belongs here. Mage/FIT1/FITK/unseen subjects are data
fixtures entering the same contracts.

Behavioral references:
- CharacterGen-class full-surface -> camera -> depth visibility separation.
- Spine-class bone/slot/attachment/draw-order/clipping/runtime separation.

No third-party runtime implementation code is copied here.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


PLAYBACK_RUNTIME_V3_SCHEMA = "RealSaS.PlaybackRuntimeV3Contract.v1"
REFERENCE_RASTER_SCHEMA = "RealSaS.ReferenceRasterContract.v1"


class AppearanceProvenance(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    OTHER_VIEW_SOURCE = "OTHER_VIEW_SOURCE"
    UNDER_RIGID_SOURCE = "UNDER_RIGID_SOURCE"
    UNSEEN = "UNSEEN"
    COMPLETION = "COMPLETION"


class AttachmentKind(str, Enum):
    DEFORMABLE_BODY = "DEFORMABLE_BODY"
    RIGID_COMPONENT = "RIGID_COMPONENT"
    CLIPPING = "CLIPPING"


class TopologyClass(str, Enum):
    STATIC = "STATIC"
    ATTACHMENT_DYNAMIC = "ATTACHMENT_DYNAMIC"
    CLIP_DYNAMIC = "CLIP_DYNAMIC"
    DRAW_ORDER_DYNAMIC = "DRAW_ORDER_DYNAMIC"


class DepthWritePolicy(str, Enum):
    ON = "ON"
    OFF = "OFF"
    CUTOUT_ONLY = "CUTOUT_ONLY"


@dataclass(frozen=True)
class ReferenceRasterContractV1:
    pixel_center: str = "HALF_INTEGER_CENTER"
    triangle_fill_rule: str = "TOP_LEFT"
    uv_origin: str = "TOP_LEFT"
    texture_filter: str = "BILINEAR"
    wrap_mode: str = "CLAMP_TO_EDGE"
    alpha_encoding: str = "STRAIGHT"
    color_space: str = "SRGB_RGBA8"
    blend_mode: str = "SOURCE_OVER"
    depth_compare: str = "LESS_EQUAL_WITH_STABLE_TIE_BREAK"
    depth_space: str = "POSED_CAMERA_FORWARD_DEPTH_SMALLER_IS_NEARER"
    face_culling: str = "DISABLED_UNLESS_EXPLICIT_ATTACHMENT_POLICY"
    per_frame_affine_refit_forbidden: bool = True
    schema_version: str = REFERENCE_RASTER_SCHEMA

    @property
    def contract_hash(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class RuntimeV3Vertex:
    x: float
    y: float
    z: float
    u: float
    v: float


@dataclass(frozen=True)
class RuntimeV3Mesh:
    view_id: str
    mesh_id: str
    slot_id: str
    attachment_id: str
    attachment_kind: AttachmentKind
    topology_class: TopologyClass
    vertices: tuple[RuntimeV3Vertex, ...]
    triangles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class RuntimeV3Slot:
    slot_id: str
    bone_id: str
    setup_order: int
    default_attachment_id: str | None


@dataclass(frozen=True)
class RuntimeV3ClipInterval:
    clip_attachment_id: str
    start_slot_id: str
    end_slot_id: str
    inverse: bool = False


@dataclass(frozen=True)
class RuntimeV3AppearancePatch:
    patch_id: str
    mesh_id: str
    face_indices: tuple[int, ...]
    provenance: AppearanceProvenance
    donor_view_index: int | None
    atlas_id: str | None
    completion_method: str | None = None


@dataclass(frozen=True)
class RuntimeV3VisibilityPolicy:
    body_depth_test: bool = True
    body_depth_write: DepthWritePolicy = DepthWritePolicy.ON
    rigid_depth_test: bool = True
    rigid_depth_write: DepthWritePolicy = DepthWritePolicy.CUTOUT_ONLY
    alpha_cutout_threshold: float = 0.5
    semantic_draw_order_required: bool = True
    body_self_occlusion_from_draw_order_forbidden: bool = True
    full_surface_geometry_authority_required: bool = True
    visibility_by_face_deletion_forbidden: bool = True


@dataclass(frozen=True)
class RuntimeV3FrameComposition:
    view_id: str
    draw_order_slot_ids: tuple[str, ...]
    active_attachment_by_slot: Mapping[str, str | None]
    clip_intervals: tuple[RuntimeV3ClipInterval, ...] = ()


@dataclass(frozen=True)
class RuntimeV3PlaybackContract:
    slots: tuple[RuntimeV3Slot, ...]
    meshes: tuple[RuntimeV3Mesh, ...]
    appearance_patches: tuple[RuntimeV3AppearancePatch, ...]
    visibility: RuntimeV3VisibilityPolicy = RuntimeV3VisibilityPolicy()
    raster: ReferenceRasterContractV1 = ReferenceRasterContractV1()
    allow_completion: bool = False
    schema_version: str = PLAYBACK_RUNTIME_V3_SCHEMA

    @property
    def contract_hash(self) -> str:
        return content_sha256(_jsonable_contract(self))


def _jsonable_contract(contract: RuntimeV3PlaybackContract) -> dict:
    return {
        "schema_version": contract.schema_version,
        "slots": [asdict(x) for x in contract.slots],
        "meshes": [
            {
                **asdict(m),
                "attachment_kind": m.attachment_kind.value,
                "topology_class": m.topology_class.value,
            }
            for m in contract.meshes
        ],
        "appearance_patches": [
            {**asdict(p), "provenance": p.provenance.value}
            for p in contract.appearance_patches
        ],
        "visibility": {
            **asdict(contract.visibility),
            "body_depth_write": contract.visibility.body_depth_write.value,
            "rigid_depth_write": contract.visibility.rigid_depth_write.value,
        },
        "raster": asdict(contract.raster),
        "allow_completion": bool(contract.allow_completion),
    }


def _view_index(view_id: str) -> int | None:
    if len(view_id) >= 2 and view_id[0] == "V" and view_id[1:].isdigit():
        return int(view_id[1:])
    return None


def validate_playback_runtime_v3_contract(
    contract: RuntimeV3PlaybackContract,
    *,
    required_view_ids: Iterable[str] = tuple(f"V{i}" for i in range(8)),
) -> str:
    """Fail-closed validation. Returns deterministic contract hash on success."""

    if contract.schema_version != PLAYBACK_RUNTIME_V3_SCHEMA:
        raise QualificationError("RUNTIME_V3_SCHEMA_MISMATCH")

    vis = contract.visibility
    if not vis.full_surface_geometry_authority_required:
        raise QualificationError("RUNTIME_V3_REQUIRES_FULL_SURFACE_GEOMETRY_AUTHORITY")
    if not vis.visibility_by_face_deletion_forbidden:
        raise QualificationError("RUNTIME_V3_FACE_DELETION_VISIBILITY_FORBIDDEN")
    if not vis.body_depth_test:
        raise QualificationError("RUNTIME_V3_BODY_DEPTH_TEST_REQUIRED")
    if vis.body_depth_write == DepthWritePolicy.OFF:
        raise QualificationError("RUNTIME_V3_BODY_DEPTH_WRITE_REQUIRED")
    if not vis.body_self_occlusion_from_draw_order_forbidden:
        raise QualificationError("RUNTIME_V3_BODY_SELF_OCCLUSION_MUST_USE_DEPTH")
    if not (0.0 <= float(vis.alpha_cutout_threshold) <= 1.0):
        raise QualificationError("RUNTIME_V3_ALPHA_CUTOUT_THRESHOLD_INVALID")
    if not contract.raster.per_frame_affine_refit_forbidden:
        raise QualificationError("RUNTIME_V3_PER_FRAME_AFFINE_REFIT_FORBIDDEN")

    required_views = tuple(str(x) for x in required_view_ids)
    if not required_views or len(required_views) != len(set(required_views)):
        raise QualificationError("RUNTIME_V3_REQUIRED_VIEW_IDS_INVALID")
    required_view_set = set(required_views)

    slots = tuple(contract.slots)
    slot_ids = [s.slot_id for s in slots]
    if not slots or len(slot_ids) != len(set(slot_ids)):
        raise QualificationError("RUNTIME_V3_SLOT_IDS_INVALID")
    if len({int(s.setup_order) for s in slots}) != len(slots):
        raise QualificationError("RUNTIME_V3_SLOT_SETUP_ORDER_MUST_BE_TOTAL")
    slot_set = set(slot_ids)

    meshes = tuple(contract.meshes)
    mesh_ids = [m.mesh_id for m in meshes]
    if not meshes or len(mesh_ids) != len(set(mesh_ids)):
        raise QualificationError("RUNTIME_V3_MESH_IDS_INVALID")
    if {m.view_id for m in meshes} != required_view_set:
        raise QualificationError("RUNTIME_V3_MESH_VIEW_SET_MISMATCH")

    # A logical attachment may have one mesh variant per view, but it must
    # always belong to the same slot/kind.
    attachment_semantics: dict[str, tuple[str, AttachmentKind]] = {}
    seen_view_attachment: set[tuple[str, str]] = set()
    mesh_by_id = {m.mesh_id: m for m in meshes}
    for mesh in meshes:
        if mesh.slot_id not in slot_set:
            raise QualificationError("RUNTIME_V3_MESH_REFERENCES_UNKNOWN_SLOT")
        pair = (mesh.view_id, mesh.attachment_id)
        if pair in seen_view_attachment:
            raise QualificationError("RUNTIME_V3_DUPLICATE_ATTACHMENT_VARIANT_IN_VIEW")
        seen_view_attachment.add(pair)
        semantic = (mesh.slot_id, mesh.attachment_kind)
        prior = attachment_semantics.setdefault(mesh.attachment_id, semantic)
        if prior != semantic:
            raise QualificationError("RUNTIME_V3_ATTACHMENT_SEMANTIC_DRIFT_ACROSS_VIEWS")
        if not mesh.vertices or not mesh.triangles:
            raise QualificationError("RUNTIME_V3_EMPTY_MESH_FORBIDDEN")
        n = len(mesh.vertices)
        for tri in mesh.triangles:
            if len(tri) != 3 or min(tri) < 0 or max(tri) >= n:
                raise QualificationError("RUNTIME_V3_TRIANGLE_INVALID")
        for v in mesh.vertices:
            vals = (v.x, v.y, v.z, v.u, v.v)
            if not all(math.isfinite(float(x)) for x in vals):
                raise QualificationError("RUNTIME_V3_NONFINITE_VERTEX")
            if not (0.0 <= float(v.u) <= 1.0 and 0.0 <= float(v.v) <= 1.0):
                raise QualificationError("RUNTIME_V3_UV_OUTSIDE_UNIT_DOMAIN")

    for slot in slots:
        if slot.default_attachment_id is None:
            continue
        semantic = attachment_semantics.get(slot.default_attachment_id)
        if semantic is None or semantic[0] != slot.slot_id:
            raise QualificationError("RUNTIME_V3_SLOT_DEFAULT_ATTACHMENT_INVALID")

    face_claims: dict[tuple[str, int], str] = {}
    patch_ids: set[str] = set()
    for patch in contract.appearance_patches:
        if patch.patch_id in patch_ids:
            raise QualificationError("RUNTIME_V3_DUPLICATE_APPEARANCE_PATCH_ID")
        patch_ids.add(patch.patch_id)
        mesh = mesh_by_id.get(patch.mesh_id)
        if mesh is None:
            raise QualificationError("RUNTIME_V3_APPEARANCE_REFERENCES_UNKNOWN_MESH")
        if not patch.face_indices:
            raise QualificationError("RUNTIME_V3_EMPTY_APPEARANCE_PATCH")

        target_index = _view_index(mesh.view_id)
        if patch.provenance == AppearanceProvenance.DIRECT_SOURCE:
            if patch.donor_view_index is None:
                raise QualificationError("RUNTIME_V3_DIRECT_SOURCE_REQUIRES_DONOR_VIEW")
            if target_index is not None and int(patch.donor_view_index) != target_index:
                raise QualificationError("RUNTIME_V3_DIRECT_SOURCE_DONOR_MUST_MATCH_TARGET_VIEW")
        elif patch.provenance == AppearanceProvenance.OTHER_VIEW_SOURCE:
            if patch.donor_view_index is None:
                raise QualificationError("RUNTIME_V3_SOURCE_PROVENANCE_REQUIRES_DONOR_VIEW")
            if target_index is not None and int(patch.donor_view_index) == target_index:
                raise QualificationError("RUNTIME_V3_OTHER_VIEW_DONOR_MUST_DIFFER_FROM_TARGET")
        elif patch.provenance == AppearanceProvenance.UNDER_RIGID_SOURCE:
            if patch.donor_view_index is None:
                raise QualificationError("RUNTIME_V3_SOURCE_PROVENANCE_REQUIRES_DONOR_VIEW")
        elif patch.provenance == AppearanceProvenance.UNSEEN:
            if patch.donor_view_index is not None or patch.atlas_id is not None:
                raise QualificationError("RUNTIME_V3_UNSEEN_MUST_REMAIN_UNBOUND")
        elif patch.provenance == AppearanceProvenance.COMPLETION:
            if not contract.allow_completion:
                raise QualificationError("RUNTIME_V3_COMPLETION_NOT_ALLOWED_BY_PRODUCT_POLICY")
            if not patch.completion_method:
                raise QualificationError("RUNTIME_V3_COMPLETION_REQUIRES_PROVENANCE_METHOD")

        for face_index in patch.face_indices:
            if face_index < 0 or face_index >= len(mesh.triangles):
                raise QualificationError("RUNTIME_V3_APPEARANCE_FACE_OUT_OF_RANGE")
            key = (mesh.mesh_id, int(face_index))
            if key in face_claims:
                raise QualificationError("RUNTIME_V3_APPEARANCE_FACE_MULTI_CLAIM")
            face_claims[key] = patch.patch_id

    for mesh in meshes:
        for face_index in range(len(mesh.triangles)):
            if (mesh.mesh_id, face_index) not in face_claims:
                raise QualificationError("RUNTIME_V3_APPEARANCE_FACE_COVERAGE_INCOMPLETE")

    return contract.contract_hash


def validate_frame_composition_v3(
    contract: RuntimeV3PlaybackContract,
    frame: RuntimeV3FrameComposition,
) -> None:
    if frame.view_id not in {m.view_id for m in contract.meshes}:
        raise QualificationError("RUNTIME_V3_FRAME_REFERENCES_UNKNOWN_VIEW")

    slot_ids = {s.slot_id for s in contract.slots}
    order = tuple(frame.draw_order_slot_ids)
    if len(order) != len(slot_ids) or set(order) != slot_ids:
        raise QualificationError("RUNTIME_V3_DRAW_ORDER_MUST_BE_EXACT_SLOT_PERMUTATION")
    if set(frame.active_attachment_by_slot) != slot_ids:
        raise QualificationError("RUNTIME_V3_ACTIVE_ATTACHMENT_MAP_MUST_COVER_ALL_SLOTS")

    attachment_semantics: dict[str, tuple[str, AttachmentKind]] = {}
    for mesh in contract.meshes:
        semantic = (mesh.slot_id, mesh.attachment_kind)
        prior = attachment_semantics.setdefault(mesh.attachment_id, semantic)
        if prior != semantic:
            raise QualificationError("RUNTIME_V3_ATTACHMENT_SLOT_DRIFT")

    active_clipping: dict[str, str] = {}
    for slot_id, attachment_id in frame.active_attachment_by_slot.items():
        if attachment_id is None:
            continue
        semantic = attachment_semantics.get(attachment_id)
        if semantic is None or semantic[0] != slot_id:
            raise QualificationError("RUNTIME_V3_ACTIVE_ATTACHMENT_SLOT_MISMATCH")
        if semantic[1] == AttachmentKind.CLIPPING:
            active_clipping[attachment_id] = slot_id

    pos = {slot_id: i for i, slot_id in enumerate(order)}
    interval_ids: set[str] = set()
    for interval in frame.clip_intervals:
        if interval.clip_attachment_id in interval_ids:
            raise QualificationError("RUNTIME_V3_DUPLICATE_CLIP_INTERVAL")
        interval_ids.add(interval.clip_attachment_id)

        semantic = attachment_semantics.get(interval.clip_attachment_id)
        if semantic is None or semantic[1] != AttachmentKind.CLIPPING:
            raise QualificationError("RUNTIME_V3_CLIP_INTERVAL_REQUIRES_CLIPPING_ATTACHMENT")
        clip_slot_id = semantic[0]
        if interval.start_slot_id != clip_slot_id:
            raise QualificationError("RUNTIME_V3_CLIP_INTERVAL_START_MUST_EQUAL_CLIP_SLOT")
        if frame.active_attachment_by_slot.get(clip_slot_id) != interval.clip_attachment_id:
            raise QualificationError("RUNTIME_V3_CLIP_INTERVAL_ATTACHMENT_NOT_ACTIVE_AT_START")
        if interval.start_slot_id not in pos or interval.end_slot_id not in pos:
            raise QualificationError("RUNTIME_V3_CLIP_INTERVAL_UNKNOWN_SLOT")
        # Spine-class semantics: the clipping attachment occupies the start
        # slot and clips subsequent slots through end_slot inclusive.
        if pos[interval.start_slot_id] >= pos[interval.end_slot_id]:
            raise QualificationError("RUNTIME_V3_CLIP_INTERVAL_MUST_COVER_SUBSEQUENT_SLOT")

    if set(active_clipping) != interval_ids:
        raise QualificationError("RUNTIME_V3_ACTIVE_CLIPPING_ATTACHMENT_REQUIRES_INTERVAL")
