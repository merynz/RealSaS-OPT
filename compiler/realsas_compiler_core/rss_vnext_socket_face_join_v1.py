from __future__ import annotations

"""Zero-discovery AXIS socket × TESSA face × directed relation join.

This is an evidence-index and candidate-compatibility court, NOT an attachment
detector. AXIS V6 anchors store PARENT-LOCAL offsets, not world-space vertex
coordinates; without a qualified parent rest frame it is mathematically wrong
to snap these offsets to the nearest source triangle.

The caller must explicitly propose an anchor index and an ORDERED held→holder
face pair. We never enumerate face pairs, select a score threshold, infer what
is held, name a region, or create an AttachmentSurfaceBindingV1. All model
outputs and exact topology are replayed against their sealed inference parent.
A separate, source-qualified relative transport and parent-frame court must
authorize the eventual executable 3D surface binding.
"""

from dataclasses import asdict, dataclass

from .hashing import content_sha256
from .rss_vnext_expert_fanout_v1 import RSSvNextExpertEvidenceFanoutV1
from .rss_vnext_face_relation_schedule_v1 import (
    compile_rss_vnext_sparse_face_relations_v1,
)
from .rss_vnext_mechanical_anchor_candidates_v1 import (
    RSSvNextMechanicalAnchorReadoutV1,
    read_rss_vnext_mechanical_anchors_v1,
)
from .rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
    compile_rss_vnext_surface_candidates_v1,
)
from .types import QualificationError


SCHEMA = "RealSaS.RSSvNext.SocketDirectedFaceEvidenceJoin.v1"
_AUTHORITY = "UNQUALIFIED_THREE_EXPERT_ATTACHMENT_HYPOTHESIS_ONLY"
_ABSTAIN_SOCKET = "ABSTAIN_ANCHOR_NOT_SOCKET_CONTROL"
_ABSTAIN_REGION = "ABSTAIN_UNKNOWN_DIRECTED_REGION_RELATION"
_ABSTAIN_OWNER = "ABSTAIN_DISTINCT_MECHANICAL_OWNERSHIP_NOT_ESTABLISHED"
_CANDIDATE = "UNQUALIFIED_ADDRESSED_CROSS_EXPERT_CANDIDATE"


@dataclass(frozen=True)
class RSSvNextExplicitSocketFaceRequestV1:
    """Explicit producer/scheduler selection, never discovered by Compiler."""
    anchor_index: int
    held_face_index: int
    holder_face_index: int


@dataclass(frozen=True)
class RSSvNextSocketFaceCandidateV1:
    anchor_index: int
    anchor_event_index: int
    socket_control_index: int | None
    socket_control_event_index: int | None
    # AXIS codec's ANCHOR is PARENT-LOCAL. Never treat as global XYZ.
    anchor_local_coord_bins: tuple[int, int, int]
    anchor_parent_kind: str
    anchor_parent_index: int
    held_face_index: int
    holder_face_index: int
    held_component_index: int
    holder_component_index: int
    known_cross_slot_support: float
    held_unknown_mass: float
    holder_unknown_mass: float
    same_drawable_affinity: float | None
    cut_affinity: float | None
    directed_attachment_affinity: float | None
    state: str
    missing_executable_witnesses: tuple[str, ...]
    anchor_coordinate_space: str = "PARENT_LOCAL_AX_V6_NOT_WORLD_SPACE"
    authority: str = _AUTHORITY
    source_contact_qualified: bool = False
    parent_frame_qualified: bool = False
    holder_held_surface_binding_qualified: bool = False
    executable_attachment_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextSocketFaceJoinV1:
    inference_evidence_hash: str
    unified_checkpoint_sha256: str
    neural_carrier_sha256: str
    unified_model_config_sha256: str
    candidate_mesh_sha256: str
    surface_evidence_sha256: str
    mechanical_readout_sha256: str
    directed_relation_schedule_sha256: str
    candidates: tuple[RSSvNextSocketFaceCandidateV1, ...]
    schema: str = SCHEMA
    authority: str = _AUTHORITY
    attachment_authority: bool = False
    source_contact_authority: bool = False
    canonical_mesh_authority: bool = False
    executable_rig_authority: bool = False


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_SOCKET_FACE_JOIN_" + code)


def compile_rss_vnext_socket_face_join_v1(
    *,
    fanout: RSSvNextExpertEvidenceFanoutV1,
    surface: RSSvNextComputedCandidateSurfaceV1,
    mechanics: RSSvNextMechanicalAnchorReadoutV1,
    requests: tuple[RSSvNextExplicitSocketFaceRequestV1, ...],
) -> RSSvNextSocketFaceJoinV1:
    """Preserve exact three-expert provenance for explicitly addressed pairs.

    Separate producer/selector must propose held face, holder face and anchor
    pairing. No threshold on a learned relation can make source contact true.
    The returned typed proposal deliberately omits barycentric anchors: their
    world-space frame and relative transport have not been qualified.
    """
    if (
        type(fanout) is not RSSvNextExpertEvidenceFanoutV1
        or type(surface) is not RSSvNextComputedCandidateSurfaceV1
        or type(mechanics) is not RSSvNextMechanicalAnchorReadoutV1
        or type(requests) is not tuple
    ):
        _fail("TYPED_PARENTS_REQUIRED")
    if (
        fanout.evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE"
        or fanout.evidence.mesh is None
        or surface.state != "UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE"
    ):
        _fail("MESH_PROPOSAL_REQUIRED")
    if (
        surface != compile_rss_vnext_surface_candidates_v1(fanout)
        or mechanics != read_rss_vnext_mechanical_anchors_v1(fanout.evidence)
    ):
        _fail("SEALED_EXPERT_READOUT_DRIFT")
    if (
        not requests
        or len(requests) > fanout.model_config.max_pair_queries
        or len(requests) > 4096
        or not surface.faces
    ):
        _fail("EXPLICIT_RESOURCE_BUDGET_OR_EMPTY_REQUEST")

    nfaces = len(surface.faces)
    seen = set()
    requested_pairs = []
    for r in requests:
        if (
            type(r) is not RSSvNextExplicitSocketFaceRequestV1
            or any(type(x) is not int for x in (
                r.anchor_index, r.held_face_index, r.holder_face_index
            ))
            or not 0 <= r.anchor_index < len(mechanics.anchors)
            or not 0 <= r.held_face_index < nfaces
            or not 0 <= r.holder_face_index < nfaces
            or r.held_face_index == r.holder_face_index
        ):
            _fail("DIRECTED_PAIR_OR_ANCHOR_ADDRESS_INVALID")
        key=(r.anchor_index,r.held_face_index,r.holder_face_index)
        if key in seen:
            _fail("DUPLICATE_ANCHOR_FACE_PAIR")
        seen.add(key)
        pair=(r.held_face_index,r.holder_face_index)
        if pair not in requested_pairs:
            requested_pairs.append(pair)

    # Source carrier + anonymous region memberships are re-verified by the
    # sparse relation compiler; no caller-supplied face affinity is trusted.
    relation_schedule = compile_rss_vnext_sparse_face_relations_v1(
        fanout,surface,requested_face_pairs=tuple(requested_pairs),
    )
    relations = {
        (row.from_face_index,row.to_face_index):row
        for row in relation_schedule.face_relations
    }
    socket_by_index = {
        row.control_index:row for row in mechanics.socket_controls
    }
    candidates = []
    for r in requests:
        anchor = mechanics.anchors[r.anchor_index]
        socket = (
            socket_by_index.get(anchor.socket_control_index)
            if anchor.socket_control_index is not None else None
        )
        rel = relations[(r.held_face_index,r.holder_face_index)]
        held = surface.faces[r.held_face_index]
        holder = surface.faces[r.holder_face_index]

        # This is NOT a general claim that connected geometry cannot be
        # independently edited. This particular bridge has no finer mechanical
        # ownership map than TESSA's predicted component addresses. It must
        # therefore abstain on same-component candidates.
        if socket is None or not socket.has_explicit_anchor:
            state = _ABSTAIN_SOCKET
        elif rel.state != "UNQUALIFIED_SPARSE_LEARNED_PAIR":
            state = _ABSTAIN_REGION
        elif held.candidate_component_index == holder.candidate_component_index:
            state = _ABSTAIN_OWNER
        else:
            state = _CANDIDATE

        missing = [
            "QUALIFIED_PARENT_REST_FRAME",
            "EXPLICIT_HELD_HOLDER_SURFACE_OWNERSHIP",
            "SOURCE_CONTACT_AND_HELDOUT_RELATIVE_TRANSPORT",
            "POSE_DEFORMATION_AND_ATTACH_DETACH_CONSEQUENCE",
            "QUALIFIED_CANONICAL_MESH_RIG_SKIN",
        ]
        if state == _ABSTAIN_SOCKET:
            missing.append("SOCKET_LINKED_AX_ANCHOR")
        elif state == _ABSTAIN_REGION:
            missing.append("KNOWN_DIRECTIONAL_REGION_RELATION")
        elif state == _ABSTAIN_OWNER:
            missing.append("DISTINCT_MECHANICAL_OWNER_EVIDENCE")

        candidates.append(RSSvNextSocketFaceCandidateV1(
            anchor_index=anchor.anchor_index,
            anchor_event_index=anchor.event_index,
            socket_control_index=anchor.socket_control_index,
            socket_control_event_index=(
                None if socket is None else socket.event_index
            ),
            anchor_local_coord_bins=anchor.quantized_xyz,
            anchor_parent_kind=anchor.parent_kind,
            anchor_parent_index=anchor.parent_index,
            held_face_index=r.held_face_index,
            holder_face_index=r.holder_face_index,
            held_component_index=held.candidate_component_index,
            holder_component_index=holder.candidate_component_index,
            known_cross_slot_support=rel.known_cross_slot_support,
            held_unknown_mass=rel.from_unknown_probability,
            holder_unknown_mass=rel.to_unknown_probability,
            same_drawable_affinity=rel.same_drawable_affinity,
            cut_affinity=rel.cut_affinity,
            directed_attachment_affinity=rel.directed_attachment_affinity,
            state=state,
            missing_executable_witnesses=tuple(missing),
        ))

    return RSSvNextSocketFaceJoinV1(
        inference_evidence_hash=fanout.evidence.evidence_hash,
        unified_checkpoint_sha256=fanout.evidence.checkpoint_sha256,
        neural_carrier_sha256=fanout.neural_carrier_sha256,
        unified_model_config_sha256=fanout.model_config_sha256,
        candidate_mesh_sha256=surface.candidate_mesh_sha256,
        surface_evidence_sha256=content_sha256(asdict(surface)),
        mechanical_readout_sha256=content_sha256(asdict(mechanics)),
        directed_relation_schedule_sha256=content_sha256(asdict(relation_schedule)),
        candidates=tuple(candidates),
    )


__all__=[
    "SCHEMA",
    "RSSvNextExplicitSocketFaceRequestV1",
    "RSSvNextSocketFaceCandidateV1",
    "RSSvNextSocketFaceJoinV1",
    "compile_rss_vnext_socket_face_join_v1",
]
