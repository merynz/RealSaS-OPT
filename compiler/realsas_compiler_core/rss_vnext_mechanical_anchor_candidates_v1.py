from __future__ import annotations

"""Zero-discovery AXIS mechanical anchor/socket -> Compiler candidate readout.

AXIS V6 owns predicted joint/control/anchor *programs*. This module extracts
their exact causal event addresses into typed proposals, while retaining the
source/mesh/checkpoint binding. It does NOT pick holder/held face pairs, fit
sockets, convert proximity to contact, or authorize attachment. A separate
explicit component/face relationship plus independent pose/transport evidence
is required before the existing rigid attachment gate can run.

The numeric event kinds/control-kind order mirror AXIS V6's published ABI.
Drift is covered by an explicit model-vocabulary regression test; the
Compiler never imports an nn.Module to inspect a saved proposal.
"""

from dataclasses import dataclass

from .hashing import content_sha256
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    validate_inference_evidence_v1,
)
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.MechanicalAnchorSocketReadout.v1"
_AX_JOINT = 2
_AX_CONTROL = 6
_AX_ANCHOR = 15
_SOCKET_CONTROL_TYPE_ID = 7
_CONTROL_KIND_COUNT = 8
_MAX_COORD_BIN = 255


@dataclass(frozen=True)
class RSSvNextSocketControlProposalV1:
    event_index: int
    control_index: int
    control_parent_index: int | None
    quantized_xyz: tuple[int, int, int]
    has_explicit_anchor: bool
    # A socket control alone does not identify a HELD object or touching face.
    holder_face_index: None = None
    held_face_index: None = None
    source_attachment_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextMechanicalAnchorProposalV1:
    event_index: int
    anchor_index: int
    parent_kind: str  # JOINT or CONTROL
    parent_index: int
    quantized_xyz: tuple[int, int, int]
    socket_control_index: int | None
    holder_face_index: None = None
    held_face_index: None = None
    source_contact_qualified: bool = False
    source_relative_transport_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextMechanicalAnchorReadoutV1:
    inference_evidence_hash: str
    unified_checkpoint_sha256: str
    candidate_mesh_sha256: str
    socket_controls: tuple[RSSvNextSocketControlProposalV1, ...]
    anchors: tuple[RSSvNextMechanicalAnchorProposalV1, ...]
    # Other non-socket controls are intentionally not reclassified.
    total_joint_count: int
    total_control_count: int
    schema: str = SCHEMA
    authority: str = "UNQUALIFIED_AX_V6_MECHANICAL_ANCHOR_PROPOSAL"
    holder_held_attachment_qualified: bool = False
    source_contact_qualified: bool = False
    executable_socket_authority: bool = False


def _fail(reason: str) -> None:
    raise QualificationError("RSS_VNEXT_MECHANICAL_SOCKET_" + reason)


def _coordinate(event: dict, event_index: int) -> tuple[int, int, int]:
    values = event["coord_indices"]
    if (len(values) != 3 or any(
        type(x) is not int or not 0 <= x <= _MAX_COORD_BIN for x in values
    )):
        _fail("QUANTIZED_ANCHOR_COORD_DOMAIN")
    return tuple(values)


def read_rss_vnext_mechanical_anchors_v1(
    evidence: RSSvNextInferenceEvidenceIRV1,
) -> RSSvNextMechanicalAnchorReadoutV1:
    """Preserve AXIS predicted socket and anchor events without new discovery.

    Accepted evidence has already passed the exact RSS export/hash court;
    this readout additionally validates AXIS pointer domains and only calls
    an ANCHOR a socket-related anchor when it targets a SOCKET_CONTROL.
    The result is NOT suitable as a completed AttachmentSurfaceBindingV1.
    """
    validate_inference_evidence_v1(evidence)
    if evidence.mesh is None or evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE":
        _fail("INDEXED_MESH_REQUIRED")
    evs = evidence.control_events
    joint_count = 0
    controls: list[tuple[int, int, tuple[int, int, int], int | None]] = []
    anchors: list[RSSvNextMechanicalAnchorProposalV1] = []
    linked_sockets: set[int] = set()
    found_anchor_section = False
    for i, ev in enumerate(evs):
        kind = ev["kind"]
        if kind == _AX_JOINT:
            if controls or anchors:
                _fail("JOINT_SEQUENCE_ORDER_DRIFT")
            joint_count += 1
        elif kind == _AX_CONTROL:
            if anchors or found_anchor_section:
                _fail("CONTROL_AFTER_ANCHOR")
            aux = ev["aux_type_id"]
            if type(aux) is not int or not 0 <= aux < _CONTROL_KIND_COUNT:
                _fail("CONTROL_TYPE_DOMAIN")
            parent = ev["pointer_a"]
            if type(parent) is not int or parent < -1 or parent >= len(controls):
                _fail("CONTROL_PARENT_POINTER_DOMAIN")
            controls.append((i, aux, _coordinate(ev, i),
                             parent if parent >= 0 else None))
        elif kind == _AX_ANCHOR:
            found_anchor_section = True
            parent_kind = ev["anchor_parent_kind"]
            if parent_kind == 0:
                parent = ev["pointer_a"]
                if (type(parent) is not int or not 0 <= parent < joint_count
                        or ev["pointer_b"] != -1):
                    _fail("ANCHOR_JOINT_PARENT_DOMAIN")
                role = "JOINT"
                linked = None
            elif parent_kind == 1:
                parent = ev["pointer_b"]
                if (type(parent) is not int or not 0 <= parent < len(controls)
                        or ev["pointer_a"] != -1):
                    _fail("ANCHOR_CONTROL_PARENT_DOMAIN")
                role = "CONTROL"
                linked = (parent if controls[parent][1] == _SOCKET_CONTROL_TYPE_ID
                          else None)
                if linked is not None:
                    linked_sockets.add(linked)
            else:
                _fail("ANCHOR_PARENT_KIND_DOMAIN")
            anchors.append(RSSvNextMechanicalAnchorProposalV1(
                event_index=i, anchor_index=len(anchors),
                parent_kind=role,parent_index=parent,
                quantized_xyz=_coordinate(ev, i),
                socket_control_index=linked,
            ))
    if not joint_count and (controls or anchors):
        _fail("ANCHOR_WITHOUT_JOINT_FOREST")
    sockets = tuple(
        RSSvNextSocketControlProposalV1(
            event_index=index,control_index=ordinal,
            control_parent_index=parent,quantized_xyz=coords,
            has_explicit_anchor=ordinal in linked_sockets,
        )
        for ordinal,(index,type_id,coords,parent) in enumerate(controls)
        if type_id == _SOCKET_CONTROL_TYPE_ID
    )
    return RSSvNextMechanicalAnchorReadoutV1(
        inference_evidence_hash=evidence.evidence_hash,
        unified_checkpoint_sha256=evidence.checkpoint_sha256,
        candidate_mesh_sha256=content_sha256(evidence.mesh),
        socket_controls=sockets, anchors=tuple(anchors),
        total_joint_count=joint_count,total_control_count=len(controls),
    )


__all__=[
    "SCHEMA",
    "RSSvNextSocketControlProposalV1",
    "RSSvNextMechanicalAnchorProposalV1",
    "RSSvNextMechanicalAnchorReadoutV1",
    "read_rss_vnext_mechanical_anchors_v1",
]
