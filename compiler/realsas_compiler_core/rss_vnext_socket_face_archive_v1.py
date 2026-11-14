from __future__ import annotations

"""RSS vNext neural socket↔face sidecar: immutable proposal, never a product.

The original RSS InferenceEvidenceIR v1 remains lossless/unmodified. This
separate, content-sealed archive binds model-selected HELD→HOLDER face pairs
to the EXACT checkpoint, TESSA mesh proposal, AXIS causal program and neural
evidence parent. It is not mechanically or source qualified.

The Compiler must re-open the original source-bound expert carrier and
recompute typed face relations; a hand-edited neural score is not authority.
"""
from dataclasses import asdict,dataclass
import json
import math
from typing import Any,Mapping

from .hashing import content_sha256
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,validate_inference_evidence_v1,
)
from .rss_vnext_mechanical_anchor_candidates_v1 import (
    read_rss_vnext_mechanical_anchors_v1,
)
from .rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
)
from .rss_vnext_expert_fanout_v1 import RSSvNextExpertEvidenceFanoutV1
from .rss_vnext_socket_face_join_v1 import (
    RSSvNextExplicitSocketFaceRequestV1,
    RSSvNextSocketFaceJoinV1,
    compile_rss_vnext_socket_face_join_v1,
)
from .types import QualificationError


SCHEMA="RealSaS.RSSvNext.NeuralSocketFaceArchive.v1"
AUTHORITY="MODEL_SELECTED_UNQUALIFIED_HELD_HOLDER_FACE_HYPOTHESES"
_PROPOSAL_FIELDS=frozenset((
    "anchor_event_index","socket_control_index",
    "held_face_index","holder_face_index","same_tessa_component",
    "known_cross_slot_support","predicted_directed_relation",
    "held_unknown_probability","holder_unknown_probability",
    "predicted_rank_log_score","state","authority",
    "source_contact_qualified","parent_rest_frame_qualified",
    "attachment_qualified",
))
_STATES=frozenset((
    "ABSTAIN_MODEL_NO_FACE_OR_PAIR",
    "UNQUALIFIED_PREDICTED_DIRECTIONAL_FACE_PAIR",
    "UNQUALIFIED_PREDICTED_PAIR_UNKNOWN_REGION",
))


def _fail(reason:str)->None:
    raise QualificationError("RSS_VNEXT_SOCKET_FACE_ARCHIVE_"+reason)


def _sha(v:object)->bool:
    return type(v) is str and len(v)==64 and all(c in "0123456789abcdef" for c in v)


@dataclass(frozen=True)
class RSSvNextNeuralSocketFaceArchiveV1:
    checkpoint_sha256:str
    inference_evidence_hash:str
    candidate_mesh_sha256:str
    axis_control_events_sha256:str
    proposals:tuple[Mapping[str,Any],...]
    schema:str=SCHEMA
    authority:str=AUTHORITY
    source_contact_qualified:bool=False
    executable_attachment_authority:bool=False
    product_revision_minted:bool=False
    archive_hash:str=""

    def to_dict(self)->dict[str,Any]:
        return asdict(self)


def seal_rss_vnext_neural_socket_face_archive_v1(
    archive:RSSvNextNeuralSocketFaceArchiveV1,
)->RSSvNextNeuralSocketFaceArchiveV1:
    from dataclasses import replace
    if type(archive) is not RSSvNextNeuralSocketFaceArchiveV1:
        _fail("ARCHIVE_TYPE_REQUIRED")
    raw=archive.to_dict()
    raw.pop("archive_hash",None)
    return replace(archive,archive_hash=content_sha256(raw))


def validate_rss_vnext_neural_socket_face_archive_v1(
    archive:RSSvNextNeuralSocketFaceArchiveV1,
    *,evidence:RSSvNextInferenceEvidenceIRV1,
)->None:
    validate_inference_evidence_v1(evidence)
    if (
        type(archive) is not RSSvNextNeuralSocketFaceArchiveV1
        or archive.schema!=SCHEMA
        or archive.authority!=AUTHORITY
        or any((archive.source_contact_qualified,
                archive.executable_attachment_authority,
                archive.product_revision_minted))
        or not all(_sha(h) for h in (
            archive.checkpoint_sha256,archive.inference_evidence_hash,
            archive.candidate_mesh_sha256,archive.axis_control_events_sha256,
            archive.archive_hash
        ))
        or evidence.mesh_status!="UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None
        or not evidence.control_events
        or archive.checkpoint_sha256!=evidence.checkpoint_sha256
        or archive.inference_evidence_hash!=evidence.evidence_hash
        or archive.candidate_mesh_sha256!=content_sha256(evidence.mesh)
        or archive.axis_control_events_sha256!=content_sha256(evidence.control_events)
    ):
        _fail("PARENT_AUTHORITY_OR_BYTES_DRIFT")
    raw=archive.to_dict()
    raw.pop("archive_hash",None)
    if content_sha256(raw)!=archive.archive_hash:
        _fail("ARCHIVE_HASH_DRIFT")
    if (
        type(archive.proposals) is not tuple
        or len(archive.proposals)>512
        or len(archive.proposals)>len(evidence.control_events)
    ):
        _fail("PROPOSAL_RESOURCE_LIMIT")

    mechanics=read_rss_vnext_mechanical_anchors_v1(evidence)
    by_event={a.event_index:a for a in mechanics.anchors}
    sockets={c.control_index:c for c in mechanics.socket_controls}
    seen=set()
    nfaces=len(evidence.mesh["faces"])
    components=evidence.mesh["face_components"]
    for row in archive.proposals:
        if (
            not isinstance(row,Mapping)
            or set(row)!=_PROPOSAL_FIELDS
            or row["authority"]!="UNQUALIFIED_NEURAL_SOCKET_FACE_CORRESPONDENCE_V1"
            or any(row[k] is not False for k in (
                "source_contact_qualified","parent_rest_frame_qualified",
                "attachment_qualified"
            ))
            or row["state"] not in _STATES
            or type(row["anchor_event_index"]) is not int
            or type(row["socket_control_index"]) is not int
        ):
            _fail("PROPOSAL_TYPED_AUTHORITY_DRIFT")
        ev=row["anchor_event_index"]
        if ev in seen:
            _fail("DUPLICATE_ANCHOR_EVENT")
        seen.add(ev)
        anchor=by_event.get(ev)
        socket=sockets.get(row["socket_control_index"])
        if (
            anchor is None or socket is None
            or anchor.socket_control_index!=socket.control_index
            or not socket.has_explicit_anchor
        ):
            _fail("AXIS_SOCKET_PARENT_BINDING_DRIFT")
        held,holder=row["held_face_index"],row["holder_face_index"]
        abstain=row["state"]=="ABSTAIN_MODEL_NO_FACE_OR_PAIR"
        if abstain:
            if (
                held is not None or holder is not None
                or row["same_tessa_component"] is not None
                or row["predicted_rank_log_score"] is not None
                or row["predicted_directed_relation"] is not None
                or row["known_cross_slot_support"]!=0.
            ):
                _fail("NO_FACE_ABSTAIN_WAS_PROMOTED")
        else:
            if (
                type(held) is not int or type(holder) is not int
                or not 0<=held<nfaces or not 0<=holder<nfaces
                or held==holder
                or type(row["same_tessa_component"]) is not bool
                or row["same_tessa_component"]!=bool(components[held]==components[holder])
                or type(row["predicted_rank_log_score"]) not in (float,int)
                or not math.isfinite(row["predicted_rank_log_score"])
            ):
                _fail("FACE_INDEX_OR_TESSA_COMPONENT_DRIFT")
            if (
                row["state"]=="UNQUALIFIED_PREDICTED_PAIR_UNKNOWN_REGION"
            ) != (row["predicted_directed_relation"] is None):
                _fail("DIRECTED_REGION_ABSTAIN_DRIFT")
        for field in (
            "known_cross_slot_support","held_unknown_probability",
            "holder_unknown_probability"
        ):
            value=row[field]
            if type(value) not in (int,float) or not math.isfinite(value):
                _fail("REGION_SUPPORT_NUMERIC_DOMAIN")
            if field!="known_cross_slot_support" and not 0<=value<=1:
                _fail("REGION_UNKNOWN_MASS_DOMAIN")
        if row["predicted_directed_relation"] is not None:
            x=row["predicted_directed_relation"]
            if type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1:
                _fail("DIRECTIONAL_AFFINITY_DOMAIN")


def encode_rss_vnext_neural_socket_face_archive_v1(
    archive:RSSvNextNeuralSocketFaceArchiveV1,
    *,evidence:RSSvNextInferenceEvidenceIRV1,
)->bytes:
    validate_rss_vnext_neural_socket_face_archive_v1(archive,evidence=evidence)
    return json.dumps(
        archive.to_dict(),sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode("utf8")


def read_rss_vnext_neural_socket_face_archive_v1(
    raw:bytes,*,evidence:RSSvNextInferenceEvidenceIRV1,
)->RSSvNextNeuralSocketFaceArchiveV1:
    if type(raw) is not bytes or len(raw)>2*1024*1024:
        _fail("ARCHIVE_BYTES_RESOURCE_LIMIT")
    try:
        value=json.loads(raw.decode("utf8"))
        if type(value) is not dict or set(value)!={k for k in RSSvNextNeuralSocketFaceArchiveV1.__dataclass_fields__}:
            _fail("ARCHIVE_FIELDS_INVALID")
        if type(value["proposals"]) is not list:
            _fail("PROPOSAL_LIST_DOMAIN")
        archive=RSSvNextNeuralSocketFaceArchiveV1(
            **{**value,"proposals":tuple(value["proposals"])},
        )
    except (ValueError,TypeError,UnicodeDecodeError,KeyError) as exc:
        _fail("ARCHIVE_JSON_DECODE_INVALID")
    validate_rss_vnext_neural_socket_face_archive_v1(archive,evidence=evidence)
    return archive


@dataclass(frozen=True)
class RSSvNextModelSelectedSocketFaceJoinV1:
    neural_archive_hash:str
    candidate_join:RSSvNextSocketFaceJoinV1|None
    abstained_anchor_event_indices:tuple[int,...]
    source_contact_qualified:bool=False
    executable_attachment_authority:bool=False


def compile_model_selected_rss_vnext_socket_face_join_v1(
    *,
    fanout:RSSvNextExpertEvidenceFanoutV1,
    surface:RSSvNextComputedCandidateSurfaceV1,
    archive:RSSvNextNeuralSocketFaceArchiveV1,
)->RSSvNextModelSelectedSocketFaceJoinV1:
    """The first non-manual, fully checkpoint-bound model→Compiler seam."""
    validate_rss_vnext_neural_socket_face_archive_v1(
        archive,evidence=fanout.evidence,
    )
    mechanics=read_rss_vnext_mechanical_anchors_v1(fanout.evidence)
    by_event={row.event_index:row for row in mechanics.anchors}
    requests=[]
    abstained=[]
    for row in archive.proposals:
        if row["state"]=="ABSTAIN_MODEL_NO_FACE_OR_PAIR":
            abstained.append(row["anchor_event_index"])
            continue
        anchor=by_event[row["anchor_event_index"]]
        requests.append(RSSvNextExplicitSocketFaceRequestV1(
            anchor_index=anchor.anchor_index,
            held_face_index=row["held_face_index"],
            holder_face_index=row["holder_face_index"],
        ))
    joined=(
        compile_rss_vnext_socket_face_join_v1(
            fanout=fanout,surface=surface,mechanics=mechanics,
            requests=tuple(requests),
        ) if requests else None
    )
    if joined is not None:
        lookup={
            (c.anchor_event_index,c.held_face_index,c.holder_face_index):c
            for c in joined.candidates
        }
        for row in archive.proposals:
            if row["held_face_index"] is None:
                continue
            key=(row["anchor_event_index"],row["held_face_index"],row["holder_face_index"])
            recomputed=lookup[key]
            for field,model_field in (
                ("known_cross_slot_support","known_cross_slot_support"),
                ("held_unknown_mass","held_unknown_probability"),
                ("holder_unknown_mass","holder_unknown_probability"),
            ):
                if abs(getattr(recomputed,field)-row[model_field])>1e-5:
                    _fail("ORIGINAL_SEMANTIC_PAIR_REPLAY_DRIFT")
            got=recomputed.directed_attachment_affinity
            predicted=row["predicted_directed_relation"]
            if (got is None)!=(predicted is None) or (
                got is not None and abs(got-predicted)>1.e-5
            ):
                _fail("ORIGINAL_DIRECTIONAL_RELATION_REPLAY_DRIFT")
    return RSSvNextModelSelectedSocketFaceJoinV1(
        neural_archive_hash=archive.archive_hash,
        candidate_join=joined,
        abstained_anchor_event_indices=tuple(abstained),
    )


__all__=[
    "SCHEMA","AUTHORITY","RSSvNextNeuralSocketFaceArchiveV1",
    "RSSvNextModelSelectedSocketFaceJoinV1",
    "seal_rss_vnext_neural_socket_face_archive_v1",
    "validate_rss_vnext_neural_socket_face_archive_v1",
    "encode_rss_vnext_neural_socket_face_archive_v1",
    "read_rss_vnext_neural_socket_face_archive_v1",
    "compile_model_selected_rss_vnext_socket_face_join_v1",
]
