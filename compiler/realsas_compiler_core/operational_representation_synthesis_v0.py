"""Research-only E+C+P -> executable representation candidates.

"Drawable" means presentation packet, not artist-authored source ownership.
These are candidates on one shared canonical mechanical carrier, not qualified
student inference or an activated production Graphics stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import re

from .hashing import content_sha256
from .types import QualificationError

SHA = re.compile(r"^[0-9a-f]{64}$")
KNOWN_OPERATIONS = frozenset({
    "INDEPENDENT_ATLAS_EDITS",
    "POSE_AWARE_SOURCE_UV",
    "LOCAL_DEPTH_SOURCE_OVER",
    "HIDDEN_EDIT_STATE_PERSISTENCE",
    "INDEPENDENT_ATTACHMENT_RELEASE",
    "SECONDARY_MOTION",
    "NONBINARY_ALPHA_SOURCE_PARITY",
})
IMPLEMENTED_OPERATIONS = frozenset({
    "INDEPENDENT_ATLAS_EDITS",
    "POSE_AWARE_SOURCE_UV",
    "LOCAL_DEPTH_SOURCE_OVER",
    "HIDDEN_EDIT_STATE_PERSISTENCE",
})
AUTHORITY_DENIAL = {
    "source_artist_owner_authorized": False,
    "corpus_materialization_authorized": False,
    "iris_training_authorized": False,
    "tessa_training_authorized": False,
    "training_authorized": False,
}


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise QualificationError(code)


@dataclass(frozen=True)
class SourceGraphicsEvidenceV0:
    """E: qualified-by-caller source identities, never reconstructed owners."""
    mesh_binding_hash: str
    skin_binding_hash: str
    source_graphics_binding_hash: str
    face_count: int
    source_face_group_ids: tuple[int, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    input_mode: str = "TEACHER_SOURCE_KNOWN"
    schema_version: str = "RealSaS.SourceGraphicsEvidence.v0"

    def validate(self):
        for h in (self.mesh_binding_hash, self.skin_binding_hash,
                  self.source_graphics_binding_hash):
            _require(isinstance(h, str) and SHA.fullmatch(h) is not None,
                     "ORS_SOURCE_SHA_REQUIRED")
        _require(type(self.face_count) is int and self.face_count > 0,
                 "ORS_SOURCE_FACE_COUNT_INVALID")
        _require(self.input_mode in {"TEACHER_SOURCE_KNOWN", "SYNTHETIC_FIXTURE",
                                      "STUDENT_INFERRED_UNQUALIFIED"},
                 "ORS_SOURCE_INPUT_MODE_INVALID")
        _require(not self.source_face_group_ids or (
            len(self.source_face_group_ids) == self.face_count
            and all(type(x) is int and x >= 0 for x in self.source_face_group_ids)),
            "ORS_SOURCE_GROUPS_INVALID")
        _require(all(isinstance(x, str) and x for x in self.provenance_refs),
                 "ORS_PROVENANCE_REFS_INVALID")


@dataclass(frozen=True)
class PuppetCapabilityContractV0:
    """C: operations required of a runtime, not observations from source."""
    independent_edit_address_count: int
    required_operations: tuple[str, ...]
    atlas_coordinate_convention: str = "TOP_LEFT_V2"
    schema_version: str = "RealSaS.PuppetCapabilityContract.v0"

    def validate(self):
        _require(type(self.independent_edit_address_count) is int
                 and 1 <= self.independent_edit_address_count <= 12,
                 "ORS_EDIT_ADDRESS_COUNT_INVALID")
        _require(bool(self.required_operations)
                 and len(set(self.required_operations)) == len(self.required_operations)
                 and set(self.required_operations) <= KNOWN_OPERATIONS,
                 "ORS_REQUIRED_OPERATIONS_INVALID")
        _require(self.atlas_coordinate_convention == "TOP_LEFT_V2",
                 "ORS_ATLAS_CONVENTION_INVALID")


@dataclass(frozen=True)
class RepresentationPolicyV0:
    """P: only packing preferences, never source-truth thresholds."""
    max_faces_per_packet: int = 32
    enable_source_group_packing: bool = True
    enable_face_stream: bool = True
    schema_version: str = "RealSaS.RepresentationPolicy.v0"

    def validate(self):
        _require(type(self.max_faces_per_packet) is int
                 and self.max_faces_per_packet >= 1,
                 "ORS_PACKET_POLICY_INVALID")


@dataclass(frozen=True)
class DrawablePacketV0:
    packet_id: str
    face_indices: tuple[int, ...]


@dataclass(frozen=True)
class RepresentationCandidateV0:
    candidate_id: str
    mesh_binding_hash: str
    skin_binding_hash: str
    source_graphics_binding_hash: str
    packets: tuple[DrawablePacketV0, ...]
    edit_control_address_indices: tuple[int, ...]
    uv_transport: str
    compositing_policy: str
    candidate_hash: str
    schema_version: str = "RealSaS.RepresentationCandidate.v0"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SynthesisResultV0:
    evidence: SourceGraphicsEvidenceV0
    contract: PuppetCapabilityContractV0
    policy: RepresentationPolicyV0
    candidates: tuple[RepresentationCandidateV0, ...]
    capability_states: dict[str, str]
    executable_candidate_ids: tuple[str, ...]
    physical_source_qualification: str = "UNKNOWN"
    measured_runtime_cost: str = "UNMEASURED_COST"
    source_artist_ownership: str = "UNKNOWN"
    research_only: bool = True
    authority: dict[str, bool] = field(default_factory=lambda: dict(AUTHORITY_DENIAL))
    schema_version: str = "RealSaS.OperationalRepresentationSynthesis.v0"

    def to_dict(self):
        return asdict(self)


def _packet_layouts(evidence: SourceGraphicsEvidenceV0,
                    policy: RepresentationPolicyV0):
    faces = tuple(range(evidence.face_count))
    yield "addressed_aggregate", (faces,)
    if policy.enable_source_group_packing and evidence.source_face_group_ids:
        groups = sorted(set(evidence.source_face_group_ids))
        yield "source_group_shared_carrier", tuple(
            tuple(i for i, label in enumerate(evidence.source_face_group_ids)
                  if label == group) for group in groups)
    chunks = tuple(faces[i:i + policy.max_faces_per_packet]
                   for i in range(0, evidence.face_count, policy.max_faces_per_packet))
    yield "bounded_hybrid", chunks
    if policy.enable_face_stream:
        yield "per_face_shared_carrier", tuple((i,) for i in faces)


def validate_candidate_v0(candidate: RepresentationCandidateV0,
                          *, face_count: int, address_count: int):
    _require(candidate.candidate_hash == content_sha256({
        k: v for k, v in candidate.to_dict().items()
        if k != "candidate_hash"
    }), "ORS_CANDIDATE_HASH_DRIFT")
    faces = tuple(face for packet in candidate.packets for face in packet.face_indices)
    _require(len(faces) == face_count and set(faces) == set(range(face_count)),
             "ORS_PACKET_COVERAGE_OR_OVERLAP")
    _require(tuple(candidate.edit_control_address_indices)
             == tuple(range(address_count)),
             "ORS_CONTROL_TIE_OR_ADDRESS_MISSING")
    _require(candidate.uv_transport == "POSED_CANONICAL_FACE_BARYCENTRIC_UV"
             and candidate.compositing_policy == "CAA_V2_LOCAL_DEPTH_K4_SOURCE_OVER",
             "ORS_UNSUPPORTED_EXECUTOR")


def synthesize_operational_representations_v0(
    evidence: SourceGraphicsEvidenceV0,
    contract: PuppetCapabilityContractV0,
    policy: RepresentationPolicyV0 = RepresentationPolicyV0(),
) -> SynthesisResultV0:
    """Produce strong layouts without reading any source target render.

    Capability states cover *implemented reference executor* operations only.
    No candidate receives physical source qualification from this function.
    """
    evidence.validate()
    contract.validate()
    policy.validate()
    candidates = []
    seen_layouts = set()
    for candidate_id, packet_faces in _packet_layouts(evidence, policy):
        layout_key = tuple(tuple(group) for group in packet_faces)
        if layout_key in seen_layouts:
            continue
        seen_layouts.add(layout_key)
        packets = tuple(DrawablePacketV0(f"packet_{i:04d}", faces)
                        for i, faces in enumerate(layout_key))
        core = dict(
            candidate_id=candidate_id,
            mesh_binding_hash=evidence.mesh_binding_hash,
            skin_binding_hash=evidence.skin_binding_hash,
            source_graphics_binding_hash=evidence.source_graphics_binding_hash,
            packets=packets,
            edit_control_address_indices=tuple(
                range(contract.independent_edit_address_count)),
            uv_transport="POSED_CANONICAL_FACE_BARYCENTRIC_UV",
            compositing_policy="CAA_V2_LOCAL_DEPTH_K4_SOURCE_OVER",
        )
        value = RepresentationCandidateV0(**core, candidate_hash="")
        hashed = content_sha256({
            k: v for k, v in value.to_dict().items()
            if k != "candidate_hash"
        })
        value = RepresentationCandidateV0(**core, candidate_hash=hashed)
        validate_candidate_v0(value, face_count=evidence.face_count,
                              address_count=contract.independent_edit_address_count)
        candidates.append(value)
    states = {
        op: ("SATISFIED" if op in IMPLEMENTED_OPERATIONS else "UNKNOWN")
        for op in contract.required_operations
    }
    # UNKNOWN is not waived: a partial executor is not a qualified full puppet.
    executable = tuple(c.candidate_id for c in candidates) if (
        all(state == "SATISFIED" for state in states.values())) else ()
    return SynthesisResultV0(
        evidence=evidence, contract=contract, policy=policy,
        candidates=tuple(candidates), capability_states=states,
        executable_candidate_ids=executable,
    )
