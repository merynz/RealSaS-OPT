from __future__ import annotations

"""Single-checkpoint student socket-face head → Compiler sidecar exporter.

Never serializes the training-only source-native face labels, GT mesh or
teacher privileged coordinates. The produced hash binds exact generated
AXIS events and TESSA mesh token/face/component identities to the original
sealed two-view neural inference proposal.
"""

from collections.abc import Mapping
from dataclasses import asdict

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,validate_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_socket_face_archive_v1 import (
    RSSvNextNeuralSocketFaceArchiveV1,
    seal_rss_vnext_neural_socket_face_archive_v1,
    encode_rss_vnext_neural_socket_face_archive_v1,
)
from models.axis.rig_program_model_v6 import AXISSequenceV6
from .attachment_face_correspondence_v1 import (
    RSSvNextSocketFaceNeuralOutputV1,
    mesh_topology_digest_v1,axis_sequence_digest_v1,
    propose_socket_face_pairs_v1,
)
from .animation_ready_mesh import RSSvNextStructuralMeshCandidate


def export_rss_vnext_neural_socket_face_archive_v1(
    prediction:Mapping[str,object],
    *,
    evidence:RSSvNextInferenceEvidenceIRV1,
)->bytes|None:
    """Optional immutable sidecar; ABSTAIN has no neural face payload."""
    validate_inference_evidence_v1(evidence)
    if not isinstance(prediction,Mapping):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_EXPORT_RESULT_REQUIRED")
    candidate=prediction.get("anchor_face_candidate")
    if candidate is None:
        if prediction.get("mesh_candidate") is None or not prediction.get("control_event_proposal"):
            return None
        raise ValueError("RSS_VNEXT_SOCKET_FACE_EXPORT_MODEL_OUTPUT_REQUIRED")
    mesh=prediction.get("mesh_candidate")
    events=prediction.get("control_event_proposal")
    if (
        type(candidate) is not RSSvNextSocketFaceNeuralOutputV1
        or type(mesh) is not RSSvNextStructuralMeshCandidate
        or mesh.source_authority!="LEARNED_PREDICTED_TOPOLOGY"
        or not isinstance(events,(list,tuple))
        or not events
        or prediction.get("mesh_status")!="UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh_status!="UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None
        or prediction.get("qualification_status")!="UNQUALIFIED_NEURAL_PROPOSAL"
        or candidate.source_correspondence_qualified
        or candidate.executable_attachment_qualified
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_EXPORT_TEACHER_OR_AUTHORITY_FORBIDDEN")
    mesh.validate()
    sequence=AXISSequenceV6.from_events(
        events,device=mesh.faces.device,
    )
    if (
        candidate.mesh_topology_digest!=mesh_topology_digest_v1(mesh)
        or candidate.axis_sequence_digest!=axis_sequence_digest_v1(sequence)
        or content_sha256(evidence.mesh["topology_tokens"])
           !=content_sha256(mesh.topology_token_ids)
        or content_sha256(evidence.mesh["faces"])
           !=content_sha256(mesh.faces.detach().cpu().tolist())
        or content_sha256(evidence.mesh["face_components"])
           !=content_sha256(mesh.face_component_ids.detach().cpu().tolist())
        or content_sha256(evidence.control_events)!=content_sha256(
            tuple({**asdict(e),"kind":int(e.kind)} for e in events)
        )
    ):
        raise ValueError("RSS_VNEXT_SOCKET_FACE_EXPORT_MODEL_MESH_OR_AXIS_DRIFT")
    predicted=propose_socket_face_pairs_v1(candidate)
    archive=RSSvNextNeuralSocketFaceArchiveV1(
        checkpoint_sha256=evidence.checkpoint_sha256,
        inference_evidence_hash=evidence.evidence_hash,
        candidate_mesh_sha256=content_sha256(evidence.mesh),
        axis_control_events_sha256=content_sha256(evidence.control_events),
        proposals=tuple(asdict(row) for row in predicted),
    )
    return encode_rss_vnext_neural_socket_face_archive_v1(
        seal_rss_vnext_neural_socket_face_archive_v1(archive),
        evidence=evidence,
    )


__all__=["export_rss_vnext_neural_socket_face_archive_v1"]
