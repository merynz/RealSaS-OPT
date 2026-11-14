from __future__ import annotations

"""One model-selected HELD→HOLDER candidate receipt for raw RSS.vNext Shape.

The original uncalibrated image cannot prove which source surface holds an
object. Neural predictions are candidates, NEVER attachment, ownership,
contact, source visibility or executable runtime authority. No second model
inference is allowed in Compiler consumers.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256

SCHEMA = "RealSaS.RSSvNext.RawNeuralSocketFaceEvidence.v1"
MODEL_AUTHORITY = "MODEL_SELECTED_UNQUALIFIED_HELD_HOLDER_FACE_HYPOTHESES"
_ABSTAIN = "ABSTAIN_NO_VALID_GENERATED_SOCKET_FACE_CANDIDATE"
_CANDIDATE = "UNQUALIFIED_GENERATED_SOCKET_FACE_CANDIDATES"
_MAX_ROWS = 512


def _sha(v: Any) -> bool:
    return type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _torch_long_digest(arr: np.ndarray) -> str:
    """Independent decoded tensor digest, matching Torch int64 serialization."""
    a = np.ascontiguousarray(arr, dtype=np.int64)
    header = json.dumps(
        {"dtype":"torch.int64", "shape":list(a.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return sha256(header+b"\0"+a.tobytes(order="C")).hexdigest()


def _mesh_digest_from_cas(scene: Mapping[str, np.ndarray]) -> str:
    keys = (
        "tessa_topology_token_ids", "tessa_candidate_faces",
        "tessa_candidate_vertex_components", "tessa_candidate_face_components",
    )
    if any(k not in scene for k in keys):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_TESSA_CAS_MISSING")
    tok, faces, vertex_ids, face_ids = (scene[k] for k in keys)
    if tok.ndim != 2 or tok.shape[0] != 1 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_TESSA_DOMAIN")
    fields = {
        "topology_tokens_sha256":sha256(json.dumps(
            tok[0].astype(np.int64).tolist(), separators=(",", ":"),
        ).encode()).hexdigest(),
        "faces_sha256":_torch_long_digest(faces),
        "components_sha256":_torch_long_digest(face_ids),
        "vertices_sha256":_torch_long_digest(vertex_ids),
    }
    return sha256(json.dumps(fields,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def encode_raw_neural_socket_faces_v1(
    proposal: Mapping[str, Any], *,
    scene_content_sha256: str, checkpoint_sha256: str,
) -> dict[str, Any]:
    """Capture actual learned output, not a Compiler-produced pair discovery."""
    from models.axis.rig_program_model_v6 import AXISSequenceV6
    from models.rss_vnext.attachment_face_correspondence_v1 import (
        RSSvNextSocketFaceNeuralOutputV1, mesh_topology_digest_v1,
        axis_sequence_digest_v1, propose_socket_face_pairs_v1,
    )
    from models.rss_vnext.animation_ready_mesh import RSSvNextStructuralMeshCandidate
    if not _sha(scene_content_sha256) or not _sha(checkpoint_sha256):
        raise ValueError("RSS_VNEXT_RAW_ANCHOR_PARENT_DIGEST")
    mesh = proposal.get("mesh_candidate")
    events = proposal.get("control_event_proposal")
    pred = proposal.get("anchor_face_candidate")
    if mesh is None or not events:
        if pred is not None:
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ABSTAIN_WITH_MODEL_OUTPUT")
        state, rows, mesh_digest, axis_digest = _ABSTAIN, (), None, None
    else:
        if (
            type(mesh) is not RSSvNextStructuralMeshCandidate
            or type(pred) is not RSSvNextSocketFaceNeuralOutputV1
            or mesh.source_authority != "LEARNED_PREDICTED_TOPOLOGY"
            or proposal.get("mesh_status") != "UNQUALIFIED_MESH_CANDIDATE"
            or proposal.get("qualification_status") != "UNQUALIFIED_NEURAL_PROPOSAL"
            or pred.source_correspondence_qualified or pred.executable_attachment_qualified
        ):
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_TEACHER_OR_MODEL_OUTPUT_MISSING")
        sequence=AXISSequenceV6.from_events(events,device=mesh.faces.device)
        if (
            pred.mesh_topology_digest != mesh_topology_digest_v1(mesh)
            or pred.axis_sequence_digest != axis_sequence_digest_v1(sequence)
        ):
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_HEAD_MESH_AXIS_DRIFT")
        rows=tuple(asdict(row) for row in propose_socket_face_pairs_v1(pred))
        if len(rows) > _MAX_ROWS:
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ROW_BUDGET")
        state = _CANDIDATE
        mesh_digest, axis_digest = pred.mesh_topology_digest, pred.axis_sequence_digest
    result={
        "schema": SCHEMA,
        "authority": MODEL_AUTHORITY,
        "scene_content_sha256": scene_content_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "state": state,
        "rows": list(rows),
        "candidate_mesh_topology_sha256": mesh_digest,
        "axis_sequence_sha256": axis_digest,
        "source_contact_qualified": False,
        "attachment_qualified": False,
        "parent_rest_frame_qualified": False,
        "product_revision_minted": False,
    }
    result["candidate_hash"] = content_sha256(result)
    return result


def validate_raw_neural_socket_faces_v1(
    candidate: Mapping[str, Any], *,
    scene: Mapping[str, np.ndarray], construction_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Compiler-only exact TESSA face/AXIS socket compatibility replay.

    No trained weights are present in Compiler. This checks a model-authored
    candidate's addressability and source lineage, not neural score accuracy.
    """
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != {
            "schema", "authority", "scene_content_sha256", "checkpoint_sha256",
            "state", "rows", "candidate_mesh_topology_sha256",
            "axis_sequence_sha256", "source_contact_qualified",
            "attachment_qualified", "parent_rest_frame_qualified",
            "product_revision_minted", "candidate_hash",
        }
        or candidate.get("schema") != SCHEMA
        or candidate.get("authority") != MODEL_AUTHORITY
        or candidate.get("scene_content_sha256") != construction_receipt.get("scene_content_sha256")
        or candidate.get("checkpoint_sha256") != construction_receipt.get("checkpoint_sha256")
        or any(candidate.get(k) is not False for k in (
            "source_contact_qualified", "attachment_qualified",
            "parent_rest_frame_qualified", "product_revision_minted",
        ))
    ):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_LINEAGE_OR_AUTHORITY_INVALID")
    digest=candidate.get("candidate_hash")
    fields={k:v for k,v in candidate.items() if k!="candidate_hash"}
    if not _sha(digest) or digest != content_sha256(fields):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_HASH_DRIFT")
    rows=candidate["rows"]
    if type(rows) is not list or len(rows)>_MAX_ROWS:
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ROW_DOMAIN")
    if candidate["state"]==_ABSTAIN:
        if rows or candidate["candidate_mesh_topology_sha256"] is not None or candidate["axis_sequence_sha256"] is not None:
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ABSTAIN_DRIFT")
        return {"state":_ABSTAIN,"candidate_count":0,"abstain_count":0,
                "attachment_qualified":False}
    if candidate["state"]!=_CANDIDATE:
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_STATE_INVALID")
    if candidate["candidate_mesh_topology_sha256"] != _mesh_digest_from_cas(scene):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_TESSA_FACE_BINDING_DRIFT")
    events=construction_receipt.get("axis_mira_control_events")
    if not isinstance(events,list):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_CAUSAL_EVENTS_REQUIRED")
    from models.axis.rig_program_model_v6 import AXISSequenceV6
    from models.rss_vnext.attachment_face_correspondence_v1 import axis_sequence_digest_v1
    from models.rss_vnext.candidate_rig_court_v1 import _events
    sequence=AXISSequenceV6.from_events(
        _events(tuple(events)),device=__import__("torch").device("cpu"),
    )
    if candidate["axis_sequence_sha256"]!=axis_sequence_digest_v1(sequence):
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_AXIS_EVENT_BINDING_DRIFT")
    face_components=scene["tessa_candidate_face_components"]
    nfaces=int(scene["tessa_candidate_faces"].shape[0])
    from models.axis.rig_program_model_v6 import RigTokenKindV6
    controls=[e for e in events if e.get("kind")==int(RigTokenKindV6.CONTROL)]
    expected_event_indices=set()
    for idx,e in enumerate(events):
        if e.get("kind")!=int(RigTokenKindV6.ANCHOR) or e.get("anchor_parent_kind")!=1:
            continue
        parent=e.get("pointer_b")
        if (
            type(parent) is int and 0 <= parent < len(controls)
            and controls[parent].get("aux_type_id")==7
        ):
            expected_event_indices.add((idx,parent))
    seen=set()
    abstain=0
    for row in rows:
        if not isinstance(row,dict) or set(row)!={
            "anchor_event_index","socket_control_index","held_face_index",
            "holder_face_index","same_tessa_component","known_cross_slot_support",
            "predicted_directed_relation","held_unknown_probability",
            "holder_unknown_probability","predicted_rank_log_score","state",
            "authority","source_contact_qualified","parent_rest_frame_qualified",
            "attachment_qualified",
        }:
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ROW_SCHEMA")
        pair=(row["anchor_event_index"],row["socket_control_index"])
        if pair not in expected_event_indices or pair in seen:
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_SOCKET_ADDRESS_INVALID")
        seen.add(pair)
        if (
            row.get("authority")!="UNQUALIFIED_NEURAL_SOCKET_FACE_CORRESPONDENCE_V1"
            or any(row.get(k) is not False for k in (
                "source_contact_qualified", "parent_rest_frame_qualified",
                "attachment_qualified",
            ))
        ):
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_ROW_FALSE_AUTHORITY")
        a,b=row["held_face_index"],row["holder_face_index"]
        if a is None or b is None:
            if a is not None or b is not None or row["state"]!="ABSTAIN_MODEL_NO_FACE_OR_PAIR":
                raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_NO_FACE_DRIFT")
            abstain+=1
            continue
        if (
            type(a) is not int or type(b) is not int or a==b
            or not (0<=a<nfaces and 0<=b<nfaces)
            or row["same_tessa_component"]!=bool(face_components[a]==face_components[b])
            or row["state"] not in (
                "UNQUALIFIED_PREDICTED_DIRECTIONAL_FACE_PAIR",
                "UNQUALIFIED_PREDICTED_PAIR_UNKNOWN_REGION",
            )
        ):
            raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_FACE_OR_DIRECTION_INVALID")
    if seen!=expected_event_indices:
        raise RuntimeError("RSS_VNEXT_RAW_ANCHOR_MISSING_SOCKET_EVENT")
    return {
        "state":_CANDIDATE,
        "candidate_count":len(rows),
        "abstain_count":abstain,
        "attachment_qualified":False,
    }


__all__=[
    "encode_raw_neural_socket_faces_v1",
    "validate_raw_neural_socket_faces_v1",
    "SCHEMA", "MODEL_AUTHORITY",
]
