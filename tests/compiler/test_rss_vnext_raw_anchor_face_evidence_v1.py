from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.rss_vnext.workers.rss_vnext_raw_anchor_face_evidence_v1 import (
    encode_raw_neural_socket_faces_v1,
    validate_raw_neural_socket_faces_v1,
)
from tests.models.test_rss_vnext_socket_face_archive_v1 import (
    _actual_student, _events,
)


def _fixture():
    proposal, _, _, _ = _actual_student()
    mesh = proposal["mesh_candidate"]
    scene = {
        "tessa_topology_token_ids": np.array(
            mesh.topology_token_ids, dtype=np.int64,
        )[None],
        "tessa_candidate_faces": mesh.faces.detach().cpu().numpy(),
        "tessa_candidate_vertex_components":
            mesh.vertex_component_ids.detach().cpu().numpy(),
        "tessa_candidate_face_components":
            mesh.face_component_ids.detach().cpu().numpy(),
    }
    receipt = {
        "scene_content_sha256": "c"*64,
        "checkpoint_sha256": "a"*64,
        "axis_mira_control_events": [
            {**asdict(e), "kind": int(e.kind)} for e in _events()
        ],
    }
    predicted = encode_raw_neural_socket_faces_v1(
        proposal, scene_content_sha256=receipt["scene_content_sha256"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )
    return predicted, scene, receipt


def test_raw_learned_faces_can_be_replayed_by_compiler_without_model():
    predicted, scene, receipt = _fixture()
    report = validate_raw_neural_socket_faces_v1(
        predicted, scene=scene, construction_receipt=receipt,
    )
    assert report["state"] == "UNQUALIFIED_GENERATED_SOCKET_FACE_CANDIDATES"
    assert report["candidate_count"] == 1
    assert not report["attachment_qualified"]
    assert not predicted["source_contact_qualified"]
    assert not predicted["parent_rest_frame_qualified"]


def test_resigned_wrong_tessa_face_and_axis_direction_cannot_be_promoted():
    predicted, scene, receipt = _fixture()
    changed = dict(predicted)
    changed["rows"] = [{**row} for row in changed["rows"]]
    changed["rows"][0]["held_face_index"] = 999
    changed["candidate_hash"] = content_sha256({
        k:v for k,v in changed.items() if k!="candidate_hash"
    })
    with pytest.raises(RuntimeError,match="FACE_OR_DIRECTION_INVALID"):
        validate_raw_neural_socket_faces_v1(
            changed,scene=scene,construction_receipt=receipt,
        )
    tampered = {**receipt,"axis_mira_control_events": [
        {**row} for row in receipt["axis_mira_control_events"]
    ]}
    tampered["axis_mira_control_events"][8]["pointer_b"] = -1
    with pytest.raises(RuntimeError,match="AXIS_EVENT_BINDING_DRIFT|SOCKET_ADDRESS_INVALID"):
        validate_raw_neural_socket_faces_v1(
            predicted,scene=scene,construction_receipt=tampered,
        )


def test_rehashed_source_mesh_face_reindex_drifts_without_qualified_repair():
    predicted, scene, receipt = _fixture()
    altered={k:v.copy() for k,v in scene.items()}
    altered["tessa_candidate_face_components"][0] += 1
    with pytest.raises(RuntimeError,match="TESSA_FACE_BINDING_DRIFT"):
        validate_raw_neural_socket_faces_v1(
            predicted,scene=altered,construction_receipt=receipt,
        )


def test_abstain_does_not_invent_contact_or_face_zero():
    candidate = encode_raw_neural_socket_faces_v1(
        {"mesh_candidate":None, "control_event_proposal":None},
        scene_content_sha256="c"*64, checkpoint_sha256="a"*64,
    )
    report=validate_raw_neural_socket_faces_v1(
        candidate,scene={},construction_receipt={
            "scene_content_sha256":"c"*64,"checkpoint_sha256":"a"*64,
        },
    )
    assert report["state"] == "ABSTAIN_NO_VALID_GENERATED_SOCKET_FACE_CANDIDATE"
    assert report["candidate_count"]==0
    assert candidate["rows"]==[]
