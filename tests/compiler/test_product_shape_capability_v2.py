from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_services.platform_worker import shape_capability_v2 as shape_v2
from compiler.realsas_compiler_services.platform_worker.shape_capability_v1 import ShapeInferenceResultV1


def _v1_payload() -> bytes:
    metadata = {
        "schema": "RealSaS.ShapeProposal.v1",
        "authority": "LEARNED_SHAPE_PROPOSAL_ONLY",
        "canonical_mesh_authority": False,
        "mesh_authority": False,
        "models": {
            "IRIS": {"model_release_id": "i", "checkpoint_content_sha256": "a" * 64},
            "TESSA": {"model_release_id": "t", "checkpoint_content_sha256": "b" * 64},
        },
    }
    stream = io.BytesIO()
    count = 3
    np.savez_compressed(
        stream,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
        token_ids=np.asarray([128, 1, 2, 3, 129, 1, 2, 3, 130, 1, 2, 3, 134], dtype=np.int64),
        surface_xyz_tessa=np.zeros((1, count, 3), dtype=np.float32),
        surface_normal=np.zeros((1, count, 3), dtype=np.float32),
        surface_h=np.zeros((1, count, 8), dtype=np.float32),
        stretch_major=np.ones((1, count), dtype=np.float32),
        stretch_minor=np.zeros((1, count), dtype=np.float32),
        tangent_direction_2theta=np.zeros((1, count, 2), dtype=np.float32),
        bend_demand=np.zeros((1, count), dtype=np.float32),
        rigidity_probability=np.zeros((1, count), dtype=np.float32),
        contact_persistence_probability=np.zeros((1, count), dtype=np.float32),
        contact_persistence_valid=np.zeros((1, count), dtype=np.uint8),
    )
    return stream.getvalue()


def test_shape_v2_serializes_decoded_topology_without_mesh_authority(monkeypatch):
    decoded = SimpleNamespace(
        vertices_normalized=np.asarray([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]], dtype=np.float64),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_component_indices=np.asarray([0, 0, 0], dtype=np.int64),
        face_component_indices=np.asarray([0], dtype=np.int64),
        component_count=1,
        restart_count=0,
    )
    monkeypatch.setattr(shape_v2, "decode_tessa_asset_sequence_v1", lambda tokens: decoded)
    v1 = ShapeInferenceResultV1(
        payload=_v1_payload(),
        diagnostics={"schema": "RealSaS.ShapeInferenceDiagnostics.v1", "canonical_mesh_authority": False},
    )
    payload = shape_v2.encode_shape_proposal_v2(v1)
    value = shape_v2.decode_shape_proposal_v2(payload)
    assert value.metadata["schema"] == "RealSaS.ShapeProposal.v2"
    assert value.metadata["canonical_mesh_authority"] is False
    assert value.metadata["mesh_authority"] is False
    assert value.metadata["decoded_topology_is_proposal_evidence"] is True
    assert value.metadata["source_truth_copied_into_shape"] is False
    assert value.metadata["decoded_vertex_count"] == 3
    assert value.metadata["decoded_face_count"] == 1
    assert value.decoded_vertices_tessa.dtype == np.float64
    assert value.decoded_faces.tolist() == [[0, 1, 2]]


def test_shape_v2_runs_v1_learned_path_once_then_decodes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shape_v2,
        "run_shape_inference_v1",
        lambda *args, **kwargs: calls.append("learned") or ShapeInferenceResultV1(
            payload=_v1_payload(), diagnostics={"canonical_mesh_authority": False, "timings": {}}
        ),
    )
    decoded = SimpleNamespace(
        vertices_normalized=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float64),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_component_indices=np.asarray([0, 0, 0], dtype=np.int64),
        face_component_indices=np.asarray([0], dtype=np.int64),
        component_count=1,
        restart_count=0,
    )
    monkeypatch.setattr(shape_v2, "decode_tessa_asset_sequence_v1", lambda tokens: calls.append("decode") or decoded)
    result = shape_v2.run_shape_inference_v2(object())
    assert calls == ["learned", "decode"]
    assert result.diagnostics["output_schema"] == "RealSaS.ShapeProposal.v2"
    assert result.diagnostics["canonical_mesh_authority"] is False
