from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_services.platform_worker import shape_capability_v2 as shape_v2


ROOT = Path(__file__).resolve().parents[2]


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
        surface_normal=np.tile(np.asarray([[[0., 0., 1.]]], dtype=np.float32), (1, count, 1)),
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


def _v1_result(**diagnostics):
    return SimpleNamespace(payload=_v1_payload(), diagnostics=dict(diagnostics))


def _decoded():
    return SimpleNamespace(
        vertices_normalized=np.asarray(
            [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_component_indices=np.asarray([0, 0, 0], dtype=np.int64),
        face_component_indices=np.asarray([0], dtype=np.int64),
        component_count=1,
        restart_count=0,
    )


def test_shape_v2_module_is_importable_without_torch():
    script = r'''
import builtins

original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("TORCH_IMPORT_FORBIDDEN_DURING_SHAPE_V2_MODULE_IMPORT")
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
from compiler.realsas_compiler_services.platform_worker import shape_capability_v2
assert shape_capability_v2.SHAPE_PROPOSAL_SCHEMA_V2 == "RealSaS.ShapeProposal.v2"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_shape_v2_serializes_scene_frame_topology_without_mesh_authority(monkeypatch):
    monkeypatch.setattr(shape_v2, "_decode_tessa_asset_sequence_v1", lambda tokens: _decoded())
    payload = shape_v2.encode_shape_proposal_v2(
        _v1_result(schema="RealSaS.ShapeInferenceDiagnostics.v1", canonical_mesh_authority=False),
        source_observation_content_sha256="a" * 64,
    )
    value = shape_v2.decode_shape_proposal_v2(payload)
    assert value.metadata["schema"] == "RealSaS.ShapeProposal.v2"
    assert value.metadata["canonical_mesh_authority"] is False
    assert value.metadata["mesh_authority"] is False
    assert value.metadata["compiler_frame_authority"] is False
    assert value.metadata["compiler_frame_binding_required"] is True
    assert value.metadata["source_geometry_lineage_claimed"] is False
    assert value.metadata["decoded_topology_is_proposal_evidence"] is True
    assert value.metadata["proposal_evidence_frame"] == "RealSaS.TESSA.SequenceNormalizedFrame.v1"
    assert value.metadata["source_truth_copied_into_shape"] is False
    assert value.metadata["decoded_vertex_count"] == 3
    assert value.metadata["decoded_face_count"] == 1
    assert len(value.metadata["decoded_topology_sha256"]) == 64
    assert value.decoded_vertices_tessa.dtype == np.float64
    assert value.decoded_faces.tolist() == [[0, 1, 2]]


def test_shape_v2_runs_learned_path_once_then_only_deterministic_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shape_v2,
        "_run_shape_inference_v1",
        lambda *args, **kwargs: calls.append("learned") or _v1_result(
            canonical_mesh_authority=False,
            timings={},
        ),
    )
    monkeypatch.setattr(
        shape_v2,
        "_decode_tessa_asset_sequence_v1",
        lambda tokens: calls.append("decode") or _decoded(),
    )
    result = shape_v2.run_shape_inference_v2(
        object(), source_observation_content_sha256="a" * 64,
    )
    assert calls == ["learned", "decode"]
    assert result.diagnostics["output_schema"] == "RealSaS.ShapeProposal.v2"
    assert result.diagnostics["canonical_mesh_authority"] is False
    assert result.diagnostics["compiler_frame_authority"] is False


def test_shape_v2_detects_decoded_topology_byte_drift(monkeypatch):
    monkeypatch.setattr(shape_v2, "_decode_tessa_asset_sequence_v1", lambda tokens: _decoded())
    payload = shape_v2.encode_shape_proposal_v2(
        _v1_result(), source_observation_content_sha256="a" * 64,
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        rows = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
    vertices = rows["decoded_vertices_tessa"]
    vertices[0, 0] += 0.125
    stream = io.BytesIO()
    np.savez_compressed(stream, **rows)
    with pytest.raises(RuntimeError, match="DECODED_TOPOLOGY_HASH_DRIFT"):
        shape_v2.decode_shape_proposal_v2(stream.getvalue())


def _encode_shape_fixture_v2(monkeypatch) -> bytes:
    monkeypatch.setattr(shape_v2, "_decode_tessa_asset_sequence_v1", lambda tokens: _decoded())
    return shape_v2.encode_shape_proposal_v2(
        _v1_result(), source_observation_content_sha256="a" * 64,
    )


def _rewrite_shape_fixture(payload: bytes, *, rewrite_array=None, rewrite_meta=None) -> bytes:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if rewrite_array is not None:
        rewrite_array(fields)
    if rewrite_meta is not None:
        meta = json.loads(str(fields["metadata_json"].reshape(-1)[0]))
        rewrite_meta(meta, fields)
        fields["metadata_json"] = np.asarray([
            json.dumps(meta, sort_keys=True, separators=(",", ":")),
        ])
    output = io.BytesIO()
    np.savez_compressed(output, **fields)
    return output.getvalue()


def test_shape_v2_consumer_replays_tessa_tokens_and_consumes_all_iris_evidence(monkeypatch):
    payload = _encode_shape_fixture_v2(monkeypatch)
    value = shape_v2.decode_shape_proposal_v2(payload)
    assert value.metadata["surface_evidence_sha256"] == shape_v2._surface_fields_sha256_v2({
        key: value.mechanical[key] if key in value.mechanical else getattr(value, key)
        for key in shape_v2._SURFACE_FIELDS_V2
    })
    assert value.decoded_faces.tolist() == [[0, 1, 2]]
    assert value.surface_normal[0, 0].tolist() == [0., 0., 1.]
    assert value.mechanical["rigidity_probability"].shape == (1, 3)
    assert value.metadata["decoded_topology_is_proposal_evidence"] is True
    assert value.metadata["source_truth_copied_into_shape"] is False


def test_shape_v2_rejects_reserialized_mechanical_payload_without_evidence_receipt(monkeypatch):
    payload = _encode_shape_fixture_v2(monkeypatch)
    changed = _rewrite_shape_fixture(
        payload, rewrite_array=lambda fields: fields["rigidity_probability"].__setitem__((0, 0), .7),
    )
    with pytest.raises(RuntimeError, match="SHAPE_V2_SURFACE_EVIDENCE_HASH_DRIFT"):
        shape_v2.decode_shape_proposal_v2(changed)


@pytest.mark.parametrize("field,mutation,error", (
    ("surface_h", "nan", "SURFACE_OR_MECHANICAL_NONFINITE"),
    ("surface_normal", "zero", "SURFACE_NORMAL_DEGENERATE"),
    ("bend_demand", "wrong_shape", "MECHANICAL_SHAPE_INVALID"),
    ("rigidity_probability", "out_of_domain", "MECHANICAL_PROBABILITY_OUT_OF_DOMAIN"),
    ("contact_persistence_valid", "invalid_mask", "MECHANICAL_VALID_MASK_INVALID"),
))
def test_shape_v2_rejects_bad_iris_or_mechanical_field_before_consuming(
    monkeypatch, field, mutation, error,
):
    payload = _encode_shape_fixture_v2(monkeypatch)

    def tamper(fields):
        value = fields[field]
        if mutation == "nan":
            value[0, 0, 0] = float("nan")
        elif mutation == "zero":
            value[0, 0, :] = 0.
        elif mutation == "wrong_shape":
            fields[field] = value[:, :2]
        elif mutation == "out_of_domain":
            value[0, 0] = 1.01
        elif mutation == "invalid_mask":
            value[0, 0] = 2
        else:
            raise AssertionError(mutation)

    changed = _rewrite_shape_fixture(payload, rewrite_array=tamper)
    with pytest.raises(RuntimeError, match=error):
        shape_v2.decode_shape_proposal_v2(changed)


def test_shape_v2_rejects_rehashed_topology_not_explained_by_tessa_tokens(monkeypatch):
    payload = _encode_shape_fixture_v2(monkeypatch)

    def rehash_topology(meta, fields):
        v = fields["decoded_vertices_tessa"]
        v[0, 0] += .125
        meta["decoded_topology_sha256"] = shape_v2._decoded_topology_sha256(
            v, fields["decoded_faces"],
            fields["decoded_vertex_component_indices"],
            fields["decoded_face_component_indices"],
        )
    altered = _rewrite_shape_fixture(payload, rewrite_meta=rehash_topology)
    with pytest.raises(RuntimeError, match="SHAPE_V2_TOPOLOGY_TOKEN_DECODE_MISMATCH"):
        shape_v2.decode_shape_proposal_v2(altered)


def test_shape_v2_rejects_forged_source_truth_metadata_even_with_valid_payload(monkeypatch):
    payload = _encode_shape_fixture_v2(monkeypatch)
    altered = _rewrite_shape_fixture(
        payload, rewrite_meta=lambda meta, _: meta.update({"source_truth_copied_into_shape": True}),
    )
    with pytest.raises(RuntimeError, match="SHAPE_V2_TOPOLOGY_AUTHORITY_OR_COUNT_DRIFT"):
        shape_v2.decode_shape_proposal_v2(altered)


def test_shape_v2_archive_rejects_unannounced_payload_family(monkeypatch):
    payload = _encode_shape_fixture_v2(monkeypatch)
    altered = _rewrite_shape_fixture(
        payload, rewrite_array=lambda fields: fields.update({
            "fabricated_source_truth": np.zeros((1, 3), dtype=np.float32),
        }),
    )
    with pytest.raises(RuntimeError, match="SHAPE_V2_ARCHIVE_FIELDS_OR_RESOURCE_LIMIT"):
        shape_v2.decode_shape_proposal_v2(altered)


def test_shape_source_parent_survives_real_shape_codec_to_mesh_consumer(monkeypatch):
    from compiler.realsas_compiler_services.platform_worker.mesh_executor_v1 import (
        validate_product_shape_source_lineage_v1,
    )
    raw=_encode_shape_fixture_v2(monkeypatch)
    shape=shape_v2.decode_shape_proposal_v2(raw)
    assert shape.metadata["source_observation_content_sha256"] == "a"*64
    validate_product_shape_source_lineage_v1(
        shape, {"content_sha256":"a"*64},
    )
    with pytest.raises(RuntimeError,match="MESH_SHAPE_SOURCE_OBSERVATION_CAS_PARENT_DRIFT"):
        validate_product_shape_source_lineage_v1(
            shape, {"content_sha256":"b"*64},
        )
    with pytest.raises(RuntimeError,match="MESH_SHAPE_SOURCE_OBSERVATION_CAS_PARENT_DRIFT"):
        validate_product_shape_source_lineage_v1(shape, {"content_sha256":""})


@pytest.mark.parametrize("source_sha", ["", "A"*64, "a"*63, "z"*64])
def test_shape_v2_cannot_mint_unbound_or_noncanonical_source_cas_sha(source_sha):
    with pytest.raises(RuntimeError,match="SOURCE_OBSERVATION_CAS_ID_REQUIRED"):
        shape_v2.encode_shape_proposal_v2(
            _v1_result(),source_observation_content_sha256=source_sha,
        )
