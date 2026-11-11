from __future__ import annotations

"""One observed source -> ONE learned Shape inference -> multiple CAS readers.

All tensors below are synthetic. These tests prove transport and consumer
boundaries, not learned generalization or source graphics qualification.
"""

from hashlib import sha256
import io
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

from compiler.realsas_compiler_services.platform_worker import shape_evidence_fanout_v1 as fanout
from compiler.realsas_compiler_services.platform_worker import shape_capability_v2 as shape_v2
from tests.compiler.test_product_shape_capability_v2 import _v1_result, _decoded


def _mock_heavy_flexicubes_for_cpu_contract_court(monkeypatch):
    # Do not load neural inference dependencies in the deterministic CPU CAS court.
    fake = ModuleType("models.iris.vnext.flexicubes_v1")
    fake.PinnedFlexiCubesV1 = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, fake.__name__, fake)


def _mock_shape_input_decoder_without_torch(monkeypatch):
    # The ShapeInput transport has independent Torch-enabled tests.
    # The cheap worker CAS court should not import its neural tensor runtime.
    fake = ModuleType(
        "compiler.realsas_compiler_services.platform_worker.scene_input_contract_v2"
    )
    fake.decode_shape_input_v2 = lambda data: SimpleNamespace(
        inputs=object(), source_observation_content_sha256="a" * 64,
    )
    monkeypatch.setitem(sys.modules, fake.__name__, fake)


def _produced(monkeypatch):
    monkeypatch.setattr(shape_v2, "_decode_tessa_asset_sequence_v1", lambda tokens: _decoded())
    with np.load(io.BytesIO(_v1_result().payload), allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]).copy() for name in archive.files}
    # A model-produced IRIS witness can legally exceed the TESSA coordinate
    # frame. The TESSA input clamps it; the IRIS sidecar must still retain
    # 1.01, not pretend that its source coordinate was exactly 1.0.
    fields["surface_xyz_tessa"][0, 0, 0] = .5
    v1_stream = io.BytesIO()
    np.savez_compressed(v1_stream, **fields)
    shape = shape_v2.encode_shape_proposal_v2(
        SimpleNamespace(payload=v1_stream.getvalue(), diagnostics={}),
        source_observation_content_sha256="a" * 64,
    )
    surface = SimpleNamespace(
        scene_planes=np.arange(1 * 3 * 8 * 2 * 2, dtype=np.float32).reshape(1, 3, 8, 2, 2),
        surface_xyz_iris=np.zeros((1, 3, 3), dtype=np.float32),
        surface_xyz_tessa=fields["surface_xyz_tessa"],
        extracted_vertex_counts=(4,),
        extracted_face_counts=(1,),
    )
    surface.surface_xyz_iris[0, 0, 0] = np.float32(1.01)
    scene = fanout.encode_iris_scene_evidence_v1(surface)
    index = fanout.seal_shape_evidence_fanout_v1(
        shape_bytes=shape, scene_bytes=scene,
        source_observation_content_sha256="a" * 64,
        shape_input_content_sha256="b" * 64,
    )
    return shape, scene, index


def test_same_neural_shape_evidence_reaches_two_independent_consumers_without_mesh_bottleneck(monkeypatch):
    shape_bytes, scene_bytes, index_bytes = _produced(monkeypatch)
    index = fanout.read_shape_evidence_fanout_v1(
        index_bytes, shape_bytes=shape_bytes, scene_bytes=scene_bytes,
        source_observation_content_sha256="a" * 64,
        shape_input_content_sha256="b" * 64,
    )
    # Product.Mesh can consume TESSA topology, IRIS normals and mechanics.
    mesh_consumer = shape_v2.decode_shape_proposal_v2(shape_bytes)
    # A separate scene/semantic/graphics research consumer can consume the
    # original IRIS planes without ever passing through CanonicalMesh.
    scene_consumer = fanout.decode_iris_scene_evidence_v1(
        scene_bytes, shape=shape_v2.decode_shape_proposal_v2(shape_bytes),
    )
    assert index["consumers_are_independent"] is True
    assert index["source_truth_or_product_pass_authority"] is False
    assert index["shape_proposal_content_sha256"] == sha256(shape_bytes).hexdigest()
    assert index["iris_scene_evidence_content_sha256"] == sha256(scene_bytes).hexdigest()
    assert mesh_consumer.decoded_faces.tolist() == [[0,1,2]]
    assert mesh_consumer.surface_h.shape == (1,3,8)
    assert "rigidity_probability" in mesh_consumer.mechanical
    assert scene_consumer["scene_planes"][0,0,0,0,1] == 1.
    assert scene_consumer["surface_xyz_iris"].shape == (1,3,3)
    assert scene_consumer["extracted_vertex_counts"].tolist() == [4]
    assert scene_consumer["extracted_face_counts"].tolist() == [1]
    assert str(scene_consumer["scene_planes_source_dtype"][0]) == "float32"
    assert scene_consumer["surface_xyz_iris"][0,0,0] == np.float32(1.01)
    assert mesh_consumer.surface_xyz_tessa[0,0,0] == np.float32(.5)
    assert scene_consumer["surface_xyz_iris"][0,0,0] != np.float32(
        2 * mesh_consumer.surface_xyz_tessa[0,0,0]
    )


def test_fanout_rejects_source_parent_swap_index_tamper_and_scene_drift(monkeypatch):
    shape, scene, index = _produced(monkeypatch)
    with pytest.raises(RuntimeError, match="SHAPE_FANOUT_PARENT_BINDING_INVALID"):
        fanout.seal_shape_evidence_fanout_v1(
            shape_bytes=shape, scene_bytes=scene,
            source_observation_content_sha256="c" * 64,
            shape_input_content_sha256="b" * 64,
        )
    with pytest.raises(RuntimeError, match="SHAPE_FANOUT_INDEX_OR_PARENT_DRIFT"):
        fanout.read_shape_evidence_fanout_v1(
            index, shape_bytes=shape, scene_bytes=scene,
            source_observation_content_sha256="a" * 64,
            shape_input_content_sha256="c" * 64,
        )
    mutated=bytearray(index)
    mutated[5] ^= 1
    with pytest.raises(RuntimeError, match="SHAPE_FANOUT_INDEX_"):
        fanout.read_shape_evidence_fanout_v1(
            bytes(mutated), shape_bytes=shape, scene_bytes=scene,
            source_observation_content_sha256="a" * 64,
            shape_input_content_sha256="b" * 64,
        )


def test_iris_sidecar_refuses_to_invent_missing_or_wrong_3d_scene_fields(monkeypatch):
    shape, scene, _ = _produced(monkeypatch)
    from io import BytesIO
    with np.load(BytesIO(scene), allow_pickle=False) as source:
        fields={name:np.asarray(source[name]).copy() for name in source.files}
    fields["surface_xyz_tessa"][0,0,0]=.2
    out=BytesIO()
    np.savez_compressed(out, **fields)
    with pytest.raises(RuntimeError,match="IRIS_TO_TESSA_FRAME_DRIFT"):
        fanout.decode_iris_scene_evidence_v1(
            out.getvalue(),shape=shape_v2.decode_shape_proposal_v2(shape),
        )


def test_registered_shape_worker_emits_three_distinct_artifacts_from_one_inference(monkeypatch):
    from compiler.realsas_compiler_services.platform_worker import shape_executor_v2 as worker
    shape, scene, _ = _produced(monkeypatch)
    calls=[]
    monkeypatch.setattr(worker,"materialize_model_releases",lambda **kwargs:[
        {"family":"IRIS"},{"family":"TESSA"},
    ])
    monkeypatch.setattr(worker,"_device_from_environment",lambda:"cpu")
    monkeypatch.setattr(worker,"_flexicubes_root_from_deployment",lambda: "/tmp/synthetic-only")
    _mock_heavy_flexicubes_for_cpu_contract_court(monkeypatch)
    _mock_shape_input_decoder_without_torch(monkeypatch)
    import compiler.realsas_compiler_services.platform_worker.shape_capability_v2 as sv
    def once(*args,**kwargs):
        calls.append("model")
        assert kwargs["source_observation_content_sha256"]=="a"*64
        return SimpleNamespace(payload=shape, scene_evidence_payload=scene, diagnostics={})
    monkeypatch.setattr(sv,"run_shape_inference_v2",once)
    cas={}
    def put(data):
        digest=sha256(data).hexdigest()
        cas[digest]=data
        return {"storage_key":"cas/"+digest,"content_sha256":digest,"size_bytes":len(data)}
    request={
        "kind":"MODULE","execution_id":"synthetic",
        "capability_id":"product.shape",
        "capability_metadata":{"executor_contract":"RealSaS.ProductShapeInference.v2"},
        "input_artifacts":[{
            "artifact_type":"RealSaS.ShapeInput","schema_version":"v2",
            "content_sha256":"b"*64,"storage_key":"cas/input",
        }],
    }
    result=worker.execute_shape_capability_v2(
        request,read_object=lambda _:b"synthetic-model-input",put_bytes=put,
    )
    assert calls==["model"]
    assert result["status"]=="PASS"
    assert {x["role"] for x in result["outputs"]}=={
        "shape_proposal","iris_scene_evidence","shape_evidence_fanout",
    }
    assert len({x["content_sha256"] for x in result["outputs"]})==3
    index_ref=next(x for x in result["outputs"] if x["role"]=="shape_evidence_fanout")
    index=fanout.read_shape_evidence_fanout_v1(
        cas[index_ref["content_sha256"]],
        shape_bytes=cas[sha256(shape).hexdigest()],
        scene_bytes=cas[sha256(scene).hexdigest()],
        source_observation_content_sha256="a"*64,
        shape_input_content_sha256="b"*64,
    )
    assert index["compiler_canonicalization_does_not_delete_parent_evidence"]
    assert result["diagnostics"]["neural_scene_preserved_for_other_consumers"]


def test_fanout_preserves_bfloat16_scene_values_without_precision_loss():
    torch = pytest.importorskip("torch")
    source = SimpleNamespace(
        scene_planes=torch.tensor(
            np.arange(24, dtype=np.float32).reshape(1, 3, 2, 2, 2),
            dtype=torch.bfloat16,
        ),
        surface_xyz_iris=torch.tensor([[[.25,0,0]]], dtype=torch.bfloat16),
        surface_xyz_tessa=torch.tensor([[[.125,0,0]]], dtype=torch.bfloat16),
        extracted_vertex_counts=(2,),
        extracted_face_counts=(1,),
    )
    raw = fanout.encode_iris_scene_evidence_v1(source)
    assert raw
    # The numeric BF16 values are exactly representable in the stored FP32.
    with np.load(io.BytesIO(raw),allow_pickle=False) as bundle:
        assert bundle["scene_planes"].dtype == np.float32
        assert np.array_equal(
            bundle["scene_planes"],
            source.scene_planes.float().cpu().numpy(),
        )
        assert str(bundle["scene_planes_source_dtype"][0]) == "torch.bfloat16"
        assert bundle["extracted_vertex_counts"].tolist() == [2]


def test_registered_mesh_consumes_index_but_never_decompresses_unused_neural_scene(monkeypatch):
    from compiler.realsas_compiler_services.platform_worker.mesh_executor_v1 import (
        validate_product_shape_fanout_refs_v1,
    )
    shape_bytes, scene_bytes, index_bytes = _produced(monkeypatch)
    proposal = shape_v2.decode_shape_proposal_v2(shape_bytes)
    shape_ref = {
        "artifact_type": "RealSaS.ShapeProposal", "schema_version": "v2",
        "content_sha256": sha256(shape_bytes).hexdigest(),
    }
    scene_ref = {
        "artifact_type": "RealSaS.IRISSceneEvidence", "schema_version": "v1",
        "content_sha256": sha256(scene_bytes).hexdigest(),
    }
    index_ref = {
        "artifact_type": "RealSaS.ShapeEvidenceFanout", "schema_version": "v1",
        "content_sha256": sha256(index_bytes).hexdigest(),
    }
    source_ref = {
        "artifact_type": "RealSaS.SourceObservation", "schema_version": "v1",
        "content_sha256": "a" * 64,
    }
    request = {"input_artifacts": [shape_ref, scene_ref, index_ref, source_ref]}
    touched = []

    def read(ref):
        touched.append(ref["artifact_type"])
        if ref["artifact_type"] == "RealSaS.ShapeEvidenceFanout":
            return index_bytes
        raise AssertionError("Product Mesh must not reload raw neural scene")

    bound = validate_product_shape_fanout_refs_v1(
        request, shape=proposal, shape_ref=shape_ref, source_ref=source_ref,
        read_object=read,
    )
    assert touched == ["RealSaS.ShapeEvidenceFanout"]
    assert bound["iris_scene_evidence_content_sha256"] == sha256(scene_bytes).hexdigest()
    assert bound["source_truth_or_product_pass_authority"] is False

    with pytest.raises(RuntimeError,match="SHAPE_FANOUT_INDEX_CAS_PARENT_DRIFT"):
        validate_product_shape_fanout_refs_v1(
            request, shape=proposal, shape_ref=shape_ref,
            source_ref={**source_ref,"content_sha256":"c"*64}, read_object=read,
        )
    with pytest.raises(RuntimeError,match="SHAPE_FANOUT_INDEX_CAS_PARENT_DRIFT"):
        validate_product_shape_fanout_refs_v1(
            {"input_artifacts":[shape_ref,{**scene_ref,"content_sha256":"d"*64},index_ref]},
            shape=proposal,shape_ref=shape_ref,source_ref=source_ref,read_object=read,
        )


def test_registered_shape_executor_fails_closed_if_scene_is_not_produced(monkeypatch):
    from compiler.realsas_compiler_services.platform_worker import shape_executor_v2 as worker
    shape, scene, _ = _produced(monkeypatch)
    monkeypatch.setattr(worker, "materialize_model_releases", lambda **kwargs: [
        {"family":"IRIS"}, {"family":"TESSA"},
    ])
    monkeypatch.setattr(worker,"_device_from_environment",lambda:"cpu")
    monkeypatch.setattr(worker,"_flexicubes_root_from_deployment",lambda:"/tmp/synthetic-only")
    _mock_heavy_flexicubes_for_cpu_contract_court(monkeypatch)
    import compiler.realsas_compiler_services.platform_worker.scene_input_contract_v2 as si
    monkeypatch.setattr(si,"decode_shape_input_v2",lambda _:SimpleNamespace(
        inputs=object(),source_observation_content_sha256="a"*64,
    ))
    import compiler.realsas_compiler_services.platform_worker.shape_capability_v2 as sv
    monkeypatch.setattr(sv,"run_shape_inference_v2",lambda *args,**kwargs:SimpleNamespace(
        payload=shape,scene_evidence_payload=None,diagnostics={},
    ))
    saved=[]
    with pytest.raises(RuntimeError,match="SHAPE_V2_IRIS_SCENE_EVIDENCE_MISSING"):
        worker.execute_shape_capability_v2({
            "kind":"MODULE","execution_id":"fake",
            "capability_metadata":{"executor_contract":"RealSaS.ProductShapeInference.v2"},
            "input_artifacts":[{
                "artifact_type":"RealSaS.ShapeInput","schema_version":"v2",
                "content_sha256":"b"*64,
            }],
        },read_object=lambda ref:b"synthetic",put_bytes=lambda data:saved.append(data))
    assert saved == []


def test_scene_sidecar_rejects_rehashed_false_extraction_counts(monkeypatch):
    shape, scene, _ = _produced(monkeypatch)
    with np.load(io.BytesIO(scene),allow_pickle=False) as archive:
        fields={name:np.asarray(archive[name]).copy() for name in archive.files}
    fields["extracted_face_counts"][0]=0
    stream=io.BytesIO()
    np.savez_compressed(stream,**fields)
    with pytest.raises(RuntimeError,match="SHAPE_FANOUT_EXTRACTION_COUNTS_INVALID"):
        fanout.decode_iris_scene_evidence_v1(
            stream.getvalue(),shape=shape_v2.decode_shape_proposal_v2(shape),
        )
