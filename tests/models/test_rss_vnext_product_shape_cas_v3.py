from __future__ import annotations

"""Real RSS unified checkpoint -> product candidate CAS fan-out integration.

Torch tiny untrained TEST backend proves the exact execution ABI, never
trained/unseen competence or product-mode vision admissibility. Production
uses a local-only FrozenDINOv2 backend and requires a release pin.
"""

from dataclasses import replace
from hashlib import sha256
import json
import uuid

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    canonicalize_two_view_source_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    encode_rss_vnext_two_view_source_v1,
    decode_rss_vnext_two_view_source_v1,
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_shape_executor_v3 import (
    EXECUTOR_CONTRACT, FANOUT_SCHEMA,
    execute_rss_vnext_shape_capability_v3,
)
from compiler.realsas_compiler_services.orchestrator import mainline
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_checkpoint_v1 import (
    save_inference_checkpoint_v1,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(447)
    torch.set_num_threads(1)


def _source():
    native = np.full((2, 4, 28, 28), 120, dtype=np.uint8)
    native[:, 3] = 255
    native[1, 0, 0, 0] = 222
    cameras = np.zeros((2, 13), dtype=np.float32)
    cameras[0, [3, 7, 11, 12]] = 1.
    cameras[1, [3, 7, 12]] = [ -1., 1., 1. ]
    cameras[1, 11] = -1.
    return canonicalize_two_view_source_v1(
        native, cameras, source_rights_claim_sha256="d" * 64,
    )


def _make_executor_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(mainline, "authority_root", lambda: tmp_path / "run-authority")
    src = _source()
    raw_source = encode_rss_vnext_two_view_source_v1(src)
    assert decode_rss_vnext_two_view_source_v1(raw_source).source_binding_sha256 == src.source_binding_sha256
    cas = {}
    def put(payload: bytes):
        digest = sha256(payload).hexdigest()
        cas[digest] = payload
        return {
            "storage_key": "cas/" + digest,
            "content_sha256": digest, "size_bytes": len(payload),
        }
    def read(ref):
        digest = ref["content_sha256"]
        payload = cas[digest]
        assert sha256(payload).hexdigest() == digest
        return payload
    source_ref = {
        **put(raw_source), "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
    }
    netcfg = RSSvNextConfig.tiny_for_test()
    network = RSSvNextUnifiedCharacterModel(
        netcfg,
        vision_encoder=RSSvNextPatchEncoderForTests(netcfg.iris.image_token_dim),
    ).eval()
    checkpoint_path = tmp_path / "unified_checkpoint.pt"
    expected = save_inference_checkpoint_v1(checkpoint_path, network)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_ref = put(checkpoint_bytes)
    model_release = {
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
        "model_release_id": str(uuid.uuid4()),
        "model_engine_release_id": str(uuid.uuid4()),
        "source_code_sha": "b" * 40,
        "checkpoint": {
            **checkpoint_ref,
            "id": str(uuid.uuid4()),
            "role": "model:RSSVNEXT",
            "artifact_type": "RealSaS.RSSvNextUnifiedInferenceCheckpoint",
            "schema_version": "v1",
            "semantic_sha256": "c" * 64,
        },
    }
    request = {
        "execution_id": "test-rss-vnext-only",
        "capability_id": "product.shape",
        "kind": "MODULE",
        "capability_metadata": {
            "executor_contract": EXECUTOR_CONTRACT,
            "maximum_new_tokens": 16,
        },
        "input_artifacts": [source_ref],
        "model_releases": [model_release],
    }
    def vision():
        return RSSvNextPatchEncoderForTests(netcfg.iris.image_token_dim)
    return src, cas, request, read, put, vision, expected, netcfg


def _run(tmp_path, monkeypatch):
    source, cas, request, read, put, vision, expected, config = (
        _make_executor_fixture(tmp_path, monkeypatch)
    )
    result = execute_rss_vnext_shape_capability_v3(
        request, read_object=read, put_bytes=put,
        test_vision_encoder_factory=vision,
    )
    return source, cas, result, request, read, put, vision, expected, config


def test_two_original_source_cas_retains_camera_alpha_and_rejects_relabelled_legacy():
    src = _source()
    data = encode_rss_vnext_two_view_source_v1(src)
    ref = {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "content_sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    decoded = validate_rss_vnext_source_ref_v1(ref, data)
    assert np.array_equal(decoded.native_rgba, src.native_rgba)
    assert np.array_equal(decoded.camera_features, src.camera_features)
    assert decoded.source_binding_sha256 == src.source_binding_sha256
    with pytest.raises(RuntimeError, match="REF_CONTENT_DRIFT"):
        validate_rss_vnext_source_ref_v1({
            **ref, "content_sha256": "b" * 64,
        }, data)
    with pytest.raises(RuntimeError, match="REF_SCHEMA_INVALID"):
        validate_rss_vnext_source_ref_v1({
            **ref, "artifact_type": "RealSaS.SourceObservation",
        }, data)
    with pytest.raises(RuntimeError, match="CAS_ARCHIVE_INVALID|CAS_DECODE_INVALID"):
        decode_rss_vnext_two_view_source_v1(b"corrupted")


def test_one_rss_checkpoint_emits_six_independent_cas_outputs_even_on_abstain(tmp_path, monkeypatch):
    src, cas, result, request, read, put, vision, checkpoint, config = (
        _run(tmp_path, monkeypatch)
    )
    assert result["status"] == "PASS"
    assert result["failure"] is None
    assert result["diagnostics"]["unified_inference_count"] == 1
    assert result["diagnostics"]["legacy_iris_tessa_model_releases_required"] is False
    assert result["diagnostics"]["original_observed_view_count"] == 2
    assert result["diagnostics"]["product_mode_test_only"] is True
    assert result["diagnostics"]["product_revision_minted"] is False
    by_role = {row["role"]: row for row in result["outputs"]}
    assert set(by_role) == {
        "original_source", "normalized_model_source", "neural_evidence",
        "neural_scene", "inference_receipt", "rss_evidence_fanout",
    }
    assert len({row["content_sha256"] for row in result["outputs"]}) == len(by_role)
    index = json.loads(read(by_role["rss_evidence_fanout"]))
    assert index["schema"] == FANOUT_SCHEMA
    assert index["unified_checkpoint_sha256"] == checkpoint
    assert index["source_binding_sha256"] == src.source_binding_sha256
    assert index["source_artifact_sha256"] == request["input_artifacts"][0]["content_sha256"]
    assert index["learned_evidence_is_not_source_truth"] is True
    assert not index["original_source_rights_verified"]
    assert not index["canonical_mesh_authority"]
    assert index["topology_abstain_retains_shared_carrier"]
    assert len(index["children"]) == 5
    assert {x["content_sha256"] for x in index["children"]} == {
        by_role[x["role"]]["content_sha256"] for x in index["children"]
    }

    receipt = json.loads(read(by_role["inference_receipt"]))
    evidence = json.loads(read(by_role["neural_evidence"]))
    assert receipt["checkpoint_sha256"] == checkpoint
    assert receipt["source_rights_claim_sha256"] == src.source_rights_claim_sha256
    assert receipt["product_revision_minted"] is False
    assert evidence["mesh_status"] == index["mesh_status"]
    assert receipt["evidence_hash"] == evidence["evidence_hash"]
    # The entire RSS common H(x) remains available to multiple consumers
    # even when the untrained topology head cannot complete a mesh.
    from models.rss_vnext.neural_scene_archive_v1 import (
        read_neural_scene_carrier_v1,
    )
    from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
        RSSvNextInferenceEvidenceIRV1,
    )
    tensors = read_neural_scene_carrier_v1(
        read(by_role["neural_scene"]),
        evidence=RSSvNextInferenceEvidenceIRV1(**evidence),
    )
    assert tensors["local_features"].shape[0] == 1
    assert tensors["scene_triplanes"].shape[1] == 3
    assert tensors["region_relation_logits"].shape[-1] == 3
    assert tensors["graphics_logits"].shape[-1] == 5
    assert torch.equal(tensors["geometry__H"], tensors["local_features"])
    assert len(cas) == 8  # original canonical source + checkpoint + six child CAS


def test_wrong_model_family_and_untrusted_source_fail_before_inference(tmp_path, monkeypatch):
    src, cas, request, read, put, vision, checkpoint, config = (
        _make_executor_fixture(tmp_path, monkeypatch)
    )
    wrong = {
        **request, "model_releases": [
            {**request["model_releases"][0], "family": "IRIS"},
        ],
    }
    with pytest.raises(RuntimeError, match="ONE_UNIFIED_MODEL_RELEASE_REQUIRED"):
        execute_rss_vnext_shape_capability_v3(
            wrong, read_object=read, put_bytes=put,
            test_vision_encoder_factory=vision,
        )
    with pytest.raises(RuntimeError, match="REF_CONTENT_DRIFT"):
        execute_rss_vnext_shape_capability_v3(
            {**request, "input_artifacts": [{
                **request["input_artifacts"][0],
                "content_sha256": "a" * 64,
            }]},
            read_object=lambda _: read(request["input_artifacts"][0]),
            put_bytes=put, test_vision_encoder_factory=vision,
        )
    with pytest.raises(RuntimeError, match="LEGACY_SOURCE_FORBIDDEN"):
        execute_rss_vnext_shape_capability_v3(
            {**request, "input_artifacts": request["input_artifacts"] + [{
                "artifact_type": "RealSaS.SourceObservation", "schema_version": "v1",
            }]}, read_object=read, put_bytes=put,
            test_vision_encoder_factory=vision,
        )
    with pytest.raises(RuntimeError, match="CALLER_CANNOT_DISABLE_PRODUCT_MODE"):
        execute_rss_vnext_shape_capability_v3({
            **request, "goal_parameters": {"product_mode": False},
        }, read_object=read, put_bytes=put,
            test_vision_encoder_factory=vision,
        )


def test_real_production_worker_fails_closed_if_pinned_local_dino_is_unavailable(tmp_path, monkeypatch):
    src, cas, request, read, put, vision, checkpoint, config = (
        _make_executor_fixture(tmp_path, monkeypatch)
    )
    monkeypatch.setenv(
        "REALSAS_RSS_DINOV2_SOURCE", str(tmp_path / "missing-pretrained-model"),
    )
    with pytest.raises(RuntimeError, match="LOCAL_DINOV2_NOT_FOUND"):
        execute_rss_vnext_shape_capability_v3(
            request, read_object=read, put_bytes=put,
        )



def test_lossless_registered_style_input_to_single_checkpoint_shape_real_e2e(tmp_path, monkeypatch):
    from compiler.realsas_compiler_services.platform_worker.rss_vnext_input_executor_v2 import (
        EXECUTOR_CONTRACT as INPUT_CONTRACT,
        execute_rss_vnext_input_capability_v2,
    )
    source, cas, shape_request, read, put, vision, checkpoint, config = (
        _make_executor_fixture(tmp_path, monkeypatch)
    )
    source_ref = shape_request["input_artifacts"][0]
    input_request = {
        "kind": "MODULE", "capability_id": "product.input",
        "execution_id": "unit-two-view",
        "capability_metadata": {"executor_contract": INPUT_CONTRACT},
        "input_artifacts": [source_ref],
        "model_releases": [],
    }
    input_result = execute_rss_vnext_input_capability_v2(
        input_request, read_object=read, put_bytes=put,
    )
    assert input_result["status"] == "PASS"
    assert len(input_result["outputs"]) == 1
    forwarded = input_result["outputs"][0]
    assert forwarded["content_sha256"] == source_ref["content_sha256"]
    assert forwarded["size_bytes"] == source_ref["size_bytes"]
    assert read(forwarded) == read(source_ref)
    assert input_result["diagnostics"]["no_reencoding_or_resampling"]
    assert not input_result["diagnostics"]["independent_dino_inference_performed"]
    # The v2 Input output is the EXACT typed artifact expected by RSS Shape,
    # not a lossy eight-view ShapeInput feature proxy.
    shape_result = execute_rss_vnext_shape_capability_v3(
        {**shape_request, "input_artifacts": [forwarded]},
        read_object=read, put_bytes=put,
        test_vision_encoder_factory=vision,
    )
    assert shape_result["status"] == "PASS"
    assert shape_result["diagnostics"]["unified_inference_count"] == 1
    index_ref = next(x for x in shape_result["outputs"] if x["role"] == "rss_evidence_fanout")
    index = json.loads(read(index_ref))
    assert index["source_artifact_sha256"] == forwarded["content_sha256"]
    assert index["source_binding_sha256"] == source.source_binding_sha256
    assert index["unified_checkpoint_sha256"] == checkpoint
    assert all(row["content_sha256"] in cas for row in index["children"])
    with pytest.raises(RuntimeError, match="EXACT_TWO_VIEW_SOURCE_REQUIRED"):
        execute_rss_vnext_input_capability_v2(
            {**input_request, "input_artifacts": [{
                **source_ref, "artifact_type": "RealSaS.SourceObservation",
            }]}, read_object=read, put_bytes=put,
        )
    with pytest.raises(RuntimeError, match="MODEL_RELEASE_FORBIDDEN"):
        execute_rss_vnext_input_capability_v2(
            {**input_request, "model_releases": shape_request["model_releases"]},
            read_object=read, put_bytes=put,
        )
