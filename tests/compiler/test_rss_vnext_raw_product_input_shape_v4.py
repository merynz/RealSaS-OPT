from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
    encode_uncalibrated_two_image_source_v1,
)
from compiler.rss_vnext.workers.rss_vnext_input_executor_v3 import (
    RAW_ARTIFACT_TYPE, EXECUTOR_CONTRACT as INPUT_V3,
    execute_rss_vnext_input_capability_v3,
)
from compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 import (
    EXECUTOR_CONTRACT as SHAPE_V4,
    execute_rss_vnext_shape_capability_v4,
)
from tests.models.test_rss_vnext_uncalibrated_neural_scene_v1 import _bundle


class MemoryCAS:
    def __init__(self):
        self.objects = {}

    def put_bytes(self, blob: bytes):
        h = sha256(blob).hexdigest()
        self.objects[h] = blob
        return {"storage_key": h, "content_sha256": h, "size_bytes": len(blob)}

    def read_object(self, ref):
        return self.objects[ref["storage_key"]]


def _png(color, size):
    buffer = io.BytesIO()
    Image.new("RGBA", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _source(cas: MemoryCAS):
    original = (
        _png((120, 35, 10, 255), (46, 32)),
        _png((9, 66, 188, 250), (22, 57)),
    )
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=original, source_rights_claim_sha256="a" * 64,
        view_claims=("FRONT", "BACK"),
    )
    packed = encode_uncalibrated_two_image_source_v1(source)
    return {
        **cas.put_bytes(packed),
        "artifact_type": RAW_ARTIFACT_TYPE,
        "schema_version": "v1",
    }, original


def _input_req(ref):
    return {
        "capability_id": "product.input",
        "kind": "MODULE",
        "execution_id": "raw-input-test",
        "model_releases": [],
        "capability_metadata": {"executor_contract": INPUT_V3},
        "input_artifacts": [ref],
    }


def _shape_req(ref):
    return {
        "capability_id": "product.shape",
        "kind": "MODULE",
        "execution_id": "raw-shape-test",
        "capability_metadata": {
            "executor_contract": SHAPE_V4,
            "maximum_new_tokens": 6,
        },
        "goal_parameters": {"product_mode": True},
        "input_artifacts": [ref],
        "model_releases": [{
            "family": "RSSVNEXT",
            "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
        }],
    }


def test_input_role_copies_exact_original_cas_and_requires_no_camera():
    cas = MemoryCAS()
    ref, original = _source(cas)
    result = execute_rss_vnext_input_capability_v3(
        _input_req(ref), read_object=cas.read_object,
        put_bytes=cas.put_bytes,
    )
    assert result["status"] == "PASS"
    output = result["outputs"][0]
    assert output["artifact_type"] == RAW_ARTIFACT_TYPE
    assert output["content_sha256"] == ref["content_sha256"]
    assert cas.read_object(output) == cas.read_object(ref)
    assert result["diagnostics"]["source_camera_calibrated"] is False
    assert result["diagnostics"]["generated_view_count"] == 0
    assert result["diagnostics"]["source_rights_verified"] is False
    assert result["diagnostics"]["product_revision_minted"] is False
    assert len(original) == 2


def test_raw_shape_runs_two_actual_generators_and_persists_only_candidate_cas(
    tmp_path, monkeypatch,
):
    torch.manual_seed(420)
    torch.set_num_threads(1)
    cas = MemoryCAS()
    ref, _ = _source(cas)
    bundle = _bundle(tmp_path)
    bundle.model._pinned_neural_bundle = bundle
    checkpoint = tmp_path / "single-release.pt"
    checkpoint.write_bytes(b"test-only-byte-pinned-injected-by-test")
    release = [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
        "behavior_identity_sha256": "c" * 64,
        "checkpoint": {
            "artifact_type": "RealSaS.RSSvNextUnifiedInferenceCheckpoint",
            "schema_version": "v2",
            "path": str(checkpoint),
            "content_sha256": "b" * 64,
        },
    }]
    import compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 as executor
    import models.rss_vnext.inference_checkpoint_v1 as loader

    monkeypatch.setattr(executor, "materialize_model_releases", lambda **kwargs: release)
    monkeypatch.setattr(
        loader, "load_inference_checkpoint_v1",
        lambda *args, **kwargs: bundle.model,
    )
    raw_request = _shape_req(ref)
    # Go may present original and product.input's byte-identical output as
    # distinct semantic artifact IDs. The candidate executor must require
    # byte identity and never treat the copied bytes as a third observation.
    raw_request["input_artifacts"].append({**ref, "artifact_id": "copy"})
    result = execute_rss_vnext_shape_capability_v4(
        raw_request, read_object=cas.read_object,
        put_bytes=cas.put_bytes,
        test_vision_encoder_factory=lambda: object(),
    )
    assert result["status"] == "PASS"
    assert result["failure"] is None
    assert len(result["outputs"]) == 3
    assert set(r["artifact_type"] for r in result["outputs"]) == {
        "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier",
        "RealSaS.RSSvNextGeneratedConditioningViews",
        "RealSaS.RSSvNextUncalibratedConstructionReceipt",
    }
    assert not any(
        x["artifact_type"] == "RealSaS.RSSvNextInferenceEvidence"
        for x in result["outputs"]
    )
    receipt = json.loads(cas.read_object(result["outputs"][2]))
    assert receipt["original_camera_calibrated"] is False
    assert receipt["generated_view_count"] == 6
    assert receipt["original_pixel_count"] == 2
    assert receipt["source_ray_evidence_qualified"] is False
    assert receipt["product_revision_minted"] is False
    with np.load(io.BytesIO(cas.read_object(result["outputs"][0]))) as archive:
        assert archive["scene_triplanes"].shape[1] == 3
        assert np.isfinite(archive["local_features"]).all()
        assert archive["graphics_logits"].shape[-1] == 5
    with np.load(io.BytesIO(cas.read_object(result["outputs"][1]))) as archive:
        assert archive["generated_rgba"].shape[1] == 6
        assert archive["generated_normals"].shape[1] == 6
        assert archive["generated_virtual_cameras"].shape == (1, 6, 13)


def test_raw_and_calibrated_sources_cannot_be_mixed_by_shape():
    cas = MemoryCAS()
    raw, _ = _source(cas)
    req = _shape_req(raw)
    req["input_artifacts"].append({
        "artifact_type": "RealSaS.RSSvNextTwoViewSource",
        "schema_version": "v1",
        **cas.put_bytes(b"forged-calibrated-source"),
    })
    with pytest.raises(RuntimeError, match="TWO_DIFFERENT_SOURCE_AUTHORITIES"):
        execute_rss_vnext_shape_capability_v4(
            req, read_object=cas.read_object, put_bytes=cas.put_bytes,
            test_vision_encoder_factory=lambda: object(),
        )


def test_raw_source_cas_tampering_is_rejected_before_checkpoint_loading():
    cas = MemoryCAS()
    raw, _ = _source(cas)
    req = _shape_req({**raw, "content_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="RAW_CAS_DRIFT"):
        execute_rss_vnext_shape_capability_v4(
            req, read_object=cas.read_object, put_bytes=cas.put_bytes,
            test_vision_encoder_factory=lambda: object(),
        )


def test_calibrated_source_delegates_to_exact_v3_path(monkeypatch):
    import compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 as executor
    sentinel = {"capability_id": "product.shape", "status": "PASS", "outputs": []}
    calls = []

    def fake_v3(req, **kwargs):
        calls.append((req, kwargs))
        return sentinel

    monkeypatch.setattr(executor, "execute_rss_vnext_shape_capability_v3", fake_v3)
    cas = MemoryCAS()
    ref = {
        "artifact_type": "RealSaS.RSSvNextTwoViewSource",
        "schema_version": "v1",
        **cas.put_bytes(b"calibrated-by-another-admission-court"),
    }
    res = execute_rss_vnext_shape_capability_v4(
        _shape_req(ref), read_object=cas.read_object,
        put_bytes=cas.put_bytes, test_vision_encoder_factory=lambda: object(),
    )
    assert res is sentinel
    assert len(calls) == 1
    assert calls[0][0]["capability_metadata"]["executor_contract"] == (
        "RealSaS.ProductShapeInference.v3"
    )
