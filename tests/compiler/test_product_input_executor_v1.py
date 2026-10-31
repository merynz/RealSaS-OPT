from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_services.platform_worker import input_executor_v1 as executor
from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    encode_source_observation_arrays_v1,
)


def _source_observation_bytes() -> bytes:
    rgba = np.zeros((1, 8, 4, 1024, 1024), dtype=np.uint8)
    cameras = []
    for view in range(8):
        rgba[0, view, :3, 100 + view : 120 + view, 200:240] = 25 + view
        rgba[0, view, 3, 100 + view : 120 + view, 200:240] = 255
        cameras.append(
            {
                "view_index": view,
                "origin": [float(view), 0.0, 2.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 0.0, 1.0],
                "forward": [0.0, 1.0, 0.0],
                "half_extent": 2.0,
                "resolution": 1024,
            }
        )
    return encode_source_observation_arrays_v1(
        native_rgba=rgba,
        camera_bundle={
            "schema": "RealSaS.CameraProjectionBundle.v1",
            "view_order": [f"V{view}" for view in range(8)],
            "common_center_xyz": [0.0, 0.0, 0.0],
            "common_half_extent": 2.0,
            "per_view_recenter": False,
            "per_view_pose_change": False,
            "cameras": cameras,
        },
    )


def test_input_executor_seals_shape_input_without_product_authority(monkeypatch, tmp_path):
    source_payload = _source_observation_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    source_ref = {
        "id": "source-id",
        "artifact_type": "RealSaS.SourceObservation",
        "schema_version": "v1",
        "storage_key": "cas/source",
        "content_sha256": source_sha256,
        "size_bytes": len(source_payload),
    }
    request = {
        "execution_id": "11111111-1111-4111-8111-111111111111",
        "capability_id": "product.input",
        "kind": "MODULE",
        "capability_metadata": {"executor_contract": "RealSaS.ProductInputInference.v1"},
        "input_artifacts": [source_ref],
    }

    deploy_root = tmp_path / "platform"
    monkeypatch.setenv("REALSAS_DEPLOY_ROOT", str(deploy_root))
    monkeypatch.setenv("REALSAS_DINOV2_SOURCE_DIR", "/must/not/be/used")
    monkeypatch.setenv("REALSAS_DINOV2_WEIGHT_PATH", "/must/not/be/used")
    monkeypatch.setattr(executor, "decode_source_observation_v1", lambda payload: object())
    monkeypatch.setattr(executor, "_device_from_environment", lambda: "cpu")
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return object(), object()

    monkeypatch.setattr(executor, "load_product_dino_v1", load)
    monkeypatch.setattr(
        executor,
        "run_input_inference_v1",
        lambda source, **kwargs: SimpleNamespace(
            payload=b"shape-input",
            diagnostics={
                "schema": "RealSaS.ProductInputInferenceDiagnostics.v1",
                "platform_view_count_invariant_claimed": False,
                "source_truth_copied_into_shape": False,
            },
        ),
    )

    def read_object(ref):
        assert ref == source_ref
        return source_payload

    sealed_payloads = []

    def put_bytes(payload):
        sealed_payloads.append(payload)
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "storage_key": f"cas/{digest}",
            "content_sha256": digest,
            "size_bytes": len(payload),
        }

    result = executor.execute_input_capability_v1(
        request, read_object=read_object, put_bytes=put_bytes
    )
    assert result["status"] == "PASS"
    assert [row["artifact_type"] for row in result["outputs"]] == [
        "RealSaS.ShapeInput",
        "RealSaS.QualifiedCameraSetIR",
        "RealSaS.QualifiedObservationSetIR",
        "RealSaS.NormalizationDomainIR",
    ]
    assert result["outputs"][0]["authority_class"] == "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"
    assert result["diagnostics"]["platform_view_count_invariant_claimed"] is False
    assert result["diagnostics"]["source_truth_copied_into_shape"] is False
    assert result["diagnostics"]["source_content_sha256"] == source_sha256
    assert result["diagnostics"]["legacy_compiler_stage_bridge_required"] is False
    assert result["diagnostics"]["product_input_output_count"] == 4
    assert len(sealed_payloads) == 4
    assert sealed_payloads[0] == b"shape-input"
    assert seen["source_dir"] == deploy_root.resolve() / "foundations/dinov2/source"
    assert seen["weight_path"] == deploy_root.resolve() / "foundations/dinov2/dinov2_vits14_pretrain.pth"
    assert seen["device"] == "cpu"


def test_input_executor_requires_exact_one_typed_source():
    try:
        executor._one_source_input({"input_artifacts": []})
    except RuntimeError as exc:
        assert "PRODUCT_INPUT_V1_SOURCE_ARTIFACT_CARDINALITY:0" in str(exc)
    else:
        raise AssertionError("missing SourceObservation must fail closed")


def test_product_input_foundation_locator_requires_deployment_root(monkeypatch):
    monkeypatch.delenv("REALSAS_DEPLOY_ROOT", raising=False)
    monkeypatch.setenv("REALSAS_DINOV2_SOURCE_DIR", "/ambient/source")
    monkeypatch.setenv("REALSAS_DINOV2_WEIGHT_PATH", "/ambient/weights")
    try:
        executor._dino_paths_from_deployment_root()
    except RuntimeError as exc:
        assert str(exc) == "PRODUCT_INPUT_V1_DEPLOY_ROOT_REQUIRED"
    else:
        raise AssertionError("ambient DINO path overrides must not become product authority")
