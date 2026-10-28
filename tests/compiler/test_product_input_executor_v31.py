from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_services.platform_worker import input_executor_v31 as executor


def test_input_v31_executor_seals_versioned_shape_input_without_activation(monkeypatch, tmp_path):
    source_ref = {
        "id": "source-id",
        "artifact_type": "RealSaS.SourceObservation",
        "schema_version": "v1",
        "storage_key": "cas/source",
        "content_sha256": "a" * 64,
        "size_bytes": 6,
    }
    request = {
        "execution_id": "11111111-1111-4111-8111-111111111111",
        "capability_id": "product.input.v31.shadow",
        "kind": "MODULE",
        "capability_metadata": {"executor_contract": "RealSaS.ProductInputInference.v3_1"},
        "input_artifacts": [source_ref],
    }

    deploy_root = tmp_path / "platform"
    monkeypatch.setenv("REALSAS_DEPLOY_ROOT", str(deploy_root))
    monkeypatch.setattr(executor, "decode_source_observation_v1", lambda payload: object())
    monkeypatch.setattr(executor, "_device_from_environment", lambda: "cpu")
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return object(), object()

    monkeypatch.setattr(executor, "load_product_dino_v1", load)

    def run(source, **kwargs):
        assert kwargs["source_content_sha256"] == "a" * 64
        assert kwargs["device"] == "cpu"
        return SimpleNamespace(
            payload=b"shape-input-v31",
            diagnostics={
                "schema": "RealSaS.ProductInputInferenceDiagnostics.v3_1",
                "source_truth_copied_into_shape": False,
                "observation_admission_minted": False,
                "model_release_minted": False,
                "product_shape_activated": False,
            },
        )

    monkeypatch.setattr(executor, "run_input_inference_v31", run)

    def read_object(ref):
        assert ref == source_ref
        return b"source"

    def put_bytes(payload):
        assert payload == b"shape-input-v31"
        return {"storage_key": "cas/shape-input-v31", "content_sha256": "b" * 64, "size_bytes": 15}

    result = executor.execute_input_capability_v31(
        request,
        read_object=read_object,
        put_bytes=put_bytes,
    )
    output = result["outputs"][0]
    assert result["status"] == "PASS"
    assert output["artifact_type"] == "RealSaS.ShapeInput"
    assert output["schema_version"] == "v3_1"
    assert output["authority_class"] == "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"
    assert result["diagnostics"]["product_shape_activated"] is False
    assert result["diagnostics"]["model_release_minted"] is False
    assert seen["source_dir"] == deploy_root.resolve() / "foundations/dinov2/source"
    assert seen["weight_path"] == deploy_root.resolve() / "foundations/dinov2/dinov2_vits14_pretrain.pth"
    assert seen["device"] == "cpu"


def test_input_v31_executor_requires_exact_one_source_observation():
    try:
        executor._one_source_input({"input_artifacts": []})
    except RuntimeError as exc:
        assert "PRODUCT_INPUT_V31_SOURCE_ARTIFACT_CARDINALITY:0" in str(exc)
    else:
        raise AssertionError("missing SourceObservation must fail closed")


def test_input_v31_foundation_locator_rejects_ambient_paths(monkeypatch):
    monkeypatch.delenv("REALSAS_DEPLOY_ROOT", raising=False)
    monkeypatch.setenv("REALSAS_DINOV2_SOURCE_DIR", "/ambient/source")
    monkeypatch.setenv("REALSAS_DINOV2_WEIGHT_PATH", "/ambient/weights")
    try:
        executor._dino_paths_from_deployment_root()
    except RuntimeError as exc:
        assert str(exc) == "PRODUCT_INPUT_V31_DEPLOY_ROOT_REQUIRED"
    else:
        raise AssertionError("ambient DINO path overrides must not become V3.1 authority")
