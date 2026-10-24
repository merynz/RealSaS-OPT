from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from compiler.realsas_compiler_services.platform_worker import input_executor_v1 as executor


def test_input_executor_seals_shape_input_without_product_authority(monkeypatch, tmp_path):
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
        return b"source"

    def put_bytes(payload):
        assert payload == b"shape-input"
        return {"storage_key": "cas/shape-input", "content_sha256": "b" * 64, "size_bytes": 11}

    result = executor.execute_input_capability_v1(
        request, read_object=read_object, put_bytes=put_bytes
    )
    assert result["status"] == "PASS"
    assert result["outputs"][0]["artifact_type"] == "RealSaS.ShapeInput"
    assert result["outputs"][0]["authority_class"] == "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"
    assert result["diagnostics"]["platform_view_count_invariant_claimed"] is False
    assert result["diagnostics"]["source_truth_copied_into_shape"] is False
    assert result["diagnostics"]["source_content_sha256"] == "a" * 64
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
