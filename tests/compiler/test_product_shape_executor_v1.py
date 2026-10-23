from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_services.platform_worker import shape_executor_v1 as executor


def test_shape_executor_materializes_attempt_models_and_seals_proposal(monkeypatch):
    seen = {}
    model_rows = [
        {"family": "IRIS", "checkpoint": {"content_sha256": "a" * 64}},
        {"family": "TESSA", "checkpoint": {"content_sha256": "b" * 64}},
    ]
    shape_ref = {
        "id": "input-id",
        "artifact_type": "RealSaS.ShapeInput",
        "schema_version": "v1",
        "storage_key": "cas/input",
        "content_sha256": "c" * 64,
        "size_bytes": 5,
    }
    request = {
        "execution_id": "11111111-1111-4111-8111-111111111111",
        "capability_id": "product.shape",
        "kind": "MODULE",
        "capability_metadata": {"executor_contract": "RealSaS.ProductShapeInference.v1"},
        "model_releases": model_rows,
        "input_artifacts": [shape_ref],
    }

    def materialize(*, manifest, models, read_object):
        seen["manifest"] = manifest
        seen["models"] = models
        manifest["_platform_model_releases"] = [{"family": "IRIS"}, {"family": "TESSA"}]
        manifest["_platform_model_snapshot_sha256"] = "d" * 64
        return manifest["_platform_model_releases"]

    monkeypatch.setattr(executor, "materialize_model_releases", materialize)
    monkeypatch.setattr(executor, "decode_shape_input_v1", lambda payload: object())
    monkeypatch.setattr(executor, "_device_from_environment", lambda: "cpu")
    monkeypatch.setattr(executor, "flexicubes_root_from_environment_v1", lambda: "/tmp/flex")
    monkeypatch.setattr(executor, "PinnedFlexiCubesV1", lambda root, device: object())
    monkeypatch.setattr(
        executor,
        "run_shape_inference_v1",
        lambda manifest, inputs, **kwargs: SimpleNamespace(
            payload=b"shape",
            diagnostics={
                "schema": "RealSaS.ShapeInferenceDiagnostics.v1",
                "canonical_mesh_authority": False,
                "timings": {"total_seconds": 0.25},
            },
        ),
    )

    def read_object(ref):
        assert ref == shape_ref
        return b"input"

    def put_bytes(payload):
        assert payload == b"shape"
        return {"storage_key": "cas/shape", "content_sha256": "e" * 64, "size_bytes": 5}

    result = executor.execute_shape_capability_v1(
        request, read_object=read_object, put_bytes=put_bytes
    )
    assert seen["models"] == model_rows
    assert seen["manifest"]["run_id"].startswith("capability-")
    assert result["status"] == "PASS"
    assert result["outputs"][0]["artifact_type"] == "RealSaS.ShapeProposal"
    assert result["outputs"][0]["authority_class"] == "LEARNED_SHAPE_PROPOSAL_ONLY"
    assert result["diagnostics"]["canonical_mesh_authority"] is False
    assert result["diagnostics"]["input_content_sha256"] == "c" * 64


def test_shape_executor_requires_exact_one_typed_shape_input():
    try:
        executor._one_shape_input({"input_artifacts": []})
    except RuntimeError as exc:
        assert "SHAPE_INPUT_ARTIFACT_CARDINALITY:0" in str(exc)
    else:
        raise AssertionError("missing Shape input must fail closed")
