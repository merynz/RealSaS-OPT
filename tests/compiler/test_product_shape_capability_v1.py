from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_services.platform_worker import shape_capability_v1 as shape
from models.iris.v2.dinov2_foundation_v2 import DINOv2FoundationAuthorityV2
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1
from models.iris.vnext.surface_v1 import IRISSurfaceEvidenceV1


def _inputs() -> IRISVNextSceneInputsV1:
    return IRISVNextSceneInputsV1(
        image_tokens=torch.zeros(1, 8, 37 * 37, 384, dtype=torch.float32),
        token_xy=torch.zeros(1, 8, 37 * 37, 2, dtype=torch.float32),
        camera_features=torch.zeros(1, 8, 13, dtype=torch.float32),
        foundation_authority_sha256=DINOv2FoundationAuthorityV2().authority_hash,
    )


def _surface() -> IRISSurfaceEvidenceV1:
    count = 4
    return IRISSurfaceEvidenceV1(
        scene_planes=torch.zeros(1, 3, 4, 2, 2),
        surface_xyz_iris=torch.zeros(1, count, 3),
        surface_xyz_tessa=torch.zeros(1, count, 3),
        surface_normal=torch.zeros(1, count, 3),
        surface_h=torch.zeros(1, count, 8),
        mechanical={
            "stretch_major": torch.ones(1, count),
            "stretch_minor": torch.zeros(1, count),
            "tangent_direction_2theta": torch.zeros(1, count, 2),
            "bend_demand": torch.zeros(1, count),
            "rigidity_probability": torch.zeros(1, count),
            "contact_persistence_probability": torch.zeros(1, count),
        },
        contact_persistence_valid=torch.zeros(1, count, dtype=torch.bool),
        extracted_vertex_counts=(12,),
        extracted_face_counts=(20,),
    )


class _Court:
    grammar_valid = True
    decode_failure = None
    finite_geometry = True
    within_scale_policy = True

    def to_dict(self):
        return {
            "schema": "RealSaS.TESSADecodedTopologyCourt.v2",
            "grammar_valid": True,
            "decode_failure": None,
            "finite_geometry": True,
            "within_scale_policy": True,
        }


def _proposal(*, completed: bool = True):
    return SimpleNamespace(
        token_ids=(128, 129, 130, 134),
        topology_court=_Court(),
        completed=completed,
        stop_reason="EOS" if completed else "MAXIMUM_NEW_TOKENS_REACHED",
    )


def _binding(family: str):
    return SimpleNamespace(
        model_ref=SimpleNamespace(
            model_release_id=("1" if family == "IRIS" else "2") * 8
            + "-1111-4111-8111-111111111111",
            checkpoint_content_sha256=("a" if family == "IRIS" else "b") * 64,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(
                extraction_training_resolution=4,
                surface_witness_count=4,
            )
        ),
    )


def test_shape_input_codec_is_pickle_free_and_exact():
    source = _inputs()
    payload = shape.encode_shape_input_v1(source)
    decoded = shape.decode_shape_input_v1(payload)
    assert torch.equal(decoded.image_tokens, source.image_tokens)
    assert torch.equal(decoded.token_xy, source.token_xy)
    assert torch.equal(decoded.camera_features, source.camera_features)
    assert decoded.foundation_authority_sha256 == source.foundation_authority_sha256


def test_shape_composes_existing_inference_primitives_without_minting_mesh(monkeypatch):
    iris = _binding("IRIS")
    tessa = _binding("TESSA")
    surface = _surface()
    calls = []

    monkeypatch.setattr(shape, "load_iris_vnext_inference_v1", lambda manifest, device="cpu": iris)
    monkeypatch.setattr(shape, "encode_scene_v1", lambda binding, inputs: calls.append("iris") or torch.zeros(1, 3, 4, 2, 2))
    monkeypatch.setattr(
        shape,
        "extract_surface_evidence_v1",
        lambda model, scene, extractor, **kwargs: calls.append("surface") or surface,
    )
    monkeypatch.setattr(shape, "load_tessa_v3_inference_v1", lambda manifest, device="cpu": tessa)
    monkeypatch.setattr(shape, "build_inputs_from_iris_v1", lambda model, evidence: calls.append("tessa_input") or object())
    monkeypatch.setattr(
        shape,
        "generate_topology_proposal_v1",
        lambda binding, inputs, policy: calls.append("tessa") or _proposal(),
    )

    result = shape.run_shape_inference_v1(
        {"run_id": "shape-test"}, _inputs(), extractor=object(), maximum_new_tokens=32
    )
    assert calls == ["iris", "surface", "tessa_input", "tessa"]
    metadata = shape.decode_shape_proposal_metadata_v1(result.payload)
    assert metadata["authority"] == "LEARNED_SHAPE_PROPOSAL_ONLY"
    assert metadata["canonical_mesh_authority"] is False
    assert metadata["mesh_authority"] is False
    assert metadata["temporary_iris_mesh_serialized"] is False
    assert metadata["models"]["IRIS"]["checkpoint_content_sha256"] == "a" * 64
    assert metadata["models"]["TESSA"]["checkpoint_content_sha256"] == "b" * 64
    assert result.diagnostics["canonical_mesh_authority"] is False
    assert set(result.diagnostics["timings"]) == {
        "iris_load_seconds",
        "iris_scene_seconds",
        "surface_seconds",
        "tessa_load_seconds",
        "tessa_input_seconds",
        "tessa_decode_seconds",
        "total_seconds",
    }

    with np.load(io.BytesIO(result.payload), allow_pickle=False) as bundle:
        assert "surface_xyz_tessa" in bundle.files
        assert "surface_h" in bundle.files
        assert "contact_persistence_valid" in bundle.files
        assert all("face" not in name.lower() and "temporary" not in name.lower() for name in bundle.files)


def test_shape_fails_closed_on_incomplete_topology(monkeypatch):
    iris = _binding("IRIS")
    tessa = _binding("TESSA")
    monkeypatch.setattr(shape, "load_iris_vnext_inference_v1", lambda manifest, device="cpu": iris)
    monkeypatch.setattr(shape, "encode_scene_v1", lambda binding, inputs: torch.zeros(1, 3, 4, 2, 2))
    monkeypatch.setattr(shape, "extract_surface_evidence_v1", lambda *args, **kwargs: _surface())
    monkeypatch.setattr(shape, "load_tessa_v3_inference_v1", lambda manifest, device="cpu": tessa)
    monkeypatch.setattr(shape, "build_inputs_from_iris_v1", lambda *args, **kwargs: object())
    monkeypatch.setattr(shape, "generate_topology_proposal_v1", lambda *args, **kwargs: _proposal(completed=False))
    with pytest.raises(RuntimeError, match="SHAPE_TOPOLOGY_PROPOSAL_INCOMPLETE"):
        shape.run_shape_inference_v1({}, _inputs(), extractor=object(), maximum_new_tokens=4)


def test_shape_source_has_no_fit_or_live_channel_authority():
    source = Path(
        "compiler/realsas_compiler_services/platform_worker/shape_capability_v1.py"
    ).read_text(encoding="utf-8")
    assert "model_channels" not in source
    assert "optimizer" not in source
    assert "cohort_" not in source
    assert "latest" not in source.lower()
    assert "canonical_mesh_authority" in source
    assert "load_iris_vnext_inference_v1" in source
    assert "load_tessa_v3_inference_v1" in source
