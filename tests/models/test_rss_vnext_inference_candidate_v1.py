from __future__ import annotations

from dataclasses import replace
import io

import numpy as np
import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_candidate_v1 import (
    AUTHORITY, build_rss_vnext_inference_evidence_v1,
    RSSvNextInferenceEvidenceBundleV1,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(55)
    torch.set_num_threads(1)


def net():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def observations(model):
    source = torch.rand(1, 2, 4, 8, 8)
    cams = torch.zeros((1, 2, 13))
    cams[:, :, 3] = cams[:, :, 7] = cams[:, :, 11] = cams[:, :, 12] = 1
    state, _, _ = model.encode_observations(source, cams)
    return state


def fixture_mesh(model, state):
    tri = encode_adjacent_teacher_mesh_v1(
        np.asarray([[-.25,-.25,0],[.25,-.25,0],[.25,.25,0]],dtype=np.float64),
        np.asarray([[0,1,2]],dtype=np.int64),
        component_id="toy",chart_id="toy",
        cfg=TESSAConfigV1(coordinate_bins=model.cfg.tessa.coordinate_bins),
    )
    return model.build_mesh_candidate(
        state, torch.tensor(tri.token_ids, dtype=torch.long),
        source_authority="LEARNED_PREDICTED_TOPOLOGY",
    )


def _result(state, mesh=None):
    return {
        "state":state,
        "mesh_candidate":mesh,
        "mesh_status":"UNQUALIFIED_MESH_CANDIDATE" if mesh is not None else "ABSTAIN",
        "qualification_status":"UNQUALIFIED_NEURAL_PROPOSAL",
        "observed_view_count":2,
    }


def _bundle(result):
    return build_rss_vnext_inference_evidence_v1(
        result, model_architecture_id=RSSvNextUnifiedCharacterModel.architecture_id,
        checkpoint_sha256="c"*64,source_image_sha256=("a"*64,"b"*64),
    )


def test_abstain_still_preserves_the_two_original_images_and_shared_scene():
    model = net()
    state = observations(model)
    artifact = _bundle(_result(state))
    assert artifact.manifest["status"] == "ABSTAIN"
    assert artifact.manifest["authority"] == AUTHORITY
    assert artifact.manifest["product_revision_id"] is None
    with np.load(io.BytesIO(artifact.array_archive_npz),allow_pickle=False) as z:
        assert z["source_rgba"].shape == (1,2,4,8,8)
        assert z["source_image_tokens"].shape == (1,2,4,32)
        assert z["shared_scene_triplanes"].shape[0:2] == (1,3)
        assert "mesh_rest_xyz" not in z
        assert np.allclose(
            z["source_rgba"], state.observed_rgba.detach().cpu().numpy(),
        )


def test_indexed_mesh_and_all_semantic_graphics_fields_survive_archive_roundtrip():
    model=net()
    state=observations(model)
    mesh=fixture_mesh(model,state)
    result=_result(state,mesh)
    artifact=_bundle(result)
    assert artifact.manifest["status"]=="MESH_CANDIDATE"
    with np.load(io.BytesIO(artifact.array_archive_npz),allow_pickle=False) as z:
        assert z["mesh_rest_xyz"].shape == (1,3,3)
        assert z["mesh_local_h"].shape == (1,3,32)
        assert z["mesh_graphics_logits"].shape[:2] == (1,3)
        assert z["mesh_source_ray_rgba"].shape == (1,3,2,4)
        assert z["mesh_source_ray_valid"].shape == (1,3,2)
        assert z["mesh_faces"].dtype == np.int64
        assert z["mesh_vertex_component"].shape == (3,)


def test_candidate_prevents_false_source_and_product_promotion():
    model=net();state=observations(model);mesh=fixture_mesh(model,state)
    artifact=_bundle(_result(state,mesh))
    claimed=dict(artifact.manifest)
    claimed["authority"]="PRODUCT_CANONICAL"
    with pytest.raises(ValueError,match="AUTHORITY_DRIFT"):
        RSSvNextInferenceEvidenceBundleV1(claimed,artifact.array_archive_npz).validate()
    claimed=dict(artifact.manifest)
    claimed["product_revision_id"]="SOMETHING"
    with pytest.raises(ValueError,match="PRODUCT_AUTHORITY_FORBIDDEN"):
        RSSvNextInferenceEvidenceBundleV1(claimed,artifact.array_archive_npz).validate()
    claimed=dict(artifact.manifest)
    claimed["array_archive_sha256"]="f"*64
    with pytest.raises(ValueError,match="ARRAY_ARCHIVE_DRIFT"):
        RSSvNextInferenceEvidenceBundleV1(claimed,artifact.array_archive_npz).validate()
    claimed=dict(artifact.manifest)
    claimed["source_image_sha256"]=["a"*64,"a"*64]
    with pytest.raises(ValueError,match="SOURCE_HASHES_INVALID"):
        RSSvNextInferenceEvidenceBundleV1(claimed,artifact.array_archive_npz).validate()
    forced=replace(mesh,source_authority="TEACHER_FORCED_TRAINING_ONLY")
    with pytest.raises(ValueError,match="TEACHER_FORCED_MESH_FORBIDDEN"):
        _bundle(_result(state,forced))


def test_generated_data_is_not_labelled_source_authority_and_must_be_two_views():
    model=net();state=observations(model)
    result=_result(state)
    result["qualification_status"]="COMPILER_CANONICAL_MESH"
    with pytest.raises(ValueError,match="NEURAL_AUTHORITY_ESCALATION"):
        _bundle(result)
    result=_result(state);result["observed_view_count"]=8
    with pytest.raises(ValueError,match="ONLY_TWO_REAL_INPUTS"):
        _bundle(result)
