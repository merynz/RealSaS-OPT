from __future__ import annotations

"""Actual model -> seeded Shape CAS -> raw Mesh court -> MIRA replay binding.

CPU tiny fixture checks addresses/provenance, not learned reconstruction.
"""
import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 import (
    _tessa_candidate_tensors_v1, _tensor_archive,
)
from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
    _validate_candidate,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesSurfaceEvidence,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    tessa_seed_from_flexicubes_v1,
)


def _model_scene_and_proposal():
    torch.manual_seed(170117)
    torch.set_num_threads(1)
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).eval()
    rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = torch.zeros(1, 2, 13)
    cameras[..., 3] = cameras[..., 7] = cameras[..., 11] = cameras[..., 12] = 1
    state, _, _ = net.encode_observations(rgba, cameras)
    # Coincident geometry is still separate when face-addresses differ.
    xyz = torch.tensor([[[-.3,.35,0.], [.3,.35,0.], [-.3,-.35,0.], [.3,-.35,0.]]])
    faces = torch.tensor([[0,2,1],[1,2,3]],dtype=torch.long)
    witness = RSSvNextFlexiCubesSurfaceEvidence(
        xyz=xyz, faces=faces,
        local_features=torch.ones(1,4,cfg.iris.local_feature_dim),
        presentation_role_logits=torch.zeros(
            1,4,cfg.semantic_role_schema().count,
        ),
        regularizer=torch.zeros(1),
        sample_grid_xyz=torch.zeros(1,4,3),
        signed_field_grid=torch.zeros(1,4),
        cube_parameters=torch.zeros(1,21),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="all",
    )
    witness.validate(
        carrier_dim=cfg.iris.local_feature_dim,
        role_count=cfg.semantic_role_schema().count,
    )
    seed = tessa_seed_from_flexicubes_v1(
        witness=witness, scene_triplanes=state.scene_triplanes,
    )
    output = net.build_seeded_mechanics_candidate_v1(state, seed)
    proposal = dict(
        mesh_candidate=output["mesh_candidate"],
        structural_token_proposal=torch.tensor([[cfg.tessa.BOS]],dtype=torch.long),
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",
        topology_candidate_abi="ONE_IRIS_SEEDED_MESH_V1",
    )
    return net, state, proposal


def _arrays_and_receipt():
    _, state, proposal = _model_scene_and_proposal()
    arrays, receipt = _tessa_candidate_tensors_v1(proposal)
    blob = _tensor_archive(**arrays, scene_triplanes=state.scene_triplanes)
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        packed = {k:z[k] for k in z.files}
    receipt["mesh_status"]="UNQUALIFIED_MESH_CANDIDATE"
    return packed, receipt


def test_real_seeded_product_candidate_has_one_source_surface_and_context():
    a, receipt = _arrays_and_receipt()
    out = _validate_candidate(a, receipt)
    assert out["mesh_state"] == "UNQUALIFIED_MESH_CANDIDATE"
    assert out["topology_candidate_abi"] == "ONE_IRIS_SEEDED_MESH_V1"
    assert out["candidate_vertices"] == 4
    assert out["candidate_faces"] == 2
    assert out["geometric_face_area_positive"]
    assert out["component_id_continuity"]
    assert receipt["tessa_mesh_candidate_authority"] == (
        "UNQUALIFIED_IRIS_SEEDED_SURFACE_ONLY"
    )
    assert np.array_equal(a["tessa_seed_vertex_parent_indices"],
                          np.array([[0,0],[1,1],[2,2],[3,3]]))
    assert np.allclose(
        a["tessa_candidate_local_features"],
        a["tessa_candidate_mechanics_local_features"],
    )


def test_raw_compiler_rejects_scene_and_parent_forgery():
    a, receipt = _arrays_and_receipt()
    changed = {**a, "scene_triplanes":a["scene_triplanes"].copy()}
    changed["scene_triplanes"][0,0,0,0,0] += 1
    with pytest.raises(RuntimeError,match="SEEDED_SCENE_HASH_DRIFT"):
        _validate_candidate(changed,receipt)
    changed={**a,"tessa_seed_vertex_parent_indices":
             a["tessa_seed_vertex_parent_indices"].copy()}
    changed["tessa_seed_vertex_parent_indices"][0,0]=999
    with pytest.raises(RuntimeError,match="SEEDED_PARENT_OR_CONTEXT_DRIFT"):
        _validate_candidate(changed,receipt)
    changed={k:v for k,v in a.items()
             if k!="tessa_candidate_mechanics_local_features"}
    with pytest.raises(RuntimeError,match="SEEDED_CONTEXT_OR_LINEAGE_REQUIRED"):
        _validate_candidate(changed,receipt)


def test_no_legacy_token_authority_when_seeded():
    a,receipt=_arrays_and_receipt()
    legacy={**receipt,"topology_candidate_abi":"LEGACY_INDEPENDENT_TESSA_GRAMMAR"}
    with pytest.raises(RuntimeError,match="PREMATURE_CANONICAL_AUTHORITY"):
        _validate_candidate(a,legacy)
