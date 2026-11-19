from __future__ import annotations

"""Source-free hybrid proposal tests, not a trained open-cloth quality proof."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.rss_vnext.open_udf_surface_v1 import tensor_bytes_sha256_v1
from models.rss_vnext.product_hybrid_open_seed_v1 import (
    ProductHybridOpenM0PolicyV1, propose_product_hybrid_open_m0_v1,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesSurfaceEvidence,
)
from tests.models.test_rss_vnext_layered_surface_compiler_v1 import (
    _test_model_and_layers,
)


def _signed(scene):
    xyz=torch.tensor([[[-.4,-.4,-.4],[.4,-.4,-.4],
                       [0.,.4,-.4],[0.,0.,.4]]],
                     device=scene.device,dtype=scene.dtype)
    faces=torch.tensor([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],
                       dtype=torch.long,device=scene.device)
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,iris_surface_faces=faces,
        component_ids_by_vertex=torch.zeros(4,dtype=torch.long,device=scene.device),
        scene_sha256=tensor_bytes_sha256_v1(scene),
        original_surface_sha256="a"*64,
    )
    return seed


def _model_and_state(monkeypatch,*,open_proposal=True):
    model,state,*_=_test_model_and_layers()
    state=replace(
        state,camera_features=None,
        scene_conditioning_authority="GENERATED_SIX_VIEW_SHARED_SCENE",
    )
    logits=(
        torch.tensor([[[ -9.,12.,-9.],[10.,-9.,-9.]]],
                     dtype=state.scene_triplanes.dtype)
        if open_proposal else
        torch.tensor([[[12.,-9.,-9.],[10.,-9.,-9.]]],
                     dtype=state.scene_triplanes.dtype)
    )
    def kind(received):
        assert received is state
        return logits.to(state.scene_triplanes.device)
    def udf(received,points_xyz,*,slot_index):
        assert received is state and slot_index==0
        assert points_xyz.shape[0]==1
        return points_xyz[...,2].abs()
    monkeypatch.setattr(model,"query_component_topology_kind_logits",kind)
    monkeypatch.setattr(model,"query_component_unsigned_distance",udf)
    return model,state


def _policy():
    return ProductHybridOpenM0PolicyV1(
        open_layer_name="cape",
        open_presentation_roles=("outer_covering",),
        resolution=(12,12,12),query_chunk=256,
    )


def test_qualified_same_H_open_slot_mints_one_indexed_m0_not_two_meshed_rigs(
    monkeypatch,
):
    model,state=_model_and_state(monkeypatch)
    signed=_signed(state.scene_triplanes)
    proposal=propose_product_hybrid_open_m0_v1(
        model=model,state=state,signed_seed=signed,policy=_policy(),
    )
    proposal.hybrid.validate()
    seed=proposal.hybrid.seed
    assert proposal.predicted_open_slot_index==0
    assert proposal.open_probability>.9
    assert proposal.product_admission_claimed is False
    assert seed.layer_names==("signed_solid","cape")
    assert seed.original_vertex_count>20
    assert seed.original_face_count>10
    assert bool((seed.faces[len(signed.faces):]>=signed.original_vertex_count).all())
    assert set(seed.vertex_component_ids.detach().cpu().tolist())=={0,1}
    assert bool((seed.vertex_layer_ids[seed.faces]
                 ==seed.vertex_layer_ids[seed.faces[:,0:1]]).all())
    # One TESSA candidate, never a second open garment rig.
    candidate=model.build_seeded_mechanics_candidate_v1(
        state,seed,structural_memory=model._structural_memory(state,None),
        adaptive_refinement=True,
    )
    mesh=candidate["mesh_candidate"]
    assert mesh.seeded_surface is not None
    assert mesh.seeded_surface.layer_names==("signed_solid","cape")
    assert mesh.seeded_surface.original_vertex_count==seed.original_vertex_count
    assert mesh.seeded_surface.original_face_count==seed.original_face_count


def test_uncertain_slot_and_non_six_view_context_fail_without_geometry(
    monkeypatch,
):
    model,state=_model_and_state(monkeypatch,open_proposal=False)
    with pytest.raises(ValueError,match="OPEN_SLOT_ABSTAIN_UNCERTAIN"):
        propose_product_hybrid_open_m0_v1(
            model=model,state=state,signed_seed=_signed(state.scene_triplanes),
            policy=_policy(),
        )
    with pytest.raises(ValueError,match="REQUIRES_SAME_GENERATED_SIX_VIEW_H"):
        propose_product_hybrid_open_m0_v1(
            model=model,
            state=replace(state,scene_conditioning_authority="DIRECT_OBSERVED_TWO_VIEWS"),
            signed_seed=_signed(state.scene_triplanes),
            policy=_policy(),
        )


def test_real_teacher_free_product_inference_uses_hybrid_mstar_not_legacy_tessa(
    monkeypatch,
):
    model,state=_model_and_state(monkeypatch)
    signed=_signed(state.scene_triplanes)
    h=model.cfg.iris.local_feature_dim
    schema=model.cfg.semantic_role_schema()
    witness=RSSvNextFlexiCubesSurfaceEvidence(
        xyz=signed.xyz,faces=signed.faces,
        local_features=torch.zeros(1,4,h),
        presentation_role_logits=torch.zeros(1,4,schema.count),
        regularizer=torch.zeros(1),
        sample_grid_xyz=torch.zeros(1,4,3),
        signed_field_grid=torch.zeros(1,4),
        cube_parameters=torch.zeros(1,21),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="all",
    )
    def generated(*args,**kwargs):
        return SimpleNamespace(
            state=state,
            flexicubes_surface_evidence=witness,
            structural_memory=model._structural_memory(state,None),
        )
    monkeypatch.setattr(model,"forward_generated_multiview",generated)
    rgba=torch.rand(1,2,4,8,8)
    six=torch.rand(1,6,4,8,8)
    camera=torch.zeros(1,6,13)
    result=model.propose_teacher_free(
        rgba,None,
        generated_rgba=six,generated_cameras=camera,
        generated_normals=torch.zeros(1,6,3,8,8),
        generation_checkpoint_sha256="a"*64,
        source_binding_sha256="b"*64,
        flexicubes_extractor=object(),
        flexicubes_layer=model.cfg.semantic_role_schema() and
            __import__("models.rss_vnext.semantic_geometry_prior",
                       fromlist=["RSSvNextSemanticLayerSpec"])
            .RSSvNextSemanticLayerSpec("all",tuple(schema.roles)),
        hybrid_open_policy=_policy(),
    )
    assert result["mesh_status"]=="UNQUALIFIED_MESH_CANDIDATE"
    assert result["mesh_candidate"].seeded_surface is not None
    assert result["semantic_layer_names"]==("signed_solid","cape")
    assert result["hybrid_open_component_surface_kinds"]==(
        "SIGNED_IRIS_H_CLOSED","UDF_IRIS_H_OPEN",
    )
    assert result["hybrid_open_selected_slot_index"]==0
    assert result["hybrid_open_candidate_product_qualified"] is False
    assert result["topology_candidate_abi"]=="ONE_IRIS_SEEDED_MESH_V1"


def test_inference_rejects_hybrid_without_generated_construction(monkeypatch):
    model,_=_model_and_state(monkeypatch)
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros(1,2,13)
    with pytest.raises(ValueError,match="HYBRID_PRODUCT_GENERATED_SCENE_REQUIRED"):
        model.propose_teacher_free(rgba,cam,hybrid_open_policy=_policy())
