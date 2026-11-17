from __future__ import annotations

"""Same H(x) disjoint body/cloth shell -> TESSA M* -> Compiler CAS proof.

Overlapping positions are INTENTIONAL. No source asset, animation, GPU
renderer, observed semantics, or source-qualified garment truth is claimed.
"""
import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesSurfaceEvidence,
)
from models.rss_vnext.layered_iris_seed_v1 import make_layered_iris_seed_v1
from models.rss_vnext.tessa_seeded_surface_v1 import (
    tessa_adaptive_split_edges_v1,
)
from compiler.rss_vnext.workers.rss_vnext_shape_executor_v4 import (
    _tessa_candidate_tensors_v1,_tensor_archive,
)
from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
    _validate_candidate,
)


def _test_model_and_layers():
    torch.manual_seed(91277)
    torch.set_num_threads(1)
    cfg=RSSvNextConfig.tiny_for_test()
    model=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).eval()
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros(1,2,13)
    cam[...,3]=cam[...,7]=cam[...,11]=cam[...,12]=1
    state,_,_=model.encode_observations(rgba,cam)
    schema=cfg.semantic_role_schema()
    roles=(
        RSSvNextSemanticLayerSpec("body",("primary_form","focal_detail")),
        RSSvNextSemanticLayerSpec("cloth",("outer_covering",)),
    )
    xyz=torch.tensor(
        [[[-.35,.35,0.],[.35,.35,0.],[-.35,-.35,0.],[.35,-.35,0.]]],
        requires_grad=True,
    )
    faces=torch.tensor([[0,2,1],[1,2,3]],dtype=torch.long)
    witnesses=[]
    for role in roles:
        # BODY AND CLOTH SHARE EXACT POSITIONS. IDs must not weld.
        witnesses.append((role,RSSvNextFlexiCubesSurfaceEvidence(
            xyz=xyz.clone(),faces=faces.clone(),
            local_features=torch.zeros(1,4,cfg.iris.local_feature_dim),
            presentation_role_logits=torch.zeros(1,4,schema.count),
            regularizer=torch.zeros(1),sample_grid_xyz=torch.zeros(1,4,3),
            signed_field_grid=torch.zeros(1,4),
            cube_parameters=torch.zeros(1,21),
            source_policy="SIGNED_HYPOTHESIS_ONLY",
            layer_name=role.name,
        )))
    joined=make_layered_iris_seed_v1(
        evidence=tuple(witnesses),schema=schema,
        scene_triplanes=state.scene_triplanes,
    )
    return model,state,joined,roles,xyz,faces,witnesses


def _archive_and_receipt():
    net,state,joined,*_=_test_model_and_layers()
    out=net.build_seeded_mechanics_candidate_v1(
        state,joined.seed,
    )
    proposal={
        "mesh_candidate":out["mesh_candidate"],
        "structural_token_proposal":torch.tensor(
            [[net.cfg.tessa.BOS]],dtype=torch.long,
        ),
        "mesh_status":"UNQUALIFIED_MESH_CANDIDATE",
        "topology_candidate_abi":"ONE_IRIS_SEEDED_MESH_V1",
    }
    arrays,receipt=_tessa_candidate_tensors_v1(proposal)
    blob=_tensor_archive(**arrays,scene_triplanes=state.scene_triplanes)
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        fields={k:z[k] for k in z.files}
    receipt["mesh_status"]="UNQUALIFIED_MESH_CANDIDATE"
    return fields,receipt,joined


def test_separate_body_cloth_exact_coincident_vertices_remain_indexed_and_trainable():
    net,state,joined,roles,xyz,faces,witnesses=_test_model_and_layers()
    seed=joined.seed
    joined.validate()
    assert seed.xyz.shape==(1,8,3)
    assert seed.original_vertex_count==8
    assert seed.faces.tolist()==[[0,2,1],[1,2,3],[4,6,5],[5,6,7]]
    torch.testing.assert_close(seed.xyz[:,0:4],seed.xyz[:,4:8])
    assert seed.vertex_component_ids.tolist()==[0,0,0,0,1,1,1,1]
    assert seed.vertex_layer_ids.tolist()==[0,0,0,0,1,1,1,1]
    assert seed.layer_names==("body","cloth")
    logits=torch.full((1,8),10.0)
    split=tessa_adaptive_split_edges_v1(
        seed,importance_logits=logits,
        max_added_vertices=4,length_gate_ratio=0.5,
    )
    split.validate()
    for idx in range(seed.original_vertex_count,split.xyz.shape[1]):
        a,b=split.parent_seed_edges[idx]
        assert int(split.vertex_layer_ids[idx])==int(seed.vertex_layer_ids[a])
        assert int(seed.vertex_layer_ids[a])==int(seed.vertex_layer_ids[b])
    split.xyz.sum().backward()
    assert xyz.grad is not None and xyz.grad.abs().sum()>0


def test_compiler_seals_semantic_layer_namespace_and_rejects_forgery():
    a,receipt,joined=_archive_and_receipt()
    result=_validate_candidate(a,receipt)
    assert result["mesh_state"]=="UNQUALIFIED_MESH_CANDIDATE"
    assert a["tessa_seed_vertex_layer_ids"].tolist()==[0,0,0,0,1,1,1,1]
    assert receipt["tessa_seed_layer_names"]==["body","cloth"]
    forged={**a,"tessa_seed_vertex_layer_ids":
            a["tessa_seed_vertex_layer_ids"].copy()}
    forged["tessa_seed_vertex_layer_ids"][4]=0
    with pytest.raises(RuntimeError,match="LAYER_ADDRESS_DRIFT"):
        _validate_candidate(forged,receipt)
    changed={**receipt,"tessa_seed_layer_names":["body","cape"]}
    with pytest.raises(RuntimeError,match="LAYER_NAMESPACE_HASH_DRIFT"):
        _validate_candidate(a,changed)
    erased={k:v for k,v in a.items() if k!="tessa_seed_vertex_layer_ids"}
    with pytest.raises(RuntimeError,match="LAYER_NAMESPACE_INCOMPLETE"):
        _validate_candidate(erased,receipt)


def test_layer_union_without_overlap_or_source_sign_fabrication():
    net,state,joined,roles,xyz,faces,witnesses=_test_model_and_layers()
    schema=net.cfg.semantic_role_schema()
    duplicated=RSSvNextSemanticLayerSpec("copy",("primary_form",))
    with pytest.raises(ValueError,match="LAYERED_OVERLAP"):
        make_layered_iris_seed_v1(
            evidence=(witnesses[0],(duplicated,replace(
                witnesses[1][1],layer_name="copy",
            ))),schema=schema,scene_triplanes=state.scene_triplanes,
        )
    with pytest.raises(ValueError,match="LAYERED_OVERLAP"):
        make_layered_iris_seed_v1(
            evidence=((roles[0],replace(witnesses[0][1],
                source_policy="OPEN_UDF_NOT_QUALIFIED",
            )),witnesses[1]),schema=schema,
            scene_triplanes=state.scene_triplanes,
        )


def test_teacher_free_model_uses_same_scene_for_two_shells(monkeypatch):
    model,_,joined,roles,*_=_test_model_and_layers()
    calls=[]
    def extract_same_scene(
        scene_triplanes, *,extractor,role_head,schema,layer,
        field_policy,high_resolution_training=False,
    ):
        assert field_policy=="SIGNED_HYPOTHESIS_ONLY"
        calls.append((layer.name,scene_triplanes.detach().clone()))
        xyz=torch.tensor([[[-.35,.35,0.],[.35,.35,0.],
                          [-.35,-.35,0.],[.35,-.35,0.]]])
        faces=torch.tensor([[0,2,1],[1,2,3]],dtype=torch.long)
        return RSSvNextFlexiCubesSurfaceEvidence(
            xyz=xyz,faces=faces,
            local_features=torch.zeros(
                1,4,model.cfg.iris.local_feature_dim,
            ),
            presentation_role_logits=torch.zeros(
                1,4,model.cfg.semantic_role_schema().count,
            ),
            regularizer=torch.zeros(1),
            sample_grid_xyz=torch.zeros(1,4,3),
            signed_field_grid=torch.zeros(1,4),
            cube_parameters=torch.zeros(1,21),
            source_policy=field_policy,
            layer_name=layer.name,
        )
    monkeypatch.setattr(
        model.fc_surface_bridge,"forward",extract_same_scene,
    )
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros(1,2,13)
    cam[...,3]=cam[...,7]=cam[...,11]=cam[...,12]=1
    out=model.propose_teacher_free(
        rgba,cam,flexicubes_extractor=object(),
        flexicubes_layer=roles[0],flexicubes_layers=roles,
    )
    assert out["mesh_status"]=="UNQUALIFIED_MESH_CANDIDATE"
    assert out["semantic_layer_names"]==("body","cloth")
    assert out["mesh_candidate"].seeded_surface is not None
    assert out["mesh_candidate"].seeded_surface.layer_names==("body","cloth")
    assert [name for name,_ in calls]==["body","cloth"]
    torch.testing.assert_close(calls[0][1],calls[1][1])


def test_teacher_free_missing_one_shell_abstains_without_legacy_mesh(monkeypatch):
    model,_,_,roles,*_=_test_model_and_layers()
    def fail_shell(
        scene_triplanes, *,extractor,role_head,schema,layer,
        field_policy,high_resolution_training=False,
    ):
        if layer.name=="cloth":
            raise ValueError("SOURCE_SIGN_SHELL_EMPTY")
        return RSSvNextFlexiCubesSurfaceEvidence(
            xyz=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]),
            faces=torch.tensor([[0,1,2]],dtype=torch.long),
            local_features=torch.zeros(
                1,3,model.cfg.iris.local_feature_dim,
            ),
            presentation_role_logits=torch.zeros(
                1,3,model.cfg.semantic_role_schema().count,
            ),
            regularizer=torch.zeros(1),
            sample_grid_xyz=torch.zeros(1,3,3),
            signed_field_grid=torch.zeros(1,3),
            cube_parameters=torch.zeros(1,21),
            source_policy=field_policy,layer_name=layer.name,
        )
    monkeypatch.setattr(model.fc_surface_bridge,"forward",fail_shell)
    rgba=torch.rand(1,2,4,8,8)
    cam=torch.zeros(1,2,13)
    cam[...,3]=cam[...,7]=cam[...,11]=cam[...,12]=1
    out=model.propose_teacher_free(
        rgba,cam,flexicubes_extractor=object(),
        flexicubes_layer=roles[0],flexicubes_layers=roles,
    )
    assert out["mesh_candidate"] is None
    assert out["mesh_status"]=="ABSTAIN"
    assert "SOURCE_SIGN_SHELL_EMPTY" in out["mesh_failure"]
    assert out["topology_candidate_abi"]=="ONE_IRIS_SEEDED_MESH_V1"
