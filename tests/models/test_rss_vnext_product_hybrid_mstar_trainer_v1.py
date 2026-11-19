from __future__ import annotations

"""Source-native open-hem + signed SDF on the ONE seeded production H/M*.

Synthetic independently constructed source face/hem receipts are regression
fixtures, NOT real corpus truth, a trained OPEN slot or GPU quality proof.
"""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.component_conditioned_udf_v1 import KIND_OPEN
from models.rss_vnext.product_hybrid_open_seed_v1 import ProductHybridOpenM0PolicyV1
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,source_surface_tensor_sha256,
)
from models.rss_vnext.training_seeded_tessa_v1 import (
    SeededTESSALossWeightsV1,_digest,
)
from models.rss_vnext.semantic_geometry_prior import RSSvNextSemanticLayerSpec
from models.rss_vnext.layered_iris_seed_v1 import layer_namespace_digest_v1
from models.rss_vnext.seeded_layer_source_court_v1 import (
    SOURCE_LAYER_AUTHORITY,SOURCE_LAYER_DERIVATION,
)
from models.rss_vnext.native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1,SCHEMA as FACE_SCHEMA,
)
from models.rss_vnext.native_layer_open_boundary_v1 import (
    NativeLayerOpenBoundaryV1,SCHEMA as HEM_SCHEMA,
    DERIVATION as HEM_DERIVATION,exact_open_boundary_edges_v1,
)
from models.rss_vnext.open_udf_surface_v1 import extract_open_udf_surface_v1
from tests.models.test_rss_vnext_product_component_udf_training_v1 import (
    _scenario_udf,
)


def _case(monkeypatch):
    net,component,_,component_registry,signed,make_base,native,generated=(
        _scenario_udf(monkeypatch)
    )
    torch.set_num_threads(1)
    first=make_base()
    source=signed.seeded_tessa_source_teacher
    assert source is not None
    open_native=extract_open_udf_surface_v1(
        scene_triplanes=torch.ones((1,3,4,4,4)),
        query_unsigned_distance=lambda xyz:xyz[:,2].abs(),
        resolution=(12,12,12),query_chunk=256,
    )
    body=RSSvNextSemanticLayerSpec("body",("primary_form",))
    cape=RSSvNextSemanticLayerSpec("cape",("outer_covering",))
    namespace=layer_namespace_digest_v1(
        (body.name,cape.name),
        (body.keep_roles,cape.keep_roles),
    )
    nv=len(source.source_xyz)
    xyz=torch.cat((source.source_xyz,open_native.xyz[0]),dim=0)
    faces=torch.cat((
        torch.tensor([[0,2,1],[0,3,2],[0,1,3],[1,2,3]],
                     dtype=torch.long),
        open_native.faces+nv,
    ),dim=0)
    vertex_components=torch.cat((
        torch.zeros(nv,dtype=torch.long),
        open_native.component_ids+1,
    ))
    face_components=vertex_components[faces[:,0]]
    layers=torch.cat((
        torch.zeros(nv,dtype=torch.long),
        torch.ones(len(open_native.xyz[0]),dtype=torch.long),
    ))
    normal=torch.cat((
        source.source_normals,
        torch.tensor([0.,0.,1.]).expand(len(open_native.xyz[0]),-1),
    ))
    sample_faces=[];sample_bary=[]
    for vertex in range(len(xyz)):
        hit=(faces==vertex).nonzero(as_tuple=False)
        assert len(hit)
        fi,corner=map(int,hit[0])
        sample_faces.append(fi)
        row=[0.,0.,0.]
        row[corner]=1.
        sample_bary.append(row)
    face_native=NativeFaceLayerSamplesV1(
        vertices=xyz.clone(),
        faces=faces,
        face_layer_ids=layers[faces[:,0]],
        face_component_ids=face_components,
        sample_face_ids=torch.tensor(sample_faces,dtype=torch.long),
        sample_barycentric=torch.tensor(sample_bary,dtype=xyz.dtype),
        source_asset_sha256=source.source_asset_sha256,
        layer_namespace_sha256=namespace,
    )
    border=exact_open_boundary_edges_v1(faces)
    border_ids=torch.unique(border.reshape(-1),sorted=True)
    boundary=NativeLayerOpenBoundaryV1(
        boundary_edges=border,boundary_vertex_ids=border_ids,
        open_layer_ids=torch.tensor([1],dtype=torch.long),
        source_asset_sha256=source.source_asset_sha256,
        layer_namespace_sha256=namespace,
    )
    teacher=replace(
        source,source_xyz=xyz,source_normals=normal,
        source_point_mask=torch.ones(len(xyz),dtype=torch.bool),
        source_importance=torch.full((len(xyz),),.5),
        source_importance_mask=torch.ones(len(xyz),dtype=torch.bool),
        source_component_ids=vertex_components,
        source_layer_ids=layers,
        native_face_layer_samples=face_native,
        native_open_boundary=boundary,
        source_boundary_xyz=xyz[border_ids],
        source_boundary_mask=torch.ones(len(border_ids),dtype=torch.bool),
    )
    source_key=signed.flexicubes_signed_source_admission_key
    open_key="source-native-open-cape"
    base_registry=native[source.source_qualification_key]
    receipt={
        **base_registry,
        "semantic_layer_name":body.name,
        "signed_layer_admission_keys":[source_key],
        "native_open_layer_admission_key":open_key,
        "source_layer_namespace_sha256":namespace,
        "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
        "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
        "source_layer_native_asset_sha256":source.source_asset_sha256,
        "native_source_layer_partition_sha256":
            source_surface_tensor_sha256(layers),
        "source_layer_independently_audited":True,
        "source_label_student_checkpoint_dependency":False,
        "source_native_face_witness_schema":FACE_SCHEMA,
        "source_native_face_witness_sha256":face_native.receipt_sha256(),
        "source_native_face_witness_derivation":
            "SOURCE_NATIVE_INDEXED_TRIANGLE_BARYCENTRIC_NOT_STUDENT",
        "source_native_open_boundary_schema":HEM_SCHEMA,
        "source_native_open_boundary_sha256":boundary.receipt_sha256(),
        "source_native_open_boundary_derivation":HEM_DERIVATION,
        **{k+"_sha256":_digest(v) for k,v in teacher.tensor_fields().items()},
    }
    signed_row={
        **dict(first.trusted_signed_source_registry[source_key]),
        "layer_name":body.name,
        "surface_target_sha256":
            source_surface_tensor_sha256(xyz[:nv][None]),
        "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
        "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
        "native_source_layer_partition_sha256":
            source_surface_tensor_sha256(layers),
        "source_layer_namespace_sha256":namespace,
        "source_layer_native_asset_sha256":source.source_asset_sha256,
        "native_source_layer_index":0,
        "source_layer_independently_audited":True,
        "source_label_student_checkpoint_dependency":False,
    }
    open_row={
        "admission_status":"QUALIFIED_SOURCE_NATIVE_OPEN_INDEXED_SHEET",
        "source_component_domain":"QUALIFIED_OPEN_MANIFOLD_SHEET",
        "source_rights_independently_verified":True,
        "generated_views_as_source_truth":False,
        "student_slot_ids_as_source_truth":False,
        "dataset_manifest_sha256":"f"*64,
        "source_qualification_key":signed.source_qualification_key,
        "source_asset_sha256":teacher.source_asset_sha256,
        "role_schema_id":net.cfg.semantic_role_schema_id,
        "layer_name":cape.name,
        "surface_target_sha256":
            source_surface_tensor_sha256(xyz[nv:][None]),
        "source_native_open_boundary_sha256":boundary.receipt_sha256(),
        "native_face_layer_samples_sha256":face_native.receipt_sha256(),
        "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
        "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
        "native_source_layer_partition_sha256":
            source_surface_tensor_sha256(layers),
        "source_layer_namespace_sha256":namespace,
        "source_layer_native_asset_sha256":source.source_asset_sha256,
        "native_source_layer_index":1,
        "source_layer_independently_audited":True,
        "source_label_student_checkpoint_dependency":False,
    }
    batch=replace(
        signed,flexicubes_layer=body,flexicubes_layers=(body,cape),
        flexicubes_signed_layer_admission_keys=(source_key,),
        product_hybrid_open_source_admission_key=open_key,
        flexicubes_reference_surface_xyz=xyz[None],
        seeded_tessa_source_teacher=teacher,
        seeded_tessa_source_face_budget=len(faces),
        component_udf_source_teacher=component.component_udf_source_teacher,
    )
    def kind(_):
        return torch.tensor([[[-9.,12.,-9.],[12.,-9.,-9.]]])
    def udf(_,points,*,slot_index):
        assert slot_index==0
        return points[...,2].abs()
    monkeypatch.setattr(net,"query_component_topology_kind_logits",kind)
    monkeypatch.setattr(net,"query_component_unsigned_distance",udf)
    policy=ProductHybridOpenM0PolicyV1(
        open_layer_name=cape.name,
        open_presentation_roles=cape.keep_roles,
        resolution=(12,12,12),query_chunk=256,
    )
    def make(*,open_registry=None,policy_override=None,weights=None):
        return RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters(),lr=1e-3),
            dataset_manifest_sha256="f"*64,
            training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            seeded_tessa_weights=weights or replace(
                SeededTESSALossWeightsV1(),boundary=.5,
            ),
            trusted_seeded_tessa_registry={
                teacher.source_qualification_key:receipt,
            },
            trusted_generated_conditioning_registry=generated,
            trusted_signed_source_registry={source_key:signed_row},
            trusted_hybrid_open_source_registry=(
                {open_key:open_row} if open_registry is None else open_registry
            ),
            product_hybrid_open_policy=(
                policy if policy_override is None else policy_override
            ),
            component_udf_weight=.8,
            trusted_component_udf_registry=component_registry,
        )
    return net,batch,make,open_row,open_key


def test_source_qualified_hybrid_mstar_trains_same_H_and_component_udf(monkeypatch):
    net,batch,make,_,_=_case(monkeypatch)
    trainer=make()
    before=net.tessa_seeded_head[-1].weight.detach().clone()
    component_before=net.component_udf.distance[0].weight.detach().clone()
    out=trainer.train_step(batch)
    assert out.hybrid_open_model_slot_index==0
    assert not out.hybrid_open_candidate_product_qualified
    assert out.seeded_tessa_loss is not None
    assert out.component_udf_loss is not None
    assert out.seeded_tessa_qualified_layer_count==2
    assert out.seeded_tessa_loss_parts["boundary"] is not None
    assert torch.isfinite(out.total_loss)
    assert not torch.equal(before,net.tessa_seeded_head[-1].weight)
    assert not torch.equal(component_before,net.component_udf.distance[0].weight)


def test_hybrid_native_boundary_missing_or_forged_abstains(monkeypatch):
    _,batch,make,row,key=_case(monkeypatch)
    trainer=make()
    missing=replace(
        batch,seeded_tessa_source_teacher=replace(
            batch.seeded_tessa_source_teacher,native_open_boundary=None,
        ),
    )
    with pytest.raises(ValueError,match="HYBRID_OPEN_NATIVE_TEACHER_REQUIRED"):
        trainer.objective(missing)
    bad={**row,"surface_target_sha256":"0"*64}
    with pytest.raises(ValueError,match="HYBRID_OPEN_SOURCE_NOT_ADMITTED"):
        make(open_registry={key:bad}).objective(batch)
    with pytest.raises(ValueError,match="HYBRID_OPEN_PHASE_POLICY"):
        make(weights=replace(SeededTESSALossWeightsV1(),boundary=0.))


def test_hybrid_open_receipt_changes_checkpoint_policy(monkeypatch,tmp_path):
    _,batch,make,row,key=_case(monkeypatch)
    first=make()
    path=tmp_path/"hybrid.pt"
    first.save_checkpoint(path,sampler_cursor=2)
    assert make().load_checkpoint(path)==2
    forged={**row,"student_slot_ids_as_source_truth":True}
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        make(open_registry={key:forged}).load_checkpoint(path)
