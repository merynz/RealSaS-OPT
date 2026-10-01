from pathlib import Path
import pytest
from backend.realsas_platform.artifacts import LocalContentAddressedStore
from backend.realsas_platform.domain import ArtifactInputIdentity, ArtifactSemanticDescriptor, ProductRevisionManifest, ProductRoleBinding, RenderRequestSpec
from backend.realsas_platform.stage_graph import StageGraph, assert_product_render_stage_subset

H1="1"*64; H2="2"*64; H3="3"*64

def test_semantic_artifact_identity_is_deterministic_and_run_independent():
    a=ArtifactSemanticDescriptor("RealSaS.QualifiedMeshIR","v1","35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",H1,H2,(ArtifactInputIdentity("skeleton",0,"RealSaS.QualifiedSkeletonIR",H3),),{"b":2,"a":1})
    b=ArtifactSemanticDescriptor(a.artifact_type,a.schema_version,a.producer_contract,H1,H2,a.inputs,{"a":1,"b":2})
    assert a.semantic_sha256==b.semantic_sha256

def test_product_revision_and_render_bind_exact_semantics():
    revision=ProductRevisionManifest("knight",(ProductRoleBinding("appearance","RealSaS.CompleteAppearanceAssetIR.v2",H1),ProductRoleBinding("geometry","RealSaS.QualifiedMeshIR.v1",H2)))
    req=RenderRequestSpec(revision.manifest_sha256,H3,{"view":"front"},{"fps":24})
    same=RenderRequestSpec(revision.manifest_sha256,H3,{"view":"front"},{"fps":24})
    assert req.semantic_sha256==same.semantic_sha256

def test_stage42_change_does_not_invalidate_upstream():
    graph=StageGraph.from_canonical_plan(Path("canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    affected=graph.descendants_including(["42_RUNTIME_PROJECTION_AND_CAA_BINDING"])
    assert affected[0]=="42_RUNTIME_PROJECTION_AND_CAA_BINDING"
    assert all(int(x.split("_",1)[0])>=42 for x in affected)
    assert "09_IRIS_FIT_PREREGISTERED" not in affected

def test_product_render_cannot_invoke_fit():
    assert_product_render_stage_subset(["42_RUNTIME_PROJECTION_AND_CAA_BINDING","43_RSS_MATERIALIZE_COMPACT","44_NATIVE_PACKAGE_OPEN_PLAYBACK"])
    with pytest.raises(ValueError,match="forbidden compiler stages"): assert_product_render_stage_subset(["10_IRIS_FIT"])

def test_local_cas_is_immutable_and_detects_corruption(tmp_path):
    store=LocalContentAddressedStore(tmp_path); a=store.put_bytes(b"artifact"); b=store.put_bytes(b"artifact"); assert a==b
    assert store.get_bytes(a.content_sha256)==b"artifact"
    store.path_for(a.content_sha256).write_bytes(b"corrupted")
    assert store.verify(a.content_sha256) is False
    with pytest.raises(RuntimeError,match="CAS_CONTENT_HASH_MISMATCH"): store.get_bytes(a.content_sha256)
