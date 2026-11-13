from __future__ import annotations

from hashlib import sha256
import io
import json

import numpy as np
import pytest

from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
    EXECUTOR_CONTRACT,
    RAW_SCENE, RAW_SOURCE, RAW_VIEWS, RAW_RECEIPT,
    execute_rss_vnext_mesh_capability_v3,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    decode_uncalibrated_two_image_source_v1,
)
from tests.compiler.test_rss_vnext_raw_product_input_shape_v4 import (
    MemoryCAS, _source,
)


def _npz(**arrays):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def _scene(*, candidate: bool = True):
    props = {
        "scene_triplanes": np.zeros((1, 3, 4, 4, 4), np.float32),
        "query_xyz": np.zeros((1, 3, 3), np.float32),
        "local_features": np.ones((1, 3, 8), np.float32),
        "semantic_logits": np.zeros((1, 3, 4), np.float32),
        "graphics_logits": np.zeros((1, 3, 5), np.float32),
        "presentation_role_logits": np.zeros((1, 3, 2), np.float32),
        "region_relation_logits": np.zeros((1, 3, 3), np.float32),
        "tessa_topology_token_ids": np.array([[7, 8, 9]], np.int64),
    }
    if candidate:
        props.update({
            "tessa_candidate_base_vertices": np.array([[
                [0, 0, 0], [.5, 0, 0], [0, .5, 0]]], np.float32),
            "tessa_candidate_rest_vertices": np.array([[
                [0, 0, 0], [.5, 0, 0], [0, .5, 0]]], np.float32),
            "tessa_candidate_faces": np.array([[0, 1, 2]], np.int64),
            "tessa_candidate_vertex_components": np.zeros(3, np.int64),
            "tessa_candidate_face_components": np.zeros(1, np.int64),
            "tessa_candidate_local_features": np.zeros((1, 3, 8), np.float32),
            "tessa_candidate_normals": np.zeros((1, 3, 3), np.float32),
            "tessa_candidate_semantic_logits": np.zeros((1, 3, 4), np.float32),
            "tessa_candidate_presentation_role_logits": np.zeros((1, 3, 2), np.float32),
            "tessa_candidate_affordance_logits": np.zeros((1, 3, 4), np.float32),
            "tessa_candidate_graphics_logits": np.zeros((1, 3, 5), np.float32),
            "tessa_candidate_static_quality_logits": np.zeros((1, 3, 4), np.float32),
            "tessa_candidate_unsigned_distance": np.ones((1, 3), np.float32),
            "tessa_candidate_sign_logits": np.zeros((1, 3), np.float32),
        })
    return _npz(**props)


def _fixture(*, candidate=True):
    cas = MemoryCAS()
    src, _ = _source(cas)
    original = decode_uncalibrated_two_image_source_v1(cas.read_object(src))
    def sealed(blob, artifact_type):
        return {**cas.put_bytes(blob), "schema_version": "v1", "artifact_type": artifact_type}
    scene = sealed(_scene(candidate=candidate), RAW_SCENE)
    view = sealed(_npz(
        canonical_rgba=np.zeros((1, 2, 4, 16, 16), np.float32),
        generated_rgba=np.zeros((1, 6, 4, 16, 16), np.float32),
        generated_normals=np.zeros((1, 6, 3, 16, 16), np.float32),
        generated_virtual_cameras=np.zeros((1, 6, 13), np.float32),
    ), RAW_VIEWS)
    receipt = {
        "schema": "RealSaS.RSSvNext.UncalibratedConstructionReceipt.v1",
        "source_content_sha256": src["content_sha256"],
        "source_binding_sha256": original.source_binding_sha256,
        "source_rights_claim_sha256": original.source_rights_claim_sha256,
        "scene_content_sha256": scene["content_sha256"],
        "generated_views_content_sha256": view["content_sha256"],
        "checkpoint_sha256": "b" * 64,
        "model_release_behavior_identity_sha256": "c" * 64,
        "original_camera_calibrated": False,
        "original_pixel_count": 2,
        "generated_view_count": 6,
        "source_ray_evidence_qualified": False,
        "generated_views_as_original_source": False,
        "canonical_mesh_qualified": False,
        "product_revision_minted": False,
        "source_rights_verified": False,
        "mesh_status": "UNQUALIFIED_MESH_CANDIDATE" if candidate else "ABSTAIN",
        "tessa_token_count": 3,
        "tessa_mesh_candidate_present": candidate,
        "tessa_mesh_candidate_vertex_count": 3 if candidate else 0,
        "tessa_mesh_candidate_face_count": 1 if candidate else 0,
        "tessa_mesh_candidate_role_schema_id": "Test.Presentation" if candidate else None,
        "tessa_mesh_candidate_authority": "UNQUALIFIED_LEARNED_PROPOSAL_ONLY",
    }
    ref = sealed((json.dumps(receipt, sort_keys=True) + "\n").encode(), RAW_RECEIPT)
    req = {
        "kind": "MODULE", "capability_id": "product.mesh",
        "capability_metadata": {"executor_contract": EXECUTOR_CONTRACT,
                                "canonical_mesh_required_for_product_execution": True},
        "model_releases": [],
        "shape_parent_binding": {
            "schema": "RealSaS.RSSvNext.ShapeParentBinding.v1",
            "source_content_sha256": src["content_sha256"],
            "checkpoint_sha256": "b" * 64,
            "model_release_behavior_sha256": "c" * 64,
        },
        "input_artifacts": [src, {**src, "artifact_id": "original_input"}, scene, view, ref],
    }
    return cas, req


def _execute(cas, req):
    return execute_rss_vnext_mesh_capability_v3(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )


def test_raw_mesh_compiler_replays_same_tessa_candidate_and_explicitly_fails_source_admission():
    cas, req = _fixture()
    result = _execute(cas, req)
    assert result["status"] == "FAIL"
    assert result["outputs"] == []
    assert result["failure"]["code"] == "RSS_VNEXT_RAW_MESH_SOURCE_QUALIFICATION_PENDING"
    d = result["diagnostics"]
    assert d["topology_token_count"] == 3
    assert d["candidate_vertices"] == 3
    assert d["candidate_faces"] == d["nondegenerate_faces"] == 1
    assert d["component_id_continuity"]
    assert not d["tessa_reinference_performed"]
    assert not d["generated_target_cameras_are_observations"]
    assert not d["source_ray_replay_admitted"]
    assert not d["canonical_mesh_minted"]


def test_raw_mesh_abstain_preserves_status_without_inventing_topology():
    cas, req = _fixture(candidate=False)
    result = _execute(cas, req)
    assert result["status"] == "FAIL"
    assert result["diagnostics"]["mesh_state"] == "ABSTAIN"
    assert result["diagnostics"]["candidate_vertices"] == 0
    assert result["diagnostics"]["candidate_faces"] == 0


def test_raw_mesh_rejects_forged_receipt_even_with_valid_cas_hash():
    cas, req = _fixture()
    ref = next(r for r in req["input_artifacts"] if r["artifact_type"] == RAW_RECEIPT)
    fake = json.loads(cas.read_object(ref))
    fake["source_ray_evidence_qualified"] = True
    forged = {**cas.put_bytes(json.dumps(fake).encode()),
              "artifact_type": RAW_RECEIPT, "schema_version": "v1"}
    req["input_artifacts"][-1] = forged
    with pytest.raises(RuntimeError, match="SOURCE_OR_CHECKPOINT_PARENT_DRIFT"):
        _execute(cas, req)


def test_raw_mesh_rejects_checkpoint_parent_drift_and_mixed_source():
    cas, req = _fixture()
    req["shape_parent_binding"]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="SOURCE_OR_CHECKPOINT_PARENT_DRIFT"):
        _execute(cas, req)
    cas, req = _fixture()
    req["input_artifacts"].append({
        **cas.put_bytes(b"forged"), "artifact_type": "RealSaS.RSSvNextTwoViewSource",
        "schema_version": "v1",
    })
    with pytest.raises(RuntimeError, match="EXACT_SOURCE_AND_THREE_CHILDREN"):
        _execute(cas, req)


def test_raw_mesh_rejects_face_address_drift_and_does_not_fallback_to_grid():
    cas, req = _fixture()
    scene_ref = next(r for r in req["input_artifacts"] if r["artifact_type"] == RAW_SCENE)
    with np.load(io.BytesIO(cas.read_object(scene_ref))) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["tessa_candidate_faces"] = np.array([[0, 1, 3]], dtype=np.int64)
    new_scene = {**cas.put_bytes(_npz(**arrays)), "artifact_type": RAW_SCENE, "schema_version": "v1"}
    req["input_artifacts"][2] = new_scene
    receipt_ref = req["input_artifacts"][-1]
    receipt = json.loads(cas.read_object(receipt_ref))
    receipt["scene_content_sha256"] = new_scene["content_sha256"]
    req["input_artifacts"][-1] = {
        **cas.put_bytes(json.dumps(receipt).encode()),
        "artifact_type": RAW_RECEIPT, "schema_version": "v1",
    }
    with pytest.raises(RuntimeError, match="INDEXED_TOPOLOGY_DOMAIN_DRIFT"):
        _execute(cas, req)


def test_calibrated_mesh_delegates_v2_without_a_second_model(monkeypatch):
    import compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 as module
    seen = []
    monkeypatch.setattr(module, "execute_rss_vnext_mesh_capability_v2", lambda req, **kwargs: (
        seen.append(req) or {"status": "DELEGATED"}
    ))
    out = _execute(MemoryCAS(), {
        "capability_id": "product.mesh", "kind": "MODULE",
        "capability_metadata": {"executor_contract": EXECUTOR_CONTRACT},
        "input_artifacts": [{
            "artifact_type": "RealSaS.RSSvNextTwoViewSource",
            "schema_version": "v1",
        }],
        "model_releases": [],
    })
    assert out["status"] == "DELEGATED"
    assert seen[0]["capability_metadata"]["executor_contract"] == "RealSaS.ProductMeshCompile.v2"
