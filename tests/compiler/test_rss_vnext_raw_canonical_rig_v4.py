from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import io
import json

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    encode_uncalibrated_two_image_source_v1,
)
from compiler.rss_vnext.workers.rss_vnext_rig_executor_v4 import (
    EXECUTOR_CONTRACT,
    RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT,
    RAW_CONFIG, RAW_DECODER, CANONICAL_MESH, STATIC_QUALIFICATION,
    execute_rss_vnext_rig_capability_v4,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
    _triangle_source_and_scene,
)
from tests.compiler.test_rss_vnext_raw_product_input_shape_v4 import MemoryCAS
from tests.models.test_rss_vnext_uncalibrated_neural_scene_v1 import _bundle
from tests.models.test_rss_vnext_canonical_rig_lowering_v1 import (
    _fixture as rig_fixture,
)


def _npz(**arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _positive_fixture(tmp_path):
    source, scene, receipt = _triangle_source_and_scene()
    cas = MemoryCAS()
    def ref(blob, artifact_type, version="v1"):
        return {**cas.put_bytes(blob), "artifact_type": artifact_type,
                "schema_version": version}
    source_ref = ref(encode_uncalibrated_two_image_source_v1(source), RAW_SOURCE)
    bundle = _bundle(tmp_path)
    from models.rss_vnext.model_config_archive_v1 import (
        encode_rss_vnext_unified_model_config_v1,
    )
    from models.rss_vnext.embedded_mira_decoder_v1 import (
        encode_embedded_mira_decoder_v1,
    )
    config_ref = ref(encode_rss_vnext_unified_model_config_v1(
        bundle.model.cfg, checkpoint_sha256="b" * 64,
    ), RAW_CONFIG)
    decoder_ref = ref(encode_embedded_mira_decoder_v1(
        bundle.model, parent_unified_checkpoint_sha256="b" * 64,
    ), RAW_DECODER)
    _, _, rig_evidence, _, _ = rig_fixture()
    events = list(rig_evidence.control_events)
    rows = json.dumps(events, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    n = int(scene["tessa_candidate_rest_vertices"].shape[1])
    scene.update({
        "scene_triplanes": np.zeros((1, 3, 4, 4, 4), np.float32),
        "query_xyz": np.zeros((1, 3, 3), np.float32),
        "local_features": np.zeros((1, 3, 8), np.float32),
        "semantic_logits": np.zeros((1, 3, 4), np.float32),
        "graphics_logits": np.zeros((1, 3, 5), np.float32),
        "presentation_role_logits": np.zeros((1, 3, 4), np.float32),
        "region_relation_logits": np.zeros((1, 3, 3), np.float32),
        "tessa_candidate_local_features": np.zeros((1, n, 8), np.float32),
        "tessa_candidate_normals": np.ones((1, n, 3), np.float32),
        "tessa_candidate_semantic_logits": np.zeros((1, n, 4), np.float32),
        "tessa_candidate_presentation_role_logits": np.zeros((1, n, 4), np.float32),
        "tessa_candidate_affordance_logits": np.zeros((1, n, 4), np.float32),
        "tessa_candidate_graphics_logits": np.zeros((1, n, 5), np.float32),
        "tessa_candidate_static_quality_logits": np.zeros((1, n, 4), np.float32),
        "tessa_candidate_unsigned_distance": np.ones((1, n), np.float32),
        "tessa_candidate_sign_logits": np.zeros((1, n), np.float32),
    })
    scene_ref = ref(_npz(**scene), RAW_SCENE)
    view_ref = ref(b"generated-view-non-authoritative-fixture", RAW_VIEWS)
    receipt.update({
        "schema": "RealSaS.RSSvNext.UncalibratedConstructionReceipt.v1",
        "source_content_sha256": source_ref["content_sha256"],
        "scene_content_sha256": scene_ref["content_sha256"],
        "generated_views_content_sha256": view_ref["content_sha256"],
        "unified_model_config_content_sha256": config_ref["content_sha256"],
        "embedded_mira_decoder_content_sha256": decoder_ref["content_sha256"],
        "original_camera_calibrated": False,
        "source_ray_evidence_qualified": False,
        "canonical_mesh_qualified": False,
        "product_revision_minted": False,
        "mesh_status": "UNQUALIFIED_MESH_CANDIDATE",
        "tessa_token_count": int(scene["tessa_topology_token_ids"].shape[1]),
        "tessa_mesh_candidate_present": True,
        "tessa_mesh_candidate_vertex_count": n,
        "tessa_mesh_candidate_face_count": int(scene["tessa_candidate_faces"].shape[0]),
        "tessa_mesh_candidate_role_schema_id": "Test.Presentation",
        "tessa_mesh_candidate_authority": "UNQUALIFIED_LEARNED_PROPOSAL_ONLY",
        "axis_mira_control_status": "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE",
        "axis_mira_control_events": events,
        "axis_mira_control_event_count": len(events),
        "axis_mira_control_events_sha256": sha256(rows).hexdigest(),
        "axis_mira_discrete_skin_candidate_present": False,
        "axis_mira_control_authority": "LEARNED_CAUSAL_CONTROL_NOT_CANONICAL_RIG_SKIN",
    })
    receipt_ref = ref(json.dumps(receipt, sort_keys=True).encode(), RAW_RECEIPT)
    mesh, static, _ = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest=scene_ref["content_sha256"],
    )
    assert mesh is not None and static is not None
    mesh_ref = ref(json.dumps(mesh.to_dict(), sort_keys=True).encode(), CANONICAL_MESH)
    static_ref = ref(json.dumps(static.to_dict(), sort_keys=True).encode(), STATIC_QUALIFICATION)
    req = {
        "kind": "MODULE", "capability_id": "product.rig",
        "capability_metadata": {"executor_contract": EXECUTOR_CONTRACT, "required_models": []},
        "model_releases": [],
        "input_artifacts": [source_ref, scene_ref, view_ref, receipt_ref,
                            config_ref, decoder_ref, mesh_ref, static_ref],
    }
    return cas, req, mesh


def _run(cas, req):
    return execute_rss_vnext_rig_capability_v4(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )


def test_raw_two_image_mesh_to_real_axis_canonical_rig_with_exact_fsq_codec(tmp_path):
    torch.manual_seed(200)
    torch.set_num_threads(1)
    cas, req, mesh = _positive_fixture(tmp_path)
    result = _run(cas, req)
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == 4
    assert result["diagnostics"]["mesh_lineage_hash"] == mesh.mesh_lineage_hash
    assert result["diagnostics"]["joint_count"] == 1
    assert result["diagnostics"]["control_count"] == 1
    assert not result["diagnostics"]["original_camera_observed"]
    assert not result["diagnostics"]["product_revision_minted"]
    rig_ref = result["outputs"][0]
    data = json.loads(cas.read_object(rig_ref))
    assert data["mesh_lineage_hash"] == mesh.mesh_lineage_hash
    assert rig_ref["artifact_type"] == "RealSaS.CanonicalRigIR"
    assert result["outputs"][1]["artifact_type"] == "RealSaS.SkinCodeProposalIR"


def test_raw_rig_refuses_different_actual_canonical_mesh_parent(tmp_path):
    torch.manual_seed(201)
    torch.set_num_threads(1)
    cas, req, _ = _positive_fixture(tmp_path)
    forged = json.loads(cas.read_object(req["input_artifacts"][-2]))
    forged["source_candidate_lineage_hash"] = "f" * 64
    req["input_artifacts"][-2] = {
        **cas.put_bytes(json.dumps(forged).encode()),
        "artifact_type": CANONICAL_MESH, "schema_version": "v1",
    }
    with pytest.raises(Exception, match="MESH_LINEAGE_HASH_DRIFT|MESH_NOT_REPRODUCIBLE"):
        _run(cas, req)


def test_calibrated_rig_still_delegates_to_existing_v3(monkeypatch):
    import compiler.rss_vnext.workers.rss_vnext_rig_executor_v4 as module
    seen = []
    monkeypatch.setattr(module, "execute_rss_vnext_rig_capability_v3", lambda req, **kw: (
        seen.append(req) or {"status": "DELEGATED"}
    ))
    result = _run(MemoryCAS(), {
        "kind": "MODULE", "capability_id": "product.rig",
        "capability_metadata": {"executor_contract": EXECUTOR_CONTRACT},
        "input_artifacts": [{"artifact_type": "RealSaS.RSSvNextTwoViewSource",
                             "schema_version": "v1"}],
        "model_releases": [],
    })
    assert result["status"] == "DELEGATED"
    assert seen[0]["capability_metadata"]["executor_contract"] == "RealSaS.ProductRigInference.v3"
