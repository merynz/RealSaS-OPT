from __future__ import annotations

"""RSS Source CAS → real unified inference → exact TESSA Compiler mesh court.

Tests run the *real* RSS scene, Semantic, Graphics and candidate-mesh heads.
Topology stream is an explicitly synthetic, codec-valid fixture in one test:
this validates integration and consequence computation, NOT unseen training
success. No mesh source rights, canonical rig, skin or drawable authority.
"""

from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    behavior_identity_sha256,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_mesh_executor_v2 import (
    PARENT_BINDING_SCHEMA,
    execute_rss_vnext_mesh_capability_v2,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_evidence_continuity_v1 import (
    validate_rss_vnext_evidence_continuity_v1,
)
from models.rss_vnext.unified_character_model import RSSvNextUnifiedCharacterModel
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from tests.models.test_rss_vnext_product_shape_cas_v3 import (
    _run as run_shape,
)


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(824)
    torch.set_num_threads(1)


def _make_mesh_request(shape_result, shape_request, *, face_pairs=()):
    release = shape_request["model_releases"][0]
    return {
        "kind": "MODULE",
        "execution_id": "rss-vnext-mesh-fixture",
        "capability_id": "product.mesh",
        "capability_metadata": {"executor_contract": "RealSaS.ProductMeshCompile.v2"},
        "input_artifacts": [
            shape_request["input_artifacts"][0],
            *shape_result["outputs"],
        ],
        "model_releases": [],
        "goal_parameters": {"requested_face_pairs": list(face_pairs)},
        "shape_parent_binding": {
            "schema": PARENT_BINDING_SCHEMA,
            "source_content_sha256": shape_request["input_artifacts"][0]["content_sha256"],
            "checkpoint_sha256": release["checkpoint"]["content_sha256"],
            "model_release_behavior_sha256": behavior_identity_sha256(release),
        },
    }


def _abstain_inference(self, rgba, camera, **_):
    with torch.no_grad():
        state, _, _ = self.encode_observations(rgba, camera)
    return {
        "state": state, "mesh_status": "ABSTAIN",
        "mesh_candidate": None, "mesh_failure": "TEST_SYNTHETIC_UNTRAINED_TOPOLOGY",
        "observed_view_count": 2,
        "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
    }


def _two_face_topology_inference(self, rgba, camera, **_):
    with torch.no_grad():
        state, _, _ = self.encode_observations(rgba, camera)
        # The ONLY synthetic teacher fixture: valid two-triangle strip.
        # All 3D H(x), region structure, geometry refinement, Graphics and
        # mechanical prior logits are produced by this same RSS checkpoint.
        vertices = np.asarray([
            [-.35, -.35, 0.], [-.15, -.35, 0.], [-.35, -.15, 0.],
            [.15, .15, 0.], [.35, .15, 0.], [.15, .35, 0.],
        ], dtype=np.float64)
        # Two separated triangles exercise cross-component candidate
        # relationships without presuming a static weld is an attachment.
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        tokens = encode_adjacent_teacher_mesh_v1(
            vertices, faces,
            component_id="TEST_ONLY", chart_id="TEST_ONLY",
            cfg=TESSAConfigV1(coordinate_bins=self.cfg.tessa.coordinate_bins),
        ).token_ids
        candidate = self.build_mesh_candidate(
            state, tuple(tokens), source_authority="LEARNED_PREDICTED_TOPOLOGY",
        )
    return {
        "state": state, "mesh_status": "UNQUALIFIED_MESH_CANDIDATE",
        "mesh_candidate": candidate, "mesh_failure": None,
        "observed_view_count": 2,
        "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
    }


def test_mesh_compiler_abstains_without_reading_or_discarding_shared_h(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RSSvNextUnifiedCharacterModel, "propose_teacher_free", _abstain_inference,
    )
    source, cas, shape, original, read, put, vision, checkpoint, cfg = (
        run_shape(tmp_path, monkeypatch)
    )
    assert shape["status"] == "PASS"
    assert shape["diagnostics"]["mesh_status"] == "ABSTAIN"
    request = _make_mesh_request(shape, original)

    scene_ref = next(x for x in shape["outputs"] if x["role"] == "neural_scene")
    read_roles = []
    def no_heavy_scene(ref):
        read_roles.append(ref.get("artifact_type"))
        if ref.get("content_sha256") == scene_ref["content_sha256"]:
            raise AssertionError("Mesh must not decompress H(x) on topology abstain")
        return read(ref)

    result = execute_rss_vnext_mesh_capability_v2(
        request, read_object=no_heavy_scene, put_bytes=put,
    )
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == 4
    assert "RealSaS.RSSvNextNeuralSceneCarrier" not in read_roles
    assert scene_ref["content_sha256"] in cas  # unchanged!
    decision = json.loads(read(result["outputs"][0]))
    assert decision["state"] == "ABSTAIN_NO_VALID_TOPOLOGY"
    assert decision["faces"] == []
    assert decision["candidate_mesh_sha256"] is None
    assert decision["neural_scene_carrier_sha256"] == scene_ref["content_sha256"]
    assert not decision["canonical_mesh_authority"]
    assert not decision["independent_editability_authority"]
    assert not result["diagnostics"]["product_revision_minted"]
    assert not result["diagnostics"]["executable_rig_skin_qualified"]
    # ABSTAIN preserves every independent expert's original artifact bytes.
    continuation = json.loads(read(next(
        r for r in result["outputs"] if r["role"] == "rss_evidence_continuity"
    )))
    assert continuation["mesh_status"] == "ABSTAIN"
    court = json.loads(read(next(x for x in result["outputs"]
        if x["role"] == "rss_cross_expert_capability_court")))
    assert court["state"] == "ABSTAIN_TOPOLOGY_EXPERTS_RETAINED"
    epistemic = json.loads(read(next(x for x in result["outputs"]
        if x["role"] == "rss_relation_epistemic_court")))
    assert epistemic["state"] == "ABSTAIN_NO_RELATION_EVIDENCE"
    assert epistemic["decisions"] == []
    assert not epistemic["attachment_authority"]
    assert court["face_count"] == 0
    assert not any(x["executable_authority"] for x in court["claims"])
    assert not continuation["has_product_authority"]
    assert not continuation["neural_scene_replaced_by_candidate"]
    assert len(continuation["preserved_children"]) == 7
    assert {r["role"] for r in continuation["preserved_children"]} == {
        "original_source", "normalized_model_source", "neural_evidence",
        "neural_scene", "inference_receipt", "unified_model_config",
        "embedded_mira_decoder",
    }
    fanout_ref = next(r for r in shape["outputs"] if r["role"] == "rss_evidence_fanout")
    index = json.loads(read(fanout_ref))
    children = {r["role"]: r for r in shape["outputs"] if r["role"] in {
        child["role"] for child in index["children"]
    }}
    validate_rss_vnext_evidence_continuity_v1(
        continuation, index=index, source_ref=original["input_artifacts"][0],
        index_ref=fanout_ref, child_refs=children,
        derived_refs=tuple(r for r in result["outputs"] if r["role"] != "rss_evidence_continuity"),
    )
    broken = json.loads(json.dumps(continuation))
    broken["preserved_children"].pop()
    with pytest.raises(RuntimeError, match="CONTINUITY_RECEIPT_HASH_OR_AUTHORITY_DRIFT"):
        validate_rss_vnext_evidence_continuity_v1(
            broken, index=index, source_ref=original["input_artifacts"][0],
            index_ref=fanout_ref, child_refs=children,
            derived_refs=tuple(r for r in result["outputs"] if r["role"] != "rss_evidence_continuity"),
        )


def test_live_shape_to_compiler_tessa_geometry_semantic_graphics_and_directed_relation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RSSvNextUnifiedCharacterModel, "propose_teacher_free", _two_face_topology_inference,
    )
    source, cas, shape, original, read, put, vision, checkpoint, cfg = (
        run_shape(tmp_path, monkeypatch)
    )
    assert shape["diagnostics"]["mesh_status"] == "UNQUALIFIED_MESH_CANDIDATE"
    request = _make_mesh_request(shape, original, face_pairs=[(0, 1), (1, 0)])
    mesh = execute_rss_vnext_mesh_capability_v2(
        request, read_object=read, put_bytes=put,
    )
    assert mesh["status"] == "PASS"
    assert len(mesh["outputs"]) == 5
    by_role = {r["role"]: r for r in mesh["outputs"]}
    assert set(by_role) == {
        "rss_computed_surface_candidate",
        "rss_sparse_face_relation_candidates",
        "rss_relation_epistemic_court",
        "rss_cross_expert_capability_court",
        "rss_evidence_continuity",
    }
    out = json.loads(read(by_role["rss_computed_surface_candidate"]))
    relations = json.loads(read(by_role["rss_sparse_face_relation_candidates"]))
    court = json.loads(read(by_role["rss_cross_expert_capability_court"]))
    epistemic = json.loads(read(by_role["rss_relation_epistemic_court"]))
    assert epistemic["state"] == "TYPED_HYPOTHESES_UNQUALIFIED"
    assert len(epistemic["decisions"]) == 6  # four hypotheses + two Compiler primitives
    assert {x["relation_type"] for x in epistemic["decisions"]} == {
        "SAME_DRAWABLE", "CUT_BOUNDARY", "DIRECTED_ATTACHMENT",
        "SHARES_INDEXED_EDGE", "SAME_CANDIDATE_COMPONENT",
    }
    assert all(not d["executable_authority"] and not d["source_truth_authority"]
               for d in epistemic["decisions"])
    assert all(x["truth"] == "UNKNOWN" for decision in epistemic["decisions"]
               for x in decision["assertions"] if x["origin"] == "LEARNED_HYPOTHESIS")
    assert all(decision["state"] in (
        "PREDICTED_RELATION_UNADMITTED", "ABSTAIN_NO_POLARITY",
        "DERIVED_CANDIDATE_GEOMETRY_NOT_SOURCE_TRUTH")
        for decision in epistemic["decisions"])
    assert court["state"] == "TYPED_CROSS_EXPERT_PROPOSALS_PRESERVED"
    assert court["face_count"] == 2
    assert court["directed_attachment_pair_count"] == 2
    assert court["max_semantic_unknown_probability"] > 0
    assert all(not c["source_truth_authority"] and not c["executable_authority"]
               for c in court["claims"])
    assert {c["capability"] for c in court["claims"]} >= {
        "SEMANTIC_REGION", "GRAPHICS_2D", "DIRECTED_ATTACHMENT",
        "ANIMATION_READY_MESH",
    }
    assert out["state"] == "UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE"
    assert out["topology_token_sequence_sha256"]
    assert out["candidate_mesh_sha256"]
    assert len(out["faces"]) == 2
    for face in out["faces"]:
        assert face["area"] > 0
        assert len(face["candidate_vertex_indices"]) == 3
        assert len(face["unit_normal_xyz"]) == 3
        assert sum(x*x for x in face["unit_normal_xyz"]) == pytest.approx(1., abs=1e-5)
        assert len(face["region_probabilities"]) == cfg.region_slots + 1
        assert sum(face["region_probabilities"]) == pytest.approx(1., abs=1e-6)
        assert face["region_probabilities"][-1] > 0
        assert len(face["presentation_role_probabilities"]) == len(cfg.semantic_role_names)
        assert len(face["graphics_logits_by_corner"]) == 3
        assert len(face["affordance_logits_by_corner"]) == 3
        assert len(face["original_rgba_by_corner"]) == 3
        assert len(face["original_projection_valid_by_corner"]) == 3
        assert not face["source_visibility_qualified"]
        assert not face["mechanical_anchor_qualified"]
    fwd, rev = relations["face_relations"]
    assert (fwd["from_face_index"], fwd["to_face_index"]) == (0, 1)
    assert (rev["from_face_index"], rev["to_face_index"]) == (1, 0)
    assert fwd["same_drawable_affinity"] == pytest.approx(rev["same_drawable_affinity"], abs=1e-6)
    assert fwd["cut_affinity"] == pytest.approx(rev["cut_affinity"], abs=1e-6)
    assert fwd["directed_attachment_affinity"] is not None
    assert rev["directed_attachment_affinity"] is not None
    assert fwd["known_cross_slot_support"] > 0
    assert fwd["from_unknown_probability"] > 0
    assert not relations["attachment_authority"]
    assert not relations["independent_drawable_authority"]
    assert mesh["diagnostics"]["tessa_replayed_from_exact_unified_checkpoint_config"]
    assert mesh["diagnostics"]["neural_scene_replaced_by_surface"] is False
    assert mesh["diagnostics"]["canonical_mesh_authority"] is False
    assert mesh["diagnostics"]["executable_rig_skin_qualified"] is False
    assert not mesh["diagnostics"]["product_revision_minted"]
    continuation = json.loads(read(by_role["rss_evidence_continuity"]))
    assert continuation["mesh_status"] == "UNQUALIFIED_MESH_CANDIDATE"
    assert len(continuation["derived_candidates"]) == 4
    assert len(continuation["preserved_children"]) == 7
    parent_index_ref = next(r for r in shape["outputs"] if r["role"] == "rss_evidence_fanout")
    parent_index = json.loads(read(parent_index_ref))
    assert continuation["evidence_hash"] == parent_index["evidence_hash"]
    assert continuation["fanout_index_ref"]["content_sha256"] == parent_index_ref["content_sha256"]
    assert continuation["source_ref"]["content_sha256"] == original["input_artifacts"][0]["content_sha256"]
    assert mesh["diagnostics"]["all_seven_neural_expert_children_preserved_by_reference"]


def test_mesh_refuses_parent_swaps_and_missing_config_child(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RSSvNextUnifiedCharacterModel, "propose_teacher_free", _abstain_inference,
    )
    _, _, shape, original, read, put, vision, checkpoint, cfg = (
        run_shape(tmp_path, monkeypatch)
    )
    req = _make_mesh_request(shape, original)
    with pytest.raises(RuntimeError, match="ATTEMPT_SHAPE_PARENT_BINDING_REQUIRED"):
        execute_rss_vnext_mesh_capability_v2(
            {**req, "shape_parent_binding": {}},
            read_object=read, put_bytes=put,
        )
    with pytest.raises(RuntimeError, match="SOURCE_OR_MODEL_BINDING_DRIFT"):
        execute_rss_vnext_mesh_capability_v2(
            {**req, "shape_parent_binding": {
                **req["shape_parent_binding"],
                "checkpoint_sha256": "f" * 64,
            }},
            read_object=read, put_bytes=put,
        )
    with pytest.raises(RuntimeError, match="EXPECTS_SOURCE_INDEX_AND_SEVEN_CHILDREN"):
        execute_rss_vnext_mesh_capability_v2(
            {**req, "input_artifacts": [
                x for x in req["input_artifacts"]
                if x.get("role") != "unified_model_config"
            ]}, read_object=read, put_bytes=put,
        )
    with pytest.raises(RuntimeError, match="OWN_MODEL_RELEASE_FORBIDDEN"):
        execute_rss_vnext_mesh_capability_v2(
            {**req, "model_releases": original["model_releases"]},
            read_object=read, put_bytes=put,
        )


def test_model_config_and_scene_drift_cannot_cross_mesh_compiler_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RSSvNextUnifiedCharacterModel, "propose_teacher_free", _two_face_topology_inference,
    )
    _, cas, shape, original, read, put, vision, checkpoint, cfg = (
        run_shape(tmp_path, monkeypatch)
    )
    req = _make_mesh_request(shape, original, face_pairs=[(0, 1)])
    config_ref = next(x for x in req["input_artifacts"] if x.get("role") == "unified_model_config")
    with pytest.raises(RuntimeError, match="MODEL_CONFIG_CAS_BYTES_DRIFT"):
        execute_rss_vnext_mesh_capability_v2(
            req,
            read_object=lambda r: read(r) + b"tamper" if r.get("content_sha256") == config_ref["content_sha256"] else read(r),
            put_bytes=put,
        )
    scene_ref = next(x for x in req["input_artifacts"] if x.get("role") == "neural_scene")
    with pytest.raises(RuntimeError, match="MESH_PARENT_CAS_BYTES_DRIFT"):
        execute_rss_vnext_mesh_capability_v2(
            req,
            read_object=lambda r: read(r) + b"tamper" if r.get("content_sha256") == scene_ref["content_sha256"] else read(r),
            put_bytes=put,
        )
    with pytest.raises(Exception, match="FACE_RELATION_PAIR_DOMAIN_INVALID|FACE_RELATION_PAIR_DUPLICATE"):
        execute_rss_vnext_mesh_capability_v2(
            {**req, "goal_parameters": {"requested_face_pairs": [[0, 0]]}},
            read_object=read, put_bytes=put,
        )
