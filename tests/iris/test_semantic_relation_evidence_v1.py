from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import CameraProjectionV3
from compiler.realsas_compiler_core.semantic_relation_evidence_v1 import (
    MeshSemanticEvidenceV1,
    detect_pose_order_conflicts_v1,
    infer_behavior_evidence_v1,
    infer_grip_candidates_v1,
    infer_source_overlap_relations_v1,
    lift_mask_evidence_to_m_v1,
)
from models.iris.semantic_evidence_v1 import (
    build_mask_evidence_v1,
    load_mask_evidence_v1,
    write_mask_evidence_v1,
)


def _camera(view: int, resolution: int = 64) -> CameraProjectionV3:
    return CameraProjectionV3(
        view_id=f"V{view}",
        view_index=view,
        origin=(0.0, 0.0, -5.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=2.0,
        resolution=resolution,
    )


def test_frozen_semantic_bundle_roundtrip(tmp_path):
    mask = np.zeros((16, 16), bool)
    mask[2:7, 3:8] = True
    evidence = build_mask_evidence_v1(
        [{"view_index": 0, "label": "Hand", "score": 0.9, "mask": mask}],
        labels=("hand", "sword"),
        provider_id="TEST_FROZEN_PROVIDER",
        source_identity="SOURCE:TEST",
        image_shape=(16, 16),
    )
    ref = write_mask_evidence_v1(tmp_path / "semantic.npz", evidence)
    loaded = load_mask_evidence_v1(ref["path"], expected_sha256=ref["sha256"])
    assert loaded.evidence_hash() == evidence.evidence_hash()
    assert loaded.labels == ("hand", "sword")
    assert np.array_equal(loaded.proposal_masks, evidence.proposal_masks)


def test_semantics_lift_to_single_m_and_propose_grip_without_geometry_mutation():
    rest = np.array(
        [
            [-1.2, -0.4, 0.0], [-0.4, -0.4, 0.0], [-0.8, 0.4, 0.0],
            [0.4, -0.4, 0.0], [1.2, -0.4, 0.0], [0.8, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    before = rest.copy()
    hand = np.zeros((64, 64), bool)
    sword = np.zeros((64, 64), bool)
    hand[:, :32] = True
    sword[:, 32:] = True
    evidence = build_mask_evidence_v1(
        [
            {"view_index": 0, "label": "hand", "score": 0.95, "mask": hand},
            {"view_index": 0, "label": "sword", "score": 0.96, "mask": sword},
        ],
        labels=("hand", "sword"),
        provider_id="TEST_FROZEN_PROVIDER",
        source_identity="SOURCE:TEST",
        image_shape=(64, 64),
    )
    mesh = lift_mask_evidence_to_m_v1(
        rest=rest,
        faces=faces,
        cameras=tuple(_camera(v) for v in range(8)),
        evidence=evidence,
        minimum_face_confidence=0.4,
    )
    assert np.array_equal(rest, before), "semantic evidence must not mutate canonical M"
    assert [mesh.label_vocab[i] for i in mesh.face_label_index] == ["hand", "sword"]
    assert len(mesh.owner_ids) == 2
    # Deliberately generous threshold for this synthetic geometry.
    grip = infer_grip_candidates_v1(
        rest=rest, faces=faces, mesh=mesh, maximum_normalized_distance=0.6
    )
    assert len(grip) == 1
    assert grip[0].relation_kind == "GRIP_CANDIDATE"
    assert grip[0].a_owner_id.startswith("OWNER:hand")
    assert grip[0].b_owner_id.startswith("OWNER:sword")
    behavior = {row.semantic_label: row.behavior_class for row in infer_behavior_evidence_v1(mesh)}
    assert behavior["hand"] == "PRIMARY_ARTICULATED"
    assert behavior["sword"] == "RIGID"


def test_source_semantic_overlap_detects_pose_order_reversal():
    # Two disconnected but screen-overlapping triangles.  Cape is closer in rest;
    # arm is behind.  In posed geometry their z order is deliberately reversed.
    rest = np.array(
        [
            [-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0],
            [-0.7, -0.7, 1.0], [0.7, -0.7, 1.0], [0.0, 0.7, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    cape_mask = np.ones((64, 64), bool)
    evidence = build_mask_evidence_v1(
        [{"view_index": 0, "label": "cape", "score": 0.99, "mask": cape_mask}],
        labels=("cape", "arm"),
        provider_id="TEST_FROZEN_PROVIDER",
        source_identity="SOURCE:TEST",
        image_shape=(64, 64),
    )
    mesh = MeshSemanticEvidenceV1(
        label_vocab=("cape", "arm"),
        face_probabilities=np.array([[1.0, 0.0], [0.0, 1.0]], np.float32),
        face_confidence=np.array([0.99, 0.95], np.float32),
        face_support_views=np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], bool),
        face_label_index=np.array([0, 1], np.int16),
        face_owner_index=np.array([0, 1], np.int32),
        owner_ids=("OWNER:cape:00", "OWNER:arm:00"),
        owner_label_index=np.array([0, 1], np.int16),
        owner_face_indices=((0,), (1,)),
        source_evidence_hash=evidence.evidence_hash(),
    )
    cameras = tuple(_camera(v) for v in range(8))
    relations = infer_source_overlap_relations_v1(
        rest=rest,
        faces=faces,
        cameras=cameras,
        evidence=evidence,
        mesh=mesh,
        resolution=64,
        minimum_support_pixels=4,
    )
    assert any(
        r.view_index == 0
        and r.a_owner_id == "OWNER:cape:00"
        and r.b_owner_id == "OWNER:arm:00"
        for r in relations
    )
    posed = rest.copy()
    posed[:3, 2] = 2.0  # cape moves behind
    posed[3:, 2] = 0.0  # arm moves in front
    conflicts = detect_pose_order_conflicts_v1(
        posed=posed,
        faces=faces,
        cameras=cameras,
        mesh=mesh,
        source_relations=relations,
        resolution=64,
    )
    assert any(
        c.relation_kind == "POSE_ORDER_CONFLICT"
        and c.a_owner_id == "OWNER:cape:00"
        and c.b_owner_id == "OWNER:arm:00"
        and c.support_count > 0
        for c in conflicts
    )
