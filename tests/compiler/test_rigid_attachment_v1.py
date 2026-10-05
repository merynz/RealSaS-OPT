from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.rigid_attachment_v1 import (
    RigidAttachmentBindingV1IR,
    apply_rigid_attachment_weights_v1,
    build_rigid_loadout_face_mask_v1,
    rigid_attachment_edge_report_v1,
    validate_rigid_attachment_bindings_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _bindings():
    return (
        RigidAttachmentBindingV1IR(
            attachment_id="sword",
            joint_id="hand.r",
            component_ids=(1,),
            variant_group_id="weapon",
            metadata={"deformable_skin_interpolation_allowed": False},
        ),
        RigidAttachmentBindingV1IR(
            attachment_id="shield.round",
            joint_id="hand.l",
            component_ids=(2,),
            variant_group_id="shield",
            metadata={"deformable_skin_interpolation_allowed": False},
        ),
        RigidAttachmentBindingV1IR(
            attachment_id="shield.spike",
            joint_id="hand.l",
            component_ids=(3,),
            variant_group_id="shield",
            metadata={"deformable_skin_interpolation_allowed": False},
        ),
    )


def test_rigid_attachment_overrides_skin_and_loadout_masks_variants():
    components = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    weights = np.full((12, 3), 1.0 / 3.0, dtype=np.float64)
    out = apply_rigid_attachment_weights_v1(
        weights,
        vertex_component_ids=components,
        bindings=_bindings(),
        joint_index_by_id={"body": 0, "hand.r": 1, "hand.l": 2},
    )
    assert np.allclose(out[components == 1], np.asarray([0.0, 1.0, 0.0]))
    assert np.allclose(out[components == 2], np.asarray([0.0, 0.0, 1.0]))
    assert np.allclose(out[components == 3], np.asarray([0.0, 0.0, 1.0]))

    faces = np.asarray([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ], dtype=np.int64)
    mask = build_rigid_loadout_face_mask_v1(
        faces,
        vertex_component_ids=components,
        bindings=_bindings(),
        selected_attachment_ids=("sword", "shield.round"),
    )
    assert mask.tolist() == [True, True, True, False]


def test_loadout_rejects_two_variants_from_same_group():
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    components = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bindings = (
        RigidAttachmentBindingV1IR(
            "a", "hand", (0,), "shield",
            {"deformable_skin_interpolation_allowed": False},
        ),
        RigidAttachmentBindingV1IR(
            "b", "hand", (1,), "shield",
            {"deformable_skin_interpolation_allowed": False},
        ),
    )
    with pytest.raises(QualificationError, match="RIGID_ATTACHMENT_LOADOUT_GROUP_CONFLICT"):
        build_rigid_loadout_face_mask_v1(
            faces,
            vertex_component_ids=components,
            bindings=bindings,
            selected_attachment_ids=("a", "b"),
        )


def test_validator_rejects_component_claim_conflict():
    bindings = (
        RigidAttachmentBindingV1IR(
            "a", "hand.l", (1,), "weapon",
            {"deformable_skin_interpolation_allowed": False},
        ),
        RigidAttachmentBindingV1IR(
            "b", "hand.r", (1,), "shield",
            {"deformable_skin_interpolation_allowed": False},
        ),
    )
    with pytest.raises(QualificationError, match="RIGID_ATTACHMENT_COMPONENT_CLAIM_CONFLICT"):
        validate_rigid_attachment_bindings_v1(
            bindings,
            known_component_ids=(0, 1),
            known_joint_ids=("hand.l", "hand.r"),
        )


def test_rigid_edge_report_detects_exact_rigid_motion():
    rest = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    components = np.asarray([1, 1, 1], dtype=np.int64)
    theta = np.deg2rad(35.0)
    R = np.asarray([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    posed = rest @ R.T + np.asarray([2.0, -1.0, 0.3])
    poses = np.stack((rest, posed), axis=0)
    binding = RigidAttachmentBindingV1IR(
        "prop", "hand", (1,), "weapon",
        {"deformable_skin_interpolation_allowed": False},
    )
    report = rigid_attachment_edge_report_v1(
        rest, poses, faces,
        vertex_component_ids=components,
        binding=binding,
    )
    assert report["rigid_edge_preservation_pass"] is True
    assert report["max_abs_edge_ratio_minus_one"] < 1e-12
