from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    FaceAppearanceAuthorityV3,
    build_face_appearance_patches_v3,
    build_full_surface_runtime_meshes_v3,
    project_points_xyz_v3,
    project_posed_full_surface_frames_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    RuntimeV3PlaybackContract,
    RuntimeV3Slot,
    validate_playback_runtime_v3_contract,
)
from compiler.realsas_compiler_core.types import QualificationError


def _camera(view_index: int, *, resolution: int = 4) -> dict:
    return {
        "view_index": view_index,
        "origin": [0.0, 0.0, -10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 1.0,
        "resolution": resolution,
    }


def _surface():
    vertices = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    # The second triangle intentionally uses its input winding/order verbatim.
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return vertices, faces


def _uv(view_ids, vertex_count):
    base = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    assert len(base) == vertex_count
    return {view_id: base.copy() for view_id in view_ids}


def test_projector_uses_runtime_half_pixel_screen_and_same_pose_for_depth():
    camera = qualify_camera_v3(_camera(0), view_id="V0", view_index=0)
    points = np.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=np.float64)
    xyz = project_points_xyz_v3(points, camera)
    assert xyz[0].tolist() == pytest.approx([2.0, 2.0, 10.0])
    assert xyz[1].tolist() == pytest.approx([2.5, 3.0, 11.0])


def test_full_surface_bridge_preserves_every_face_order_and_winding_in_all_8_views():
    vertices, faces = _surface()
    views = tuple(f"V{i}" for i in range(8))
    cameras = {view_id: _camera(i) for i, view_id in enumerate(views)}
    bridge = build_full_surface_runtime_meshes_v3(
        vertices,
        faces,
        cameras,
        _uv(views, len(vertices)),
        slot_id="body_slot",
        attachment_id="body_attachment",
    )
    assert bridge.vertex_count == len(vertices)
    assert bridge.face_count == len(faces)
    assert len(bridge.meshes) == 8
    expected_faces = tuple(tuple(map(int, row)) for row in faces)
    for i, mesh in enumerate(bridge.meshes):
        assert mesh.view_id == f"V{i}"
        assert mesh.mesh_id == f"V{i}:body_attachment"
        assert len(mesh.vertices) == len(vertices)
        assert mesh.triangles == expected_faces


def test_product_full_surface_api_has_no_face_subset_parameter():
    params = inspect.signature(build_full_surface_runtime_meshes_v3).parameters
    assert "selected_face_indices" not in params
    assert "face_indices" not in params


def test_uv_is_explicit_appearance_input_not_invented_by_geometry_bridge():
    vertices, faces = _surface()
    with pytest.raises(TypeError):
        build_full_surface_runtime_meshes_v3(
            vertices,
            faces,
            {"V0": _camera(0)},
            slot_id="body_slot",
            attachment_id="body_attachment",
            required_view_ids=("V0",),
        )
    bad_uv = {"V0": np.full((len(vertices), 2), 1.5, dtype=np.float64)}
    with pytest.raises(QualificationError, match="FULL_SURFACE_V3_UV_OUTSIDE_UNIT_DOMAIN"):
        build_full_surface_runtime_meshes_v3(
            vertices,
            faces,
            {"V0": _camera(0)},
            bad_uv,
            slot_id="body_slot",
            attachment_id="body_attachment",
            required_view_ids=("V0",),
        )


def test_posed_projection_recomputes_xy_and_depth_from_same_3d_points():
    vertices, _faces = _surface()
    posed = vertices.copy()
    posed[:, 0] += 0.25
    posed[:, 2] += 1.5
    out = project_posed_full_surface_frames_v3(
        posed,
        {"V0": _camera(0)},
        attachment_id="body_attachment",
        expected_vertex_count=len(vertices),
        required_view_ids=("V0",),
    )
    xyz = np.asarray(out["V0:body_attachment"], dtype=np.float64)
    rest_camera = qualify_camera_v3(_camera(0), view_id="V0", view_index=0)
    expected = project_points_xyz_v3(posed, rest_camera)
    assert xyz == pytest.approx(expected)
    assert np.all(xyz[:, 2] == pytest.approx(11.5))


def test_face_appearance_builder_preserves_unseen_as_unbound_and_contract_validates():
    vertices, faces = _surface()
    bridge = build_full_surface_runtime_meshes_v3(
        vertices,
        faces,
        {"V0": _camera(0)},
        _uv(("V0",), len(vertices)),
        slot_id="body_slot",
        attachment_id="body_attachment",
        required_view_ids=("V0",),
    )
    mesh = bridge.meshes[0]
    patches = build_face_appearance_patches_v3(
        mesh=mesh,
        face_authorities=(
            FaceAppearanceAuthorityV3(AppearanceProvenance.DIRECT_SOURCE, 0, "V0"),
            FaceAppearanceAuthorityV3(AppearanceProvenance.UNSEEN, None, None),
        ),
    )
    assert sum(len(p.face_indices) for p in patches) == len(faces)
    unseen = next(p for p in patches if p.provenance == AppearanceProvenance.UNSEEN)
    assert unseen.donor_view_index is None
    assert unseen.atlas_id is None

    contract = RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        meshes=bridge.meshes,
        appearance_patches=patches,
    )
    assert len(validate_playback_runtime_v3_contract(contract, required_view_ids=("V0",))) == 64


def test_face_appearance_builder_does_not_choose_missing_source_donor():
    vertices, faces = _surface()
    bridge = build_full_surface_runtime_meshes_v3(
        vertices,
        faces,
        {"V0": _camera(0)},
        _uv(("V0",), len(vertices)),
        slot_id="body_slot",
        attachment_id="body_attachment",
        required_view_ids=("V0",),
    )
    with pytest.raises(QualificationError, match="SOURCE_APPEARANCE_REQUIRES_DONOR_AND_ATLAS"):
        build_face_appearance_patches_v3(
            mesh=bridge.meshes[0],
            face_authorities=(
                FaceAppearanceAuthorityV3(AppearanceProvenance.DIRECT_SOURCE, None, None),
                FaceAppearanceAuthorityV3(AppearanceProvenance.UNSEEN, None, None),
            ),
        )


def test_bridge_is_subject_agnostic():
    source = inspect.getsource(__import__(
        "compiler.realsas_compiler_core.playback_full_surface_v3",
        fromlist=["dummy"],
    )).lower()
    assert "mage" not in source
