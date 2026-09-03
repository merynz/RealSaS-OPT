from __future__ import annotations

from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation, QualificationError
import pytest


RESOLUTION = 1024
TARGET_VIEW = 0
CAMERA_HASH = "CAM:PIXEL_CENTER:0"


def _surface() -> RiggingSurfaceIR:
    pixels = {
        "a": (255.5, 255.5),
        "b": (767.5, 255.5),
        "c": (255.5, 767.5),
        "d": (767.5, 767.5),
    }
    world = {
        "a": (-0.5, -0.5, 0.0),
        "b": (0.5, -0.5, 0.0),
        "c": (-0.5, 0.5, 0.0),
        "d": (0.5, 0.5, 0.0),
    }
    nodes = tuple(
        SurfaceNode(
            sid,
            world[sid],
            (TARGET_VIEW,),
            (f"PROV:{sid}",),
            (f"OBS:{sid}",),
            ((TARGET_VIEW, pixels[sid]),),
            f"QPV2:{sid}",
            derived_normal=(0.0, 0.0, 1.0),
        )
        for sid in sorted(pixels)
    )
    ids = tuple(sorted(pixels))
    edges = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    relations = tuple(
        SurfaceRelation(
            f"REL:{i}",
            a,
            b,
            "OBSERVED_LOCAL_RASTER_NEIGHBOR",
            1.0,
            {"crosses_unknown": False, "unknown_bridge": False, "source_mesh_used": False},
        )
        for i, (a, b) in enumerate(edges)
    )
    return RiggingSurfaceIR(
        nodes,
        relations,
        "S:PIXEL_CENTER",
        metadata={"raster_coordinate_system": "PIXEL_CENTER_XY", "resolution": RESOLUTION},
    )


def _mesh(surface):
    candidate = build_mwb2_candidate(surface, view_index=TARGET_VIEW, camera_binding_hash=CAMERA_HASH)
    mesh = qualify_mwb2_mesh(surface, candidate)
    assert len(mesh.faces) == 2
    return mesh


def _expected_uv(pixel_xy):
    x, y = map(float, pixel_xy)
    return ((x + 0.5) / float(RESOLUTION), 1.0 - (y + 0.5) / float(RESOLUTION))


def _corner_source(surface, mesh, face_index, corner_index):
    vertices = {v.canonical_mesh_vertex_id: v for v in mesh.vertices}
    nodes = {n.surface_id: n for n in surface.surface_nodes}
    vid = mesh.faces[face_index][corner_index]
    binding = vertices[vid].support_binding
    assert binding.mode == "IDENTITY_SURFACE_NODE"
    assert len(binding.coefficients) == 1 and binding.coefficients[0][1] == 1.0
    sid = binding.coefficients[0][0]
    node = nodes[sid]
    raster = [xy for v, xy in node.raster_bindings if int(v) == TARGET_VIEW]
    assert len(raster) == 1
    return sid, tuple(map(float, raster[0]))


def test_observed_appearance_respects_native_pixel_center_contract_and_provenance():
    surface = _surface()
    mesh = _mesh(surface)
    first = build_observed_appearance_binding(
        surface=surface,
        mesh=mesh,
        target_view_index=TARGET_VIEW,
        camera_binding_hash=CAMERA_HASH,
        observation_hash_by_view={TARGET_VIEW: "OBSERVATION_HASH_A"},
        atlas_payload_hash="ATLAS:A",
    )

    expected_keys = {(fi, ci) for fi, face in enumerate(mesh.faces) for ci in range(len(face))}
    actual_keys = {(c.face_index, c.corner_index) for c in first.corner_bindings}
    assert actual_keys == expected_keys
    assert len(first.corner_bindings) == 6
    assert first.metadata["camera_refit"] is False
    assert first.metadata["source_mesh_uv_used"] is False
    assert first.metadata["unknown_completion_used"] is False

    first_by_key = {(c.face_index, c.corner_index): c for c in first.corner_bindings}
    for key, corner in first_by_key.items():
        _, pixel_xy = _corner_source(surface, mesh, *key)
        expected_uv = _expected_uv(pixel_xy)
        assert corner.authority_class == "OBSERVED_LOCAL"
        assert corner.donor_view_index == TARGET_VIEW
        assert tuple(corner.donor_raster_xy) == pixel_xy
        assert abs(float(corner.material_uv[0]) - expected_uv[0]) <= 1e-12
        assert abs(float(corner.material_uv[1]) - expected_uv[1]) <= 1e-12
        assert 0.0 <= float(corner.material_uv[0]) <= 1.0
        assert 0.0 <= float(corner.material_uv[1]) <= 1.0

    second = build_observed_appearance_binding(
        surface=surface,
        mesh=mesh,
        target_view_index=TARGET_VIEW,
        camera_binding_hash=CAMERA_HASH,
        observation_hash_by_view={TARGET_VIEW: "OBSERVATION_HASH_B"},
        atlas_payload_hash="ATLAS:A",
    )
    second_by_key = {(c.face_index, c.corner_index): c for c in second.corner_bindings}
    assert second.appearance_lineage_hash != first.appearance_lineage_hash
    for key in expected_keys:
        a = first_by_key[key]
        b = second_by_key[key]
        assert a.source_observation_hash != b.source_observation_hash
        assert a.donor_raster_xy == b.donor_raster_xy
        assert a.material_uv == b.material_uv


def test_observed_appearance_camera_mismatch_fails_closed():
    surface = _surface()
    mesh = _mesh(surface)
    with pytest.raises(QualificationError, match="APPEARANCE_CAMERA_LINEAGE_MISMATCH"):
        build_observed_appearance_binding(
            surface=surface,
            mesh=mesh,
            target_view_index=TARGET_VIEW,
            camera_binding_hash="CAM:WRONG",
            observation_hash_by_view={TARGET_VIEW: "OBSERVATION_HASH_A"},
        )
