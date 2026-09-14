from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.appearance_atlas import (
    build_sprite_panel_appearance,
    rebind_local_observed_appearance_to_atlas_panel,
)
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    QualificationError,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import build_appearance_binding
from compiler.realsas_compiler_core.v4_types import AppearanceCornerBinding


def _mesh():
    verts = (
        QualifiedMeshVertex("V0", (0, 0, 0), SurfaceSupportBinding("X", (("S0", 1.0),)), metadata={"raster_xy": (0.0, 0.0), "source_raster_xy": (0.0, 0.0)}),
        QualifiedMeshVertex("V1", (1, 0, 0), SurfaceSupportBinding("X", (("S1", 1.0),)), metadata={"raster_xy": (3.0, 0.0), "source_raster_xy": (3.0, 0.0)}),
        QualifiedMeshVertex("V2", (0, 1, 0), SurfaceSupportBinding("X", (("S2", 1.0),)), metadata={"raster_xy": (0.0, 3.0), "source_raster_xy": (0.0, 3.0)}),
    )
    mesh = QualifiedEditableMeshIR(
        verts,
        (("V0", "V1", "V2"),),
        (("V0", "V1"), ("V1", "V2"), ("V2", "V0")),
        "S",
        0,
        "CAM",
        {"status": "PASS"},
        "",
    )
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def _appearance(mesh):
    corners = (
        AppearanceCornerBinding(0, 0, (0.125, 0.875), 0, (0.0, 0.0), "OBS", "OBSERVED_LOCAL"),
        AppearanceCornerBinding(0, 1, (0.875, 0.875), 0, (3.0, 0.0), "OBS", "OBSERVED_LOCAL"),
        AppearanceCornerBinding(0, 2, (0.125, 0.125), 0, (0.0, 3.0), "OBS", "OBSERVED_LOCAL"),
    )
    return build_appearance_binding(
        target_view_index=0,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash="CAM",
        corner_bindings=corners,
        atlas_payload_hash="OLD",
        metadata={"material_uv_convention": "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1"},
    )


def test_rebind_preserves_donor_selection_and_moves_only_atlas_coordinates():
    mesh = _mesh()
    source = _appearance(mesh)
    out = rebind_local_observed_appearance_to_atlas_panel(
        source,
        source_width=4,
        source_height=4,
        atlas_width=20,
        atlas_height=4,
        panel_x_offset=0,
        atlas_payload_hash="ATLAS",
    )
    assert out.mesh_binding_hash == source.mesh_binding_hash
    assert out.metadata["source_appearance_lineage_hash"] == source.appearance_lineage_hash
    assert [corner.source_observation_hash for corner in out.corner_bindings] == ["OBS"] * 3
    assert out.corner_bindings[1].donor_raster_xy == (3.0, 0.0)
    assert out.corner_bindings[1].material_uv == ((3.0 + 0.5) / 20.0, 1.0 - (0.0 + 0.5) / 4.0)


def test_sprite_panel_appearance_uses_exact_panel_offset():
    mesh = _mesh()
    out = build_sprite_panel_appearance(
        mesh,
        target_view_index=0,
        source_observation_hash="OBS",
        source_width=4,
        source_height=4,
        atlas_width=20,
        atlas_height=4,
        panel_x_offset=8,
        atlas_payload_hash="ATLAS",
    )
    assert out.corner_bindings[0].donor_raster_xy == (8.0, 0.0)
    assert out.corner_bindings[1].donor_raster_xy == (11.0, 0.0)
    assert out.metadata["new_pixels_generated"] is False


def test_atlas_rebind_rejects_cross_view_donor():
    mesh = _mesh()
    source = _appearance(mesh)
    corners = (
        replace(source.corner_bindings[0], donor_view_index=1, authority_class="OBSERVED_CROSS_VIEW"),
    ) + source.corner_bindings[1:]
    bad = build_appearance_binding(
        target_view_index=0,
        mesh_binding_hash=source.mesh_binding_hash,
        camera_binding_hash=source.camera_binding_hash,
        corner_bindings=corners,
        atlas_payload_hash=source.atlas_payload_hash,
        metadata=source.metadata,
    )
    with pytest.raises(QualificationError, match="LOCAL_OBSERVED_DONOR"):
        rebind_local_observed_appearance_to_atlas_panel(
            bad,
            source_width=4,
            source_height=4,
            atlas_width=20,
            atlas_height=4,
            panel_x_offset=0,
            atlas_payload_hash="ATLAS",
        )
