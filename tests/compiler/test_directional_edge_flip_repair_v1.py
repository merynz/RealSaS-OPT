from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.mesh.directional_edge_flip_repair import (
    repair_directional_mesh_by_edge_flips_v1,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _fixture():
    points = (
        (0.0, 0.0, 0.0),
        (1.1021617230097327, 0.08824696449157315, 0.0),
        (2.000811344156495, 1.259851093044035, 0.0),
        (0.023198170560102815, 0.5431310724699959, 0.0),
    )
    surface_hash = "a" * 64
    nodes = tuple(
        SurfaceNode(
            f"S{i}",
            p,
            (0,),
            (),
            (),
            raster_bindings=((0, (p[0], p[1])),),
        )
        for i, p in enumerate(points)
    )
    surface = RiggingSurfaceIR(nodes, geometry_lineage_hash=surface_hash)
    vertices = tuple(
        QualifiedMeshVertex(
            f"V{i}",
            p,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((f"S{i}", 1.0),)),
            metadata={"raster_xy": (p[0], p[1])},
        )
        for i, p in enumerate(points)
    )
    mesh = QualifiedEditableMeshIR(
        vertices=vertices,
        faces=(("V0", "V1", "V2"), ("V0", "V2", "V3")),
        edges=(
            ("V0", "V1"),
            ("V1", "V2"),
            ("V2", "V3"),
            ("V3", "V0"),
            ("V0", "V2"),
        ),
        surface_binding_hash=surface_hash,
        view_index=0,
        camera_binding_hash="b" * 64,
        qualification_report={"status": "TEST"},
        mesh_lineage_hash="",
    )
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))
    return surface, mesh


def _tri_union_area(mesh):
    xy = {
        v.canonical_mesh_vertex_id: v.metadata["raster_xy"]
        for v in mesh.vertices
    }
    def area(face):
        a,b,c=(xy[x] for x in face)
        return abs(
            (b[0]-a[0])*(c[1]-a[1])
            - (b[1]-a[1])*(c[0]-a[0])
        ) * 0.5
    return sum(area(face) for face in mesh.faces)


def test_edge_flip_improves_bad_quad_without_changing_vertices_or_coverage_union():
    surface, mesh = _fixture()
    repaired = repair_directional_mesh_by_edge_flips_v1(
        mesh,
        surface,
        min_angle_deg=20.0,
        max_aspect=4.5,
    )
    report = repaired.qualification_report["directional_edge_flip_repair"]
    assert report["accepted_flip_count"] == 1
    assert report["bad_face_count_before"] > report["bad_face_count_after"]
    assert report["bad_face_count_after"] == 0
    assert tuple(v.to_dict() for v in repaired.vertices) == tuple(v.to_dict() for v in mesh.vertices)
    assert _tri_union_area(repaired) == _tri_union_area(mesh)
    assert ("V0", "V2") not in {
        tuple(sorted(edge)) for edge in repaired.edges
    }  # edges are normalized by the repair in the next assertion path


def test_edge_flip_is_deterministic():
    surface, mesh = _fixture()
    a = repair_directional_mesh_by_edge_flips_v1(
        mesh, surface, min_angle_deg=20.0, max_aspect=4.5
    )
    b = repair_directional_mesh_by_edge_flips_v1(
        mesh, surface, min_angle_deg=20.0, max_aspect=4.5
    )
    assert a.mesh_lineage_hash == b.mesh_lineage_hash
    assert a.faces == b.faces
