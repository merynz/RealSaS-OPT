from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
)
from compiler.realsas_compiler_services.export.current_v4_directional_runtime_v4 import (
    project_directional_product_bakes_to_runtime_v4,
)
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
)
from compiler.realsas_compiler_services.proof.motion_bake import (
    bind_qualification_owned_motion_bake,
)


VIEWS = ("V0", "V1")
BODY = "BODY_UNDERLAY"
FG = "HAT_FOREGROUND"


def _camera(view_id: str, index: int) -> CameraProjectionV3:
    return CameraProjectionV3(
        view_id=view_id,
        view_index=index,
        origin=(0.0, 0.0, -5.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=32.0,
        resolution=64,
    )


def _texture(index: int) -> RuntimeTexturePayloadV1:
    image_hash = ("%x" % (index + 1)) * 64
    relpath = f"textures/V{index}.png"
    return RuntimeTexturePayloadV1(
        view_index=index,
        image_relpath=relpath,
        image_sha256=image_hash,
        image_crc32=index + 10,
        width=64,
        height=64,
        atlas_payload_hash=content_sha256({
            "image_sha256": image_hash,
            "image_relpath": relpath,
        }),
    )


def _component(
    view: int,
    component_id: str,
    setup_order: int,
    atlas_hash: str,
    *,
    pinched: bool = False,
):
    xy = (
        ((10.0, 10.0), (22.0, 10.0), (10.0, 22.0), (34.0, 10.0), (34.0, 22.0))
        if pinched
        else ((10.0, 10.0), (22.0, 10.0), (10.0, 22.0))
    )
    vertices = tuple(
        SimpleNamespace(
            canonical_mesh_vertex_id=f"{component_id}:V{view}:{i}",
            P=(float(i == 1), 0.0, float(i == 2)),
            metadata={"raster_xy": p, "source_raster_xy": p},
        )
        for i, p in enumerate(xy)
    )
    faces = (
        (
            (vertices[0].canonical_mesh_vertex_id, vertices[1].canonical_mesh_vertex_id, vertices[2].canonical_mesh_vertex_id),
            (vertices[0].canonical_mesh_vertex_id, vertices[3].canonical_mesh_vertex_id, vertices[4].canonical_mesh_vertex_id),
        )
        if pinched
        else ((vertices[0].canonical_mesh_vertex_id, vertices[1].canonical_mesh_vertex_id, vertices[2].canonical_mesh_vertex_id),)
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        view_index=view,
        mesh_lineage_hash=content_sha256({"mesh": component_id, "view": view}),
        qualification_report={
            "source_alpha_recall": 1.0,
            "precision_inside_alpha": 1.0,
        },
    )
    mesh_skin = SimpleNamespace(
        mesh_skin_lineage_hash=content_sha256({"skin": component_id, "view": view}),
        metadata={"parent_joint_id": "root"} if component_id != BODY else {},
    )
    vertex_raster = {
        vertex.canonical_mesh_vertex_id: xy[i]
        for i, vertex in enumerate(vertices)
    }
    corners = tuple(
        SimpleNamespace(
            face_index=face_index,
            corner_index=corner_index,
            material_uv=(0.0, 0.0),
            donor_view_index=view,
            donor_raster_xy=vertex_raster[vertex_id],
            authority_class="OBSERVED_LOCAL",
        )
        for face_index, face in enumerate(faces)
        for corner_index, vertex_id in enumerate(face)
    )
    appearance = SimpleNamespace(
        target_view_index=view,
        atlas_payload_hash=atlas_hash,
        appearance_lineage_hash=content_sha256({"appearance": component_id, "view": view}),
        corner_bindings=corners,
    )
    return SimpleNamespace(
        component_id=component_id,
        view_index=view,
        mesh=mesh,
        mesh_skin=mesh_skin,
        appearance=appearance,
        setup_order=setup_order,
        component_state_hash=content_sha256({"component": component_id, "view": view}),
    )


def _fixture(*, body_first: bool = True, pinched_body: bool = False):
    textures = tuple(_texture(i) for i in range(2))
    directions = []
    for view in range(2):
        body = _component(
            view,
            BODY,
            0,
            textures[view].atlas_payload_hash,
            pinched=pinched_body,
        )
        fg = _component(view, FG, 1, textures[view].atlas_payload_hash)
        directions.append(SimpleNamespace(
            view_index=view,
            components=(body, fg),
        ))

    continuity_views = tuple(
        {
            "view_index": i,
            "substrate_component_id": BODY,
            "metadata": {
                "new_pixels_generated": False,
                "topology_mutated": False,
                "weights_mutated": False,
                "pixel_authority": "EXACT_SUBSTRATE_APPEARANCE_REUSE_ONLY",
            },
        }
        for i in range(2)
    )
    continuity_payload = {
        "views": continuity_views,
        "schema_version": "RealSaS.QualifiedContinuityUnderlaySetIR.v1",
    }
    renderables = SimpleNamespace(
        directions=tuple(directions),
        metadata={
            "continuity_underlay_qualified": True,
            "continuity_underlay_set_hash": "c" * 64,
            "mechanical_continuity_underlay": continuity_payload,
        },
    )
    product = SimpleNamespace(
        product_state_hash="a" * 64,
        directional_renderables=renderables,
        mechanical_state=SimpleNamespace(
            skeleton=SimpleNamespace(
                joints=(SimpleNamespace(
                    canonical_joint_id="root",
                    parent_canonical_id=None,
                ),)
            )
        ),
    )

    rest = {}
    triangles = {}
    frames = []
    for view in range(2):
        for component_id in (BODY, FG):
            mesh_id = f"V{view}:{component_id}"
            offset = 0.0 if component_id == BODY else 4.0
            if component_id == BODY and pinched_body:
                pts = (
                    (10.0, 10.0),
                    (22.0, 10.0),
                    (10.0, 22.0),
                    (34.0, 10.0),
                    (34.0, 22.0),
                )
                tris = ((0, 1, 2), (0, 3, 4))
            else:
                pts = (
                    (10.0 + offset, 10.0),
                    (22.0 + offset, 10.0),
                    (10.0 + offset, 22.0),
                )
                tris = ((0, 1, 2),)
            rest[mesh_id] = pts
            triangles[mesh_id] = tris

    for time_seconds, delta in ((0.0, 0.0), (1.0, 1.0)):
        meshes = {
            mesh_id: tuple((x + delta, y + 0.25 * delta) for x, y in points)
            for mesh_id, points in rest.items()
        }
        orders = {
            f"V{view}": (
                (f"V{view}:{BODY}", f"V{view}:{FG}")
                if body_first
                else (f"V{view}:{FG}", f"V{view}:{BODY}")
            )
            for view in range(2)
        }
        frames.append({
            "time_seconds": time_seconds,
            "mesh_vertices_by_id": meshes,
            "render_order_by_view": orders,
        })

    bake = bind_qualification_owned_motion_bake(
        source_product_state_hash=product.product_state_hash,
        proof_plan_hash="b" * 64,
        clip_id="idle",
        duration_seconds=1.0,
        fps=1.0,
        loop=True,
        evaluator_semantic_version="TEST_DIRECTIONAL_ASSEMBLY_EVALUATOR.v1",
        evaluator_binding_hash="d" * 64,
        sampling_policy="TEST_TWO_ENDPOINTS",
        rest_mesh_vertices_by_id=rest,
        triangles_by_mesh_id=triangles,
        frame_rows=tuple(frames),
    )
    cameras = {view_id: _camera(view_id, i) for i, view_id in enumerate(VIEWS)}
    return product, (bake,), textures, cameras


def test_directional_assembly_bridge_roundtrips_qualification_bake_xy_exactly():
    product, bakes, textures, cameras = _fixture()
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    assert len(projection.contract.assets) == 4
    assert len(projection.contract.slots) == 2
    assert projection.body_component_id == BODY

    bake_frames = bakes[0].frames
    for runtime_frame, bake_frame in zip(projection.clips[0].frames, bake_frames):
        mesh_xy = dict(bake_frame.mesh_vertices_by_id)
        for asset in projection.contract.assets:
            owner = projection.owner_view_by_asset[asset.asset_id]
            component_id = projection.component_id_by_asset[asset.asset_id]
            source_indices = projection.runtime_source_indices_by_asset[asset.asset_id]
            expected = np.asarray(
                [mesh_xy[f"V{owner}:{component_id}"][i] for i in source_indices],
                dtype=np.float64,
            )
            actual = project_points_xyz_v3(
                runtime_frame.canonical_posed_xyz_by_asset[asset.asset_id],
                projection.contract.views[owner].camera,
            )[:, :2]
            assert np.allclose(actual, expected, atol=2e-5, rtol=0.0)



def test_reference_runtime_projection_accepts_pinched_triangle_soup_without_global_boundary_theorem():
    # Two body triangles share one canonical vertex and otherwise form separate fans.
    # This is legal raster input; no single degree-2 global boundary loop is required.
    product, bakes, textures, cameras = _fixture(pinched_body=True)
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEWS,
        body_component_id=BODY,
    )
    assert len(projection.clips) == 1
    assert projection.clips[0].runtime_qualified is False
    body_assets = [
        asset for asset in projection.contract.assets
        if asset.slot_id == BODY
    ]
    assert len(body_assets) == len(VIEWS)
    assert all(asset.face_count == 2 for asset in body_assets)

