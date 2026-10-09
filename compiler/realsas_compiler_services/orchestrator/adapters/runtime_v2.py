from __future__ import annotations

"""V2 Stage42-45 deterministic runtime projection, package, native proof and DVI."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    complete_appearance_asset_from_dict,
    complete_appearance_qualification_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    render_caa_reference,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    qualified_dynamic_motion_v2_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.dynamic_appearance_conditioning_v2 import (
    dynamic_face_conditioning_metrics,
    screen_to_texture_max_texels_per_pixel,
    validate_dynamic_appearance_policy,
)
from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import (
    dynamic_visibility_load_gate,
    interior_shared_edge_projection_continuity,
    projected_orientation_flip,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_mesh_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    complete_puppet_state_v2_from_dict,
)
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    DynamicVisualIntegrityV2IR,
    NativePlaybackProbeV2IR,
    NativePlaybackV2IR,
    RuntimeClipV2IR,
    RuntimePackageSealV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    dynamic_visual_integrity_hash,
    native_playback_hash,
    native_playback_probe_hash,
    native_playback_from_dict,
    runtime_package_hash,
    runtime_package_seal_from_dict,
    runtime_projection_from_dict,
    runtime_projection_hash,
)
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    SourceOwnedVisualDynamicIntegrityV1IR,
    SourceOwnedVisualRuntimeClipV1IR,
    SourceOwnedVisualRuntimeProjectionV1IR,
    SourceOwnedVisualRuntimeViewV1IR,
    source_owned_visual_dynamic_integrity_hash,
    source_owned_visual_runtime_projection_hash,
    source_owned_visual_runtime_projection_from_dict,
)
from compiler.realsas_compiler_core.visual_presentation_v1 import (
    load_qualified_visual_presentation_view,
    qualified_visual_presentation_set_from_dict,
)
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    bind_region_visual_vertices_to_mechanical_affine_v1,
    evaluate_region_visual_binding_v1,
)
from compiler.realsas_compiler_core.visual_domain_v2 import (
    OPERATOR_ID, POLICY, DEPTH_CONTRACT, build_domain_binding,
    evaluate_domain_binding, domain_binding_from_arrays,
    presentation_condition_metrics,
)
from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import (
    OPERATOR_ID as ATTACHMENT_OPERATOR, POLICY as ATTACHMENT_POLICY,
    evaluate_attachment_motion, attachment_slot_carrier_residual,
)
from compiler.realsas_compiler_core.presentation_attachment_v1 import visual_vertex_attachment_owners
from compiler.realsas_compiler_core.visual_motion_blend_v1 import (
    OPERATOR_ID as MOTION_BLEND_OPERATOR, POLICY as MOTION_BLEND_POLICY,
    build_motion_blend_coefficients, evaluate_motion_blend, canonical_pose_palette_residual,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_rss_v2_entries,
    build_source_owned_visual_rss_v2_entries,
    read_rss_v2,
    write_rss_v2,
)
from compiler.realsas_compiler_core.visibility_v2 import (
    VISIBILITY_CONTRACT_V2_HASH,
    rasterize_visible_owner,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def _save_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def _projection_arrays(projection):
    path = resolved_path(projection.projection_npz_path)
    if not path.is_file() or sha256_file(path) != projection.projection_npz_sha256:
        raise QualificationError("RUNTIME_V2_PROJECTION_ARRAY_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}



def _runtime_view_page_rows(view) -> tuple[dict, ...]:
    metadata = dict(view.metadata or {})
    rows = tuple(metadata.get("texture_pages") or ())
    if not rows:
        return (
            {
                "page_index": 0,
                "path": str(view.texture_path),
                "sha256": str(view.texture_sha256),
            },
        )
    ordered = tuple(
        sorted((dict(row) for row in rows), key=lambda row: int(row["page_index"]))
    )
    if tuple(int(row["page_index"]) for row in ordered) != tuple(range(len(ordered))):
        raise QualificationError("RUNTIME_V2_TEXTURE_PAGE_INDEX_DRIFT")
    if (
        str(ordered[0]["path"]) != str(view.texture_path)
        or str(ordered[0]["sha256"]) != str(view.texture_sha256)
    ):
        raise QualificationError("RUNTIME_V2_PRIMARY_TEXTURE_PAGE_DRIFT")
    return ordered


def _load_runtime_view_texture(view) -> np.ndarray:
    rows = _runtime_view_page_rows(view)
    images = []
    for row in rows:
        path = resolved_path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise QualificationError("RUNTIME_V2_TEXTURE_PAGE_BYTES_DRIFT")
        images.append(np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8))
    if len({image.shape for image in images}) != 1:
        raise QualificationError("RUNTIME_V2_TEXTURE_PAGE_SHAPE_DRIFT")
    if bool(dict(view.metadata or {}).get("paged_atlas")):
        return np.stack(images, axis=0)
    if len(images) != 1:
        raise QualificationError("RUNTIME_V2_LEGACY_TEXTURE_PAGE_COUNT_DRIFT")
    return images[0]

def _largest_connected_fraction(mask: np.ndarray, *, denominator: int) -> float:
    grid = np.asarray(mask, dtype=bool)
    if grid.ndim != 2:
        raise QualificationError("RUNTIME_V2_CONNECTED_MASK_DIMENSION_INVALID")
    if denominator <= 0 or not np.any(grid):
        return 0.0
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    largest = 0
    for y0, x0 in np.argwhere(grid):
        y0 = int(y0)
        x0 = int(x0)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [(y0, x0)]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and grid[yy, xx]
                    and not seen[yy, xx]
                ):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        largest = max(largest, size)
    return float(largest) / float(denominator)


def _pixel_face_counts(
    face_grid: np.ndarray,
    valid_grid: np.ndarray,
    *,
    face_count: int,
) -> np.ndarray:
    faces = np.asarray(face_grid, dtype=np.int64)
    valid = np.asarray(valid_grid, dtype=bool)
    if faces.shape != valid.shape or faces.ndim != 3:
        raise QualificationError("RUNTIME_V2_PIXEL_FACE_GRID_SHAPE_INVALID")
    height, width, _ = faces.shape
    ys, xs, slots = np.nonzero(valid & (faces >= 0))
    if not len(ys):
        return np.zeros(int(face_count), dtype=np.int64)
    selected = faces[ys, xs, slots]
    if np.any(selected >= int(face_count)):
        raise QualificationError("RUNTIME_V2_PIXEL_FACE_INDEX_DRIFT")
    pixel = ys.astype(np.int64) * int(width) + xs.astype(np.int64)
    packed = selected * (int(height) * int(width)) + pixel
    unique = np.unique(packed)
    unique_faces = unique // (int(height) * int(width))
    return np.bincount(unique_faces, minlength=int(face_count))



def _visual_mesh_motion_metrics(
    rest_positions: np.ndarray,
    posed_positions: np.ndarray,
    faces: np.ndarray,
) -> dict:
    rest = np.asarray(rest_positions, dtype=np.float64)
    posed = np.asarray(posed_positions, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if (
        rest.ndim != 2
        or rest.shape[1] != 2
        or posed.shape != rest.shape
        or tri.ndim != 2
        or tri.shape[1] != 3
        or np.any(tri < 0)
        or np.any(tri >= len(rest))
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_MOTION_METRIC_INPUT_INVALID"
        )
    if not np.isfinite(rest).all() or not np.isfinite(posed).all():
        raise QualificationError("SOURCE_VISUAL_RUNTIME_NONFINITE_POSITION")
    edges = set()
    for face in tri.tolist():
        for a, b in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edges.add(tuple(sorted((int(a), int(b)))))
    if not edges:
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_MOTION_METRIC_EDGE_EMPTY"
        )
    edge_rows = np.asarray(sorted(edges), dtype=np.int64)
    rest_len = np.linalg.norm(
        rest[edge_rows[:, 1]] - rest[edge_rows[:, 0]], axis=1
    )
    posed_len = np.linalg.norm(
        posed[edge_rows[:, 1]] - posed[edge_rows[:, 0]], axis=1
    )
    valid = (rest_len > 1.0e-12) & (posed_len > 1.0e-12)
    ratio = np.ones_like(rest_len)
    ratio[valid] = np.maximum(
        posed_len[valid] / rest_len[valid],
        rest_len[valid] / posed_len[valid],
    )
    ratio[~valid & (rest_len > 1.0e-12)] = np.inf

    def signed_double_area(points):
        a = points[tri[:, 0]]
        b = points[tri[:, 1]]
        c = points[tri[:, 2]]
        return (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    rest_area = signed_double_area(rest)
    posed_area = signed_double_area(posed)
    measurable = np.abs(rest_area) > 1.0e-12
    flipped = measurable & (
        np.sign(rest_area) != np.sign(posed_area)
    )
    finite_ratio = ratio[np.isfinite(ratio)]
    return {
        "edge_count": int(len(edge_rows)),
        "flipped_triangle_count": int(np.count_nonzero(flipped)),
        "edge_gt_4_count": int(np.count_nonzero(ratio > 4.0)),
        "edge_gt_10_count": int(np.count_nonzero(ratio > 10.0)),
        "p95_edge_ratio": (
            float(np.percentile(finite_ratio, 95.0))
            if len(finite_ratio)
            else float("inf")
        ),
        "max_edge_ratio": (
            float(np.max(finite_ratio))
            if len(finite_ratio)
            else float("inf")
        ),
    }


def _build_source_owned_visual_runtime_projection(
    ctx: dict,
    *,
    complete,
    mesh,
    dynamic,
    asset,
    appearance,
    cameras,
) -> dict:
    qualified_visual = qualified_visual_presentation_set_from_dict(
        stage_output_payload(
            ctx,
            "37_QUALIFIED_PRESENTATION_STRUCTURE",
            "RealSaS.QualifiedVisualPresentationSetIR.v1",
        )
    )
    complete_visual_hash = str(
        dict(complete.metadata or {}).get(
            "visual_mesh_set_binding_hash"
        )
        or ""
    )
    if complete_visual_hash != qualified_visual.set_hash:
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_COMPLETE_PUPPET_BINDING_DRIFT"
        )
    if (
        qualified_visual.mechanical_mesh_binding_hash
        != mesh.mesh_lineage_hash
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_MECHANICAL_MESH_DRIFT"
        )
    if (
        qualified_visual.appearance_asset_binding_hash
        != asset.asset_hash
        or qualified_visual.appearance_qualification_binding_hash
        != appearance.qualification_hash
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_APPEARANCE_BINDING_DRIFT"
        )
    if (
        dynamic.mechanical_state_binding_hash
        != complete.mechanical_state_binding_hash
        or dynamic.presentation_binding_hash
        != complete.presentation_graph_binding_hash
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_DYNAMIC_BINDING_DRIFT"
        )

    vertex_ids = [
        str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices
    ]
    vertex_index = {
        vertex_id: index for index, vertex_id in enumerate(vertex_ids)
    }
    if len(vertex_index) != len(vertex_ids):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_DUPLICATE_MECHANICAL_VERTEX_ID"
        )
    rest_mechanical = np.asarray(
        [vertex.P for vertex in mesh.vertices], dtype=np.float64
    )
    mechanical_faces = np.asarray(
        [
            [vertex_index[str(vertex_id)] for vertex_id in face]
            for face in mesh.faces
        ],
        dtype=np.int64,
    )
    camera_by_view = {
        int(row.view_index): row for row in cameras.cameras
    }
    texture_by_view = {
        int(row.direction_index): row for row in asset.textures
    }
    visual_by_view = {
        int(row.view_index): row for row in qualified_visual.views
    }
    if (
        set(camera_by_view) != set(range(8))
        or set(texture_by_view) != set(range(8))
        or set(visual_by_view) != set(range(8))
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_REQUIRES_EXACT_V0_V7"
        )

    operator_policy = dict(POLICY)
    operator_policy_hash = content_sha256(operator_policy)

    arrays = {}
    runtime_views = []
    bindings = {}
    qa_summary = {}
    for view_index in range(8):
        camera = camera_by_view[view_index]
        visual_row = visual_by_view[view_index]
        loaded = load_qualified_visual_presentation_view(visual_row)
        visual_mesh = loaded["mesh"]
        visibility = rasterize_visible_owner(
            mesh,
            camera,
            width=int(visual_row.width),
            height=int(visual_row.height),
            max_layers=4,
        )
        binding = build_domain_binding(
            points_source_xy=np.asarray(visual_mesh.positions, dtype=np.float64),
            visual_faces=np.asarray(visual_mesh.faces, dtype=np.int64),
            vertex_region_id=loaded["vertex_region_id"],
            seed_region_labels=loaded["seed_region_labels"],
            owner_face_index=visibility.owner_face_index,
            mechanical_positions_xyz=rest_mechanical,
            mechanical_faces=mechanical_faces,
            camera=camera,
        )
        rest_eval = evaluate_domain_binding(
            binding, visual_faces=visual_mesh.faces,
            posed_mechanical_positions_xyz=rest_mechanical, camera=camera,
        )[:, :2]
        rest_error = np.linalg.norm(
            np.asarray(rest_eval, dtype=np.float64)
            - np.asarray(visual_mesh.positions, dtype=np.float64),
            axis=1,
        )
        if (
            not np.isfinite(rest_error).all()
            or float(np.max(rest_error, initial=0.0)) > 1.0e-7
        ):
            raise QualificationError(
                "SOURCE_VISUAL_RUNTIME_REST_BINDING_DRIFT"
            )
        prefix = f"view_{view_index}"
        arrays[f"{prefix}_faces"] = np.asarray(
            visual_mesh.faces, dtype=np.uint32
        )
        arrays[f"{prefix}_uv"] = np.asarray(
            visual_mesh.uv, dtype=np.float64
        )
        arrays[f"{prefix}_rest_positions"] = np.asarray(
            visual_mesh.positions, dtype=np.float64
        )
        arrays[f"{prefix}_vertex_region_id"] = np.asarray(
            loaded["vertex_region_id"], dtype=np.int32
        )
        arrays[f"{prefix}_face_region_id"] = np.asarray(
            loaded["face_region_id"], dtype=np.int32
        )
        for name, value in binding.items():
            arrays[f"{prefix}_domain_{name}"] = np.asarray(value)
        bindings[view_index] = binding

        texture = texture_by_view[view_index]
        texture_path = resolved_path(texture.transport_png_path)
        if (
            not texture_path.is_file()
            or sha256_file(texture_path)
            != texture.transport_png_sha256
        ):
            raise QualificationError(
                "SOURCE_VISUAL_RUNTIME_TEXTURE_BYTES_DRIFT"
            )
        texture_meta = dict(texture.metadata or {})
        if (
            str(
                texture_meta.get("visual_mesh_set_binding_hash")
                or ""
            )
            != qualified_visual.source_visual_mesh_set_binding_hash
        ):
            raise QualificationError(
                "SOURCE_VISUAL_RUNTIME_TEXTURE_SOURCE_BINDING_DRIFT"
            )
        runtime_views.append(
            SourceOwnedVisualRuntimeViewV1IR(
                view_index=view_index,
                view_id=f"V{view_index}",
                camera=asdict(camera),
                source_width=int(visual_row.width),
                source_height=int(visual_row.height),
                texture_path=str(texture_path),
                texture_sha256=str(texture.transport_png_sha256),
                visual_mesh_npz_path=str(visual_row.mesh_npz_path),
                visual_mesh_npz_sha256=str(
                    visual_row.mesh_npz_sha256
                ),
                visual_mesh_hash=str(visual_row.visual_mesh_hash),
                visual_vertex_count=int(visual_row.vertex_count),
                visual_face_count=int(visual_row.face_count),
                metadata={
                    "source_owned_visual_mesh_mode": True,
                    "mechanical_mesh_render_authority": False,
                    "fixed_source_raster_uv": True,
                    "source_visual_mesh_hash": (
                        visual_row.source_visual_mesh_hash
                    ),
                },
            )
        )
        qa_summary[f"V{view_index}"] = {
            "domain_count": int(len(np.unique(binding["domain_id"]))),
            "anchor_count": int(len(binding["anchor_vertex"])),
            "max_rest_reconstruction_error_px": float(np.max(rest_error, initial=0)),
            "visibility_layer_overflow_pixel_count": int(np.count_nonzero(visibility.layer_overflow)),
        }

    clips = []
    dynamic_qa = {}
    for clip_index, clip in enumerate(dynamic.clips):
        prefix = f"clip_{clip_index}"
        times = np.asarray(
            [frame.time_seconds for frame in clip.frames],
            dtype=np.float64,
        )
        arrays[f"{prefix}_times"] = times
        clip_qa = {}
        for view_index in range(8):
            visual_row = visual_by_view[view_index]
            loaded = load_qualified_visual_presentation_view(
                visual_row
            )
            visual_mesh = loaded["mesh"]
            frames = np.empty(
                (
                    len(clip.frames),
                    int(visual_row.vertex_count),
                    2,
                ),
                dtype=np.float64,
            )
            depths = np.empty((len(clip.frames), int(visual_row.vertex_count)), dtype=np.float64)
            metrics = []
            for frame_index, frame in enumerate(clip.frames):
                by_id = {
                    str(vertex_id): tuple(map(float, xyz))
                    for vertex_id, xyz in frame.posed_vertex_xyz
                }
                if set(by_id) != set(vertex_ids):
                    raise QualificationError(
                        "SOURCE_VISUAL_RUNTIME_DYNAMIC_VERTEX_SET_DRIFT"
                    )
                posed_mechanical = np.asarray(
                    [by_id[vertex_id] for vertex_id in vertex_ids],
                    dtype=np.float64,
                )
                field = evaluate_domain_binding(
                    bindings[view_index], visual_faces=visual_mesh.faces,
                    posed_mechanical_positions_xyz=posed_mechanical,
                    camera=camera_by_view[view_index],
                )
                posed_visual = field[:, :2]
                frames[frame_index] = posed_visual
                depths[frame_index] = field[:, 2]
                metrics.append(
                    _visual_mesh_motion_metrics(
                        np.asarray(
                            visual_mesh.positions, dtype=np.float64
                        ),
                        np.asarray(posed_visual, dtype=np.float64),
                        np.asarray(
                            visual_mesh.faces, dtype=np.int64
                        ),
                    )
                )
            arrays[
                f"{prefix}_view_{view_index}_positions"
            ] = frames
            arrays[f"{prefix}_view_{view_index}_depths"] = depths
            clip_qa[f"V{view_index}"] = {
                "maximum_flipped_triangle_count": max(
                    row["flipped_triangle_count"] for row in metrics
                ),
                "maximum_edge_gt_4_count": max(
                    row["edge_gt_4_count"] for row in metrics
                ),
                "maximum_edge_gt_10_count": max(
                    row["edge_gt_10_count"] for row in metrics
                ),
                "maximum_p95_edge_ratio": max(
                    row["p95_edge_ratio"] for row in metrics
                ),
                "maximum_edge_ratio": max(
                    row["max_edge_ratio"] for row in metrics
                ),
                "frames": metrics,
            }
        dynamic_qa[str(clip.clip_id)] = clip_qa
        clips.append(
            SourceOwnedVisualRuntimeClipV1IR(
                clip_id=str(clip.clip_id),
                duration_seconds=float(clip.duration_seconds),
                loop=bool(clip.loop),
                frame_count=len(clip.frames),
                array_prefix=prefix,
                metadata={
                    "classification": str(clip.classification),
                    "source_dynamic_clip_proof_hash": str(
                        clip.clip_proof_hash
                    ),
                    "playback_sampling_contract": (
                        "SEALED_FRAME_INDEX_ONLY"
                    ),
                    "host_interpolation_authorized": False,
                },
            )
        )

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    array_path = root / "source_owned_visual_projection_arrays_v1.npz"
    array_sha = _save_npz(array_path, **arrays)
    projection = SourceOwnedVisualRuntimeProjectionV1IR(
        complete_puppet_binding_hash=str(complete.complete_puppet_hash),
        mechanical_state_binding_hash=str(
            complete.mechanical_state_binding_hash
        ),
        mechanical_mesh_binding_hash=str(mesh.mesh_lineage_hash),
        qualified_visual_presentation_binding_hash=str(
            qualified_visual.set_hash
        ),
        dynamic_motion_binding_hash=str(dynamic.dynamic_motion_hash),
        appearance_asset_binding_hash=str(asset.asset_hash),
        appearance_qualification_binding_hash=str(
            appearance.qualification_hash
        ),
        camera_set_binding_hash=str(cameras.camera_set_hash),
        visual_deformation_operator_id=str(
            operator_policy["operator_id"]
        ),
        visual_deformation_policy_hash=operator_policy_hash,
        projection_npz_path=str(array_path),
        projection_npz_sha256=array_sha,
        views=tuple(runtime_views),
        clips=tuple(clips),
        projection_hash="",
        metadata={
            "presentation_geometry_mode": (
                "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
            ),
            "mechanical_mesh_render_authority": False,
            "runtime_visual_mesh_rebuild": False,
            "runtime_binding_solve": False,
            "runtime_generation": False,
            "donor_search_at_runtime": False,
            "dynamic_quality_authority": "STAGE45",
            "operator_policy": operator_policy,
            "rest_binding_qa": qa_summary,
            "dynamic_motion_qa": dynamic_qa,
        },
    )
    projection = replace(
        projection,
        projection_hash=source_owned_visual_runtime_projection_hash(
            projection
        ),
    )
    root.mkdir(parents=True, exist_ok=True)
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "source_owned_visual_runtime_projection_v1.json",
                projection,
                authority_class=(
                    "SOURCE_OWNED_VISUAL_RUNTIME_PROJECTION_V1"
                ),
            ),
            {
                "path": str(array_path),
                "sha256": array_sha,
                "authority_class": (
                    "SOURCE_OWNED_VISUAL_RUNTIME_ARRAYS_V1"
                ),
                "schema": (
                    "RealSaS.SourceOwnedVisualRuntimeProjectionArrays.v1"
                ),
            },
        ],
        "diagnostics": {
            "projection_hash": projection.projection_hash,
            "qualified_visual_presentation_binding_hash": (
                qualified_visual.set_hash
            ),
            "clip_count": len(clips),
            "view_count": len(runtime_views),
            "presentation_geometry_mode": (
                "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
            ),
            "visual_deformation_operator_id": (
                projection.visual_deformation_operator_id
            ),
            "mechanical_mesh_render_authority": False,
            "runtime_generation": False,
        },
    }


def build_runtime_projection_stage(ctx: dict) -> dict:
    complete = complete_puppet_state_v2_from_dict(
        stage_output_payload(
            ctx,
            "38_CANONICAL_PUPPET_SEALED",
            "RealSaS.CompletePuppetStateIR.v2",
        )
    )
    mesh = qualified_mesh_from_dict(
        stage_output_payload(
            ctx,
            "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            "RealSaS.QualifiedMeshIR.v1",
        )
    )
    dynamic = qualified_dynamic_motion_v2_from_dict(
        stage_output_payload(
            ctx,
            "41_MOTION_DYNAMIC_PROOF",
            "RealSaS.QualifiedDynamicMotionIR.v2",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    appearance = complete_appearance_qualification_from_dict(
        stage_output_payload(
            ctx,
            "24_COMPLETE_APPEARANCE_QUALIFIED",
            "RealSaS.CompleteAppearanceQualificationIR.v2",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )

    if complete.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise QualificationError("RUNTIME_V2_COMPLETE_PUPPET_MESH_DRIFT")
    if complete.complete_appearance_asset_binding_hash != asset.asset_hash:
        raise QualificationError("RUNTIME_V2_COMPLETE_PUPPET_APPEARANCE_DRIFT")
    if (
        complete.complete_appearance_qualification_binding_hash
        != appearance.qualification_hash
    ):
        raise QualificationError("RUNTIME_V2_COMPLETE_PUPPET_APPEARANCE_QUAL_DRIFT")
    if dynamic.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise QualificationError("RUNTIME_V2_DYNAMIC_MESH_DRIFT")

    asset_meta = dict(asset.metadata or {})
    if asset_meta.get("source_owned_visual_mesh_mode") is True:
        return _build_source_owned_visual_runtime_projection(
            ctx,
            complete=complete,
            mesh=mesh,
            dynamic=dynamic,
            asset=asset,
            appearance=appearance,
            cameras=cameras,
        )
    if (
        dynamic.mechanical_state_binding_hash
        != complete.mechanical_state_binding_hash
    ):
        raise QualificationError("RUNTIME_V2_DYNAMIC_MECHANICAL_STATE_DRIFT")
    if (
        dynamic.presentation_binding_hash
        != complete.presentation_graph_binding_hash
    ):
        raise QualificationError("RUNTIME_V2_DYNAMIC_PRESENTATION_DRIFT")

    vertex_ids = [str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices]
    vertex_index = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}
    if len(vertex_index) != len(vertex_ids):
        raise QualificationError("RUNTIME_V2_DUPLICATE_MESH_VERTEX_ID")
    vertices = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float64)
    faces = np.asarray(
        [[vertex_index[str(vertex_id)] for vertex_id in face] for face in mesh.faces],
        dtype=np.uint32,
    )
    uv_path = resolved_path(asset.uv_npz_path)
    if not uv_path.is_file() or sha256_file(uv_path) != asset.uv_npz_sha256:
        raise QualificationError("RUNTIME_V2_CAA_UV_BYTES_DRIFT")
    with np.load(uv_path, allow_pickle=False) as data:
        if "face_uv" not in data.files:
            raise QualificationError("RUNTIME_V2_CAA_FACE_UV_MISSING")
        face_uv = np.asarray(data["face_uv"], dtype=np.float64)
        face_page_index = (
            np.asarray(data["face_page_index"], dtype=np.int32)
            if "face_page_index" in data.files
            else np.zeros((len(mesh.faces),), dtype=np.int32)
        )
    if face_uv.shape != (len(mesh.faces), 3, 2):
        raise QualificationError("RUNTIME_V2_FACE_UV_TOPOLOGY_DRIFT")
    if (
        face_page_index.shape != (len(mesh.faces),)
        or np.any(face_page_index < 0)
    ):
        raise QualificationError("RUNTIME_V2_FACE_PAGE_INDEX_INVALID")

    arrays = {
        "vertices": vertices,
        "faces": faces,
        "face_uv": face_uv,
        "face_page_index": face_page_index,
    }
    clips = []
    for clip_index, clip in enumerate(dynamic.clips):
        times = np.asarray([frame.time_seconds for frame in clip.frames], dtype=np.float64)
        positions = np.zeros(
            (len(clip.frames), len(vertex_ids), 3),
            dtype=np.float64,
        )
        for frame_index, frame in enumerate(clip.frames):
            by_id = {
                str(vertex_id): tuple(map(float, xyz))
                for vertex_id, xyz in frame.posed_vertex_xyz
            }
            if set(by_id) != set(vertex_ids):
                raise QualificationError("RUNTIME_V2_DYNAMIC_VERTEX_ID_SET_DRIFT")
            positions[frame_index] = np.asarray(
                [by_id[vertex_id] for vertex_id in vertex_ids],
                dtype=np.float64,
            )
        prefix = f"clip_{clip_index}"
        arrays[f"{prefix}_times"] = times
        arrays[f"{prefix}_positions"] = positions
        clips.append(
            RuntimeClipV2IR(
                clip_id=clip.clip_id,
                duration_seconds=float(clip.duration_seconds),
                loop=bool(clip.loop),
                frame_count=len(clip.frames),
                array_prefix=prefix,
                metadata={
                    "classification": clip.classification,
                    "source_dynamic_clip_proof_hash": clip.clip_proof_hash,
                    "playback_sampling_contract": "SEALED_FRAME_INDEX_ONLY",
                    "host_interpolation_authorized": False,
                },
            )
        )

    by_texture = {row.direction_index: row for row in asset.textures}
    views = []
    for camera in sorted(cameras.cameras, key=lambda row: row.view_index):
        texture = by_texture[int(camera.view_index)]
        metadata = dict(texture.metadata or {})
        raw_pages = tuple(metadata.get("pages") or ())
        if raw_pages:
            page_rows = tuple(
                sorted((dict(row) for row in raw_pages), key=lambda row: int(row["page_index"]))
            )
        else:
            page_rows = (
                {
                    "page_index": 0,
                    "path": str(texture.transport_png_path),
                    "sha256": str(texture.transport_png_sha256),
                    "width": int(texture.width),
                    "height": int(texture.height),
                },
            )
        if tuple(int(row["page_index"]) for row in page_rows) != tuple(range(len(page_rows))):
            raise QualificationError("RUNTIME_V2_TEXTURE_PAGE_INDEX_DRIFT")
        for row in page_rows:
            page_path = resolved_path(str(row["path"]))
            if not page_path.is_file() or sha256_file(page_path) != str(row["sha256"]):
                raise QualificationError("RUNTIME_V2_TEXTURE_PAGE_BYTES_DRIFT")
        if np.any(face_page_index >= len(page_rows)):
            raise QualificationError("RUNTIME_V2_FACE_PAGE_OUT_OF_RANGE")
        primary = page_rows[0]
        path = resolved_path(str(primary["path"]))
        views.append(
            RuntimeViewV2IR(
                view_index=int(camera.view_index),
                view_id=str(camera.view_id),
                camera=asdict(camera),
                texture_path=str(path),
                texture_sha256=str(primary["sha256"]),
                metadata={
                    "visibility": "CANONICAL_POSED_XYZ_ZBUFFER",
                    "appearance": "SEALED_CAA_V2",
                    "runtime_generation": False,
                    "paged_atlas": bool(metadata.get("paged_atlas", False)),
                    "page_count": len(page_rows),
                    "texture_pages": [dict(row) for row in page_rows],
                },
            )
        )

    provenance_path = resolved_path(asset.provenance_npz_path)
    if (
        not provenance_path.is_file()
        or sha256_file(provenance_path) != asset.provenance_npz_sha256
    ):
        raise QualificationError("RUNTIME_V2_PROVENANCE_BYTES_DRIFT")

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    array_path = root / "runtime_projection_arrays.npz"
    array_sha = _save_npz(array_path, **arrays)
    projection = RuntimeProjectionV2IR(
        complete_puppet_binding_hash=complete.complete_puppet_hash,
        mechanical_state_binding_hash=complete.mechanical_state_binding_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        dynamic_motion_binding_hash=dynamic.dynamic_motion_hash,
        appearance_asset_binding_hash=asset.asset_hash,
        appearance_qualification_binding_hash=appearance.qualification_hash,
        camera_set_binding_hash=cameras.camera_set_hash,
        visibility_contract_hash=VISIBILITY_CONTRACT_V2_HASH,
        projection_npz_path=str(array_path),
        projection_npz_sha256=array_sha,
        provenance_npz_path=str(provenance_path),
        provenance_npz_sha256=asset.provenance_npz_sha256,
        views=tuple(views),
        clips=tuple(clips),
        projection_hash="",
        metadata={
            "single_mesh_truth": True,
            "posed_xyz_from_stage41_exact": True,
            "skin_resolve_at_runtime": False,
            "donor_search_at_runtime": False,
            "runtime_generation": False,
            "relighting": False,
            "playback_sampling_contract": "SEALED_FRAME_INDEX_ONLY",
            "host_interpolation_authorized": False,
            "geometry_uv_position_precision": "IEEE754_FLOAT64",
            "appearance_paging_contract": "FACE_INDEX_TO_FIXED_PHYSICAL_PAGE_V1",
            "appearance_page_count": int(np.max(face_page_index) + 1) if len(face_page_index) else 1,
        },
    )
    projection = replace(
        projection, projection_hash=runtime_projection_hash(projection)
    )
    root.mkdir(parents=True, exist_ok=True)
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "runtime_projection_v2.json",
                projection,
                authority_class="RUNTIME_PROJECTION_AND_CAA_BINDING_V2",
            ),
            {
                "path": str(array_path),
                "sha256": array_sha,
                "authority_class": "RUNTIME_PROJECTION_ARRAYS_V2",
                "schema": "RealSaS.RuntimeProjectionArrays.v2",
            },
        ],
        "diagnostics": {
            "projection_hash": projection.projection_hash,
            "clip_count": len(clips),
            "view_count": len(views),
            "runtime_generation": False,
            "runtime_skin_solve": False,
        },
    }


def materialize_runtime_package_stage(ctx: dict) -> dict:
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    source_owned_visual = bool(
        dict(asset.metadata or {}).get("source_owned_visual_mesh_mode")
    )
    if source_owned_visual:
        projection = source_owned_visual_runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1",
            )
        )
        arrays = resolved_path(projection.projection_npz_path)
        if sha256_file(arrays) != projection.projection_npz_sha256:
            raise QualificationError(
                "RUNTIME_V2_VISUAL_PACKAGE_PROJECTION_BYTES_DRIFT"
            )
        for view in projection.views:
            texture = resolved_path(view.texture_path)
            if (
                not texture.is_file()
                or sha256_file(texture) != view.texture_sha256
            ):
                raise QualificationError(
                    "RUNTIME_V2_VISUAL_PACKAGE_TEXTURE_BYTES_DRIFT"
                )
        entries = build_source_owned_visual_rss_v2_entries(projection)
        presentation_mode = "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
    else:
        projection = runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.RuntimeProjectionIR.v2",
            )
        )
        arrays = resolved_path(projection.projection_npz_path)
        provenance = resolved_path(projection.provenance_npz_path)
        if sha256_file(arrays) != projection.projection_npz_sha256:
            raise QualificationError(
                "RUNTIME_V2_PACKAGE_PROJECTION_BYTES_DRIFT"
            )
        if sha256_file(provenance) != projection.provenance_npz_sha256:
            raise QualificationError(
                "RUNTIME_V2_PACKAGE_PROVENANCE_BYTES_DRIFT"
            )
        for view in projection.views:
            _load_runtime_view_texture(view)
        entries = build_rss_v2_entries(projection)
        presentation_mode = "MECHANICAL_CANONICAL_DEPTH_V2"

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    archive = root / "product_runtime_v2.rss"
    result = write_rss_v2(archive, entries)
    replay = read_rss_v2(archive)
    if tuple(replay.keys()) != tuple(entries.keys()):
        raise QualificationError("RUNTIME_V2_PACKAGE_ENTRY_REPLAY_DRIFT")
    seal = RuntimePackageSealV2IR(
        projection_binding_hash=projection.projection_hash,
        archive_path=str(archive),
        archive_sha256=str(result["archive_sha256"]),
        archive_bytes=int(result["archive_bytes"]),
        package_format=str(result["package_format"]),
        entry_names=tuple(result["entry_names"]),
        package_hash="",
        metadata={
            "container_compression": "NONE_V1",
            "presentation_geometry_mode": presentation_mode,
            "contains_only_sealed_runtime_authorities": True,
            "source_owned_visual_mesh_mode": source_owned_visual,
            "mechanical_mesh_render_authority": (
                False if source_owned_visual else True
            ),
        },
    )
    seal = replace(seal, package_hash=runtime_package_hash(seal))
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "runtime_package_seal_v2.json",
                seal,
                authority_class="QUALIFIED_RUNTIME_PACKAGE_V2",
            ),
            {
                "path": str(archive),
                "sha256": seal.archive_sha256,
                "authority_class": "RUNTIME_V2_RSS_PACKAGE",
                "schema": "application/x-realsas-rss-v2",
            },
        ],
        "diagnostics": {
            "package_hash": seal.package_hash,
            "archive_sha256": seal.archive_sha256,
            "archive_bytes": seal.archive_bytes,
            "entry_count": len(seal.entry_names),
            "presentation_geometry_mode": presentation_mode,
            "mechanical_mesh_render_authority": (
                False if source_owned_visual else True
            ),
        },
    }


def _native_player(ctx: dict):
    cfg = dict(ctx["run_manifest"].get("runtime") or {})
    ref = dict(cfg.get("native_player") or {})
    path = resolved_path(str(ref.get("path") or ""))
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise QualificationError("RUNTIME_V2_NATIVE_PLAYER_REF_INVALID")
    return path, expected


def _run_native(
    *,
    player: Path,
    package: Path,
    clip_id: str,
    view_id: str,
    frame_index: int,
    root: Path,
):
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{clip_id}__{view_id}__{frame_index:04d}"
    rgba = root / f"{stem}.rgba"
    provenance = root / f"{stem}.prov"
    owner = root / f"{stem}.owner"
    completed = subprocess.run(
        [
            str(player),
            str(package),
            "--clip",
            clip_id,
            "--view",
            view_id,
            "--frame",
            str(frame_index),
            "--out-rgba",
            str(rgba),
            "--out-provenance",
            str(provenance),
            "--out-owner",
            str(owner),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise QualificationError(
            "RUNTIME_V2_NATIVE_PLAYER_FAIL:" + completed.stderr.strip()
        )
    renderer_tokens = (
        "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH",
        "renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D",
    )
    if not any(token in completed.stdout for token in renderer_tokens):
        raise QualificationError("RUNTIME_V2_NATIVE_RENDERER_CONTRACT_DRIFT")
    return rgba, provenance, owner, completed.stdout


def _native_parallel_workers(ctx: dict) -> int:
    """Bound native proof concurrency without changing render semantics."""
    cfg = dict(ctx["run_manifest"].get("runtime") or {})
    raw = int(cfg.get("native_parallel_workers", 2))
    return max(1, min(raw, 8))


def _run_native_many(
    *,
    player: Path,
    package: Path,
    requests,
    root: Path,
    max_workers: int,
):
    rows = tuple(dict(row) for row in requests)
    if not rows:
        return []
    workers = max(1, min(int(max_workers), len(rows)))
    if workers == 1:
        return [
            _run_native(
                player=player,
                package=package,
                clip_id=row["clip_id"],
                view_id=row["view_id"],
                frame_index=int(row["frame_index"]),
                root=root,
            )
            for row in rows
        ]

    def invoke(row):
        return _run_native(
            player=player,
            package=package,
            clip_id=row["clip_id"],
            view_id=row["view_id"],
            frame_index=int(row["frame_index"]),
            root=root,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(invoke, rows))


def _numeric_mesh(arrays):
    vertices = [
        SimpleNamespace(canonical_mesh_vertex_id=f"v{index}", P=tuple(map(float, xyz)))
        for index, xyz in enumerate(np.asarray(arrays["vertices"]))
    ]
    faces = tuple(
        tuple(f"v{int(index)}" for index in face)
        for face in np.asarray(arrays["faces"], dtype=np.int64)
    )
    return SimpleNamespace(vertices=vertices, faces=faces)


def _build_reference_render_context(projection, arrays):
    """Cache invariant Python reference-render inputs for one Stage44/45 run.

    This is a performance-only transport optimization: mesh topology, CAA pages,
    provenance and qualified camera objects are immutable across every frame.
    Keeping them resident avoids repeated PNG/NPZ decode and object reconstruction
    without changing any rendered bytes or qualification semantics.
    """
    mesh = _numeric_mesh(arrays)
    provenance_path = resolved_path(projection.provenance_npz_path)
    if (
        not provenance_path.is_file()
        or sha256_file(provenance_path) != projection.provenance_npz_sha256
    ):
        raise QualificationError("RUNTIME_V2_PROVENANCE_BYTES_DRIFT")
    with np.load(provenance_path, allow_pickle=False) as data:
        if "provenance" not in data.files:
            raise QualificationError("RUNTIME_V2_PROVENANCE_ARRAY_MISSING")
        provenance_all = np.asarray(data["provenance"], dtype=np.uint8).copy()

    face_page_index = np.asarray(
        arrays.get("face_page_index", np.zeros((len(mesh.faces),), dtype=np.int32)),
        dtype=np.int32,
    )
    return {
        "mesh": mesh,
        "face_uv": np.asarray(arrays["face_uv"], dtype=np.float64),
        "face_page_index": face_page_index,
        "provenance_all": provenance_all,
        "texture_by_view_id": {
            view.view_id: _load_runtime_view_texture(view)
            for view in projection.views
        },
        "camera_by_view_id": {
            view.view_id: qualify_camera_v3(
                dict(view.camera),
                view_id=view.view_id,
                view_index=view.view_index,
            )
            for view in projection.views
        },
    }


def _reference_frame(
    projection,
    arrays,
    *,
    clip,
    view,
    frame_index,
    reference_context=None,
):
    context = (
        reference_context
        if reference_context is not None
        else _build_reference_render_context(projection, arrays)
    )
    positions = np.asarray(
        arrays[f"{clip.array_prefix}_positions"][frame_index],
        dtype=np.float64,
    )
    return render_caa_reference(
        mesh=context["mesh"],
        camera=context["camera_by_view_id"][view.view_id],
        face_uv=context["face_uv"],
        texture_rgba_u8=context["texture_by_view_id"][view.view_id],
        provenance_atlas=np.asarray(
            context["provenance_all"][view.view_index],
            dtype=np.uint8,
        ),
        face_page_index=context["face_page_index"],
        positions=positions,
    )



def _source_owned_visual_reference_frame(
    projection,
    arrays,
    *,
    clip,
    view,
    frame_index: int,
):
    vi = int(view.view_index)
    uv = np.asarray(arrays[f"view_{vi}_uv"], dtype=np.float64)
    faces = np.asarray(arrays[f"view_{vi}_faces"], dtype=np.int64)
    positions = np.asarray(
        arrays[f"{clip.array_prefix}_view_{vi}_positions"][frame_index],
        dtype=np.float64,
    )
    if uv.shape != (int(view.visual_vertex_count), 2):
        raise QualificationError("RUNTIME_V2_VISUAL_REFERENCE_UV_SHAPE_INVALID")
    if faces.shape != (int(view.visual_face_count), 3):
        raise QualificationError("RUNTIME_V2_VISUAL_REFERENCE_FACE_SHAPE_INVALID")
    if positions.shape != (int(view.visual_vertex_count), 2):
        raise QualificationError(
            "RUNTIME_V2_VISUAL_REFERENCE_POSITION_SHAPE_INVALID"
        )

    from compiler.realsas_compiler_core.visual_attachment_depth_v1 import OPERATOR_ID as SLOT_DEPTH_OPERATOR
    if projection.visual_deformation_operator_id in (OPERATOR_ID, ATTACHMENT_OPERATOR, MOTION_BLEND_OPERATOR, SLOT_DEPTH_OPERATOR):
        key = f"{clip.array_prefix}_view_{vi}_depths"
        if key not in arrays:
            raise QualificationError("SOURCE_VISUAL_CANONICAL_DEPTH_MISSING")
        return render_visual_depth(
            positions=positions, depths=arrays[key][frame_index], faces=faces, uv=uv,
            texture=np.asarray(Image.open(resolved_path(view.texture_path)).convert("RGBA")),
            resolution=int(view.camera["resolution"]),
        )

    texture_path = resolved_path(view.texture_path)
    texture = np.asarray(
        Image.open(texture_path).convert("RGBA"),
        dtype=np.uint8,
    )
    if texture.shape != (
        int(view.source_height),
        int(view.source_width),
        4,
    ):
        raise QualificationError(
            "RUNTIME_V2_VISUAL_REFERENCE_TEXTURE_SHAPE_INVALID"
        )
    resolution = int(dict(view.camera).get("resolution") or 0)
    if resolution <= 0:
        raise QualificationError(
            "RUNTIME_V2_VISUAL_REFERENCE_RESOLUTION_INVALID"
        )

    posed = np.asarray(positions, dtype=np.float64).copy()
    posed[:, 0] = (
        posed[:, 0] + 0.5
    ) * float(resolution) / float(view.source_width)
    posed[:, 1] = (
        posed[:, 1] + 0.5
    ) * float(resolution) / float(view.source_height)

    accum = np.zeros((resolution, resolution, 4), dtype=np.float64)
    owner = np.full((resolution, resolution), -1, dtype=np.int32)

    def orient(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (
            b[1] - a[1]
        ) * (x - a[0])

    def top_left(a, b):
        dy = float(b[1] - a[1])
        dx = float(b[0] - a[0])
        return dy < 0.0 or (abs(dy) <= 1.0e-12 and dx > 0.0)

    def sample_bilinear(u, v):
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        x = u * float(view.source_width - 1)
        y = v * float(view.source_height - 1)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, int(view.source_width) - 1)
        y1 = min(y0 + 1, int(view.source_height) - 1)
        tx = float(x - x0)
        ty = float(y - y0)
        p00 = texture[y0, x0].astype(np.float64) / 255.0
        p10 = texture[y0, x1].astype(np.float64) / 255.0
        p01 = texture[y1, x0].astype(np.float64) / 255.0
        p11 = texture[y1, x1].astype(np.float64) / 255.0
        a = p00 * (1.0 - tx) + p10 * tx
        b = p01 * (1.0 - tx) + p11 * tx
        return a * (1.0 - ty) + b * ty

    for face_index, face in enumerate(faces):
        a, b, c = posed[face]
        area = float(orient(a, b, c[0], c[1]))
        if abs(area) <= 1.0e-12:
            continue
        sign = 1.0 if area > 0.0 else -1.0
        positive = area > 0.0
        min_x = max(
            0,
            int(np.floor(float(np.min((a[0], b[0], c[0]))) - 0.5)),
        )
        max_x = min(
            resolution - 1,
            int(np.ceil(float(np.max((a[0], b[0], c[0]))) - 0.5)),
        )
        min_y = max(
            0,
            int(np.floor(float(np.min((a[1], b[1], c[1]))) - 0.5)),
        )
        max_y = min(
            resolution - 1,
            int(np.ceil(float(np.max((a[1], b[1], c[1]))) - 0.5)),
        )
        if min_x > max_x or min_y > max_y:
            continue
        tl0 = top_left(b, c) if positive else top_left(c, b)
        tl1 = top_left(c, a) if positive else top_left(a, c)
        tl2 = top_left(a, b) if positive else top_left(b, a)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px = float(x) + 0.5
                py = float(y) + 0.5
                q0 = sign * orient(b, c, px, py)
                q1 = sign * orient(c, a, px, py)
                q2 = sign * orient(a, b, px, py)
                accept0 = q0 > 1.0e-12 or (
                    abs(q0) <= 1.0e-12 and tl0
                )
                accept1 = q1 > 1.0e-12 or (
                    abs(q1) <= 1.0e-12 and tl1
                )
                accept2 = q2 > 1.0e-12 or (
                    abs(q2) <= 1.0e-12 and tl2
                )
                if not (accept0 and accept1 and accept2):
                    continue
                w0 = orient(b, c, px, py) / area
                w1 = orient(c, a, px, py) / area
                w2 = 1.0 - w0 - w1
                u = w0 * uv[face[0], 0] + w1 * uv[face[1], 0] + w2 * uv[
                    face[2], 0
                ]
                v = w0 * uv[face[0], 1] + w1 * uv[face[1], 1] + w2 * uv[
                    face[2], 1
                ]
                sample = sample_bilinear(u, v)
                alpha = float(np.clip(sample[3], 0.0, 1.0))
                transmission = 1.0 - alpha
                accum[y, x, 0] = (
                    sample[0] * alpha + accum[y, x, 0] * transmission
                )
                accum[y, x, 1] = (
                    sample[1] * alpha + accum[y, x, 1] * transmission
                )
                accum[y, x, 2] = (
                    sample[2] * alpha + accum[y, x, 2] * transmission
                )
                accum[y, x, 3] = alpha + accum[y, x, 3] * transmission
                if alpha > 1.0e-12:
                    owner[y, x] = int(face_index)

    alpha = np.clip(accum[:, :, 3], 0.0, 1.0)
    inv = np.zeros_like(alpha)
    visible = alpha > 1.0e-12
    inv[visible] = 1.0 / alpha[visible]
    straight = np.zeros_like(accum)
    straight[:, :, :3] = accum[:, :, :3] * inv[:, :, None]
    straight[:, :, 3] = alpha
    rgba = np.floor(
        np.clip(straight, 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    provenance = np.full((resolution, resolution), 255, dtype=np.uint8)
    provenance[visible] = 0
    return SimpleNamespace(
        straight_rgba_u8=rgba,
        provenance_code=provenance,
        owner_face_index=owner,
    )


def prove_native_package_playback_stage(ctx: dict) -> dict:
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    source_owned_visual = bool(
        dict(asset.metadata or {}).get("source_owned_visual_mesh_mode")
    )
    if source_owned_visual:
        projection = source_owned_visual_runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1",
            )
        )
        renderer_id = "REALSAS_V2_SOURCE_OWNED_VISUAL_2D"
    else:
        projection = runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.RuntimeProjectionIR.v2",
            )
        )
        renderer_id = "REALSAS_V2_CAA_CANONICAL_DEPTH"

    package = runtime_package_seal_from_dict(
        stage_output_payload(
            ctx,
            "43_RSS_MATERIALIZE_COMPACT",
            "RealSaS.RuntimePackageSealIR.v2",
        )
    )
    if package.projection_binding_hash != projection.projection_hash:
        raise QualificationError("RUNTIME_V2_NATIVE_PACKAGE_PROJECTION_DRIFT")
    package_mode = str(
        dict(package.metadata or {}).get("presentation_geometry_mode") or ""
    )
    expected_mode = (
        "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
        if source_owned_visual
        else "MECHANICAL_CANONICAL_DEPTH_V2"
    )
    if package_mode != expected_mode:
        raise QualificationError(
            "RUNTIME_V2_NATIVE_PACKAGE_PRESENTATION_MODE_DRIFT"
        )
    archive = resolved_path(package.archive_path)
    if not archive.is_file() or sha256_file(archive) != package.archive_sha256:
        raise QualificationError("RUNTIME_V2_NATIVE_PACKAGE_BYTES_DRIFT")
    player, player_sha = _native_player(ctx)
    arrays = _projection_arrays(projection)
    reference_context = (
        None
        if source_owned_visual
        else _build_reference_render_context(projection, arrays)
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]

    probes = []
    outputs = []
    native_workers = _native_parallel_workers(ctx)
    view_by_id = {view.view_id: view for view in projection.views}
    for clip in projection.clips:
        frame_index = int(clip.frame_count // 2)
        views = tuple(projection.views)
        native_rows = _run_native_many(
            player=player,
            package=archive,
            requests=(
                {
                    "clip_id": clip.clip_id,
                    "view_id": view.view_id,
                    "frame_index": frame_index,
                }
                for view in views
            ),
            root=root / "probes",
            max_workers=native_workers,
        )
        for view, (rgba, provenance, owner, stdout) in zip(
            views, native_rows
        ):
            resolution = int(view.camera["resolution"])
            if rgba.stat().st_size != resolution * resolution * 4:
                raise QualificationError("RUNTIME_V2_NATIVE_RGBA_SIZE_DRIFT")
            if provenance.stat().st_size != resolution * resolution:
                raise QualificationError(
                    "RUNTIME_V2_NATIVE_PROVENANCE_SIZE_DRIFT"
                )
            if owner.stat().st_size != resolution * resolution * 4:
                raise QualificationError("RUNTIME_V2_NATIVE_OWNER_SIZE_DRIFT")
            if f"renderer={renderer_id}" not in stdout:
                raise QualificationError(
                    "RUNTIME_V2_NATIVE_RENDERER_CONTRACT_DRIFT"
                )

            native_rgba = np.frombuffer(
                rgba.read_bytes(), dtype=np.uint8
            ).reshape(resolution, resolution, 4)
            native_provenance = np.frombuffer(
                provenance.read_bytes(), dtype=np.uint8
            ).reshape(resolution, resolution)
            native_owner = np.frombuffer(
                owner.read_bytes(), dtype="<i4"
            ).reshape(resolution, resolution)

            if source_owned_visual:
                reference = _source_owned_visual_reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view_by_id[view.view_id],
                    frame_index=frame_index,
                )
            else:
                reference = _reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view_by_id[view.view_id],
                    frame_index=frame_index,
                    reference_context=reference_context,
                )
            mismatch_mask = (
                np.any(native_rgba != reference.straight_rgba_u8, axis=2)
                | (native_provenance != reference.provenance_code)
                | (native_owner != reference.owner_face_index)
            )
            mismatch = int(np.count_nonzero(mismatch_mask))
            if mismatch:
                raise QualificationError(
                    "RUNTIME_V2_NATIVE_REFERENCE_PARITY_FAIL:"
                    f"{clip.clip_id}:{view.view_id}:{mismatch}"
                )

            probe = NativePlaybackProbeV2IR(
                clip_id=clip.clip_id,
                view_id=view.view_id,
                frame_index=frame_index,
                rgba_raw_path=str(rgba),
                rgba_raw_sha256=sha256_file(rgba),
                provenance_raw_path=str(provenance),
                provenance_raw_sha256=sha256_file(provenance),
                owner_raw_path=str(owner),
                owner_raw_sha256=sha256_file(owner),
                stdout_sha256=hashlib.sha256(
                    stdout.encode("utf-8")
                ).hexdigest(),
                probe_hash="",
                metadata={
                    "native_reference_mismatch_pixels": 0,
                    "renderer": renderer_id,
                    "presentation_geometry_mode": expected_mode,
                    "mechanical_mesh_render_authority": (
                        False if source_owned_visual else True
                    ),
                },
            )
            probe = replace(
                probe,
                probe_hash=native_playback_probe_hash(probe),
            )
            probes.append(probe)
            for output_path, authority, schema in (
                (rgba, "NATIVE_V2_RGBA", "application/x-rgba8"),
                (
                    provenance,
                    "NATIVE_V2_PROVENANCE",
                    "application/x-u8-mask",
                ),
                (owner, "NATIVE_V2_OWNER", "application/x-i32-owner"),
            ):
                outputs.append(
                    {
                        "path": str(output_path),
                        "sha256": sha256_file(output_path),
                        "authority_class": authority,
                        "schema": schema,
                    }
                )

    playback = NativePlaybackV2IR(
        package_binding_hash=package.package_hash,
        projection_binding_hash=projection.projection_hash,
        native_player_sha256=player_sha,
        probes=tuple(probes),
        playback_hash="",
        metadata={
            "native_package_opened_directly": True,
            "midpoint_probe_per_clip_view": True,
            "python_reference_byte_parity": True,
            "native_parallel_workers": native_workers,
            "renderer": renderer_id,
            "presentation_geometry_mode": expected_mode,
            "mechanical_mesh_render_authority": (
                False if source_owned_visual else True
            ),
        },
    )
    playback = replace(
        playback,
        playback_hash=native_playback_hash(playback),
    )
    outputs.insert(
        0,
        write_ir(
            root / "qualified_native_playback_v2.json",
            playback,
            authority_class="QUALIFIED_NATIVE_PLAYBACK_V2",
        ),
    )
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "playback_hash": playback.playback_hash,
            "probe_count": len(probes),
            "native_reference_mismatch_pixels": 0,
            "native_player_sha256": player_sha,
            "native_parallel_workers": native_workers,
            "renderer": renderer_id,
            "presentation_geometry_mode": expected_mode,
        },
    }


def prove_visual_domain_matrix(projection, arrays, *, mesh, dynamic, attachment_witness=None):
    """Recompute every field against the upstream canonical motion witness."""
    motion_blend_mode = projection.visual_deformation_operator_id == MOTION_BLEND_OPERATOR
    attachment_mode = projection.visual_deformation_operator_id in (ATTACHMENT_OPERATOR, MOTION_BLEND_OPERATOR)
    policy = MOTION_BLEND_POLICY if motion_blend_mode else ATTACHMENT_POLICY if attachment_mode else POLICY
    if (projection.visual_deformation_operator_id not in (OPERATOR_ID, ATTACHMENT_OPERATOR, MOTION_BLEND_OPERATOR)
            or projection.visual_deformation_policy_hash != content_sha256(policy)
            or projection.mechanical_mesh_binding_hash != mesh.mesh_lineage_hash
            or projection.dynamic_motion_binding_hash != dynamic.dynamic_motion_hash):
        raise QualificationError("SOURCE_VISUAL_PRESENTATION_AUTHORITY_REQUIRED")
    ids = [str(v.canonical_mesh_vertex_id) for v in mesh.vertices]
    id_index = {key: i for i, key in enumerate(ids)}
    canonical_faces = {tuple(sorted(id_index[str(k)] for k in face)) for face in mesh.faces}
    rest = np.asarray([v.P for v in mesh.vertices], dtype=np.float64)
    if attachment_mode and (attachment_witness is None
            or not np.array_equal(attachment_witness.get("vertices"), rest)
            or not projection.metadata.get("target_attachments")):
        raise QualificationError("SOURCE_VISUAL_ATTACHMENT_SEALED_WITNESS_REQUIRED")
    if len(ids) != len(set(ids)) or sorted(int(v.view_index) for v in projection.views) != list(range(8)):
        raise QualificationError("SOURCE_VISUAL_FRAME_MATRIX_VIEW_OR_VERTEX_DRIFT")
    witness = {str(c.clip_id): c for c in dynamic.clips}
    if set(witness) != {str(c.clip_id) for c in projection.clips}:
        raise QualificationError("SOURCE_VISUAL_FRAME_MATRIX_CLIP_DRIFT")
    count = area_bad = condition_bad = 0
    residual = 0.0
    slot_residual = 0.0
    palette_residual = 0.0
    matrix = []
    for view in projection.views:
        vi = int(view.view_index)
        binding = domain_binding_from_arrays(arrays, vi)
        if any(tuple(sorted(map(int, face))) not in canonical_faces
               for face in binding["anchor_mechanical_vertices"]):
            raise QualificationError("SOURCE_VISUAL_DOMAIN_ANCHOR_NOT_CANONICAL_FACE")
        if not np.array_equal(binding.get("mechanical_rest_xyz"), rest):
            raise QualificationError("SOURCE_VISUAL_DOMAIN_CANONICAL_REST_DRIFT")
        faces = arrays[f"view_{vi}_faces"]
        source_camera = dict(view.camera)
        source_camera["resolution"] = int(view.source_width)
        camera = qualify_camera_v3(source_camera, view_id=view.view_id, view_index=vi)
        if attachment_mode:
            vertex_owners = visual_vertex_attachment_owners(binding,
                attachment_witness["presentation_attachment_vertex_owner"])
        if motion_blend_mode:
            blend = build_motion_blend_coefficients(binding, visual_faces=faces,
                mechanical_weights=attachment_witness["canonical_motion_weights"])
            if not np.array_equal(blend, arrays[f"view_{vi}_motion_blend_coefficients"]):
                raise QualificationError("SOURCE_VISUAL_CANONICAL_MOTION_COEFFICIENT_DRIFT")
        for clip in projection.clips:
            source = witness[str(clip.clip_id)]
            if len(source.frames) != int(clip.frame_count):
                raise QualificationError("SOURCE_VISUAL_FRAME_MATRIX_FRAME_DRIFT")
            times = np.asarray([frame.time_seconds for frame in source.frames])
            if not np.array_equal(arrays[f"{clip.array_prefix}_times"], times):
                raise QualificationError("SOURCE_VISUAL_FRAME_MATRIX_TIME_DRIFT")
            depth_key = f"{clip.array_prefix}_view_{vi}_depths"
            if depth_key not in arrays:
                raise QualificationError("SOURCE_VISUAL_CANONICAL_DEPTH_MISSING")
            for fi, frame in enumerate(source.frames):
                by_id = {str(k): xyz for k, xyz in frame.posed_vertex_xyz}
                if len(by_id) != len(ids) or set(by_id) != set(ids):
                    raise QualificationError("SOURCE_VISUAL_DOMAIN_WITNESS_VERTEX_DRIFT")
                posed = np.asarray([by_id[k] for k in ids])
                expected = evaluate_domain_binding(
                    binding, visual_faces=faces,
                    posed_mechanical_positions_xyz=posed,
                    camera=camera,
                )
                if attachment_mode:
                    if not np.array_equal(attachment_witness[f"{clip.array_prefix}_canonical_xyz"][fi], posed):
                        raise QualificationError("SOURCE_VISUAL_ATTACHMENT_MOTION_WITNESS_DRIFT")
                    matrices = attachment_witness[f"{clip.array_prefix}_skin_matrices_source"][fi]
                    if motion_blend_mode:
                        palette_residual = max(palette_residual, canonical_pose_palette_residual(
                            rest_xyz=rest, posed_xyz=posed, mechanical_weights=attachment_witness["canonical_motion_weights"],
                            skin_matrices_source=matrices))
                        expected = evaluate_motion_blend(expected,
                            rest_source_xy=binding["rest_positions"], coefficients=blend,
                            axis_positions_source=attachment_witness["axis_positions_source"],
                            skin_matrices_source=matrices, camera=camera)
                    slot_residual = max(slot_residual, attachment_slot_carrier_residual(
                        projection.metadata["target_attachments"], rest_xyz=rest,
                        posed_xyz=posed, skin_matrices_source=matrices))
                    expected = evaluate_attachment_motion(expected,
                        rest_source_xy=binding["rest_positions"], vertex_attachment_owner=vertex_owners,
                        attachments=projection.metadata["target_attachments"],
                        axis_positions_source=attachment_witness["axis_positions_source"],
                        skin_matrices_source=matrices, camera=camera)
                actual = np.column_stack((
                    arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi],
                    arrays[depth_key][fi],
                ))
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise QualificationError("SOURCE_VISUAL_DOMAIN_FRAME_INVALID")
                delta = float(np.max(np.abs(actual - expected), initial=0))
                residual = max(residual, delta)
                metrics = presentation_condition_metrics(binding["rest_positions"], actual[:, :2], faces)
                area_bad += metrics["area_collapse_count"]
                condition_bad += metrics["condition_failure_count"]
                matrix.append({"clip_id": clip.clip_id, "view_id": view.view_id,
                               "frame_index": fi, "field_residual": delta, **metrics})
                count += 1
    expected_count = sum(int(c.frame_count) for c in projection.clips) * 8
    if count != expected_count or count <= 0:
        raise QualificationError("SOURCE_VISUAL_FRAME_MATRIX_INCOMPLETE")
    return {"domain_coherence_passed": residual <= 1e-7,
            "canonical_pose_palette_passed": not motion_blend_mode or palette_residual <= policy["maximum_canonical_pose_palette_residual"],
            "maximum_canonical_pose_palette_residual": palette_residual,
            "attachment_slot_motion_passed": not attachment_mode or slot_residual <= policy["maximum_carrier_slot_relative_residual"],
            "maximum_attachment_slot_carrier_relative_residual": slot_residual,
            "attachment_motion_operator_id": policy.get("attachment_motion_operator_id"),
            "frame_view_matrix_complete": count == expected_count,
            "area_condition_passed": area_bad == 0 and condition_bad == 0,
            "area_collapse_count": area_bad, "condition_failure_count": condition_bad,
            "maximum_field_residual": residual, "expected_frame_view_count": expected_count,
            "minimum_signed_area_ratio_threshold": POLICY["minimum_signed_area_ratio"],
            "maximum_jacobian_condition_threshold": POLICY["maximum_jacobian_condition"],
            "frames": matrix}


def _prove_source_owned_visual_dynamic_integrity(
    ctx: dict,
    *,
    projection,
    package,
    playback,
) -> dict:
    if playback.package_binding_hash != package.package_hash:
        raise QualificationError(
            "SOURCE_VISUAL_DVI_PLAYBACK_PACKAGE_DRIFT"
        )
    if playback.projection_binding_hash != projection.projection_hash:
        raise QualificationError(
            "SOURCE_VISUAL_DVI_PLAYBACK_PROJECTION_DRIFT"
        )
    if (
        str(dict(package.metadata or {}).get("presentation_geometry_mode") or "")
        != "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
    ):
        raise QualificationError(
            "SOURCE_VISUAL_DVI_PACKAGE_PRESENTATION_MODE_DRIFT"
        )
    if bool(
        dict(package.metadata or {}).get("mechanical_mesh_render_authority", True)
    ):
        raise QualificationError(
            "SOURCE_VISUAL_DVI_MECHANICAL_RENDER_AUTHORITY_FORBIDDEN"
        )

    player, player_sha = _native_player(ctx)
    if player_sha != playback.native_player_sha256:
        raise QualificationError("SOURCE_VISUAL_DVI_NATIVE_PLAYER_DRIFT")
    archive = resolved_path(package.archive_path)
    if not archive.is_file() or sha256_file(archive) != package.archive_sha256:
        raise QualificationError("SOURCE_VISUAL_DVI_PACKAGE_BYTES_DRIFT")
    arrays = _projection_arrays(projection)
    presentation_proof = prove_visual_domain_matrix(
        projection, arrays,
        mesh=qualified_mesh_from_dict(stage_output_payload(ctx, "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED", "RealSaS.QualifiedMeshIR.v1")),
        dynamic=qualified_dynamic_motion_v2_from_dict(stage_output_payload(
            ctx, "41_MOTION_DYNAMIC_PROOF", "RealSaS.QualifiedDynamicMotionIR.v2")),
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    native_workers = _native_parallel_workers(ctx)

    evaluated_frame_view_count = 0
    rendered_visible_pixel_count = 0
    empty_frame_view_count = 0
    flipped_triangle_count = 0
    edge_gt_4_count = 0
    edge_gt_10_count = 0
    maximum_p95_edge_ratio = 1.0
    maximum_edge_ratio = 1.0
    native_reference_mismatch_pixel_count = 0
    direct_source_provenance_mismatch_pixel_count = 0
    outputs = []
    unresolved_depth_tie_count = fragment_overflow_count = 0

    for clip in projection.clips:
        views = tuple(projection.views)
        for frame_index in range(int(clip.frame_count)):
            native_rows = _run_native_many(
                player=player,
                package=archive,
                requests=(
                    {
                        "clip_id": clip.clip_id,
                        "view_id": view.view_id,
                        "frame_index": frame_index,
                    }
                    for view in views
                ),
                root=root / "frames",
                max_workers=native_workers,
            )
            for view, (
                rgba_path,
                prov_path,
                owner_path,
                stdout,
            ) in zip(views, native_rows):
                if (
                    "renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D"
                    not in stdout
                ):
                    raise QualificationError(
                        "SOURCE_VISUAL_DVI_RENDERER_CONTRACT_DRIFT"
                    )
                resolution = int(view.camera["resolution"])
                rgba = np.frombuffer(
                    rgba_path.read_bytes(), dtype=np.uint8
                ).reshape(resolution, resolution, 4)
                provenance = np.frombuffer(
                    prov_path.read_bytes(), dtype=np.uint8
                ).reshape(resolution, resolution)
                owner = np.frombuffer(
                    owner_path.read_bytes(), dtype="<i4"
                ).reshape(resolution, resolution)

                reference = _source_owned_visual_reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view,
                    frame_index=frame_index,
                )
                unresolved_depth_tie_count += int(reference.unresolved_depth_tie_count)
                fragment_overflow_count += int(reference.fragment_overflow_count)
                mismatch = (
                    np.any(
                        rgba != reference.straight_rgba_u8,
                        axis=2,
                    )
                    | (provenance != reference.provenance_code)
                    | (owner != reference.owner_face_index)
                )
                native_reference_mismatch_pixel_count += int(
                    np.count_nonzero(mismatch)
                )

                visible = rgba[:, :, 3] > 0
                visible_count = int(np.count_nonzero(visible))
                rendered_visible_pixel_count += visible_count
                if visible_count <= 0:
                    empty_frame_view_count += 1
                provenance_bad = (
                    (visible & (provenance != 0))
                    | ((~visible) & (provenance != 255))
                )
                direct_source_provenance_mismatch_pixel_count += int(
                    np.count_nonzero(provenance_bad)
                )

                vi = int(view.view_index)
                rest = np.asarray(
                    arrays[f"view_{vi}_rest_positions"],
                    dtype=np.float64,
                )
                faces = np.asarray(
                    arrays[f"view_{vi}_faces"],
                    dtype=np.int64,
                )
                posed = np.asarray(
                    arrays[
                        f"{clip.array_prefix}_view_{vi}_positions"
                    ][frame_index],
                    dtype=np.float64,
                )
                metrics = _visual_mesh_motion_metrics(
                    rest,
                    posed,
                    faces,
                )
                flipped_triangle_count += int(
                    metrics["flipped_triangle_count"]
                )
                edge_gt_4_count += int(metrics["edge_gt_4_count"])
                edge_gt_10_count += int(metrics["edge_gt_10_count"])
                maximum_p95_edge_ratio = max(
                    maximum_p95_edge_ratio,
                    float(metrics["p95_edge_ratio"]),
                )
                maximum_edge_ratio = max(
                    maximum_edge_ratio,
                    float(metrics["max_edge_ratio"]),
                )
                evaluated_frame_view_count += 1

                for output_path, authority, schema in (
                    (
                        rgba_path,
                        "SOURCE_VISUAL_DVI_NATIVE_RGBA",
                        "application/x-rgba8",
                    ),
                    (
                        prov_path,
                        "SOURCE_VISUAL_DVI_NATIVE_PROVENANCE",
                        "application/x-u8-mask",
                    ),
                    (
                        owner_path,
                        "SOURCE_VISUAL_DVI_NATIVE_OWNER",
                        "application/x-i32-owner",
                    ),
                ):
                    outputs.append(
                        {
                            "path": str(output_path),
                            "sha256": sha256_file(output_path),
                            "authority_class": authority,
                            "schema": schema,
                        }
                    )

    passed = (
        evaluated_frame_view_count > 0
        and rendered_visible_pixel_count > 0
        and empty_frame_view_count == 0
        and flipped_triangle_count == 0
        and edge_gt_4_count == 0
        and native_reference_mismatch_pixel_count == 0
        and direct_source_provenance_mismatch_pixel_count == 0
        and presentation_proof["domain_coherence_passed"]
        and presentation_proof["frame_view_matrix_complete"]
        and presentation_proof["area_condition_passed"]
        and unresolved_depth_tie_count == 0
        and fragment_overflow_count == 0
    )
    value = SourceOwnedVisualDynamicIntegrityV1IR(
        package_binding_hash=package.package_hash,
        projection_binding_hash=projection.projection_hash,
        native_playback_binding_hash=playback.playback_hash,
        evaluated_frame_view_count=evaluated_frame_view_count,
        rendered_visible_pixel_count=rendered_visible_pixel_count,
        empty_frame_view_count=empty_frame_view_count,
        flipped_triangle_count=flipped_triangle_count,
        edge_gt_4_count=edge_gt_4_count,
        edge_gt_10_count=edge_gt_10_count,
        maximum_p95_edge_ratio=maximum_p95_edge_ratio,
        maximum_edge_ratio=maximum_edge_ratio,
        native_reference_mismatch_pixel_count=(
            native_reference_mismatch_pixel_count
        ),
        direct_source_provenance_mismatch_pixel_count=(
            direct_source_provenance_mismatch_pixel_count
        ),
        qualification_report={
            **presentation_proof,
            "canonical_depth_ownership_passed": unresolved_depth_tie_count == 0 and fragment_overflow_count == 0,
            "depth_ownership_contract": DEPTH_CONTRACT,
            "unresolved_depth_tie_count": unresolved_depth_tie_count,
            "fragment_overflow_count": fragment_overflow_count,
            "status": (
                "PASS_SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY"
                if passed
                else "FAIL_SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY"
            ),
            "native_reference_byte_parity_passed": (
                native_reference_mismatch_pixel_count == 0
            ),
            "direct_source_provenance_passed": (
                direct_source_provenance_mismatch_pixel_count == 0
            ),
            "all_frame_views_nonempty": empty_frame_view_count == 0,
            "visual_orientation_passed": flipped_triangle_count == 0,
            "catastrophic_edge_stretch_passed": edge_gt_4_count == 0,
            "catastrophic_edge_ratio_threshold": 4.0,
            "p95_edge_ratio_is_diagnostic": True,
            "maximum_edge_ratio_is_diagnostic_below_catastrophic_threshold": True,
            "perceptual_optimality_claimed": False,
            "subject_identity_used_for_thresholds": False,
        },
        integrity_hash="",
        metadata={
            "presentation_geometry_mode": (
                "SOURCE_OWNED_VISUAL_PRESENTATION_V1"
            ),
            "renderer": "REALSAS_V2_SOURCE_OWNED_VISUAL_2D",
            "mechanical_mesh_render_authority": False,
            "runtime_generation": False,
            "dynamic_quality_scope": (
                "STRUCTURAL_RUNTIME_INTEGRITY__NOT_PERCEPTUAL_OPTIMALITY"
            ),
            "native_parallel_workers": native_workers,
        },
    )
    value = replace(
        value,
        integrity_hash=source_owned_visual_dynamic_integrity_hash(value),
    )
    if not passed:
        return {
            "status": "FAIL",
            "blockers": ["SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY_FAILED"],
            "diagnostics": value.to_dict(),
        }
    outputs.insert(
        0,
        write_ir(
            root / "source_owned_visual_dynamic_integrity_v1.json",
            value,
            authority_class=(
                "QUALIFIED_SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY_V1"
            ),
        ),
    )
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "integrity_hash": value.integrity_hash,
            "evaluated_frame_view_count": evaluated_frame_view_count,
            "rendered_visible_pixel_count": rendered_visible_pixel_count,
            "flipped_triangle_count": flipped_triangle_count,
            "edge_gt_4_count": edge_gt_4_count,
            "maximum_p95_edge_ratio": maximum_p95_edge_ratio,
            "maximum_edge_ratio": maximum_edge_ratio,
            "native_reference_mismatch_pixel_count": (
                native_reference_mismatch_pixel_count
            ),
            "direct_source_provenance_mismatch_pixel_count": (
                direct_source_provenance_mismatch_pixel_count
            ),
        },
    }


def prove_dynamic_visual_integrity_stage(ctx: dict) -> dict:
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    if bool(dict(asset.metadata or {}).get("source_owned_visual_mesh_mode")):
        projection = source_owned_visual_runtime_projection_from_dict(
            stage_output_payload(
                ctx,
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1",
            )
        )
        package = runtime_package_seal_from_dict(
            stage_output_payload(
                ctx,
                "43_RSS_MATERIALIZE_COMPACT",
                "RealSaS.RuntimePackageSealIR.v2",
            )
        )
        playback = native_playback_from_dict(
            stage_output_payload(
                ctx,
                "44_NATIVE_PACKAGE_OPEN_PLAYBACK",
                "RealSaS.NativePlaybackIR.v2",
            )
        )
        return _prove_source_owned_visual_dynamic_integrity(
            ctx,
            projection=projection,
            package=package,
            playback=playback,
        )

    projection = runtime_projection_from_dict(
        stage_output_payload(
            ctx,
            "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
            "RealSaS.RuntimeProjectionIR.v2",
        )
    )
    package = runtime_package_seal_from_dict(
        stage_output_payload(
            ctx,
            "43_RSS_MATERIALIZE_COMPACT",
            "RealSaS.RuntimePackageSealIR.v2",
        )
    )
    playback = native_playback_from_dict(
        stage_output_payload(
            ctx,
            "44_NATIVE_PACKAGE_OPEN_PLAYBACK",
            "RealSaS.NativePlaybackIR.v2",
        )
    )
    if playback.package_binding_hash != package.package_hash:
        raise QualificationError("RUNTIME_V2_DVI_PLAYBACK_PACKAGE_DRIFT")
    if playback.projection_binding_hash != projection.projection_hash:
        raise QualificationError("RUNTIME_V2_DVI_PLAYBACK_PROJECTION_DRIFT")

    appearance_qualification = stage_output_payload(
        ctx,
        "24_COMPLETE_APPEARANCE_QUALIFIED",
        "RealSaS.CompleteAppearanceQualificationIR.v2",
    )
    policy = dict(appearance_qualification.get("metadata", {}).get("policy") or {})
    required_dynamic_visibility = (
        "dynamic_max_compiled_unobserved_visible_fraction",
        "dynamic_max_frame_compiled_unobserved_visible_fraction",
        "dynamic_max_connected_compiled_unobserved_visible_fraction",
        "dynamic_max_micro_visible_pixel_fraction_per_frame",
        "dynamic_max_unmeasurable_consequential_visible_face_count",
        "dynamic_max_exact_depth_ambiguous_fraction",
        "dynamic_max_visible_orientation_flip_face_count",
        "dynamic_max_visibility_layer_overflow_pixel_count",
    )
    if any(key not in policy for key in required_dynamic_visibility):
        raise QualificationError("RUNTIME_V2_DVI_VISIBILITY_POLICY_INCOMPLETE")
    exposure_budget = float(
        policy["dynamic_max_compiled_unobserved_visible_fraction"]
    )
    frame_exposure_budget = float(
        policy["dynamic_max_frame_compiled_unobserved_visible_fraction"]
    )
    connected_exposure_budget = float(
        policy["dynamic_max_connected_compiled_unobserved_visible_fraction"]
    )
    micro_visible_budget = float(
        policy["dynamic_max_micro_visible_pixel_fraction_per_frame"]
    )
    max_unmeasurable_consequential = int(
        policy["dynamic_max_unmeasurable_consequential_visible_face_count"]
    )
    exact_depth_ambiguity_budget = float(
        policy["dynamic_max_exact_depth_ambiguous_fraction"]
    )
    max_orientation_flip_faces = int(
        policy["dynamic_max_visible_orientation_flip_face_count"]
    )
    max_layer_overflow_pixels = int(
        policy["dynamic_max_visibility_layer_overflow_pixel_count"]
    )
    for value in (
        exposure_budget,
        frame_exposure_budget,
        connected_exposure_budget,
        micro_visible_budget,
        exact_depth_ambiguity_budget,
    ):
        if not (0.0 <= value <= 1.0):
            raise QualificationError("RUNTIME_V2_DVI_VISIBILITY_BUDGET_INVALID")
    if (
        max_unmeasurable_consequential < 0
        or max_orientation_flip_faces < 0
        or max_layer_overflow_pixels < 0
    ):
        raise QualificationError("RUNTIME_V2_DVI_COUNT_BUDGET_INVALID")
    conditioning_policy = validate_dynamic_appearance_policy(policy)
    min_visible_pixels = int(
        conditioning_policy["dynamic_min_visible_pixels_per_face"]
    )
    min_projected_area2 = float(
        conditioning_policy["dynamic_min_projected_double_area_px2"]
    )

    player, player_sha = _native_player(ctx)
    if player_sha != playback.native_player_sha256:
        raise QualificationError("RUNTIME_V2_DVI_NATIVE_PLAYER_DRIFT")
    archive = resolved_path(package.archive_path)
    arrays = _projection_arrays(projection)
    reference_context = _build_reference_render_context(projection, arrays)
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    native_workers = _native_parallel_workers(ctx)

    faces = np.asarray(arrays["faces"], dtype=np.int64)
    rest_vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    face_uv = np.asarray(arrays["face_uv"], dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("RUNTIME_V2_DVI_FACE_ARRAY_INVALID")
    if face_uv.shape != (len(faces), 3, 2):
        raise QualificationError("RUNTIME_V2_DVI_FACE_UV_ARRAY_INVALID")
    cameras = {}
    for view in projection.views:
        camera = qualify_camera_v3(
            dict(view.camera),
            view_id=view.view_id,
            view_index=view.view_index,
        )
        cameras[view.view_id] = camera
    rest_screen_by_view = {
        view_id: project_points_xyz_v3(rest_vertices, camera)[:, :2]
        for view_id, camera in cameras.items()
    }
    texture_shape_by_view = {}
    for view in projection.views:
        texture_path = resolved_path(view.texture_path)
        with Image.open(texture_path) as image:
            width, height = image.size
        texture_shape_by_view[view.view_id] = (int(width), int(height))

    geometry_visible = 0
    alpha_transparent = 0
    compiled_visible = 0
    undefined_visible = 0
    unsupported_abstain_visible_pixels = 0
    padding_visible_pixels = 0
    mismatch_pixels = 0
    max_mismatch_fraction = 0.0
    max_frame_compiled_fraction = 0.0
    max_connected_compiled_fraction = 0.0
    max_frame_micro_visible_fraction = 0.0
    exact_depth_ambiguous_pixels = 0
    max_frame_exact_depth_ambiguous_fraction = 0.0
    visibility_layer_overflow_pixels = 0
    visible_orientation_flip_faces = 0
    frame_count = 0
    conditioning_sample_count = 0
    relative_conditioning_sample_count = 0
    temporal_conditioning_sample_count = 0
    consequential_visible_face_count = 0
    unmeasurable_visible_face_count = 0
    max_uv_to_surface_condition = 1.0
    max_relative_surface_condition = 1.0
    max_relative_surface_principal_stretch = 1.0
    max_adjacent_frame_surface_principal_stretch = 1.0
    max_texture_texels_per_output_pixel = 0.0
    interior_shared_edge_frame_view_count = 0
    interior_shared_edge_instance_count = 0
    mismatched_interior_shared_edge_count = 0
    maximum_shared_edge_endpoint_error_px = 0.0
    nonmanifold_shared_edge_count_diagnostic = 0
    previous_consequential_faces = {}
    outputs = []

    for clip in projection.clips:
        positions_all = np.asarray(
            arrays[f"{clip.array_prefix}_positions"],
            dtype=np.float64,
        )
        if positions_all.shape != (clip.frame_count, len(rest_vertices), 3):
            raise QualificationError("RUNTIME_V2_DVI_DYNAMIC_POSITION_ARRAY_INVALID")
        for frame_index in range(clip.frame_count):
            posed_vertices = positions_all[frame_index]
            previous_vertices = (
                None if frame_index <= 0 else positions_all[frame_index - 1]
            )
            views = tuple(projection.views)
            native_rows = _run_native_many(
                player=player,
                package=archive,
                requests=(
                    {
                        "clip_id": clip.clip_id,
                        "view_id": view.view_id,
                        "frame_index": frame_index,
                    }
                    for view in views
                ),
                root=root / "frames",
                max_workers=native_workers,
            )
            for view, (rgba_path, prov_path, owner_path, _stdout) in zip(
                views,
                native_rows,
            ):
                resolution = int(view.camera["resolution"])
                rgba = np.frombuffer(rgba_path.read_bytes(), dtype=np.uint8).reshape(
                    resolution, resolution, 4
                )
                prov = np.frombuffer(prov_path.read_bytes(), dtype=np.uint8).reshape(
                    resolution, resolution
                )
                owner = np.frombuffer(owner_path.read_bytes(), dtype="<i4").reshape(
                    resolution, resolution
                )
                reference = _reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view,
                    frame_index=frame_index,
                    reference_context=reference_context,
                )
                parity_mismatch = (
                    np.any(rgba != reference.straight_rgba_u8, axis=2)
                    | (prov != reference.provenance_code)
                    | (owner != reference.owner_face_index)
                )
                visible = np.asarray(reference.geometry_visible, dtype=bool)
                visible_count = int(np.count_nonzero(visible))
                geometry_visible += visible_count
                alpha_transparent += int(
                    np.count_nonzero(
                        visible & (reference.straight_rgba_u8[:, :, 3] == 0)
                    )
                )
                compiled_mask = visible & np.isin(
                    reference.provenance_code,
                    np.asarray(
                        (
                            int(CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]),
                            int(CAA_PROVENANCE["CANONICAL_GLOBAL_COMPLETION"]),
                        ),
                        dtype=np.uint8,
                    ),
                )
                compiled_count = int(np.count_nonzero(compiled_mask))
                compiled_visible += compiled_count
                if visible_count > 0:
                    max_frame_compiled_fraction = max(
                        max_frame_compiled_fraction,
                        float(compiled_count) / float(visible_count),
                    )
                    max_connected_compiled_fraction = max(
                        max_connected_compiled_fraction,
                        _largest_connected_fraction(
                            compiled_mask,
                            denominator=visible_count,
                        ),
                    )
                unsupported_visible_mask = visible & (
                    reference.provenance_code
                    == int(CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"])
                )
                padding_visible_mask = visible & (
                    reference.provenance_code == 255
                )
                unsupported_visible_count = int(
                    np.count_nonzero(unsupported_visible_mask)
                )
                padding_visible_count = int(
                    np.count_nonzero(padding_visible_mask)
                )
                unsupported_abstain_visible_pixels += unsupported_visible_count
                padding_visible_pixels += padding_visible_count
                undefined_visible += (
                    unsupported_visible_count + padding_visible_count
                )

                exact_depth_count = int(
                    np.count_nonzero(reference.exact_depth_ambiguity)
                )
                exact_depth_ambiguous_pixels += exact_depth_count
                visibility_layer_overflow_pixels += int(
                    np.count_nonzero(reference.layer_overflow)
                )
                if visible_count > 0:
                    max_frame_exact_depth_ambiguous_fraction = max(
                        max_frame_exact_depth_ambiguous_fraction,
                        float(exact_depth_count) / float(visible_count),
                    )
                mismatch = parity_mismatch
                mismatch_count = int(np.count_nonzero(mismatch))
                mismatch_pixels += mismatch_count
                frame_pixels = resolution * resolution
                max_mismatch_fraction = max(
                    max_mismatch_fraction,
                    float(mismatch_count) / float(frame_pixels),
                )
                frame_count += 1

                if visible_count > 0:
                    sample_owner = np.asarray(
                        reference.coverage_sample_owner_face_index,
                        dtype=np.int64,
                    )
                    if (
                        sample_owner.ndim != 3
                        or sample_owner.shape[:2] != owner.shape
                        or sample_owner.shape[2]
                        != int(reference.coverage_sample_count)
                    ):
                        raise QualificationError(
                            "RUNTIME_V2_DVI_COVERAGE_SAMPLE_SHAPE_DRIFT"
                        )
                    front_counts = _pixel_face_counts(
                        sample_owner,
                        sample_owner >= 0,
                        face_count=len(faces),
                    )
                    layer_owner = np.asarray(
                        reference.layer_owner_face_index,
                        dtype=np.int64,
                    )
                    contribution_mask = np.asarray(
                        reference.contributing_layer_mask,
                        dtype=bool,
                    )
                    if (
                        layer_owner.shape != contribution_mask.shape
                        or layer_owner.shape[:2] != owner.shape
                    ):
                        raise QualificationError(
                            "RUNTIME_V2_DVI_LAYER_CONTRIBUTION_SHAPE_DRIFT"
                        )
                    contribution_counts = _pixel_face_counts(
                        layer_owner,
                        contribution_mask,
                        face_count=len(faces),
                    )
                    # A face is consequential when it is geometrically frontmost
                    # OR actually contributes color/alpha through the qualified
                    # multi-layer composite. This prevents transparent front
                    # geometry from hiding a deformed deeper surface from DVI.
                    counts = np.maximum(front_counts, contribution_counts)
                    consequential = {
                        int(index)
                        for index in np.nonzero(counts >= min_visible_pixels)[0]
                    }
                    visual_consequential = {
                        int(index)
                        for index in np.nonzero(
                            contribution_counts >= min_visible_pixels
                        )[0]
                    }
                    micro = np.nonzero(
                        (counts > 0) & (counts < min_visible_pixels)
                    )[0]
                    if len(micro) == 0:
                        micro_pixels = 0
                    else:
                        micro_pixel_mask = np.any(
                            np.isin(sample_owner, micro),
                            axis=2,
                        )
                        contributed_micro = contribution_mask & np.isin(
                            layer_owner,
                            micro,
                        )
                        micro_pixel_mask |= np.any(
                            contributed_micro,
                            axis=2,
                        )
                        micro_pixels = int(np.count_nonzero(micro_pixel_mask))
                    max_frame_micro_visible_fraction = max(
                        max_frame_micro_visible_fraction,
                        float(micro_pixels) / float(visible_count),
                    )
                    consequential_visible_face_count += len(consequential)
                else:
                    consequential = set()
                    visual_consequential = set()

                camera = cameras[view.view_id]
                rest_screen = rest_screen_by_view[view.view_id]
                posed_screen = project_points_xyz_v3(
                    posed_vertices, camera
                )[:, :2]
                shared_edge = interior_shared_edge_projection_continuity(
                    faces=faces,
                    projected_face_vertices=posed_screen[faces],
                )
                interior_shared_edge_frame_view_count += 1
                interior_shared_edge_instance_count += int(
                    shared_edge["interior_shared_edge_count"]
                )
                mismatched_interior_shared_edge_count += int(
                    shared_edge["mismatched_interior_shared_edge_count"]
                )
                maximum_shared_edge_endpoint_error_px = max(
                    maximum_shared_edge_endpoint_error_px,
                    float(shared_edge["maximum_projected_endpoint_error_px"]),
                )
                nonmanifold_shared_edge_count_diagnostic += int(
                    shared_edge["nonmanifold_shared_edge_count_diagnostic"]
                )
                previous_screen = (
                    None
                    if previous_vertices is None
                    else project_points_xyz_v3(previous_vertices, camera)[:, :2]
                )
                prior_key = (clip.clip_id, view.view_id)
                prior_consequential = previous_consequential_faces.get(
                    prior_key, set()
                )
                for face_index in sorted(consequential):
                    face = faces[face_index]
                    rest_tri = rest_screen[face]
                    posed_tri = posed_screen[face]
                    if projected_orientation_flip(
                        rest_screen_triangle=rest_tri,
                        posed_screen_triangle=posed_tri,
                        min_projected_double_area_px2=min_projected_area2,
                    ):
                        visible_orientation_flip_faces += 1
                    metrics = dynamic_face_conditioning_metrics(
                        uv_triangle=face_uv[face_index],
                        posed_xyz_triangle=posed_vertices[face],
                        rest_xyz_triangle=rest_vertices[face],
                        posed_screen_triangle=posed_screen[face],
                        previous_xyz_triangle=(
                            previous_vertices[face]
                            if previous_vertices is not None
                            and face_index in prior_consequential
                            else None
                        ),
                        min_projected_double_area_px2=min_projected_area2,
                    )
                    if not bool(metrics["measurable"]):
                        unmeasurable_visible_face_count += 1
                        continue
                    if face_index in visual_consequential:
                        texture_width, texture_height = texture_shape_by_view[
                            view.view_id
                        ]
                        footprint = screen_to_texture_max_texels_per_pixel(
                            uv_triangle=face_uv[face_index],
                            screen_triangle=posed_screen[face],
                            texture_width=texture_width,
                            texture_height=texture_height,
                        )
                        max_texture_texels_per_output_pixel = max(
                            max_texture_texels_per_output_pixel,
                            float(footprint),
                        )
                    conditioning_sample_count += 1
                    max_uv_to_surface_condition = max(
                        max_uv_to_surface_condition,
                        float(metrics["uv_to_surface_condition_number"]),
                    )
                    if metrics["relative_surface_condition_number"] is not None:
                        relative_conditioning_sample_count += 1
                        max_relative_surface_condition = max(
                            max_relative_surface_condition,
                            float(metrics["relative_surface_condition_number"]),
                        )
                        max_relative_surface_principal_stretch = max(
                            max_relative_surface_principal_stretch,
                            float(metrics["relative_surface_principal_stretch"]),
                        )
                    if metrics["adjacent_frame_surface_principal_stretch"] is not None:
                        temporal_conditioning_sample_count += 1
                        max_adjacent_frame_surface_principal_stretch = max(
                            max_adjacent_frame_surface_principal_stretch,
                            float(metrics["adjacent_frame_surface_principal_stretch"]),
                        )
                previous_consequential_faces[prior_key] = consequential

                for path, authority, schema in (
                    (rgba_path, "DVI_NATIVE_RGBA", "application/x-rgba8"),
                    (prov_path, "DVI_NATIVE_PROVENANCE", "application/x-u8-mask"),
                    (owner_path, "DVI_NATIVE_OWNER", "application/x-i32-owner"),
                ):
                    outputs.append(
                        {
                            "path": str(path),
                            "sha256": sha256_file(path),
                            "authority_class": authority,
                            "schema": schema,
                        }
                    )

    exposure_fraction = (
        0.0
        if geometry_visible <= 0
        else float(compiled_visible) / float(geometry_visible)
    )
    transparent_fraction = (
        0.0
        if geometry_visible <= 0
        else float(alpha_transparent) / float(geometry_visible)
    )
    exact_depth_ambiguous_fraction = (
        0.0
        if geometry_visible <= 0
        else float(exact_depth_ambiguous_pixels) / float(geometry_visible)
    )
    visibility_load = dynamic_visibility_load_gate(
        maximum_frame_micro_visible_pixel_fraction=(
            max_frame_micro_visible_fraction
        ),
        unmeasurable_consequential_visible_face_count=(
            unmeasurable_visible_face_count
        ),
        max_micro_visible_pixel_fraction_per_frame=micro_visible_budget,
        max_unmeasurable_consequential_visible_face_count=(
            max_unmeasurable_consequential
        ),
    )
    micro_face_passed = bool(
        visibility_load["micro_visible_face_load_passed"]
    )
    unmeasurable_face_passed = bool(
        visibility_load["unmeasurable_consequential_face_load_passed"]
    )
    conditioning_passed = (
        conditioning_sample_count > 0
        and unmeasurable_face_passed
        and max_uv_to_surface_condition
        <= float(conditioning_policy["dynamic_max_uv_to_surface_condition_number"])
        and max_relative_surface_condition
        <= float(conditioning_policy["dynamic_max_relative_surface_condition_number"])
        and max_relative_surface_principal_stretch
        <= float(conditioning_policy["dynamic_max_relative_surface_principal_stretch"])
        and max_adjacent_frame_surface_principal_stretch
        <= float(
            conditioning_policy["dynamic_max_adjacent_frame_surface_principal_stretch"]
        )
    )
    minification_passed = (
        max_texture_texels_per_output_pixel
        <= float(conditioning_policy["dynamic_max_texture_texels_per_output_pixel"])
    )
    exposure_passed = (
        exposure_fraction <= exposure_budget
        and max_frame_compiled_fraction <= frame_exposure_budget
        and max_connected_compiled_fraction <= connected_exposure_budget
    )
    exact_depth_ambiguity_passed = (
        exact_depth_ambiguous_fraction <= exact_depth_ambiguity_budget
        and max_frame_exact_depth_ambiguous_fraction
        <= exact_depth_ambiguity_budget
    )
    layered_visibility_passed = (
        visibility_layer_overflow_pixels <= max_layer_overflow_pixels
    )
    sidedness_passed = (
        visible_orientation_flip_faces <= max_orientation_flip_faces
    )
    shared_edge_continuity_passed = (
        mismatched_interior_shared_edge_count == 0
        and maximum_shared_edge_endpoint_error_px == 0.0
    )
    passed = (
        geometry_visible > 0
        and undefined_visible == 0
        and mismatch_pixels == 0
        and exposure_passed
        and micro_face_passed
        and exact_depth_ambiguity_passed
        and layered_visibility_passed
        and sidedness_passed
        and shared_edge_continuity_passed
        and minification_passed
        and conditioning_passed
    )
    value = DynamicVisualIntegrityV2IR(
        package_binding_hash=package.package_hash,
        projection_binding_hash=projection.projection_hash,
        native_playback_binding_hash=playback.playback_hash,
        frame_count=frame_count,
        geometry_visible_pixel_count=geometry_visible,
        final_alpha_hole_pixel_count=alpha_transparent,
        final_alpha_hole_fraction=transparent_fraction,
        compiled_unobserved_visible_pixel_count=compiled_visible,
        compiled_unobserved_visible_fraction=exposure_fraction,
        maximum_frame_compiled_unobserved_visible_fraction=max_frame_compiled_fraction,
        maximum_connected_compiled_unobserved_visible_fraction=max_connected_compiled_fraction,
        maximum_frame_micro_visible_pixel_fraction=max_frame_micro_visible_fraction,
        consequential_visible_face_count=consequential_visible_face_count,
        unmeasurable_consequential_visible_face_count=unmeasurable_visible_face_count,
        exact_depth_ambiguous_pixel_count=exact_depth_ambiguous_pixels,
        exact_depth_ambiguous_fraction=exact_depth_ambiguous_fraction,
        maximum_frame_exact_depth_ambiguous_fraction=max_frame_exact_depth_ambiguous_fraction,
        visibility_layer_overflow_pixel_count=visibility_layer_overflow_pixels,
        visible_orientation_flip_face_count=visible_orientation_flip_faces,
        native_reference_mismatch_pixel_count=mismatch_pixels,
        maximum_frame_native_reference_mismatch_fraction=max_mismatch_fraction,
        dynamic_conditioning_sample_count=conditioning_sample_count,
        relative_conditioning_sample_count=relative_conditioning_sample_count,
        temporal_conditioning_sample_count=temporal_conditioning_sample_count,
        maximum_uv_to_surface_condition_number=max_uv_to_surface_condition,
        maximum_relative_surface_condition_number=max_relative_surface_condition,
        maximum_relative_surface_principal_stretch=max_relative_surface_principal_stretch,
        maximum_adjacent_frame_surface_principal_stretch=max_adjacent_frame_surface_principal_stretch,
        qualification_report={
            "status": (
                "PASS_DYNAMIC_VISUAL_INTEGRITY"
                if passed
                else "FAIL_DYNAMIC_VISUAL_INTEGRITY"
            ),
            "undefined_visible_pixel_count": undefined_visible,
            "unsupported_abstain_visible_pixel_count": int(
                unsupported_abstain_visible_pixels
            ),
            "padding_visible_pixel_count": int(
                padding_visible_pixels
            ),
            "compiled_unobserved_exposure_budget": exposure_budget,
            "compiled_unobserved_exposure_passed": exposure_passed,
            "maximum_frame_compiled_unobserved_visible_fraction": max_frame_compiled_fraction,
            "maximum_connected_compiled_unobserved_visible_fraction": max_connected_compiled_fraction,
            "maximum_frame_micro_visible_pixel_fraction": max_frame_micro_visible_fraction,
            "micro_visible_face_load_passed": micro_face_passed,
            "unmeasurable_consequential_face_load_passed": (
                unmeasurable_face_passed
            ),
            "exact_depth_ambiguous_fraction": exact_depth_ambiguous_fraction,
            "maximum_frame_exact_depth_ambiguous_fraction": (
                max_frame_exact_depth_ambiguous_fraction
            ),
            "exact_depth_ambiguity_passed": exact_depth_ambiguity_passed,
            "visibility_layer_overflow_pixel_count": visibility_layer_overflow_pixels,
            "layered_visibility_passed": layered_visibility_passed,
            "visible_orientation_flip_face_count": visible_orientation_flip_faces,
            "surface_sidedness_passed": sidedness_passed,
            "interior_shared_edge_continuity_mode": (
                "TOPOLOGY_OWNED_INTERIOR_SHARED_EDGE_EXACT_PROJECTION_V1"
            ),
            "interior_shared_edge_frame_view_count": (
                interior_shared_edge_frame_view_count
            ),
            "interior_shared_edge_instance_count": (
                interior_shared_edge_instance_count
            ),
            "mismatched_interior_shared_edge_count": (
                mismatched_interior_shared_edge_count
            ),
            "maximum_shared_edge_endpoint_error_px": (
                maximum_shared_edge_endpoint_error_px
            ),
            "nonmanifold_shared_edge_count_diagnostic": (
                nonmanifold_shared_edge_count_diagnostic
            ),
            "interior_shared_edge_continuity_passed": (
                shared_edge_continuity_passed
            ),
            "cross_component_background_gap_is_crack_authority": False,
            "maximum_texture_texels_per_output_pixel": (
                max_texture_texels_per_output_pixel
            ),
            "texture_minification_passed": minification_passed,
            "native_reference_byte_parity_passed": mismatch_pixels == 0,
            "dynamic_appearance_conditioning_passed": conditioning_passed,
            "dynamic_appearance_policy": dict(conditioning_policy),
            "consequential_visible_face_count": consequential_visible_face_count,
            "unmeasurable_consequential_visible_face_count": (
                unmeasurable_visible_face_count
            ),
            "alpha_transparency_is_diagnostic_not_undefinedness": True,
            "unsupported_abstention_is_hard_undefined_visibility": True,
            "physical_padding_is_hard_undefined_visibility": True,
            "renderer": "REALSAS_V2_CAA_CANONICAL_DEPTH",
        },
        visual_integrity_hash="",
        metadata={
            "geometry_visibility_appearance_sampling_attribution": True,
            "final_alpha_transparent_fraction_diagnostic": transparent_fraction,
            "dynamic_appearance_conditioning": (
                "RIGID_INVARIANT_CAA_UV_TO_POSED_SURFACE_METRIC"
            ),
            "perceptual_optimality_claimed": False,
            "subject_identity_used_for_thresholds": False,
            "dynamic_crack_authority": (
                "TOPOLOGY_OWNED_INTERIOR_SHARED_EDGES_ONLY"
            ),
            "raw_owner_negative_background_area_gated_as_crack": False,
            "cross_component_non_detachability_authority_claimed": False,
            "native_parallel_workers": native_workers,
        },
    )
    value = replace(
        value, visual_integrity_hash=dynamic_visual_integrity_hash(value)
    )
    if not passed:
        return {
            "status": "FAIL",
            "blockers": ["DYNAMIC_VISUAL_INTEGRITY_FAILED"],
            "diagnostics": value.to_dict(),
        }
    outputs.insert(
        0,
        write_ir(
            root / "dynamic_visual_integrity_v2.json",
            value,
            authority_class="QUALIFIED_DYNAMIC_VISUAL_INTEGRITY_V2",
        ),
    )
    return {
        "status": "PASS",
        "outputs": outputs,
        "diagnostics": {
            "visual_integrity_hash": value.visual_integrity_hash,
            "frame_view_count": frame_count,
            "compiled_unobserved_visible_fraction": exposure_fraction,
            "maximum_frame_compiled_unobserved_visible_fraction": max_frame_compiled_fraction,
            "maximum_connected_compiled_unobserved_visible_fraction": max_connected_compiled_fraction,
            "maximum_frame_micro_visible_pixel_fraction": max_frame_micro_visible_fraction,
            "consequential_visible_face_count": consequential_visible_face_count,
            "unmeasurable_consequential_visible_face_count": unmeasurable_visible_face_count,
            "native_reference_mismatch_pixel_count": 0,
            "undefined_visible_pixel_count": 0,
            "dynamic_conditioning_sample_count": conditioning_sample_count,
            "maximum_uv_to_surface_condition_number": max_uv_to_surface_condition,
            "maximum_relative_surface_condition_number": (
                max_relative_surface_condition
            ),
            "maximum_relative_surface_principal_stretch": max_relative_surface_principal_stretch,
            "maximum_adjacent_frame_surface_principal_stretch": (
                max_adjacent_frame_surface_principal_stretch
            ),
            "transparent_visible_fraction_diagnostic": transparent_fraction,
            "exact_depth_ambiguous_fraction": exact_depth_ambiguous_fraction,
            "maximum_frame_exact_depth_ambiguous_fraction": (
                max_frame_exact_depth_ambiguous_fraction
            ),
            "visibility_layer_overflow_pixel_count": visibility_layer_overflow_pixels,
            "visible_orientation_flip_face_count": visible_orientation_flip_faces,
            "interior_shared_edge_instance_count": interior_shared_edge_instance_count,
            "mismatched_interior_shared_edge_count": (
                mismatched_interior_shared_edge_count
            ),
            "maximum_shared_edge_endpoint_error_px": (
                maximum_shared_edge_endpoint_error_px
            ),
            "interior_shared_edge_continuity_passed": (
                shared_edge_continuity_passed
            ),
        },
    }
