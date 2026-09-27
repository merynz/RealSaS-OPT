from __future__ import annotations

"""Fast Knight alpha/occupancy witness.

Diagnostic only. Reuses sealed Stage18/20/21/23/28/32 artifacts.
CAA RGB remains untouched. Alpha is separated into a canonical occupancy field:
- any direct source observation is immutable authority (foreground alpha or background zero),
- only source-unobserved samples may be solved,
- unknown occupancy may fill only through the same bounded surface-local harmonic policy,
- policy-violating / boundary-less regions abstain to transparent.

This falsifies whether CAA RGBA completion is improperly creating silhouette.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE,
    caa_compile_artifact_from_dict,
    caa_preregistration_from_dict,
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_bake_v2 import (
    bake_direction_adaptive_atlas_pages,
    bake_direction_atlas_pages,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    bounded_surface_harmonic_fill,
    surface_sample_neighbors,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_compile_arrays,
    _load_texture_pages,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)


def _canonical_occupancy_pages(ctx, candidate, appearance):
    prereg = caa_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "20_CAA_BACKEND_PREREGISTERED",
            "RealSaS.CAACompilePreregistrationIR.v2",
        )
    )
    artifact = caa_compile_artifact_from_dict(
        stage_output_payload(
            ctx,
            "21_CAA_COMPILE",
            "RealSaS.CAACompileArtifactIR.v2",
        )
    )
    arrays = _load_compile_arrays(
        artifact,
        required_names={
            "sample_positions",
            "sample_component_index",
            "direct_valid",
            "direct_rgba",
        },
    )

    direct = np.asarray(arrays["direct_valid"], dtype=bool)
    direct_rgba = np.asarray(arrays["direct_rgba"], dtype=np.uint8)
    if direct.shape != direct_rgba.shape[:2] or direct_rgba.shape[2] != 4:
        raise RuntimeError("OCCUPANCY_DIRECT_SHAPE_DRIFT")

    # Immutable source occupancy authority:
    # foreground contributes its authored alpha; observed transparent
    # background contributes zero. Multiple views combine by maximum authored
    # alpha so an oblique transparent observation cannot erase a foreground
    # observation from another qualified view.
    observed = np.any(direct, axis=0)
    source_alpha = np.where(direct, direct_rgba[:, :, 3], 0).max(axis=0)

    occ_rgba = np.zeros((len(observed), 4), dtype=np.uint8)
    occ_rgba[:, 3] = source_alpha
    occ_provenance = np.full(
        (len(observed),),
        CAA_PROVENANCE["DIRECT_SOURCE"],
        dtype=np.uint8,
    )
    occ_source_view = np.zeros((len(observed),), dtype=np.int16)
    missing = ~observed

    sample_mode = str(
        dict(artifact.metadata or {}).get("sample_count_mode")
        or "UNIFORM_FACE_LATTICE_V1"
    )
    adaptive = sample_mode == "PER_FACE_ADAPTIVE_V1"
    offsets = (
        np.asarray(arrays["face_sample_offsets"], dtype=np.int64)
        if adaptive
        else None
    )
    resolutions = (
        np.asarray(arrays["face_tile_resolutions"], dtype=np.int32)
        if adaptive
        else None
    )
    face_vertex_ids = tuple(tuple(map(str, face)) for face in candidate.faces)
    graph = surface_sample_neighbors(
        positions=np.asarray(arrays["sample_positions"], dtype=np.float64),
        face_count=artifact.face_count,
        tile_resolution=None if adaptive else int(artifact.tile_resolution),
        face_sample_offsets=offsets,
        face_tile_resolutions=resolutions,
        face_vertex_ids=face_vertex_ids,
    )

    policy = dict(prereg.completion_quality_policy)
    stats = bounded_surface_harmonic_fill(
        rgba=occ_rgba,
        provenance=occ_provenance,
        source_view=occ_source_view,
        missing=missing,
        observed_mask=observed,
        sample_component=np.asarray(
            arrays["sample_component_index"],
            dtype=np.int32,
        ),
        neighbors=graph,
        max_region_samples=int(policy["max_local_harmonic_region_samples"]),
        max_graph_hops=int(policy["max_local_harmonic_graph_hops"]),
        abstain_on_policy_violation=True,
        abstain_provenance_code=CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"],
        abstain_source_view_value=-4,
    )

    bleed = int(prereg.compile_policy["bleed_px"])
    max_page = int(prereg.compile_policy["max_atlas_resolution"])
    dummy_provenance = np.zeros((len(occ_rgba),), dtype=np.uint8)
    if adaptive:
        pages, _prov, uv, face_page, layout = bake_direction_adaptive_atlas_pages(
            face_sample_rgba=occ_rgba,
            face_sample_provenance=dummy_provenance,
            face_tile_resolutions=resolutions,
            face_sample_offsets=offsets,
            bleed_px=bleed,
            max_page_resolution=max_page,
        )
    else:
        pages, _prov, uv, face_page, layout = bake_direction_atlas_pages(
            face_sample_rgba=occ_rgba,
            face_sample_provenance=dummy_provenance,
            face_count=artifact.face_count,
            tile_resolution=int(artifact.tile_resolution),
            bleed_px=bleed,
            max_page_resolution=max_page,
        )

    asset_uv = load_face_uv(appearance)
    asset_page = load_face_page_index(appearance)
    if not np.array_equal(uv, asset_uv) or not np.array_equal(face_page, asset_page):
        raise RuntimeError("OCCUPANCY_ATLAS_LAYOUT_DRIFT")
    if dict(layout) != dict(appearance.atlas_layout):
        raise RuntimeError("OCCUPANCY_ATLAS_METADATA_DRIFT")

    return pages[..., 3].astype(np.uint8), {
        "observed_sample_count": int(np.count_nonzero(observed)),
        "observed_opaque_sample_count": int(
            np.count_nonzero(observed & (source_alpha > 0))
        ),
        "observed_transparent_sample_count": int(
            np.count_nonzero(observed & (source_alpha == 0))
        ),
        "initial_unknown_sample_count": int(np.count_nonzero(~observed)),
        "final_opaque_sample_count": int(np.count_nonzero(occ_rgba[:, 3] > 0)),
        "final_transparent_sample_count": int(np.count_nonzero(occ_rgba[:, 3] == 0)),
        "bounded_solve": stats,
        "policy": {
            "max_local_harmonic_region_samples": int(
                policy["max_local_harmonic_region_samples"]
            ),
            "max_local_harmonic_graph_hops": int(
                policy["max_local_harmonic_graph_hops"]
            ),
        },
    }


def _gif_frame(v0: Image.Image, v2: Image.Image, target_height: int = 512):
    combined = Image.new("RGBA", (v0.width + v2.width, max(v0.height, v2.height)))
    combined.alpha_composite(v0, (0, 0))
    combined.alpha_composite(v2, (v0.width, 0))
    if combined.height != target_height:
        width = max(1, round(combined.width * target_height / combined.height))
        combined = combined.resize((width, target_height), Image.Resampling.LANCZOS)
    return combined.convert("P", palette=Image.Palette.ADAPTIVE, colors=255)


def run(*, authority_root: Path, run_id: str, out_dir: Path, resolution: int):
    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1")
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1")
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras_full = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    cameras = tuple(replace(c, resolution=int(resolution)) for c in cameras_full)
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )

    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}
    occupancy_alpha_pages, occupancy_stats = _canonical_occupancy_pages(
        ctx, candidate, appearance
    )

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "RealSaS.KnightOccupancyWitness.v1",
        "status": "MEASURED_DEMO_ONLY",
        "run_id": run_id,
        "hypothesis": "CAA_RGB_COMPLETION_MUST_NOT_OWN_SILHOUETTE",
        "occupancy_authority": "DIRECT_SOURCE_ALPHA_PLUS_BOUNDED_SURFACE_LOCAL_ALPHA_ONLY_COMPLETION",
        "rgb_authority": "UNCHANGED_STAGE23_COMPLETE_APPEARANCE",
        "stage35_failure_preserved": True,
        "product_authority_claimed": False,
        "render_resolution": int(resolution),
        "occupancy_stats": occupancy_stats,
        "clips": [],
    }

    clip_specs = [
        ("demo_idle_v1", "IDLE", 833),
        ("demo_run_v1", "RUN", 208),
        ("demo_slash_v1", "SLASH", 278),
    ]
    for clip_id, short, duration_ms in clip_specs:
        motion_path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
        tracks, mapping_details = _tracks_for_clip(
            payload, skeleton, cameras_full, source_report
        )
        duration = float(payload["duration_seconds"])
        sample_times = np.linspace(
            0.0,
            duration,
            4,
            endpoint=not bool(payload.get("loop")),
        )

        by_view = {}
        metrics = []
        for view in (0, 2):
            texture = _load_texture_pages(texture_by_view[view]).copy()
            if texture.ndim != 4:
                raise RuntimeError("OCCUPANCY_WITNESS_REQUIRES_PAGED_TEXTURE")
            if texture.shape[:3] != occupancy_alpha_pages.shape:
                raise RuntimeError("OCCUPANCY_PAGE_SHAPE_DRIFT")
            texture[..., 3] = occupancy_alpha_pages
            frames = []
            for frame_index, time_seconds in enumerate(sample_times):
                skin_mats, _joint_pos, _frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras_full,
                )
                posed = _skin(rest, W, joint_ids, skin_mats)
                render = render_caa_reference(
                    mesh=candidate,
                    camera=cameras[view],
                    face_uv=face_uv,
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,
                    positions=posed,
                )
                rgba = Image.fromarray(
                    np.asarray(render.straight_rgba_u8, dtype=np.uint8),
                    mode="RGBA",
                )
                frames.append(rgba)
                metrics.append({
                    "view_index": view,
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                    "geometry_visible_pixel_count": int(
                        np.count_nonzero(render.geometry_visible)
                    ),
                    "final_alpha_pixel_count": int(
                        np.count_nonzero(render.final_alpha)
                    ),
                })
            by_view[view] = frames

        gif_frames = [
            _gif_frame(by_view[0][i], by_view[2][i])
            for i in range(len(sample_times))
        ]
        gif_path = out_dir / f"KNIGHT_{short}_OCCUPANCY_WITNESS_V1.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(duration_ms),
            loop=0,
            disposal=2,
            optimize=False,
            transparency=0,
        )
        report["clips"].append({
            "clip_id": clip_id,
            "duration_seconds": duration,
            "gif_path": str(gif_path),
            "mapping": mapping_details,
            "frame_metrics": metrics,
        })

    report_path = out_dir / "KNIGHT_OCCUPANCY_WITNESS_REPORT_V1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_OCCUPANCY_WITNESS_PASS", json.dumps({
        "gif_count": 3,
        "resolution": int(resolution),
        "initial_unknown_sample_count": occupancy_stats["initial_unknown_sample_count"],
        "final_opaque_sample_count": occupancy_stats["final_opaque_sample_count"],
    }, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir),
        resolution=int(a.resolution),
    )


if __name__ == "__main__":
    main()
