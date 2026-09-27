from __future__ import annotations

"""Native fast-path for Knight occupancy witness.

Builds a diagnostic RSS from already-sealed upstream artifacts and renders
V0/V2 frames with the C++ Runtime V2 reference player. Python only prepares
posed vertices and occupancy-gated atlas pages; it does not rasterize frames.
"""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    RuntimeClipV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    runtime_projection_hash,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_rss_v2_entries,
    write_rss_v2,
)
from compiler.realsas_compiler_core.visibility_v2 import VISIBILITY_CONTRACT_V2_HASH
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_texture_pages,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)
from tools.demo.render_knight_occupancy_witness_v1 import (
    _canonical_occupancy_pages,
)


def _camera_dict(camera, *, resolution: int) -> dict:
    return {
        "view_index": int(camera.view_index),
        "origin": list(map(float, camera.origin)),
        "right": list(map(float, camera.right)),
        "screen_up": list(map(float, camera.screen_up)),
        "forward": list(map(float, camera.forward)),
        "half_extent": float(camera.half_extent),
        "resolution": int(resolution),
    }


def _faces(candidate) -> np.ndarray:
    index = {
        str(vertex.candidate_vertex_id): i
        for i, vertex in enumerate(candidate.vertices)
    }
    if len(index) != len(candidate.vertices):
        raise RuntimeError("NATIVE_WITNESS_VERTEX_ID_DUPLICATE")
    out = []
    for face in candidate.faces:
        try:
            out.append(tuple(index[str(vertex_id)] for vertex_id in face))
        except KeyError as exc:
            raise RuntimeError("NATIVE_WITNESS_FACE_VERTEX_UNKNOWN") from exc
    value = np.asarray(out, dtype=np.uint32)
    if value.shape != (len(candidate.faces), 3):
        raise RuntimeError("NATIVE_WITNESS_FACE_SHAPE_INVALID")
    return value


def _page_rows(texture):
    rows = tuple(
        sorted(
            (dict(row) for row in dict(texture.metadata or {}).get("pages") or ()),
            key=lambda row: int(row["page_index"]),
        )
    )
    if not rows:
        rows = (
            {
                "page_index": 0,
                "path": str(texture.transport_png_path),
                "sha256": str(texture.transport_png_sha256),
            },
        )
    return rows


def _build_modified_view_pages(
    *,
    texture,
    occupancy_alpha_pages: np.ndarray,
    target_root: Path,
    view_index: int,
):
    original = _load_texture_pages(texture)
    if original.ndim != 4 or original.shape[:3] != occupancy_alpha_pages.shape:
        raise RuntimeError("NATIVE_WITNESS_TEXTURE_OCCUPANCY_SHAPE_DRIFT")
    value = original.copy()
    value[..., 3] = occupancy_alpha_pages
    rows = []
    target_root.mkdir(parents=True, exist_ok=True)
    for page_index in range(value.shape[0]):
        path = target_root / f"V{view_index}_p{page_index}.png"
        Image.fromarray(value[page_index], mode="RGBA").save(
            path,
            format="PNG",
            optimize=False,
            compress_level=1,
        )
        rows.append(
            {
                "page_index": int(page_index),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return tuple(rows)


def _render_native(player: Path, package: Path, request: dict):
    root = Path(request["root"])
    root.mkdir(parents=True, exist_ok=True)
    stem = f'{request["clip"]}_{request["view"]}_f{request["frame"]}'
    rgba = root / f"{stem}.rgba"
    prov = root / f"{stem}.prov"
    owner = root / f"{stem}.owner"
    cmd = [
        str(player),
        str(package),
        "--clip", str(request["clip"]),
        "--view", str(request["view"]),
        "--frame", str(int(request["frame"])),
        "--out-rgba", str(rgba),
        "--out-provenance", str(prov),
        "--out-owner", str(owner),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"NATIVE_WITNESS_RENDER_FAIL:{request['clip']}:{request['view']}:{request['frame']}\n"
            + proc.stdout
        )
    return {
        **request,
        "rgba_path": rgba,
        "stdout": proc.stdout,
    }


def _gif_frame(v0: Image.Image, v2: Image.Image, target_height: int = 512):
    combined = Image.new("RGBA", (v0.width + v2.width, max(v0.height, v2.height)))
    combined.alpha_composite(v0, (0, 0))
    combined.alpha_composite(v2, (v0.width, 0))
    if combined.height != target_height:
        width = max(1, round(combined.width * target_height / combined.height))
        combined = combined.resize((width, target_height), Image.Resampling.LANCZOS)
    return combined.convert("P", palette=Image.Palette.ADAPTIVE, colors=255)


def run(
    *,
    authority_root: Path,
    run_id: str,
    out_dir: Path,
    resolution: int,
    native_player: Path,
    workers: int,
):
    ctx = _ctx(authority_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )

    occupancy_alpha_pages, occupancy_stats = _canonical_occupancy_pages(
        ctx,
        candidate,
        appearance,
    )

    texture_by_view = {
        int(row.direction_index): row
        for row in appearance.textures
    }
    modified_root = out_dir / "occupancy_texture_pages"
    runtime_views = []
    for view_index, camera in enumerate(cameras):
        texture = texture_by_view[view_index]
        if view_index in (0, 2):
            page_rows = _build_modified_view_pages(
                texture=texture,
                occupancy_alpha_pages=occupancy_alpha_pages,
                target_root=modified_root,
                view_index=view_index,
            )
        else:
            page_rows = _page_rows(texture)

        primary = dict(page_rows[0])
        runtime_views.append(
            RuntimeViewV2IR(
                view_index=int(view_index),
                view_id=f"V{view_index}",
                camera=_camera_dict(camera, resolution=resolution),
                texture_path=str(primary["path"]),
                texture_sha256=str(primary["sha256"]),
                metadata={
                    "paged_atlas": True,
                    "texture_pages": [dict(row) for row in page_rows],
                    "diagnostic_occupancy_alpha_override": bool(
                        view_index in (0, 2)
                    ),
                },
            )
        )

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    clip_specs = [
        ("demo_idle_v1", "IDLE", 833),
        ("demo_run_v1", "RUN", 208),
        ("demo_slash_v1", "SLASH", 278),
    ]
    arrays = {
        "vertices": rest,
        "faces": _faces(candidate),
        "face_uv": load_face_uv(appearance),
        "face_page_index": load_face_page_index(appearance),
    }
    runtime_clips = []
    gif_duration_by_clip = {}
    for clip_id, short, gif_ms in clip_specs:
        path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(path.read_text())
        tracks, _mapping = _tracks_for_clip(
            payload,
            skeleton,
            cameras,
            source_report,
        )
        duration = float(payload["duration_seconds"])
        times = np.linspace(
            0.0,
            duration,
            4,
            endpoint=not bool(payload.get("loop")),
            dtype=np.float64,
        )
        posed_frames = []
        for time_seconds in times:
            skin_mats, _joint_pos, _frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(time_seconds),
                cameras=cameras,
            )
            posed_frames.append(
                _skin(rest, W, joint_ids, skin_mats)
            )
        prefix = short.lower()
        arrays[f"{prefix}_times"] = times
        arrays[f"{prefix}_positions"] = np.stack(posed_frames, axis=0)
        runtime_clips.append(
            RuntimeClipV2IR(
                clip_id=clip_id,
                duration_seconds=duration,
                loop=bool(payload.get("loop")),
                frame_count=4,
                array_prefix=prefix,
                metadata={"diagnostic_only": True},
            )
        )
        gif_duration_by_clip[clip_id] = int(gif_ms)

    projection_npz = out_dir / "native_witness_projection.npz"
    np.savez(projection_npz, **arrays)
    projection_sha = sha256_file(projection_npz)

    projection = RuntimeProjectionV2IR(
        complete_puppet_binding_hash="diagnostic",
        mechanical_state_binding_hash="diagnostic",
        mesh_binding_hash=str(getattr(candidate, "candidate_lineage_hash", "diagnostic")),
        dynamic_motion_binding_hash="diagnostic",
        appearance_asset_binding_hash=appearance.asset_hash,
        appearance_qualification_binding_hash="diagnostic",
        camera_set_binding_hash=camera_set.camera_set_hash,
        visibility_contract_hash=VISIBILITY_CONTRACT_V2_HASH,
        projection_npz_path=str(projection_npz.resolve()),
        projection_npz_sha256=projection_sha,
        provenance_npz_path=str(Path(appearance.provenance_npz_path).resolve()),
        provenance_npz_sha256=str(appearance.provenance_npz_sha256),
        views=tuple(runtime_views),
        clips=tuple(runtime_clips),
        projection_hash="",
        metadata={
            "diagnostic_only": True,
            "occupancy_witness": True,
            "product_authority_claimed": False,
        },
    )
    projection = replace(
        projection,
        projection_hash=runtime_projection_hash(projection),
    )

    package = out_dir / "knight_occupancy_witness.rss"
    package_meta = write_rss_v2(
        package,
        build_rss_v2_entries(projection),
    )

    requests = []
    for clip in runtime_clips:
        for frame_index in range(clip.frame_count):
            for view_id in ("V0", "V2"):
                requests.append(
                    {
                        "clip": clip.clip_id,
                        "view": view_id,
                        "frame": int(frame_index),
                        "root": out_dir / "native_frames",
                    }
                )

    max_workers = max(1, min(int(workers), len(requests)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rendered = list(
            pool.map(
                lambda row: _render_native(native_player, package, row),
                requests,
            )
        )

    resolution = int(resolution)
    by_key = {}
    for row in rendered:
        raw = np.frombuffer(
            Path(row["rgba_path"]).read_bytes(),
            dtype=np.uint8,
        )
        if raw.size != resolution * resolution * 4:
            raise RuntimeError("NATIVE_WITNESS_RGBA_SIZE_DRIFT")
        image = Image.fromarray(
            raw.reshape(resolution, resolution, 4),
            mode="RGBA",
        )
        by_key[(row["clip"], row["view"], int(row["frame"]))] = image

    outputs = []
    for clip_id, short, _gif_ms in clip_specs:
        frames = [
            _gif_frame(
                by_key[(clip_id, "V0", frame_index)],
                by_key[(clip_id, "V2", frame_index)],
            )
            for frame_index in range(4)
        ]
        gif_path = out_dir / f"KNIGHT_{short}_OCCUPANCY_NATIVE_WITNESS_V2.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration_by_clip[clip_id],
            loop=0,
            disposal=2,
            optimize=False,
            transparency=0,
        )
        outputs.append(str(gif_path))

    report = {
        "schema": "RealSaS.KnightOccupancyNativeWitness.v2",
        "status": "MEASURED_DEMO_ONLY",
        "run_id": run_id,
        "renderer": "CXX_RUNTIME_V2_CAA_CANONICAL_DEPTH",
        "python_rasterization_used": False,
        "diagnostic_rss_path": str(package),
        "diagnostic_rss_sha256": str(package_meta["archive_sha256"]),
        "render_resolution": int(resolution),
        "native_workers": int(max_workers),
        "occupancy_stats": occupancy_stats,
        "outputs": outputs,
        "product_authority_claimed": False,
    }
    report_path = out_dir / "KNIGHT_OCCUPANCY_NATIVE_WITNESS_REPORT_V2.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "KNIGHT_OCCUPANCY_NATIVE_WITNESS_PASS",
        json.dumps(
            {
                "gif_count": len(outputs),
                "native_workers": int(max_workers),
                "resolution": int(resolution),
                "rss_bytes": int(package_meta["archive_bytes"]),
            },
            sort_keys=True,
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--native-player", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir),
        resolution=int(a.resolution),
        native_player=Path(a.native_player).expanduser().resolve(),
        workers=int(a.workers),
    )


if __name__ == "__main__":
    main()
