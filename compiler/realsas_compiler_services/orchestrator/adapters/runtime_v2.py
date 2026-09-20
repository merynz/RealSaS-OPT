from __future__ import annotations

"""V2 Stage42-45 deterministic runtime projection, package, native proof and DVI."""

from dataclasses import asdict, replace
import hashlib
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
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
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
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_rss_v2_entries,
    read_rss_v2,
    write_rss_v2,
)
from compiler.realsas_compiler_core.visibility_v2 import (
    VISIBILITY_CONTRACT_V2_HASH,
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

    vertex_ids = [str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices]
    vertex_index = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}
    if len(vertex_index) != len(vertex_ids):
        raise QualificationError("RUNTIME_V2_DUPLICATE_MESH_VERTEX_ID")
    vertices = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float32)
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
        face_uv = np.asarray(data["face_uv"], dtype=np.float32)
    if face_uv.shape != (len(mesh.faces), 3, 2):
        raise QualificationError("RUNTIME_V2_FACE_UV_TOPOLOGY_DRIFT")

    arrays = {
        "vertices": vertices,
        "faces": faces,
        "face_uv": face_uv,
    }
    clips = []
    for clip_index, clip in enumerate(dynamic.clips):
        times = np.asarray([frame.time_seconds for frame in clip.frames], dtype=np.float64)
        positions = np.zeros(
            (len(clip.frames), len(vertex_ids), 3),
            dtype=np.float32,
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
                dtype=np.float32,
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
                },
            )
        )

    by_texture = {row.direction_index: row for row in asset.textures}
    views = []
    for camera in sorted(cameras.cameras, key=lambda row: row.view_index):
        texture = by_texture[int(camera.view_index)]
        path = resolved_path(texture.transport_png_path)
        if not path.is_file() or sha256_file(path) != texture.transport_png_sha256:
            raise QualificationError("RUNTIME_V2_TEXTURE_BYTES_DRIFT")
        views.append(
            RuntimeViewV2IR(
                view_index=int(camera.view_index),
                view_id=str(camera.view_id),
                camera=asdict(camera),
                texture_path=str(path),
                texture_sha256=texture.transport_png_sha256,
                metadata={
                    "visibility": "CANONICAL_POSED_XYZ_ZBUFFER",
                    "appearance": "SEALED_CAA_V2",
                    "runtime_generation": False,
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
        raise QualificationError("RUNTIME_V2_PACKAGE_PROJECTION_BYTES_DRIFT")
    if sha256_file(provenance) != projection.provenance_npz_sha256:
        raise QualificationError("RUNTIME_V2_PACKAGE_PROVENANCE_BYTES_DRIFT")
    for view in projection.views:
        if sha256_file(resolved_path(view.texture_path)) != view.texture_sha256:
            raise QualificationError("RUNTIME_V2_PACKAGE_TEXTURE_BYTES_DRIFT")

    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    archive = root / "product_runtime_v2.rss"
    entries = build_rss_v2_entries(projection)
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
            "compression": "NONE_V1",
            "native_reader_dependency_free": True,
            "contains_only_sealed_runtime_authorities": True,
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
    if "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH" not in completed.stdout:
        raise QualificationError("RUNTIME_V2_NATIVE_RENDERER_CONTRACT_DRIFT")
    return rgba, provenance, owner, completed.stdout


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


def _reference_frame(projection, arrays, *, clip, view, frame_index):
    mesh = _numeric_mesh(arrays)
    positions = np.asarray(
        arrays[f"{clip.array_prefix}_positions"][frame_index],
        dtype=np.float64,
    )
    camera = qualify_camera_v3(
        dict(view.camera), view_id=view.view_id, view_index=view.view_index
    )
    texture = np.asarray(
        Image.open(resolved_path(view.texture_path)).convert("RGBA"),
        dtype=np.uint8,
    )
    with np.load(projection.provenance_npz_path, allow_pickle=False) as data:
        provenance = np.asarray(data["provenance"][view.view_index], dtype=np.uint8)
    return render_caa_reference(
        mesh=mesh,
        camera=camera,
        face_uv=np.asarray(arrays["face_uv"], dtype=np.float64),
        texture_rgba_u8=texture,
        provenance_atlas=provenance,
        positions=positions,
    )


def prove_native_package_playback_stage(ctx: dict) -> dict:
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
    if package.projection_binding_hash != projection.projection_hash:
        raise QualificationError("RUNTIME_V2_NATIVE_PACKAGE_PROJECTION_DRIFT")
    archive = resolved_path(package.archive_path)
    if not archive.is_file() or sha256_file(archive) != package.archive_sha256:
        raise QualificationError("RUNTIME_V2_NATIVE_PACKAGE_BYTES_DRIFT")
    player, player_sha = _native_player(ctx)
    arrays = _projection_arrays(projection)
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]

    probes = []
    outputs = []
    view_by_id = {view.view_id: view for view in projection.views}
    for clip in projection.clips:
        frame_index = int(clip.frame_count // 2)
        for view in projection.views:
            rgba, provenance, owner, stdout = _run_native(
                player=player,
                package=archive,
                clip_id=clip.clip_id,
                view_id=view.view_id,
                frame_index=frame_index,
                root=root / "probes",
            )
            resolution = int(view.camera["resolution"])
            if rgba.stat().st_size != resolution * resolution * 4:
                raise QualificationError("RUNTIME_V2_NATIVE_RGBA_SIZE_DRIFT")
            if provenance.stat().st_size != resolution * resolution:
                raise QualificationError("RUNTIME_V2_NATIVE_PROVENANCE_SIZE_DRIFT")
            if owner.stat().st_size != resolution * resolution * 4:
                raise QualificationError("RUNTIME_V2_NATIVE_OWNER_SIZE_DRIFT")

            native_rgba = np.frombuffer(rgba.read_bytes(), dtype=np.uint8).reshape(
                resolution, resolution, 4
            )
            reference = _reference_frame(
                projection,
                arrays,
                clip=clip,
                view=view_by_id[view.view_id],
                frame_index=frame_index,
            )
            mismatch = int(
                np.count_nonzero(
                    np.any(native_rgba != reference.straight_rgba_u8, axis=2)
                )
            )
            if mismatch:
                raise QualificationError(
                    f"RUNTIME_V2_NATIVE_REFERENCE_PARITY_FAIL:{clip.clip_id}:{view.view_id}:{mismatch}"
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
                stdout_sha256=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                probe_hash="",
                metadata={
                    "native_reference_mismatch_pixels": 0,
                    "renderer": "REALSAS_V2_CAA_CANONICAL_DEPTH",
                },
            )
            probe = replace(probe, probe_hash=native_playback_probe_hash(probe))
            probes.append(probe)
            for path, authority, schema in (
                (rgba, "NATIVE_V2_RGBA", "application/x-rgba8"),
                (provenance, "NATIVE_V2_PROVENANCE", "application/x-u8-mask"),
                (owner, "NATIVE_V2_OWNER", "application/x-i32-owner"),
            ):
                outputs.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
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
        },
    )
    playback = replace(playback, playback_hash=native_playback_hash(playback))
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
        },
    }


def prove_dynamic_visual_integrity_stage(ctx: dict) -> dict:
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
    if "dynamic_max_compiled_unobserved_visible_fraction" not in policy:
        raise QualificationError("RUNTIME_V2_DVI_EXPOSURE_BUDGET_MISSING")
    exposure_budget = float(
        policy["dynamic_max_compiled_unobserved_visible_fraction"]
    )
    if not (0.0 <= exposure_budget <= 1.0):
        raise QualificationError("RUNTIME_V2_DVI_EXPOSURE_BUDGET_INVALID")

    player, player_sha = _native_player(ctx)
    if player_sha != playback.native_player_sha256:
        raise QualificationError("RUNTIME_V2_DVI_NATIVE_PLAYER_DRIFT")
    archive = resolved_path(package.archive_path)
    arrays = _projection_arrays(projection)
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]

    geometry_visible = 0
    alpha_transparent = 0
    compiled_visible = 0
    undefined_visible = 0
    mismatch_pixels = 0
    max_mismatch_fraction = 0.0
    frame_count = 0
    outputs = []

    for clip in projection.clips:
        for frame_index in range(clip.frame_count):
            for view in projection.views:
                rgba_path, prov_path, owner_path, _stdout = _run_native(
                    player=player,
                    package=archive,
                    clip_id=clip.clip_id,
                    view_id=view.view_id,
                    frame_index=frame_index,
                    root=root / "frames",
                )
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
                visible = owner >= 0
                visible_count = int(np.count_nonzero(visible))
                geometry_visible += visible_count
                alpha_transparent += int(
                    np.count_nonzero(visible & (rgba[:, :, 3] == 0))
                )
                compiled_mask = visible & (
                    (prov == int(CAA_PROVENANCE["COMPILED_NEAREST_SURFACE"]))
                    | (prov == int(CAA_PROVENANCE["COMPILED_GLOBAL_SURFACE"]))
                )
                compiled_visible += int(np.count_nonzero(compiled_mask))
                undefined_visible += int(np.count_nonzero(visible & (prov == 255)))

                reference = _reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view,
                    frame_index=frame_index,
                )
                mismatch = np.any(rgba != reference.straight_rgba_u8, axis=2)
                mismatch_count = int(np.count_nonzero(mismatch))
                mismatch_pixels += mismatch_count
                frame_pixels = resolution * resolution
                max_mismatch_fraction = max(
                    max_mismatch_fraction,
                    float(mismatch_count) / float(frame_pixels),
                )
                frame_count += 1

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
    passed = (
        geometry_visible > 0
        and undefined_visible == 0
        and mismatch_pixels == 0
        and exposure_fraction <= exposure_budget
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
        native_reference_mismatch_pixel_count=mismatch_pixels,
        maximum_frame_native_reference_mismatch_fraction=max_mismatch_fraction,
        qualification_report={
            "status": "PASS_DYNAMIC_VISUAL_INTEGRITY" if passed else "FAIL_DYNAMIC_VISUAL_INTEGRITY",
            "undefined_visible_pixel_count": undefined_visible,
            "compiled_unobserved_exposure_budget": exposure_budget,
            "compiled_unobserved_exposure_passed": exposure_fraction
            <= exposure_budget,
            "native_reference_byte_parity_passed": mismatch_pixels == 0,
            "alpha_transparency_is_diagnostic_not_undefinedness": True,
            "renderer": "REALSAS_V2_CAA_CANONICAL_DEPTH",
        },
        visual_integrity_hash="",
        metadata={
            "geometry_visibility_appearance_sampling_attribution": True,
            "final_alpha_transparent_fraction_diagnostic": transparent_fraction,
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
            "native_reference_mismatch_pixel_count": 0,
            "undefined_visible_pixel_count": 0,
            "transparent_visible_fraction_diagnostic": transparent_fraction,
        },
    }
