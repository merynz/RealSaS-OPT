from __future__ import annotations

"""Materialize TEST_SUBJECT_001 BODY-only Runtime-v3 package on the generic stack.

This is a subject fixture, not product-core policy. It wires the sealed Mage FIT2
mechanical authorities into the generic D1/full-surface Runtime-v3 path.

Strict first-smoke scope:
- complete dense zero-surface topology; no face subset/coverage selector;
- current FIT2 skeleton + skin, replayed exactly onto dense vertices;
- phase-authored Mage motion reinterpreted through an explicit D0 3D axis contract;
- same-view source appearance only; every non-qualified face is UNSEEN;
- no appearance completion and no rigid-component restoration;
- runtime_qualified remains false until D1 semantic/scientific gates are closed.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import zlib

import numpy as np

from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import (
    replay_dense_zero_surface_compaction,
)
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import (
    compile_motion_clip_to_d1_v1,
)
from compiler.realsas_compiler_core.motion_locomotion import (
    build_mage_historical_phase_motion,
)
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    FaceAppearanceAuthorityV3,
    build_face_appearance_patches_v3,
    build_full_surface_runtime_meshes_v3,
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3PlaybackContract,
    RuntimeV3Slot,
    TopologyClass,
    validate_playback_runtime_v3_contract,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.playback_v3_bridge import (
    build_runtime_v3_clip_from_d1,
)
from compiler.realsas_compiler_services.export.runtime_v3 import (
    RuntimeV3TexturePayload,
    materialize_runtime_v3_archive,
)

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1


SCHEMA = "RealSaS.TestSubject001PlaybackV3Materialization.v1"
APPEARANCE_AUTHORITY_SCHEMA = "RealSaS.TestSubject001SameViewAppearanceAuthority.v1"
APPEARANCE_POLICY = "SAME_VIEW_DIRECT_SOURCE_ONLY__ALL_OTHER_FACES_UNSEEN"
EXPECTED_ZERO_SURFACE_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
EXPECTED_SURFACE_FILE_SHA256 = "170b8e4712fd78ef0721462f19f80ccb94eb008206fc6a7061908bfdc36f202f"
DEFAULT_PRODUCT_STATE_HASH = "6c45c8d6f82148d2874f1486f04719a204733b1a76c508918a9cd4d2691ea3a9"
DEFAULT_PROOF_BUNDLE_HASH = "ba9761fe7c6bd4d463057fc3b8582d2dd8e01f36d28c8884845911adf21cd8c1"
VIEW_IDS = tuple(f"V{i}" for i in range(8))
BODY_SLOT_ID = "BODY_FULL_SURFACE_SLOT"
BODY_ATTACHMENT_ID = "BODY_FULL_SURFACE"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii"))
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if len(raw) != 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise RuntimeError(f"TEST_SUBJECT_001_TEXTURE_NOT_PNG:{path}")
    return struct.unpack(">II", raw[16:24])


def _load_zero_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if _sha(path) != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_ZERO_SURFACE_SHA_DRIFT")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(f"TEST_SUBJECT_001_ZERO_SURFACE_PAYLOAD_DRIFT:{sorted(z.files)}")
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("TEST_SUBJECT_001_ZERO_SURFACE_VERTEX_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("TEST_SUBJECT_001_ZERO_SURFACE_FACE_INVALID")
    return vertices, faces


def _load_cameras(paths: tuple[Path, ...]) -> dict[str, dict]:
    if len(paths) != 8:
        raise RuntimeError("TEST_SUBJECT_001_REQUIRES_8_CAMERAS")
    out: dict[str, dict] = {}
    for view_index, path in enumerate(paths):
        if _sha(path) != ceiling_v1.CAMERA_SHA256[view_index]:
            raise RuntimeError(f"TEST_SUBJECT_001_CAMERA_SHA_DRIFT_V{view_index}")
        value = _load_json(path)
        if int(value.get("view_index", -1)) != view_index or int(value.get("resolution", 0)) != 1024:
            raise RuntimeError(f"TEST_SUBJECT_001_CAMERA_CONTRACT_DRIFT_V{view_index}")
        out[f"V{view_index}"] = value
    return out


def _validate_source_textures(paths: tuple[Path, ...]) -> dict[str, dict]:
    if len(paths) != 8:
        raise RuntimeError("TEST_SUBJECT_001_REQUIRES_8_SOURCE_TEXTURES")
    rows: dict[str, dict] = {}
    for view_index, path in enumerate(paths):
        digest = _sha(path)
        if digest != ceiling_v1.OBSERVATION_SHA256[view_index]:
            raise RuntimeError(f"TEST_SUBJECT_001_SOURCE_TEXTURE_SHA_DRIFT_V{view_index}")
        width, height = _png_dimensions(path)
        if (width, height) != (1024, 1024):
            raise RuntimeError(f"TEST_SUBJECT_001_SOURCE_TEXTURE_SIZE_DRIFT_V{view_index}:{width}x{height}")
        rows[f"V{view_index}"] = {
            "path": path,
            "sha256": digest,
            "width": width,
            "height": height,
        }
    return rows


def _surface_skin_matrix(surface, skin, joint_ids: tuple[str, ...]) -> np.ndarray:
    rows = {str(r.surface_id): r for r in skin.rows}
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}
    if len(joint_index) != len(joint_ids):
        raise RuntimeError("TEST_SUBJECT_001_JOINT_ORDER_DUPLICATE")
    W = np.zeros((len(surface.surface_nodes), len(joint_ids)), dtype=np.float64)
    for row_index, node in enumerate(surface.surface_nodes):
        skin_row = rows.get(str(node.surface_id))
        if skin_row is None:
            raise RuntimeError(f"TEST_SUBJECT_001_SKIN_ROW_MISSING:{node.surface_id}")
        for joint_id, weight in skin_row.influences:
            joint_id = str(joint_id)
            if joint_id not in joint_index:
                raise RuntimeError(f"TEST_SUBJECT_001_SKIN_JOINT_UNKNOWN:{joint_id}")
            W[row_index, joint_index[joint_id]] = float(weight)
    if not np.isfinite(W).all() or np.any(W < -1.0e-10):
        raise RuntimeError("TEST_SUBJECT_001_SURFACE_SKIN_INVALID")
    row_error = float(np.max(np.abs(W.sum(axis=1) - 1.0), initial=0.0))
    if row_error > 1.0e-7:
        raise RuntimeError(f"TEST_SUBJECT_001_SURFACE_SKIN_SIMPLEX_DRIFT:{row_error}")
    return W


def _load_same_view_appearance_authority(
    path: Path,
    *,
    face_count: int,
    source_texture_rows: dict[str, dict],
) -> tuple[dict[str, tuple[int, ...]], str]:
    raw = _load_json(path)
    if raw.get("schema") != APPEARANCE_AUTHORITY_SCHEMA:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_SCHEMA_DRIFT")
    if raw.get("status") != "PASS":
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_NOT_PASS")
    if raw.get("policy") != APPEARANCE_POLICY:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_POLICY_DRIFT")
    if raw.get("source_zero_surface_sha256") != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_ZERO_SURFACE_DRIFT")
    if int(raw.get("face_count", -1)) != int(face_count):
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_FACE_COUNT_DRIFT")
    views = raw.get("views")
    if not isinstance(views, dict) or set(views) != set(VIEW_IDS):
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_VIEW_SET_DRIFT")

    out: dict[str, tuple[int, ...]] = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        row = views[view_id]
        if not isinstance(row, dict):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_VIEW_ROW_INVALID:{view_id}")
        if row.get("source_texture_sha256") != source_texture_rows[view_id]["sha256"]:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_TEXTURE_SHA_DRIFT:{view_id}")
        values = tuple(int(x) for x in (row.get("direct_source_face_indices") or ()))
        if tuple(sorted(set(values))) != values:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_FACE_SET_NOT_SORTED_UNIQUE:{view_id}")
        if values and (values[0] < 0 or values[-1] >= face_count):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_FACE_OUT_OF_RANGE:{view_id}")
        if not values:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DIRECT_SOURCE_EMPTY:{view_id}")
        out[view_id] = values
    return out, _sha(path)


def source_pixel_center_uv_v1(
    projected_xyz: np.ndarray,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Runtime-v3 screen pixel-center coordinates to native bilinear UV.

    Native sampling uses x=u*(width-1), y=v*(height-1), while screen pixel centers
    are 0.5, 1.5, ... . Therefore u=(screen_x-0.5)/(width-1).
    The unclamped result is returned for source-authority validation; the clamped
    copy is legal storage for vertices that occur only in UNSEEN faces.
    """

    xyz = np.asarray(projected_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError("TEST_SUBJECT_001_PROJECTED_XYZ_INVALID")
    resolution = int(resolution)
    if resolution <= 1:
        raise QualificationError("TEST_SUBJECT_001_TEXTURE_RESOLUTION_INVALID")
    raw = np.stack(
        (
            (xyz[:, 0] - 0.5) / float(resolution - 1),
            (xyz[:, 1] - 0.5) / float(resolution - 1),
        ),
        axis=1,
    )
    return raw, np.clip(raw, 0.0, 1.0)


def _build_uv_and_face_authorities(
    dense_vertices: np.ndarray,
    dense_faces: np.ndarray,
    cameras: dict[str, dict],
    direct_faces_by_view: dict[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[FaceAppearanceAuthorityV3, ...]], dict]:
    uv_by_view: dict[str, np.ndarray] = {}
    authority_by_view: dict[str, tuple[FaceAppearanceAuthorityV3, ...]] = {}
    counts: dict[str, dict] = {}
    face_count = len(dense_faces)

    for view_index, view_id in enumerate(VIEW_IDS):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        xyz = project_points_xyz_v3(dense_vertices, camera)
        raw_uv, stored_uv = source_pixel_center_uv_v1(xyz, resolution=camera.resolution)
        direct = np.asarray(direct_faces_by_view[view_id], dtype=np.int64)
        if len(direct):
            direct_vertices = np.unique(dense_faces[direct].reshape(-1))
            uv = raw_uv[direct_vertices]
            eps = 1.0e-9
            if np.any(uv < -eps) or np.any(uv > 1.0 + eps):
                raise RuntimeError(f"TEST_SUBJECT_001_DIRECT_SOURCE_UV_OUT_OF_BOUNDS:{view_id}")
        direct_set = set(map(int, direct.tolist()))
        rows = tuple(
            FaceAppearanceAuthorityV3(
                AppearanceProvenance.DIRECT_SOURCE,
                view_index,
                f"SOURCE_TEXTURE:{view_id}",
            )
            if face_index in direct_set
            else FaceAppearanceAuthorityV3(AppearanceProvenance.UNSEEN, None, None)
            for face_index in range(face_count)
        )
        uv_by_view[view_id] = np.ascontiguousarray(stored_uv, dtype=np.float64)
        authority_by_view[view_id] = rows
        counts[view_id] = {
            "direct_source_faces": len(direct_set),
            "unseen_faces": face_count - len(direct_set),
        }
    return uv_by_view, authority_by_view, counts


def _root_joint_id(skeleton) -> str:
    roots = [str(j.canonical_joint_id) for j in skeleton.joints if j.parent_canonical_id is None]
    if len(roots) != 1:
        raise RuntimeError(f"TEST_SUBJECT_001_EXACTLY_ONE_ROOT_REQUIRED:{roots}")
    return roots[0]


def _composition(frame_count: int) -> tuple[dict[str, RuntimeV3FrameComposition], ...]:
    row = {
        view_id: RuntimeV3FrameComposition(
            view_id=view_id,
            draw_order_slot_ids=(BODY_SLOT_ID,),
            active_attachment_by_slot={BODY_SLOT_ID: BODY_ATTACHMENT_ID},
            clip_intervals=(),
        )
        for view_id in VIEW_IDS
    }
    return tuple(dict(row) for _ in range(int(frame_count)))


def _uniform_times(duration: float, sample_count: int) -> tuple[float, ...]:
    sample_count = int(sample_count)
    if sample_count < 3 or sample_count % 2 == 0:
        raise RuntimeError("TEST_SUBJECT_001_SAMPLE_COUNT_MUST_BE_ODD_AND_AT_LEAST_3")
    return tuple(map(float, np.linspace(0.0, float(duration), sample_count)))


def _stage_textures(
    source_rows: dict[str, dict],
    texture_root: Path,
) -> tuple[RuntimeV3TexturePayload, ...]:
    rows = []
    for view_id in VIEW_IDS:
        src = Path(source_rows[view_id]["path"])
        rel = Path("textures") / f"{view_id}.png"
        dst = texture_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        payload = dst.read_bytes()
        rows.append(RuntimeV3TexturePayload(
            view_id=view_id,
            texture_path=rel.as_posix(),
            texture_sha256=_sha(dst),
            texture_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            width=int(source_rows[view_id]["width"]),
            height=int(source_rows[view_id]["height"]),
        ))
    return tuple(rows)


def materialize(args) -> dict:
    zero_surface = Path(args.zero_surface).resolve()
    surface_path = Path(args.surface).resolve()
    skeleton_path = Path(args.skeleton).resolve()
    skin_path = Path(args.skin).resolve()
    axis_path = Path(args.axis_contract).resolve()
    appearance_path = Path(args.appearance_authority).resolve()
    out_path = Path(args.out).resolve()
    cameras_paths = tuple(Path(x).resolve() for x in args.cameras)
    texture_paths = tuple(Path(x).resolve() for x in args.source_textures)

    if fit2io.sha256_file(surface_path) != EXPECTED_SURFACE_FILE_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_SURFACE_FILE_SHA_DRIFT")
    dense_vertices, dense_faces = _load_zero_surface(zero_surface)
    cameras = _load_cameras(cameras_paths)
    source_texture_rows = _validate_source_textures(texture_paths)
    direct_faces_by_view, appearance_authority_sha = _load_same_view_appearance_authority(
        appearance_path,
        face_count=len(dense_faces),
        source_texture_rows=source_texture_rows,
    )
    axis_contract = _load_json(axis_path)

    surface, skeleton, skin, mechanical = fit2io.build_exact_mechanical(
        surface_path, skeleton_path, skin_path
    )
    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    replay = replay_dense_zero_surface_compaction(
        surface,
        dense_vertices,
        source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256,
    )
    surface_W = _surface_skin_matrix(surface, skin, joint_ids)
    dense_W = np.ascontiguousarray(
        surface_W[replay.dense_vertex_to_compact_index],
        dtype=np.float64,
    )

    uv_by_view, authority_by_view, appearance_counts = _build_uv_and_face_authorities(
        dense_vertices, dense_faces, cameras, direct_faces_by_view
    )
    full_surface = build_full_surface_runtime_meshes_v3(
        dense_vertices,
        dense_faces,
        cameras,
        uv_by_view,
        slot_id=BODY_SLOT_ID,
        attachment_id=BODY_ATTACHMENT_ID,
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        required_view_ids=VIEW_IDS,
    )
    appearance_patches = []
    for mesh in full_surface.meshes:
        appearance_patches.extend(build_face_appearance_patches_v3(
            mesh=mesh,
            face_authorities=authority_by_view[mesh.view_id],
        ))

    contract = RuntimeV3PlaybackContract(
        slots=(RuntimeV3Slot(
            slot_id=BODY_SLOT_ID,
            bone_id=_root_joint_id(skeleton),
            setup_order=0,
            default_attachment_id=BODY_ATTACHMENT_ID,
        ),),
        meshes=full_surface.meshes,
        appearance_patches=tuple(appearance_patches),
        allow_completion=False,
    )
    contract_hash = validate_playback_runtime_v3_contract(
        contract, required_view_ids=VIEW_IDS
    )

    authored_motion = build_mage_historical_phase_motion(mechanical)
    motion = compile_motion_quality(authored_motion, mechanical)
    runtime_clips = []
    d1_rows = []
    bridge_rows = []
    for clip_id, display_name, intent in (
        ("mage_fit1_idle_v2", "Idle D1 Full Surface", "IDLE"),
        ("mage_fit1_run_v2", "Run D1 Full Surface", "RUN"),
    ):
        source_clip = next(c for c in motion.clips if str(c.clip_id) == clip_id)
        sample_times = _uniform_times(float(source_clip.duration_sec), int(args.sample_count))
        d1 = compile_motion_clip_to_d1_v1(
            skeleton=skeleton,
            axis_contract=axis_contract,
            motion_state=motion,
            clip_id=clip_id,
            sample_times=sample_times,
        )
        if d1.joint_ids != joint_ids:
            raise RuntimeError("TEST_SUBJECT_001_D1_JOINT_ORDER_DRIFT")
        fps = float(len(d1.times) - 1) / float(d1.duration_seconds)
        runtime_clip, bridge_report = build_runtime_v3_clip_from_d1(
            d1_clip=d1,
            dense_vertices=dense_vertices,
            dense_weights=dense_W,
            weight_joint_ids=joint_ids,
            cameras=cameras,
            attachment_id=BODY_ATTACHMENT_ID,
            composition_by_frame=_composition(len(d1.times)),
            display_name=display_name,
            intent=intent,
            nominal_fps=fps,
            runtime_qualified=False,
            required_view_ids=VIEW_IDS,
        )
        runtime_clips.append(runtime_clip)
        d1_rows.append({
            "clip_id": clip_id,
            "duration_seconds": d1.duration_seconds,
            "frame_count": len(d1.times),
            "axis_contract_hash": d1.axis_contract_hash,
            "source_motion_state_hash": d1.source_motion_state_hash,
            "compile_hash": d1.compile_hash,
        })
        bridge_rows.append({
            "clip_id": clip_id,
            "bridge_hash": bridge_report.bridge_hash,
            "runtime_qualified": bridge_report.runtime_qualified,
            "max_weight_row_sum_error": bridge_report.max_weight_row_sum_error,
        })

    texture_root = out_path.parent / (out_path.stem + "_texture_staging")
    if texture_root.exists():
        shutil.rmtree(texture_root)
    textures = _stage_textures(source_texture_rows, texture_root)
    package = materialize_runtime_v3_archive(
        out_path=out_path,
        texture_root=texture_root,
        contract=contract,
        textures=textures,
        clips=tuple(runtime_clips),
        source_product_state_hash=str(args.source_product_state_hash),
        source_proof_bundle_hash=str(args.source_proof_bundle_hash),
        required_views=VIEW_IDS,
    )

    report = {
        "schema": SCHEMA,
        "subject_id": "TEST_SUBJECT_001",
        "scope": "BODY_ONLY__FULL_DENSE_ZERO_SURFACE__D1_3D_MOTION__SAME_VIEW_SOURCE_APPEARANCE",
        "status": "MATERIALIZED_DIAGNOSTIC_RUNTIME__SCIENTIFIC_AND_FOUNDER_VISUAL_PASS_NOT_CLAIMED",
        "source_product_state_hash": str(args.source_product_state_hash),
        "source_proof_bundle_hash": str(args.source_proof_bundle_hash),
        "source_proof_semantics": "LINEAGE_IDENTITY_ONLY__DOES_NOT_QUALIFY_NEW_D1_RUNTIME",
        "zero_surface_sha256": _sha(zero_surface),
        "surface_file_sha256": fit2io.sha256_file(surface_path),
        "skeleton_file_sha256": fit2io.sha256_file(skeleton_path),
        "skin_file_sha256": fit2io.sha256_file(skin_path),
        "axis_contract_file_sha256": _sha(axis_path),
        "appearance_authority_file_sha256": appearance_authority_sha,
        "dense_vertex_count": len(dense_vertices),
        "dense_face_count": len(dense_faces),
        "dense_vertex_array_sha256": _array_sha(dense_vertices),
        "dense_face_array_sha256": _array_sha(dense_faces),
        "dense_weight_array_sha256": _array_sha(dense_W),
        "compaction_replay": replay.summary(),
        "full_surface": {
            "topology_hash": full_surface.topology_hash,
            "camera_hash": full_surface.camera_hash,
            "canonical_vertex_sha256": full_surface.canonical_vertex_sha256,
            "canonical_face_sha256": full_surface.canonical_face_sha256,
            "view_count": len(full_surface.view_ids),
        },
        "appearance_policy": APPEARANCE_POLICY,
        "appearance_counts_by_view": appearance_counts,
        "completion_used": False,
        "selected_face_authority_used": False,
        "visibility_by_face_deletion_used": False,
        "rigid_components_included": False,
        "runtime_qualified": False,
        "founder_visual_pass_claimed": False,
        "d1_clips": d1_rows,
        "runtime_bridges": bridge_rows,
        "playback_contract_hash": contract_hash,
        "package": package,
    }
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--skin", required=True)
    p.add_argument("--axis-contract", required=True)
    p.add_argument("--appearance-authority", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--source-textures", nargs=8, required=True)
    p.add_argument("--sample-count", type=int, default=9)
    p.add_argument("--source-product-state-hash", default=DEFAULT_PRODUCT_STATE_HASH)
    p.add_argument("--source-proof-bundle-hash", default=DEFAULT_PROOF_BUNDLE_HASH)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = _parser().parse_args()
    report = materialize(args)
    print("TEST_SUBJECT_001_RUNTIME_V3_MATERIALIZED")
    print(json.dumps({
        "archive_path": report["package"]["archive_path"],
        "archive_sha256": report["package"]["archive_sha256"],
        "report_path": report["report_path"],
        "playback_contract_hash": report["playback_contract_hash"],
        "runtime_qualified": report["runtime_qualified"],
        "founder_visual_pass_claimed": report["founder_visual_pass_claimed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
