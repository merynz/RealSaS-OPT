from __future__ import annotations

"""Seal conservative SAME_VIEW source appearance authority for TEST_SUBJECT_001.

Geometry authority remains the complete canonical dense zero-surface. This R2
fixture only decides which full-surface faces have direct same-view source evidence
at rest. It never deletes geometry and never fills unseen appearance.

Frozen v1 admission rule, chosen before the Mage result is inspected:
- the face must be a full-surface Z-buffer first hit for at least one rest pixel;
- every pixel where that face is first hit must be source BODY (owner == 0);
- every such pixel must have source alpha >= 8;
- otherwise the entire face is UNSEEN for that view.

No percentage threshold, donor search, completion, face-selection recall target, or
motion result participates in this authority decision.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.types import QualificationError
from experiments.playback_stack_v1.materialize_test_subject_001_v1 import (
    APPEARANCE_AUTHORITY_SCHEMA,
    APPEARANCE_POLICY,
    CAMERA_SHA256,
    EXPECTED_ZERO_SURFACE_SHA256,
    OBSERVATION_SHA256,
    VIEW_IDS,
)


SCHEMA = "RealSaS.TestSubject001SameViewAppearanceAuthorityBuilder.v1"
EXPECTED_OWNER_MANIFEST_SHA256 = "4df8d42a273840021624c854bd86414e148f6a5861730e3c3d53bade039d9167"
EXPECTED_OWNER_MANIFEST_STATUS = "PASS__EXACT_SOURCE_FIT_TARGET_COMPONENT_OWNER_RASTER_V0_V7"
EXPECTED_NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
SOURCE_ALPHA_THRESHOLD = 8
DIRECT_RULE = "FIRST_HIT_PIXELS_NONEMPTY_AND_ALL_OWNER_BODY0_AND_ALL_ALPHA_GE_8"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
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


def _load_zero_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if _sha(path) != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("R2_APPEARANCE_ZERO_SURFACE_SHA_DRIFT")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(f"R2_APPEARANCE_ZERO_SURFACE_PAYLOAD_DRIFT:{sorted(z.files)}")
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("R2_APPEARANCE_ZERO_SURFACE_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("R2_APPEARANCE_ZERO_SURFACE_FACES_INVALID")
    return vertices, faces


def _load_cameras(paths: tuple[Path, ...]) -> dict[str, dict]:
    if len(paths) != 8:
        raise RuntimeError("R2_APPEARANCE_REQUIRES_8_CAMERAS")
    out = {}
    for view_index, path in enumerate(paths):
        if _sha(path) != CAMERA_SHA256[view_index]:
            raise RuntimeError(f"R2_APPEARANCE_CAMERA_SHA_DRIFT_V{view_index}")
        row = _load_json(path)
        if int(row.get("view_index", -1)) != view_index or int(row.get("resolution", 0)) != 1024:
            raise RuntimeError(f"R2_APPEARANCE_CAMERA_CONTRACT_DRIFT_V{view_index}")
        out[f"V{view_index}"] = row
    return out


def _load_source_alpha(paths: tuple[Path, ...]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if len(paths) != 8:
        raise RuntimeError("R2_APPEARANCE_REQUIRES_8_SOURCE_TEXTURES")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - real fixture execution dependency.
        raise RuntimeError("R2_APPEARANCE_PIL_REQUIRED_FOR_SOURCE_ALPHA") from exc
    masks, hashes = {}, {}
    for view_index, path in enumerate(paths):
        digest = _sha(path)
        if digest != OBSERVATION_SHA256[view_index]:
            raise RuntimeError(f"R2_APPEARANCE_SOURCE_TEXTURE_SHA_DRIFT_V{view_index}")
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        if rgba.shape != (1024, 1024, 4):
            raise RuntimeError(f"R2_APPEARANCE_SOURCE_TEXTURE_SHAPE_DRIFT_V{view_index}:{rgba.shape}")
        view_id = f"V{view_index}"
        masks[view_id] = np.ascontiguousarray(rgba[..., 3] >= SOURCE_ALPHA_THRESHOLD)
        hashes[view_id] = digest
    return masks, hashes


def _load_owner_rasters(manifest_path: Path, paths: tuple[Path, ...]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if _sha(manifest_path) != EXPECTED_OWNER_MANIFEST_SHA256:
        raise RuntimeError("R2_APPEARANCE_OWNER_MANIFEST_SHA_DRIFT")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != EXPECTED_OWNER_MANIFEST_STATUS:
        raise RuntimeError("R2_APPEARANCE_OWNER_MANIFEST_STATUS_DRIFT")
    artifacts = {str(row["path"]): str(row["sha256"]) for row in (manifest.get("artifacts") or ())}
    if len(paths) != 8:
        raise RuntimeError("R2_APPEARANCE_REQUIRES_8_OWNER_RASTERS")
    owners, hashes = {}, {}
    for view_index, path in enumerate(paths):
        expected_name = f"V{view_index}_SOURCE_COMPONENT_OWNER_RASTER.npz"
        digest = _sha(path)
        if path.name != expected_name or artifacts.get(expected_name) != digest:
            raise RuntimeError(f"R2_APPEARANCE_OWNER_RASTER_SHA_DRIFT_V{view_index}")
        with np.load(path, allow_pickle=False) as z:
            if "owner" not in z.files:
                raise RuntimeError(f"R2_APPEARANCE_OWNER_RASTER_PAYLOAD_DRIFT_V{view_index}")
            owner = np.asarray(z["owner"], dtype=np.int16)
        if owner.shape != (1024, 1024):
            raise RuntimeError(f"R2_APPEARANCE_OWNER_RASTER_SHAPE_DRIFT_V{view_index}:{owner.shape}")
        view_id = f"V{view_index}"
        owners[view_id] = np.ascontiguousarray(owner)
        hashes[view_id] = digest
    return owners, hashes


def classify_direct_source_faces_v1(
    first_hit_face_ids,
    owner,
    alpha_mask,
    *,
    face_count: int,
) -> tuple[tuple[int, ...], dict]:
    """Pure conservative face classifier used by the sealed R2 builder."""

    ids = np.asarray(first_hit_face_ids, dtype=np.int64)
    own = np.asarray(owner, dtype=np.int16)
    alpha = np.asarray(alpha_mask, dtype=np.bool_)
    if ids.shape != own.shape or ids.shape != alpha.shape or ids.ndim != 2:
        raise QualificationError("R2_APPEARANCE_CLASSIFIER_RASTER_SHAPE_MISMATCH")
    face_count = int(face_count)
    if face_count <= 0:
        raise QualificationError("R2_APPEARANCE_CLASSIFIER_FACE_COUNT_INVALID")
    if np.any(ids < -1) or np.any(ids >= face_count):
        raise QualificationError("R2_APPEARANCE_CLASSIFIER_FACE_ID_OUT_OF_RANGE")

    visible = ids >= 0
    visible_ids = ids[visible]
    hit_count = np.bincount(visible_ids, minlength=face_count).astype(np.int64)
    bad_pixel = visible & ((own != 0) | (~alpha))
    bad_count = np.bincount(ids[bad_pixel], minlength=face_count).astype(np.int64)
    direct_mask = (hit_count > 0) & (bad_count == 0)
    direct = tuple(map(int, np.flatnonzero(direct_mask).tolist()))

    body_fail_pixel = visible & (own != 0)
    alpha_fail_pixel = visible & (~alpha)
    stats = {
        "first_hit_pixel_count": int(np.count_nonzero(visible)),
        "visible_face_count": int(np.count_nonzero(hit_count > 0)),
        "direct_source_face_count": len(direct),
        "rejected_visible_face_count": int(np.count_nonzero((hit_count > 0) & ~direct_mask)),
        "first_hit_non_body_pixel_count": int(np.count_nonzero(body_fail_pixel)),
        "first_hit_alpha_fail_pixel_count": int(np.count_nonzero(alpha_fail_pixel)),
        "direct_rule": DIRECT_RULE,
        "source_alpha_threshold": SOURCE_ALPHA_THRESHOLD,
    }
    return direct, stats


def _inside_triangle_mask_runtime(points_xy: np.ndarray, resolution: int) -> np.ndarray:
    """Synthetic calibration helper; avoids edge-equality cases by construction."""

    tri = np.asarray(points_xy, dtype=np.float64)
    if tri.shape != (3, 2):
        raise ValueError("tri must be 3x2")
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    px = xx.astype(np.float64) + 0.5
    py = yy.astype(np.float64) + 0.5
    a, b, c = tri
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(area) <= 1.0e-12:
        raise ValueError("degenerate calibration triangle")
    e0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
    e1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
    e2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
    if area > 0:
        return (e0 > 0.0) & (e1 > 0.0) & (e2 > 0.0)
    return (e0 < 0.0) & (e1 < 0.0) & (e2 < 0.0)


def _projected_xyz_to_clip(projected_xyz: np.ndarray, *, resolution: int) -> np.ndarray:
    xyz = np.asarray(projected_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError("R2_APPEARANCE_PROJECTED_XYZ_INVALID")
    resolution = int(resolution)
    if resolution <= 1:
        raise QualificationError("R2_APPEARANCE_RESOLUTION_INVALID")
    zmin, zmax = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    span = max(zmax - zmin, 1.0e-9)
    x_ndc = 2.0 * xyz[:, 0] / float(resolution) - 1.0
    y_ndc = 1.0 - 2.0 * xyz[:, 1] / float(resolution)
    z_ndc = -0.9 + 1.8 * (xyz[:, 2] - zmin) / span
    return np.ascontiguousarray(np.stack((x_ndc, y_ndc, z_ndc, np.ones(len(xyz))), axis=1), dtype=np.float32)


def _nvdiffrast_raw_face_ids(projected_xyz, faces, *, resolution: int, ctx, torch, dr) -> np.ndarray:
    clip = _projected_xyz_to_clip(np.asarray(projected_xyz), resolution=resolution)
    tri = np.asarray(faces, dtype=np.int32)
    if tri.ndim != 2 or tri.shape[1] != 3 or np.any(tri < 0) or np.any(tri >= len(clip)):
        raise QualificationError("R2_APPEARANCE_RASTER_FACES_INVALID")
    device = torch.device("cuda")
    pos_t = torch.as_tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
    tri_t = torch.as_tensor(np.ascontiguousarray(tri), dtype=torch.int32, device=device)
    rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[int(resolution), int(resolution)])
    return rast[0, :, :, 3].to(torch.int64).cpu().numpy() - 1


def calibrate_nvdiffrast_runtime_mapping_v1(*, ctx, torch, dr) -> dict:
    """Synthetic orientation + near-depth calibration independent of subject data."""

    resolution = 8
    tri_xy = np.asarray([[1.2, 1.3], [6.4, 1.8], [2.1, 6.3]], dtype=np.float64)
    expected = _inside_triangle_mask_runtime(tri_xy, resolution)
    xyz = np.concatenate((tri_xy, np.ones((3, 1), dtype=np.float64)), axis=1)
    raw = _nvdiffrast_raw_face_ids(xyz, np.asarray([[0, 1, 2]], dtype=np.int32), resolution=resolution, ctx=ctx, torch=torch, dr=dr)
    raw_mask = raw >= 0
    raw_ok = np.array_equal(raw_mask, expected)
    flipped_ok = np.array_equal(np.flipud(raw_mask), expected)
    if raw_ok == flipped_ok:
        raise RuntimeError(f"R2_APPEARANCE_NVDIFFRAST_ORIENTATION_CALIBRATION_FAIL:raw={raw_ok}:flip={flipped_ok}")
    vertical_flip = bool(flipped_ok)

    near = np.concatenate((tri_xy, np.ones((3, 1), dtype=np.float64)), axis=1)
    far = np.concatenate((tri_xy, np.full((3, 1), 2.0, dtype=np.float64)), axis=1)
    stacked = np.concatenate((far, near), axis=0)
    overlap_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    depth_ids = _nvdiffrast_raw_face_ids(stacked, overlap_faces, resolution=resolution, ctx=ctx, torch=torch, dr=dr)
    if vertical_flip:
        depth_ids = np.flipud(depth_ids)
    if not np.all(depth_ids[expected] == 1):
        raise RuntimeError("R2_APPEARANCE_NVDIFFRAST_DEPTH_DIRECTION_CALIBRATION_FAIL")
    return {
        "status": "PASS",
        "vertical_flip": vertical_flip,
        "synthetic_resolution": resolution,
        "nearer_runtime_z_wins": True,
    }


def render_first_hit_face_ids_v1(vertices, faces, cameras: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict]:
    try:
        import torch
        import nvdiffrast.torch as dr
    except ImportError as exc:  # pragma: no cover - real GPU execution dependency.
        raise RuntimeError("R2_APPEARANCE_TORCH_NVDIFFRAST_REQUIRED") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("R2_APPEARANCE_CUDA_REQUIRED")
    ctx = dr.RasterizeCudaContext(device=torch.device("cuda"))
    calibration = calibrate_nvdiffrast_runtime_mapping_v1(ctx=ctx, torch=torch, dr=dr)
    out = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        xyz = project_points_xyz_v3(vertices, camera)
        ids = _nvdiffrast_raw_face_ids(xyz, faces, resolution=camera.resolution, ctx=ctx, torch=torch, dr=dr)
        if calibration["vertical_flip"]:
            ids = np.flipud(ids)
        out[view_id] = np.ascontiguousarray(ids, dtype=np.int64)
    return out, calibration


def build_authority(args) -> dict:
    zero_path = Path(args.zero_surface).resolve()
    camera_paths = tuple(Path(x).resolve() for x in args.cameras)
    source_paths = tuple(Path(x).resolve() for x in args.source_textures)
    owner_manifest = Path(args.owner_manifest).resolve()
    owner_paths = tuple(Path(x).resolve() for x in args.owner_rasters)
    out_path = Path(args.out).resolve()
    if str(args.nvdiffrast_commit) != EXPECTED_NVDIFFRAST_COMMIT:
        raise RuntimeError("R2_APPEARANCE_NVDIFFRAST_PIN_DRIFT")

    vertices, faces = _load_zero_surface(zero_path)
    cameras = _load_cameras(camera_paths)
    alpha_masks, texture_hashes = _load_source_alpha(source_paths)
    owners, owner_hashes = _load_owner_rasters(owner_manifest, owner_paths)
    first_hit_by_view, calibration = render_first_hit_face_ids_v1(vertices, faces, cameras)

    views = {}
    for view_id in VIEW_IDS:
        direct, stats = classify_direct_source_faces_v1(
            first_hit_by_view[view_id],
            owners[view_id],
            alpha_masks[view_id],
            face_count=len(faces),
        )
        if not direct:
            raise RuntimeError(f"R2_APPEARANCE_DIRECT_SOURCE_EMPTY:{view_id}")
        views[view_id] = {
            "source_texture_sha256": texture_hashes[view_id],
            "owner_raster_sha256": owner_hashes[view_id],
            "direct_source_face_indices": list(direct),
            "stats": stats,
        }

    payload = {
        "schema": APPEARANCE_AUTHORITY_SCHEMA,
        "status": "PASS",
        "policy": APPEARANCE_POLICY,
        "builder_schema": SCHEMA,
        "direct_rule": DIRECT_RULE,
        "source_alpha_threshold": SOURCE_ALPHA_THRESHOLD,
        "source_zero_surface_sha256": EXPECTED_ZERO_SURFACE_SHA256,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "owner_manifest_sha256": _sha(owner_manifest),
        "nvdiffrast_commit": EXPECTED_NVDIFFRAST_COMMIT,
        "rasterizer_calibration": calibration,
        "geometry_policy": "FULL_ZERO_SURFACE_PRESERVED__FIRST_HIT_USED_ONLY_FOR_APPEARANCE_PROVENANCE",
        "completion_used": False,
        "other_view_donor_used": False,
        "views": views,
    }
    _write_json(out_path, payload)
    return {**payload, "out_path": str(out_path), "out_sha256": _sha(out_path)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--source-textures", nargs=8, required=True)
    p.add_argument("--owner-manifest", required=True)
    p.add_argument("--owner-rasters", nargs=8, required=True)
    p.add_argument("--nvdiffrast-commit", default=EXPECTED_NVDIFFRAST_COMMIT)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    report = build_authority(_parser().parse_args())
    print("TEST_SUBJECT_001_SAME_VIEW_APPEARANCE_AUTHORITY_PASS")
    print(json.dumps({
        "out_path": report["out_path"],
        "out_sha256": report["out_sha256"],
        "face_count": report["face_count"],
        "policy": report["policy"],
        "direct_face_counts": {
            view_id: report["views"][view_id]["stats"]["direct_source_face_count"]
            for view_id in VIEW_IDS
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
