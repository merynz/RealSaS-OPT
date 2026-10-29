#!/usr/bin/env python3
"""Build canonical Truth139 dense Graphics front-surface truth.

This is the Graphics-domain successor to the legacy dense-appearance G-buffer producer.  It keeps
source color as one evidence family and adds the geometry/provenance fields needed by the shared
IRIS scene: exact source/canonical surface address, geometric orientation, view-facing confidence,
contour coverage and explicit observation provenance.

No semantic body-part/material/owner label is exposed as a student target.  Privileged face metadata
is preserved only in a separate teacher file.  SOURCE_OBSERVATION is direct source authority;
CONSTRUCTED_OBSERVATION and LEARNED_FILL are never allowed to mint direct source RGBA authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus import build_truth139_dense_appearance_gbuffer_v1 as legacy
from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot

SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
VIEW_SCHEMA = "RealSaS.Truth139DenseGraphicsView.v1"
GRAPHICS_CONTRACT = "canonical/TRUTH139_DENSE_GRAPHICS_TEACHER_V1.json"
PROVENANCE_CODES = {
    "SOURCE_OBSERVATION": 1,
    "CONSTRUCTED_OBSERVATION": 2,
    "LEARNED_FILL": 3,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("TRUTH139_DENSE_GRAPHICS_MATRIX:" + label)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0):
        raise ValueError("TRUTH139_DENSE_GRAPHICS_AFFINE_MATRIX:" + label)
    if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_SINGULAR_MATRIX:" + label)
    return matrix


def transform_points(points: np.ndarray, source_world_to_canonical: np.ndarray) -> np.ndarray:
    """Transform source-world points with the imported-scene affine map."""
    value = np.asarray(points, dtype=np.float64)
    matrix = _matrix(source_world_to_canonical, "source_world_to_canonical")
    if value.ndim != 2 or value.shape[1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("TRUTH139_DENSE_GRAPHICS_POINT_LAYOUT")
    homogeneous = np.concatenate((value, np.ones((len(value), 1), dtype=np.float64)), axis=1)
    out = homogeneous @ matrix.T
    w = out[:, 3]
    if not np.allclose(w, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("TRUTH139_DENSE_GRAPHICS_AFFINE_POINT_W")
    return out[:, :3].astype(np.float32)


def transform_normals(normals: np.ndarray, source_world_to_canonical: np.ndarray) -> np.ndarray:
    """Transform normals by inverse-transpose of the source->canonical linear map."""
    value = np.asarray(normals, dtype=np.float64)
    matrix = _matrix(source_world_to_canonical, "source_world_to_canonical")
    if value.ndim != 2 or value.shape[1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("TRUTH139_DENSE_GRAPHICS_NORMAL_LAYOUT")
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    out = (normal_matrix @ value.T).T
    length = np.linalg.norm(out, axis=1)
    valid = length > 1.0e-12
    normalized = np.zeros_like(out)
    normalized[valid] = out[valid] / length[valid, None]
    return normalized.astype(np.float32)


def reconstruct_surface_points(vertices: np.ndarray, faces: np.ndarray, triangle_id: np.ndarray, barycentric: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri_id = np.asarray(triangle_id, dtype=np.int64)
    bc = np.asarray(barycentric, dtype=np.float64)
    if tri_id.ndim != 1 or bc.shape != (len(tri_id), 3):
        raise ValueError("TRUTH139_DENSE_GRAPHICS_BARY_LAYOUT")
    if np.any(tri_id < 0) or np.any(tri_id >= len(f)) or not np.isfinite(bc).all():
        raise ValueError("TRUTH139_DENSE_GRAPHICS_BARY_ADDRESS")
    if np.max(np.abs(bc.sum(axis=1) - 1.0), initial=0.0) > 2.0e-4:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_BARY_SUM")
    return np.einsum("ni,nij->nj", bc, v[f[tri_id]]).astype(np.float32)


def view_facing(face_normals_source_world: np.ndarray, triangle_id: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(face_normals_source_world, dtype=np.float64)[np.asarray(triangle_id, dtype=np.int64)]
    forward = np.asarray(camera["forward"], dtype=np.float64)
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("TRUTH139_DENSE_GRAPHICS_CAMERA_FORWARD")
    norm = np.linalg.norm(forward)
    if norm <= 1.0e-12:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_CAMERA_FORWARD_ZERO")
    # camera.forward is camera -> scene; surface -> camera is therefore -forward.
    view_direction = -forward / norm
    signed = np.clip(normals @ view_direction, -1.0, 1.0)
    # Confidence is orientation confidence, not front/back classification.  Grazing incidence is
    # low confidence on either winding side; signed cosine is retained separately.
    confidence = np.abs(signed)
    return signed.astype(np.float32), confidence.astype(np.float32)


def provenance_direct_source_authority(provenance: str) -> bool:
    if provenance not in PROVENANCE_CODES:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_PROVENANCE:" + str(provenance))
    return provenance == "SOURCE_OBSERVATION"


def _load_scene_snapshot(path: Path) -> tuple[dict, np.ndarray]:
    snapshot = json.loads(path.read_text())
    validate_imported_scene_snapshot(snapshot)
    matrix = _matrix(snapshot["source_world_to_canonical"], "source_world_to_canonical")
    return snapshot, matrix


def _erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius < 0:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_EROSION_RADIUS")
    value = np.asarray(mask, dtype=bool)
    if radius == 0:
        return value.copy()
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(value.astype(np.float32))[None, None]
    # min-pool via complement max-pool.
    inverse = 1.0 - t
    dilated_inverse = F.max_pool2d(inverse, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (1.0 - dilated_inverse[0, 0].numpy()) > 0.5


def _write_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(
    *,
    subject_id: str,
    geometry_npz: Path,
    source_scene_snapshot: Path,
    camera_bundle: Path,
    observation_dir: Path,
    output_dir: Path,
    device: str,
    mask_alignment_radius_px: int,
    observation_provenance: str,
) -> dict:
    geometry_npz = geometry_npz.resolve()
    source_scene_snapshot = source_scene_snapshot.resolve()
    camera_bundle = camera_bundle.resolve()
    observation_dir = observation_dir.resolve()
    output_dir = output_dir.resolve()
    for path in (geometry_npz, source_scene_snapshot, camera_bundle):
        if not path.is_file():
            raise FileNotFoundError("TRUTH139_DENSE_GRAPHICS_INPUT_MISSING:" + str(path))
    if not observation_dir.is_dir():
        raise FileNotFoundError("TRUTH139_DENSE_GRAPHICS_OBSERVATION_DIR_MISSING")
    if not subject_id:
        raise ValueError("TRUTH139_DENSE_GRAPHICS_SUBJECT_ID")
    direct_authority = provenance_direct_source_authority(observation_provenance)
    provenance_code = PROVENANCE_CODES[observation_provenance]

    vertices_source, faces, teacher = legacy._load_geometry(geometry_npz)
    cameras = legacy._load_camera_bundle(camera_bundle)
    _snapshot, source_to_canonical = _load_scene_snapshot(source_scene_snapshot)
    vertices_canonical = transform_points(vertices_source, source_to_canonical)
    normals_source = legacy.geometric_face_normals(vertices_source, faces)
    normals_canonical = transform_normals(normals_source, source_to_canonical)

    output_dir.mkdir(parents=True, exist_ok=True)
    face_meta = output_dir / "FACE_TEACHER_METADATA.npz"
    _write_npz(
        face_meta,
        face_normal_source_world=normals_source,
        face_normal_canonical=normals_canonical,
        triangle_uv=teacher["triangle_uv"],
        triangle_uv_valid=teacher["triangle_uv_valid"],
        face_owner_privileged=teacher["face_owner_privileged"],
        face_material_privileged=teacher["face_material_privileged"],
    )

    views: list[dict] = []
    totals = {
        "source_foreground_pixels": 0,
        "geometry_covered_pixels": 0,
        "direct_source_rgba_pixels": 0,
        "alpha_boundary_pixels": 0,
        "silhouette_boundary_pixels": 0,
    }
    for index, camera in enumerate(cameras):
        png = observation_dir / f"V{index}.png"
        mask_path = observation_dir / f"V{index}.mask.bin"
        if not png.is_file() or not mask_path.is_file():
            raise FileNotFoundError("TRUTH139_DENSE_GRAPHICS_VIEW_INPUT_MISSING:" + str(index))
        rgba_u8 = legacy._read_observation_rgba(png)
        observation_mask = legacy._read_mask(mask_path)
        face_image, depth_image, bary_image = legacy._rasterize_view(vertices_source, faces, camera, device=device)
        geometry_mask = face_image >= 0
        geometry_dilated = legacy._dilate(geometry_mask, int(mask_alignment_radius_px))
        observation_without_geometry = observation_mask & ~geometry_dilated
        if direct_authority and np.any(observation_without_geometry):
            raise ValueError(
                f"TRUTH139_DENSE_GRAPHICS_SOURCE_WITHOUT_GEOMETRY:V{index}:"
                f"{int(np.count_nonzero(observation_without_geometry))}"
            )

        alpha_u8 = rgba_u8[..., 3]
        direct_source = direct_authority & observation_mask & geometry_mask & (alpha_u8 == 255)
        alpha_boundary = observation_mask & geometry_mask & (alpha_u8 > 0) & (alpha_u8 < 255)
        silhouette_boundary_mask = geometry_mask & ~_erode(geometry_mask, radius=1)

        covered = np.flatnonzero(geometry_mask.reshape(-1))
        ys = covered // legacy.EXPECTED_RESOLUTION
        xs = covered % legacy.EXPECTED_RESOLUTION
        tri = face_image[ys, xs].astype(np.int32)
        bc = bary_image[ys, xs].astype(np.float32)
        if np.any(tri < 0) or np.any(tri >= len(faces)) or not np.isfinite(bc).all():
            raise ValueError("TRUTH139_DENSE_GRAPHICS_RASTER_ADDRESS_INVALID")

        surface_source = reconstruct_surface_points(vertices_source, faces, tri, bc)
        surface_canonical = reconstruct_surface_points(vertices_canonical, faces, tri, bc)
        normal_source = normals_source[tri]
        normal_canonical = normals_canonical[tri]
        signed_cosine, facing_confidence = view_facing(normals_source, tri, camera)

        rgb_srgb = rgba_u8[ys, xs, :3].astype(np.float32) / 255.0
        source_rgba_linear = np.concatenate(
            (legacy.srgb_to_linear(rgb_srgb), rgba_u8[ys, xs, 3:4].astype(np.float32) / 255.0), axis=1
        )
        source_foreground = observation_mask[ys, xs]
        direct_flat = np.asarray(direct_source[ys, xs], dtype=np.bool_)
        boundary_flat = np.asarray(alpha_boundary[ys, xs], dtype=np.bool_)
        silhouette_flat = np.asarray(silhouette_boundary_mask[ys, xs], dtype=np.bool_)
        source_alpha = rgba_u8[ys, xs, 3].astype(np.float32) / 255.0
        provenance = np.full(len(covered), provenance_code, dtype=np.uint8)

        # A constructed/learned view can still provide proposal pixels, but it cannot assert source
        # color authority.  This invariant is encoded in the dense rows, not only the manifest.
        if not direct_authority and np.any(direct_flat):
            raise AssertionError("TRUTH139_DENSE_GRAPHICS_CONSTRUCTED_SOURCE_AUTHORITY")

        out = output_dir / f"V{index}_DENSE_GRAPHICS_GBUFFER.npz"
        _write_npz(
            out,
            pixel_linear_index=covered.astype(np.int32),
            triangle_id=tri,
            barycentric_uv=bc[:, :2].astype(np.float32),
            surface_xyz_source_world=surface_source,
            surface_xyz_canonical=surface_canonical,
            depth_source_camera=depth_image[ys, xs].astype(np.float32),
            normal_source_world=normal_source.astype(np.float32),
            normal_canonical=normal_canonical.astype(np.float32),
            front_facing_signed_cosine=signed_cosine,
            view_facing_confidence=facing_confidence,
            source_rgba_linear=source_rgba_linear.astype(np.float32),
            source_alpha=source_alpha.astype(np.float32),
            source_foreground=source_foreground.astype(np.bool_),
            direct_source_rgba_valid=direct_flat,
            alpha_boundary=boundary_flat,
            silhouette_boundary_band=silhouette_flat,
            edge_coverage=source_alpha.astype(np.float32),
            observation_provenance=provenance,
            resolution=np.asarray([legacy.EXPECTED_RESOLUTION, legacy.EXPECTED_RESOLUTION], dtype=np.int32),
            origin=np.asarray(["TOP_LEFT"]),
        )

        row = {
            "schema": VIEW_SCHEMA,
            "view_index": index,
            "gbuffer_path": str(out),
            "gbuffer_sha256": sha256(out),
            "source_rgba_path": str(png),
            "source_rgba_sha256": sha256(png),
            "source_mask_path": str(mask_path),
            "source_mask_sha256": sha256(mask_path),
            "observation_provenance": observation_provenance,
            "direct_source_authority": direct_authority,
            "geometry_covered_pixel_count": int(len(covered)),
            "source_foreground_pixel_count": int(np.count_nonzero(observation_mask)),
            "direct_source_rgba_count": int(np.count_nonzero(direct_flat)),
            "alpha_boundary_count": int(np.count_nonzero(boundary_flat)),
            "silhouette_boundary_count": int(np.count_nonzero(silhouette_flat)),
            "source_foreground_without_geometry_after_dilation": int(np.count_nonzero(observation_without_geometry)),
            "mask_alignment_radius_px": int(mask_alignment_radius_px),
        }
        views.append(row)
        totals["source_foreground_pixels"] += row["source_foreground_pixel_count"]
        totals["geometry_covered_pixels"] += row["geometry_covered_pixel_count"]
        totals["direct_source_rgba_pixels"] += row["direct_source_rgba_count"]
        totals["alpha_boundary_pixels"] += row["alpha_boundary_count"]
        totals["silhouette_boundary_pixels"] += row["silhouette_boundary_count"]
        print(
            f"[DENSE_GRAPHICS] {subject_id} V{index} geometry={row['geometry_covered_pixel_count']} "
            f"source_fg={row['source_foreground_pixel_count']} direct_source={row['direct_source_rgba_count']} "
            f"provenance={observation_provenance}",
            flush=True,
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING",
        "subject_id": subject_id,
        "graphics_contract": GRAPHICS_CONTRACT,
        "geometry_npz": {"path": str(geometry_npz), "sha256": sha256(geometry_npz)},
        "source_scene_snapshot": {"path": str(source_scene_snapshot), "sha256": sha256(source_scene_snapshot)},
        "camera_bundle": {"path": str(camera_bundle), "sha256": sha256(camera_bundle)},
        "face_teacher_metadata": {"path": str(face_meta), "sha256": sha256(face_meta)},
        "observation_provenance": {
            "codebook": {str(code): name for name, code in PROVENANCE_CODES.items()},
            "view_value": observation_provenance,
            "direct_source_authority": direct_authority,
        },
        "coordinate_contract": {
            "source_world_to_canonical": source_to_canonical.tolist(),
            "point_transform": "AFFINE_HOMOGENEOUS",
            "normal_transform": "INVERSE_TRANSPOSE_THEN_NORMALIZE",
        },
        "rasterizer": {
            "implementation": "PYTORCH3D_RASTERIZE_MESHES",
            "required_git_commit": legacy.PYTORCH3D_PIN,
            "license": "BSD",
            "faces_per_pixel": 1,
            "perspective_correct": False,
            "student_semantic_targets": False,
        },
        "views": views,
        "counts": {"view_count": len(views), **totals},
        "teacher_only_fields": [
            "face_owner_privileged",
            "face_material_privileged",
            "triangle_uv",
            "triangle_uv_valid",
        ],
        "semantic_part_taxonomy_used": False,
        "visibility_occlusion_teacher_pending": True,
        "independent_front_surface_geometry_court_pending": True,
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["graphics_identity_sha256"] = digest_json(payload)
    manifest = output_dir / "DENSE_GRAPHICS_GBUFFER_MANIFEST.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "subject_id": subject_id,
        "direct_source_rgba_pixels": totals["direct_source_rgba_pixels"],
        "graphics_identity_sha256": payload["graphics_identity_sha256"],
    }, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--source-scene-snapshot", required=True, type=Path)
    parser.add_argument("--camera-bundle", required=True, type=Path)
    parser.add_argument("--observation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-alignment-radius-px", type=int, default=2)
    parser.add_argument(
        "--observation-provenance",
        choices=sorted(PROVENANCE_CODES),
        default="SOURCE_OBSERVATION",
        help="Truth139 controlled 8-view uses SOURCE_OBSERVATION. Future internal views must declare CONSTRUCTED_OBSERVATION.",
    )
    args = parser.parse_args()
    build(
        subject_id=args.subject_id,
        geometry_npz=args.geometry_npz,
        source_scene_snapshot=args.source_scene_snapshot,
        camera_bundle=args.camera_bundle,
        observation_dir=args.observation_dir,
        output_dir=args.output_dir,
        device=args.device,
        mask_alignment_radius_px=args.mask_alignment_radius_px,
        observation_provenance=args.observation_provenance,
    )


if __name__ == "__main__":
    main()
