#!/usr/bin/env python3
"""Build dense, surface-addressed Truth139 appearance authority from admitted observations.

This producer complements Stage48; it does not replace the expensive renderer-intervention court.
Stage48 remains the source-equivalence / ambiguous multi-hit witness.  This producer supplies the
coverage Stage48 deliberately lacks by rasterizing exact admitted rest geometry into the exact
8-view camera bundle and binding every visible source pixel to a triangle+barycentric surface
address.

No semantic body-part, owner or material id is exposed as a student target.  Privileged material
and owner arrays remain in the Stage47 teacher package and may be used only for audit/grouping.

PyTorch3D is imported lazily so ordinary CPU source tests do not need the dependency.  The external
worker notebook must install the repository-pinned BSD build before running this producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "RealSaS.Truth139DenseAppearanceGBuffer.v1"
VIEW_SCHEMA = "RealSaS.Truth139DenseAppearanceView.v1"
EXPECTED_VIEW_COUNT = 8
EXPECTED_RESOLUTION = 1024
PYTORCH3D_PIN = "978cd99221b9e0a6a568f1d427854d73363265cf"


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


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 inverse transfer, vectorized over [..., 3]."""
    value = np.asarray(rgb, dtype=np.float32)
    if value.shape[-1] != 3 or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_SRGB_DOMAIN")
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4).astype(np.float32)


def world_to_pytorch3d_ndc(vertices: np.ndarray, camera: dict) -> np.ndarray:
    """Project source-world vertices into PyTorch3D NDC for the controlled orthographic camera.

    Controlled camera `right` points to image-right.  PyTorch3D NDC +X points image-left, hence
    the explicit sign flip.  Both systems use +Y image-up.  `forward` points camera->scene, which
    is PyTorch3D's positive depth direction.
    """
    v = np.asarray(vertices, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    half = float(camera["half_extent"])
    if v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all():
        raise ValueError("TRUTH139_DENSE_APPEARANCE_VERTEX_LAYOUT")
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (origin, right, up, forward)):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_CAMERA_BASIS")
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("TRUTH139_DENSE_APPEARANCE_CAMERA_HALF_EXTENT")
    rel = v - origin[None, :]
    x = -(rel @ right) / half
    y = (rel @ up) / half
    z = rel @ forward
    ndc = np.stack((x, y, z), axis=-1)
    if not np.isfinite(ndc).all():
        raise ValueError("TRUTH139_DENSE_APPEARANCE_NDC_NONFINITE")
    return ndc.astype(np.float32)


def geometric_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    valid = length > 1.0e-12
    out = np.zeros_like(n, dtype=np.float64)
    out[valid] = n[valid] / length[valid, None]
    return out.astype(np.float32)


def _load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    required = {
        "rest_vertices_source_world",
        "faces",
        "face_owner_privileged",
        "face_material_privileged",
        "triangle_uv",
        "triangle_uv_valid",
    }
    with np.load(path, allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_DENSE_APPEARANCE_STAGE47_FIELDS_MISSING")
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int32)
        teacher = {
            "face_owner_privileged": np.asarray(z["face_owner_privileged"], dtype=np.int32),
            "face_material_privileged": np.asarray(z["face_material_privileged"], dtype=np.int32),
            "triangle_uv": np.asarray(z["triangle_uv"], dtype=np.float32),
            "triangle_uv_valid": np.asarray(z["triangle_uv_valid"], dtype=np.bool_),
        }
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) == 0
        or len(faces) == 0
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_GEOMETRY_INVALID")
    if teacher["face_owner_privileged"].shape != (len(faces),) or teacher["face_material_privileged"].shape != (len(faces),):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_PRIVILEGED_FACE_LAYOUT")
    if teacher["triangle_uv"].shape != (len(faces), 3, 2) or teacher["triangle_uv_valid"].shape != (len(faces),):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_UV_LAYOUT")
    return vertices, faces, teacher


def _load_camera_bundle(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise ValueError("TRUTH139_DENSE_APPEARANCE_CAMERA_SCHEMA")
    cameras = sorted(list(value.get("cameras") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cameras) != EXPECTED_VIEW_COUNT or [int(row.get("view_index", -1)) for row in cameras] != list(range(EXPECTED_VIEW_COUNT)):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_CAMERA_CARDINALITY")
    for camera in cameras:
        if int(camera.get("resolution", -1)) != EXPECTED_RESOLUTION:
            raise ValueError("TRUTH139_DENSE_APPEARANCE_RESOLUTION_DRIFT")
    return cameras


def _read_observation_rgba(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if value.shape != (EXPECTED_RESOLUTION, EXPECTED_RESOLUTION, 4):
        raise ValueError("TRUTH139_DENSE_APPEARANCE_RGBA_LAYOUT")
    return value


def _read_mask(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    expected = EXPECTED_RESOLUTION * EXPECTED_RESOLUTION
    if len(raw) != expected:
        raise ValueError("TRUTH139_DENSE_APPEARANCE_MASK_SIZE")
    return np.frombuffer(raw, dtype=np.uint8).reshape(EXPECTED_RESOLUTION, EXPECTED_RESOLUTION).astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("TRUTH139_DENSE_APPEARANCE_DILATION_RADIUS")
    if radius == 0:
        return np.asarray(mask, dtype=bool)
    import torch
    import torch.nn.functional as F

    value = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    out = F.max_pool2d(value, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out[0, 0].numpy() > 0.5


def _rasterize_view(vertices: np.ndarray, faces: np.ndarray, camera: dict, *, device: str):
    try:
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    except Exception as exc:  # pragma: no cover - external worker dependency
        raise RuntimeError("TRUTH139_DENSE_APPEARANCE_PYTORCH3D_REQUIRED") from exc

    ndc = world_to_pytorch3d_ndc(vertices, camera)
    verts = torch.from_numpy(ndc).to(device=device, dtype=torch.float32)
    tri = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device=device)
    mesh = Meshes(verts=[verts], faces=[tri])
    pix_to_face, zbuf, bary, _dist = rasterize_meshes(
        mesh,
        image_size=EXPECTED_RESOLUTION,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None,
        perspective_correct=False,
        clip_barycentric_coords=False,
        cull_backfaces=False,
    )
    face = pix_to_face[0, ..., 0].detach().cpu().numpy().astype(np.int32)
    depth = zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    bc = bary[0, ..., 0, :].detach().cpu().numpy().astype(np.float32)
    return face, depth, bc


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(
    *,
    subject_id: str,
    geometry_npz: Path,
    camera_bundle: Path,
    observation_dir: Path,
    output_dir: Path,
    device: str,
    mask_alignment_radius_px: int,
) -> dict:
    geometry_npz = geometry_npz.resolve()
    camera_bundle = camera_bundle.resolve()
    observation_dir = observation_dir.resolve()
    output_dir = output_dir.resolve()
    if not geometry_npz.is_file() or not camera_bundle.is_file() or not observation_dir.is_dir():
        raise FileNotFoundError("TRUTH139_DENSE_APPEARANCE_INPUT_MISSING")
    if not subject_id:
        raise ValueError("TRUTH139_DENSE_APPEARANCE_SUBJECT_ID")

    vertices, faces, teacher = _load_geometry(geometry_npz)
    cameras = _load_camera_bundle(camera_bundle)
    face_normals = geometric_face_normals(vertices, faces)
    output_dir.mkdir(parents=True, exist_ok=True)
    face_meta = output_dir / "FACE_TEACHER_METADATA.npz"
    _write_npz(
        face_meta,
        face_normal_source_world=face_normals,
        triangle_uv=teacher["triangle_uv"],
        triangle_uv_valid=teacher["triangle_uv_valid"],
        face_owner_privileged=teacher["face_owner_privileged"],
        face_material_privileged=teacher["face_material_privileged"],
    )

    views = []
    total_source_foreground = 0
    total_geometry_coverage = 0
    total_direct_opaque = 0
    total_alpha_boundary = 0
    for index, camera in enumerate(cameras):
        png = observation_dir / f"V{index}.png"
        mask_path = observation_dir / f"V{index}.mask.bin"
        if not png.is_file() or not mask_path.is_file():
            raise FileNotFoundError("TRUTH139_DENSE_APPEARANCE_VIEW_INPUT_MISSING:" + str(index))
        rgba_u8 = _read_observation_rgba(png)
        source_mask = _read_mask(mask_path)
        face_image, depth_image, bary_image = _rasterize_view(vertices, faces, camera, device=device)
        geometry_mask = face_image >= 0
        # Source antialias/filter coverage may extend outside the exact geometric pixel set by a
        # bounded amount.  The admitted observation policy already allows a two-pixel mask gap;
        # use the same bounded domain here rather than inventing a new alignment tolerance.
        geometry_dilated = _dilate(geometry_mask, int(mask_alignment_radius_px))
        source_without_geometry = source_mask & ~geometry_dilated
        if np.any(source_without_geometry):
            raise ValueError(
                f"TRUTH139_DENSE_APPEARANCE_SOURCE_WITHOUT_GEOMETRY:V{index}:"
                f"{int(np.count_nonzero(source_without_geometry))}"
            )

        alpha_u8 = rgba_u8[..., 3]
        direct_opaque = source_mask & geometry_mask & (alpha_u8 == 255)
        alpha_boundary = source_mask & geometry_mask & (alpha_u8 > 0) & (alpha_u8 < 255)
        covered = np.flatnonzero(geometry_mask.reshape(-1))
        ys = covered // EXPECTED_RESOLUTION
        xs = covered % EXPECTED_RESOLUTION
        tri = face_image[ys, xs]
        bc = bary_image[ys, xs]
        # Store only two barycentric coordinates; the third is exact-by-contract as 1-u-v.
        bary_uv = bc[:, :2].astype(np.float32)
        if np.any(tri < 0) or np.any(tri >= len(faces)) or not np.isfinite(bary_uv).all():
            raise ValueError("TRUTH139_DENSE_APPEARANCE_RASTER_ADDRESS_INVALID")
        source_rgb = rgba_u8[ys, xs, :3].astype(np.float32) / 255.0
        source_rgba_linear = np.concatenate(
            (srgb_to_linear(source_rgb), rgba_u8[ys, xs, 3:4].astype(np.float32) / 255.0), axis=1
        )
        direct_flat = direct_opaque[ys, xs]
        boundary_flat = alpha_boundary[ys, xs]
        source_foreground_flat = source_mask[ys, xs]

        out = output_dir / f"V{index}_DENSE_APPEARANCE_GBUFFER.npz"
        _write_npz(
            out,
            pixel_linear_index=covered.astype(np.int32),
            triangle_id=tri.astype(np.int32),
            barycentric_uv=bary_uv,
            depth_source_camera=depth_image[ys, xs].astype(np.float32),
            source_rgba_linear=source_rgba_linear.astype(np.float32),
            source_foreground=source_foreground_flat.astype(np.bool_),
            direct_opaque_rgba_valid=direct_flat.astype(np.bool_),
            alpha_boundary=boundary_flat.astype(np.bool_),
            resolution=np.asarray([EXPECTED_RESOLUTION, EXPECTED_RESOLUTION], dtype=np.int32),
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
            "geometry_covered_pixel_count": int(len(covered)),
            "source_foreground_pixel_count": int(np.count_nonzero(source_mask)),
            "direct_opaque_rgba_count": int(np.count_nonzero(direct_flat)),
            "alpha_boundary_count": int(np.count_nonzero(boundary_flat)),
            "source_foreground_without_geometry_after_dilation": 0,
            "mask_alignment_radius_px": int(mask_alignment_radius_px),
        }
        views.append(row)
        total_source_foreground += row["source_foreground_pixel_count"]
        total_geometry_coverage += row["geometry_covered_pixel_count"]
        total_direct_opaque += row["direct_opaque_rgba_count"]
        total_alpha_boundary += row["alpha_boundary_count"]
        print(
            f"[DENSE_APPEARANCE] {subject_id} V{index} geometry={row['geometry_covered_pixel_count']} "
            f"source_fg={row['source_foreground_pixel_count']} direct_opaque={row['direct_opaque_rgba_count']} "
            f"alpha_boundary={row['alpha_boundary_count']}",
            flush=True,
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_DENSE_SURFACE_ADDRESSED_APPEARANCE_GBUFFER__STAGE48_INTERVENTION_WITNESS_STILL_REQUIRED",
        "subject_id": subject_id,
        "authority_contract": "canonical/TRUTH139_DENSE_APPEARANCE_AUTHORITY_V1.json",
        "geometry_npz": {"path": str(geometry_npz), "sha256": sha256(geometry_npz)},
        "camera_bundle": {"path": str(camera_bundle), "sha256": sha256(camera_bundle)},
        "face_teacher_metadata": {"path": str(face_meta), "sha256": sha256(face_meta)},
        "rasterizer": {
            "implementation": "PYTORCH3D_RASTERIZE_MESHES",
            "required_git_commit": PYTORCH3D_PIN,
            "license": "BSD",
            "faces_per_pixel": 1,
            "perspective_correct": False,
            "student_semantic_targets": False,
        },
        "color_contract": {
            "source_png_rgb": "STANDARD_VIEW_TRANSFORM_SRGB_ENCODED_8BIT",
            "stored_rgb": "IEC_61966_2_1_INVERSE_TRANSFER_TO_SCENE_LINEAR",
            "stored_alpha": "SOURCE_PNG_ALPHA_0_TO_1",
            "direct_student_rgba": "OPAQUE_SOURCE_FOREGROUND_WITH_EXACT_GEOMETRY_ADDRESS_ONLY",
            "alpha_boundary": "SEPARATE_TYPED_STRATUM_NOT_SILENTLY_TREATED_AS_OPAQUE",
        },
        "views": views,
        "counts": {
            "view_count": len(views),
            "source_foreground_pixels": total_source_foreground,
            "geometry_covered_pixels": total_geometry_coverage,
            "direct_opaque_rgba_pixels": total_direct_opaque,
            "alpha_boundary_pixels": total_alpha_boundary,
        },
        "teacher_only_fields": [
            "face_owner_privileged",
            "face_material_privileged",
            "triangle_uv",
            "triangle_uv_valid",
        ],
        "semantic_part_taxonomy_used": False,
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
        "stage48_required_for_ambiguous_multihit_and_source_equivalence": True,
    }
    payload["authority_identity_sha256"] = digest_json(payload)
    manifest = output_dir / "DENSE_APPEARANCE_GBUFFER_MANIFEST.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "subject_id": subject_id,
                "direct_opaque_rgba_pixels": total_direct_opaque,
                "alpha_boundary_pixels": total_alpha_boundary,
                "authority_identity_sha256": payload["authority_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-npz", required=True, type=Path)
    parser.add_argument("--camera-bundle", required=True, type=Path)
    parser.add_argument("--observation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-alignment-radius-px", type=int, default=2)
    args = parser.parse_args()
    build(
        subject_id=args.subject_id,
        geometry_npz=args.geometry_npz,
        camera_bundle=args.camera_bundle,
        observation_dir=args.observation_dir,
        output_dir=args.output_dir,
        device=args.device,
        mask_alignment_radius_px=args.mask_alignment_radius_px,
    )


if __name__ == "__main__":
    main()
