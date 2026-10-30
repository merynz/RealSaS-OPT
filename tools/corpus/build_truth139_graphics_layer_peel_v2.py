#!/usr/bin/env python3
"""Build Truth139 K-hit geometric peel from canonical Dense Graphics G-buffer V1.

V2 removes the legacy DenseGraphicsTeacher.v1 parent.  The canonical parent is the
court-validated RealSaS.Truth139DenseGraphicsGBuffer.v1 produced by the current
Truth139 V2 materialization chain.  CUDA/PyTorch3D selects face ids; barycentric
surface addresses and depth are independently reconstructed at exact PyTorch3D
pixel centres in float64 before any truth is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import (
    EXPECTED_RESOLUTION,
    PYTORCH3D_PIN,
    sha256,
    world_to_pytorch3d_ndc,
)

SCHEMA = "RealSaS.Truth139GraphicsLayerPeelTeacher.v1"
VIEW_SCHEMA = "RealSaS.Truth139GraphicsLayerPeelView.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
GRAPHICS_STATUS = "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING"
DENSE_COURT_SCHEMA = "RealSaS.Truth139DenseGraphicsGeometryCourt.v1"
DENSE_COURT_STATUS = "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER"
STATUS = "PASS_K_HIT_GEOMETRIC_LAYER_PEEL_CANDIDATE__CPU_PREFIX_COURT_REQUIRED__RGBA_COMPLETION_SEPARATE"
PRODUCER_REVISION = "V2_DENSE_GBUFFER_PARENT__FLOAT64_PIXEL_CENTER_BARY"
CONTRACT = "canonical/TRUTH139_GRAPHICS_LAYER_DECOMPOSITION_TEACHER_V1.json"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _write_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def _load_parent(graphics_manifest: Path, dense_court: Path) -> tuple[dict, dict, Path, Path]:
    graphics_path = graphics_manifest.resolve()
    court_path = dense_court.resolve()
    if not graphics_path.is_file():
        raise FileNotFoundError("TRUTH139_LAYER_PEEL_V2_GRAPHICS_MANIFEST_MISSING")
    if not court_path.is_file():
        raise FileNotFoundError("TRUTH139_LAYER_PEEL_V2_DENSE_COURT_MISSING")
    graphics = json.loads(graphics_path.read_text())
    court = json.loads(court_path.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or graphics.get("status") != GRAPHICS_STATUS:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_GRAPHICS_PARENT")
    if court.get("schema") != DENSE_COURT_SCHEMA or court.get("status") != DENSE_COURT_STATUS:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_DENSE_COURT_STATUS")
    subject_id = str(graphics.get("subject_id") or "")
    if not subject_id or str(court.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_SUBJECT_BINDING")
    graphics_identity = str(graphics.get("graphics_identity_sha256") or "")
    if len(graphics_identity) != 64 or str(court.get("dense_graphics_identity_sha256") or "") != graphics_identity:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_DENSE_COURT_IDENTITY")
    geometry_path = _bound_path(dict(graphics.get("geometry_npz") or {}), "TRUTH139_LAYER_PEEL_V2_GEOMETRY_DRIFT")
    camera_path = _bound_path(dict(graphics.get("camera_bundle") or {}), "TRUTH139_LAYER_PEEL_V2_CAMERA_DRIFT")
    return graphics, court, geometry_path, camera_path


def _float64_surface_address(
    raw_face: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute barycentrics/depth for CUDA-selected faces at exact pixel centres.

    Returns face, depth, bary, raw_stack_saturated.  A mathematically zero-area
    CUDA leak is removed.  If the raw K capacity was saturated, completeness stays
    unknown even if such a leak is filtered.
    """
    face = np.asarray(raw_face, dtype=np.int32).copy()
    if face.ndim != 3:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_FACE_LAYOUT_INTERNAL")
    h, w, k = face.shape
    raw_stack_saturated = np.all(face >= 0, axis=-1)

    ndc = np.asarray(world_to_pytorch3d_ndc(vertices, camera), dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if ndc.shape != (len(vertices), 3):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_NDC_LAYOUT")

    depth = np.full(face.shape, -1.0, dtype=np.float32)
    bary = np.zeros((*face.shape, 3), dtype=np.float32)
    keep = face >= 0

    for slot in range(k):
        mask = keep[..., slot]
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        fids = face[ys, xs, slot].astype(np.int64)
        if np.any(fids < 0) or np.any(fids >= len(f)):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_FACE_RANGE")

        tri_xy = ndc[f[fids], :2]
        x0, y0 = tri_xy[:, 0, 0], tri_xy[:, 0, 1]
        x1, y1 = tri_xy[:, 1, 0], tri_xy[:, 1, 1]
        x2, y2 = tri_xy[:, 2, 0], tri_xy[:, 2, 1]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        stable = np.isfinite(den) & (np.abs(den) > 1.0e-12)

        if np.any(~stable):
            by, bx = ys[~stable], xs[~stable]
            keep[by, bx, slot] = False

        if not np.any(stable):
            continue
        sy, sx, sf, sden = ys[stable], xs[stable], fids[stable], den[stable]
        px = 1.0 - (2.0 * sx.astype(np.float64) + 1.0) / float(w)
        py = 1.0 - (2.0 * sy.astype(np.float64) + 1.0) / float(h)
        sx0, sy0 = x0[stable], y0[stable]
        sx1, sy1 = x1[stable], y1[stable]
        sx2, sy2 = x2[stable], y2[stable]

        w0 = ((sy1 - sy2) * (px - sx2) + (sx2 - sx1) * (py - sy2)) / sden
        w1 = ((sy2 - sy0) * (px - sx2) + (sx0 - sx2) * (py - sy2)) / sden
        w2 = 1.0 - w0 - w1
        bc64 = np.stack((w0, w1, w2), axis=-1)
        if not np.isfinite(bc64).all():
            raise ValueError("TRUTH139_LAYER_PEEL_V2_BARY_NONFINITE")
        if np.max(np.abs(bc64.sum(axis=1) - 1.0), initial=0.0) > 1.0e-12:
            raise ValueError("TRUTH139_LAYER_PEEL_V2_BARY_SUM_INTERNAL")
        z = np.einsum("ni,ni->n", bc64, ndc[f[sf], 2])
        if not np.isfinite(z).all():
            raise ValueError("TRUTH139_LAYER_PEEL_V2_DEPTH_NONFINITE")
        bary[sy, sx, slot] = bc64.astype(np.float32)
        depth[sy, sx, slot] = z.astype(np.float32)

    face[~keep] = -1
    depth[~keep] = -1.0
    bary[~keep] = 0.0
    sort_key = np.where(keep, depth.astype(np.float64), np.inf)
    order = np.argsort(sort_key, axis=-1, kind="stable")
    face = np.take_along_axis(face, order, axis=-1)
    depth = np.take_along_axis(depth, order, axis=-1)
    bary = np.take_along_axis(bary, order[..., None], axis=-2)
    keep = face >= 0
    face[~keep] = -1
    depth[~keep] = -1.0
    bary[~keep] = 0.0
    return face, depth, bary, raw_stack_saturated


def _rasterize_k(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict,
    *,
    k: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TRUTH139_LAYER_PEEL_V2_PYTORCH3D_REQUIRED") from exc

    ndc = world_to_pytorch3d_ndc(vertices, camera)
    verts = torch.from_numpy(ndc).to(device=device, dtype=torch.float32)
    tri = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device=device)
    mesh = Meshes(verts=[verts], faces=[tri])
    pix_to_face, _zbuf, _bary, _dist = rasterize_meshes(
        mesh,
        image_size=EXPECTED_RESOLUTION,
        blur_radius=0.0,
        faces_per_pixel=int(k),
        bin_size=None,
        max_faces_per_bin=None,
        perspective_correct=False,
        clip_barycentric_coords=False,
        cull_backfaces=False,
    )
    raw_face = pix_to_face[0].detach().cpu().numpy().astype(np.int32)
    return _float64_surface_address(raw_face, vertices, faces, camera)


def build(
    *,
    dense_graphics_manifest: Path,
    dense_graphics_court: Path,
    output_dir: Path,
    k: int = 8,
    device: str = "cuda",
    tie_epsilon_relative: float = 1.0e-7,
) -> dict:
    if k < 2 or k > 64:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_K_RANGE")
    if not math.isfinite(tie_epsilon_relative) or tie_epsilon_relative <= 0:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_TIE_EPSILON")

    graphics, court, geometry_path, camera_path = _load_parent(
        dense_graphics_manifest, dense_graphics_court
    )
    subject_id = str(graphics["subject_id"])
    output_dir = output_dir.resolve()

    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if (
        vertices.ndim != 2 or vertices.shape[1:] != (3,)
        or faces.ndim != 2 or faces.shape[1:] != (3,)
        or len(vertices) == 0 or len(faces) == 0
        or np.any(faces < 0) or np.any(faces >= len(vertices))
    ):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_GEOMETRY_INVALID")
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TRUTH139_LAYER_PEEL_V2_GEOMETRY_SCALE")
    tie_epsilon = max(scale * float(tie_epsilon_relative), 1.0e-9)

    camera_payload = json.loads(camera_path.read_text())
    cameras = sorted(
        list(camera_payload.get("cameras") or ()),
        key=lambda row: int(row.get("view_index", -1)),
    )
    if len(cameras) != 8 or [int(row.get("view_index", -1)) for row in cameras] != list(range(8)):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_CAMERA_COUNT")
    if any(int(row.get("resolution", -1)) != EXPECTED_RESOLUTION for row in cameras):
        raise ValueError("TRUTH139_LAYER_PEEL_V2_RESOLUTION_DRIFT")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_hit_pixels = total_truncated = total_tied = total_filtered_zero_area = 0
    max_hits_observed = 0
    layer_pixel_counts = np.zeros(k, dtype=np.int64)

    for view_index, camera in enumerate(cameras):
        face, depth, bary, raw_saturated = _rasterize_k(
            vertices, faces, camera, k=k, device=device
        )
        valid = face >= 0
        hit_count_image = np.sum(valid, axis=-1).astype(np.uint8)
        hit_pixels = np.flatnonzero(hit_count_image.reshape(-1) > 0)
        y = hit_pixels // EXPECTED_RESOLUTION
        x = hit_pixels % EXPECTED_RESOLUTION
        face_sparse = face[y, x]
        depth_sparse = depth[y, x]
        bary_sparse = bary[y, x]
        valid_sparse = valid[y, x]
        hit_count = hit_count_image[y, x]
        raw_saturated_sparse = raw_saturated[y, x]

        if np.any(face_sparse[valid_sparse] >= len(faces)):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_FACE_RANGE")
        if not np.isfinite(depth_sparse[valid_sparse]).all() or not np.isfinite(bary_sparse[valid_sparse]).all():
            raise ValueError("TRUTH139_LAYER_PEEL_V2_NONFINITE")
        bary_sum = np.sum(bary_sparse, axis=-1)
        if np.any(np.abs(bary_sum[valid_sparse] - 1.0) > 2.0e-4):
            raise ValueError("TRUTH139_LAYER_PEEL_V2_BARY_SUM")

        for slot in range(k - 1):
            pair_valid = valid_sparse[:, slot] & valid_sparse[:, slot + 1]
            if np.any(depth_sparse[pair_valid, slot + 1] + 1.0e-7 < depth_sparse[pair_valid, slot]):
                raise ValueError("TRUTH139_LAYER_PEEL_V2_DEPTH_ORDER")

        adjacent_gap = np.full((len(hit_pixels), max(k - 1, 1)), np.nan, dtype=np.float32)
        tie_pair = np.zeros((len(hit_pixels), max(k - 1, 1)), dtype=np.bool_)
        if k > 1:
            adjacent_gap[:, : k - 1] = depth_sparse[:, 1:] - depth_sparse[:, :-1]
            pair_valid = valid_sparse[:, 1:] & valid_sparse[:, :-1]
            tie_pair[:, : k - 1] = pair_valid & (adjacent_gap[:, : k - 1] <= tie_epsilon)
        has_depth_tie = np.any(tie_pair, axis=1)

        stack_complete = ~raw_saturated_sparse
        truncated_or_unknown = ~stack_complete
        peel_layer_valid = valid_sparse.copy()
        for slot in range(k):
            layer_pixel_counts[slot] += int(np.count_nonzero(peel_layer_valid[:, slot]))

        filtered_zero_area = int(np.count_nonzero(raw_saturated_sparse & (hit_count < k)))
        total_filtered_zero_area += filtered_zero_area

        out = output_dir / f"V{view_index}_LAYER_PEEL_K{k}.npz"
        _write_npz(
            out,
            pixel_linear_index=hit_pixels.astype(np.int32),
            hit_count_lower_bound=hit_count,
            raw_stack_saturated=raw_saturated_sparse.astype(np.bool_),
            stack_complete=stack_complete.astype(np.bool_),
            truncated_or_overflow_unknown=truncated_or_unknown.astype(np.bool_),
            has_depth_tie=has_depth_tie.astype(np.bool_),
            hit_valid=valid_sparse.astype(np.bool_),
            hit_triangle_id=face_sparse.astype(np.int32),
            hit_barycentric=bary_sparse.astype(np.float32),
            hit_depth=depth_sparse.astype(np.float32),
            adjacent_depth_gap=adjacent_gap,
            adjacent_depth_tie=tie_pair,
            peel_layer_valid=peel_layer_valid.astype(np.bool_),
            resolution=np.asarray([EXPECTED_RESOLUTION, EXPECTED_RESOLUTION], dtype=np.int32),
            origin=np.asarray(["TOP_LEFT"]),
        )
        row = {
            "schema": VIEW_SCHEMA,
            "view_index": view_index,
            "path": str(out),
            "sha256": sha256(out),
            "hit_pixel_count": int(len(hit_pixels)),
            "maximum_hit_count_lower_bound": int(hit_count.max(initial=0)),
            "complete_stack_pixel_count": int(np.count_nonzero(stack_complete)),
            "truncated_or_overflow_unknown_pixel_count": int(np.count_nonzero(truncated_or_unknown)),
            "depth_tie_pixel_count": int(np.count_nonzero(has_depth_tie)),
            "raw_capacity_saturated_pixel_count": int(np.count_nonzero(raw_saturated_sparse)),
            "raw_saturated_after_zero_area_filter_pixel_count": filtered_zero_area,
            "per_layer_valid_pixel_count": [
                int(np.count_nonzero(peel_layer_valid[:, slot])) for slot in range(k)
            ],
        }
        rows.append(row)
        total_hit_pixels += row["hit_pixel_count"]
        total_truncated += row["truncated_or_overflow_unknown_pixel_count"]
        total_tied += row["depth_tie_pixel_count"]
        max_hits_observed = max(max_hits_observed, row["maximum_hit_count_lower_bound"])
        print(
            f"[LAYER_PEEL_V2] {subject_id} V{view_index} hit_pixels={row['hit_pixel_count']} "
            f"max_hits>={row['maximum_hit_count_lower_bound']} "
            f"truncated={row['truncated_or_overflow_unknown_pixel_count']} "
            f"ties={row['depth_tie_pixel_count']}",
            flush=True,
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": STATUS,
        "producer_revision": PRODUCER_REVISION,
        "subject_id": subject_id,
        "contract": CONTRACT,
        "parents": {
            "dense_graphics_gbuffer_manifest": {
                "path": str(Path(dense_graphics_manifest).resolve()),
                "sha256": sha256(Path(dense_graphics_manifest).resolve()),
            },
            "dense_graphics_identity_sha256": str(graphics["graphics_identity_sha256"]),
            "dense_graphics_geometry_court": {
                "path": str(Path(dense_graphics_court).resolve()),
                "sha256": sha256(Path(dense_graphics_court).resolve()),
            },
            "dense_graphics_geometry_court_identity_sha256": str(court.get("court_identity_sha256") or ""),
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "camera_bundle": {"path": str(camera_path), "sha256": sha256(camera_path)},
        },
        "rasterizer": {
            "implementation": "PYTORCH3D_FACE_SELECTION__FLOAT64_PIXEL_CENTER_SURFACE_ADDRESS",
            "required_git_commit": PYTORCH3D_PIN,
            "faces_per_pixel": int(k),
            "perspective_correct": False,
            "cull_backfaces": False,
            "cuda_barycentric_is_truth_authority": False,
            "bounded_stack_is_complete_when_raw_k_capacity_saturated": False,
        },
        "policy": {
            "semantic_layer_names_used": False,
            "peel_masks_are_depth_order_masks_not_body_part_labels": True,
            "hidden_rgba_materialized": False,
            "hidden_rgba_missing_is_zero": False,
            "raw_k_capacity_saturation_preserves_overflow_unknown": True,
            "depth_ties_are_typed_not_force_ordered": True,
            "student_semantic_taxonomy_used": False,
            "training_performed": False,
        },
        "k": int(k),
        "tie_epsilon": float(tie_epsilon),
        "views": rows,
        "counts": {
            "hit_pixels_all_views": int(total_hit_pixels),
            "truncated_or_overflow_unknown_pixels_all_views": int(total_truncated),
            "depth_tie_pixels_all_views": int(total_tied),
            "raw_saturated_after_zero_area_filter_pixels_all_views": int(total_filtered_zero_area),
            "maximum_hit_count_lower_bound": int(max_hits_observed),
            "per_layer_valid_pixels_all_views": [int(v) for v in layer_pixel_counts.tolist()],
        },
        "required_next_courts": [
            "PASS_LAYER_PEEL_V2_GPU_FACE_SELECTION_MATCHES_COMPLETE_CPU_RAY_STACK",
            "RGBA_COMPLETION_COURT_IF_EXACT_SOURCE_MATERIAL_EVALUATOR_IS_AVAILABLE",
        ],
        "qualification_minted": False,
    }
    payload["layer_peel_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "LAYER_PEEL_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-manifest", required=True, type=Path)
    parser.add_argument("--dense-graphics-court", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tie-epsilon-relative", type=float, default=1.0e-7)
    args = parser.parse_args()
    result = build(
        dense_graphics_manifest=args.dense_graphics_manifest,
        dense_graphics_court=args.dense_graphics_court,
        output_dir=args.output_dir,
        k=args.k,
        device=args.device,
        tie_epsilon_relative=args.tie_epsilon_relative,
    )
    print(json.dumps({
        "status": result["status"],
        "producer_revision": result["producer_revision"],
        "subject_id": result["subject_id"],
        "k": result["k"],
        "counts": result["counts"],
        "layer_peel_teacher_identity_sha256": result["layer_peel_teacher_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
