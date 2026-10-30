#!/usr/bin/env python3
"""Rescue Truth139 K-hit peel by overfetching CUDA candidates and exact-filtering coverage.

The V2 producer asks PyTorch3D for exactly K faces per pixel, then reconstructs
barycentrics/depth after CUDA face selection. A float32 raster edge candidate can
therefore occupy the K-th capacity slot even when the exact pixel centre lies
outside that triangle; with a full K buffer the true K-th CPU-ray hit is no longer
available for replacement. The independent CPU court exposed this only at S7
(K=8) on the failed Full139 extension subjects.

V3 keeps CUDA as candidate discovery only, overfetches a guarded candidate stack,
then re-evaluates pixel-centre triangle coverage, barycentrics and depth from the
original world geometry + serialized camera in float64 before selecting the first
K qualified hits. It does not relax the CPU court and preserves overflow/unknown
typing whenever the guarded candidate buffer is saturated or more than K exact
candidates survive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus import build_truth139_graphics_layer_peel_v2 as v2

PRODUCER_REVISION = "V3_OVERFETCH16__DIRECT_FLOAT64_CAMERA_COVERAGE__K_PREFIX"
EXACT_INTERSECTION_EPSILON = 1.0e-10


def _candidate_k(output_k: int) -> int:
    return min(64, max(int(output_k) * 2, int(output_k) + 8))


def _world_to_ndc_float64(vertices: np.ndarray, camera: dict) -> np.ndarray:
    """Project directly in float64; unlike the CUDA path no float32 round-trip is allowed."""
    v = np.asarray(vertices, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    half = float(camera["half_extent"])
    if v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all():
        raise ValueError("TRUTH139_LAYER_PEEL_V3_VERTEX_LAYOUT")
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (origin, right, up, forward)):
        raise ValueError("TRUTH139_LAYER_PEEL_V3_CAMERA_BASIS")
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("TRUTH139_LAYER_PEEL_V3_CAMERA_HALF_EXTENT")
    rel = v - origin[None, :]
    return np.stack((-(rel @ right) / half, (rel @ up) / half, rel @ forward), axis=-1)


def _float64_surface_address_guarded(
    raw_face: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict,
    *,
    output_k: int,
    intersection_epsilon: float = EXACT_INTERSECTION_EPSILON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an exact-filtered K prefix from an overfetched CUDA candidate stack."""
    candidate_face = np.asarray(raw_face, dtype=np.int32).copy()
    if candidate_face.ndim != 3:
        raise ValueError("TRUTH139_LAYER_PEEL_V3_FACE_LAYOUT_INTERNAL")
    h, w, candidate_k = candidate_face.shape
    if output_k < 2 or candidate_k <= output_k:
        raise ValueError("TRUTH139_LAYER_PEEL_V3_GUARD_CAPACITY")
    if not np.isfinite(intersection_epsilon) or intersection_epsilon <= 0.0:
        raise ValueError("TRUTH139_LAYER_PEEL_V3_INTERSECTION_EPSILON")

    candidate_buffer_saturated = np.all(candidate_face >= 0, axis=-1)
    ndc = _world_to_ndc_float64(vertices, camera)
    tri_index = np.asarray(faces, dtype=np.int64)
    if ndc.shape != (len(vertices), 3):
        raise ValueError("TRUTH139_LAYER_PEEL_V3_NDC_LAYOUT")

    keep = candidate_face >= 0
    candidate_depth = np.full(candidate_face.shape, np.inf, dtype=np.float64)
    candidate_bary = np.zeros((*candidate_face.shape, 3), dtype=np.float64)

    for slot in range(candidate_k):
        mask = keep[..., slot]
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        fids = candidate_face[ys, xs, slot].astype(np.int64)
        if np.any(fids < 0) or np.any(fids >= len(tri_index)):
            raise ValueError("TRUTH139_LAYER_PEEL_V3_FACE_RANGE")

        tri_xy = ndc[tri_index[fids], :2]
        x0, y0 = tri_xy[:, 0, 0], tri_xy[:, 0, 1]
        x1, y1 = tri_xy[:, 1, 0], tri_xy[:, 1, 1]
        x2, y2 = tri_xy[:, 2, 0], tri_xy[:, 2, 1]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        stable = np.isfinite(den) & (np.abs(den) > 1.0e-12)

        # Exact PyTorch3D pixel centres in its image convention (+X points image-left).
        px = 1.0 - (2.0 * xs.astype(np.float64) + 1.0) / float(w)
        py = 1.0 - (2.0 * ys.astype(np.float64) + 1.0) / float(h)
        bc64 = np.full((len(fids), 3), np.nan, dtype=np.float64)
        if np.any(stable):
            s = stable
            w0 = ((y1[s] - y2[s]) * (px[s] - x2[s]) + (x2[s] - x1[s]) * (py[s] - y2[s])) / den[s]
            w1 = ((y2[s] - y0[s]) * (px[s] - x2[s]) + (x0[s] - x2[s]) * (py[s] - y2[s])) / den[s]
            w2 = 1.0 - w0 - w1
            bc64[s] = np.stack((w0, w1, w2), axis=-1)

        # Match the independent CPU oracle's barycentric inclusion semantics.
        exact_inside = (
            stable
            & np.isfinite(bc64).all(axis=1)
            & np.all(bc64 >= -float(intersection_epsilon), axis=1)
            & np.all(bc64 <= 1.0 + float(intersection_epsilon), axis=1)
        )
        rejected = ~exact_inside
        if np.any(rejected):
            keep[ys[rejected], xs[rejected], slot] = False
        if not np.any(exact_inside):
            continue

        sy = ys[exact_inside]
        sx = xs[exact_inside]
        sf = fids[exact_inside]
        bc = np.clip(bc64[exact_inside], 0.0, 1.0)
        norm = bc.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
            raise ValueError("TRUTH139_LAYER_PEEL_V3_BARY_NORMALIZATION")
        bc = bc / norm
        z = np.einsum("ni,ni->n", bc, ndc[tri_index[sf], 2])
        if not np.isfinite(z).all():
            raise ValueError("TRUTH139_LAYER_PEEL_V3_DEPTH_NONFINITE")
        candidate_bary[sy, sx, slot] = bc
        candidate_depth[sy, sx, slot] = z

    candidate_face[~keep] = -1
    candidate_depth[~keep] = np.inf
    candidate_bary[~keep] = 0.0

    order = np.argsort(candidate_depth, axis=-1, kind="stable")
    sorted_face = np.take_along_axis(candidate_face, order, axis=-1)
    sorted_depth = np.take_along_axis(candidate_depth, order, axis=-1)
    sorted_bary = np.take_along_axis(candidate_bary, order[..., None], axis=-2)
    sorted_valid = sorted_face >= 0
    exact_count = np.sum(sorted_valid, axis=-1)

    # Stored K is complete only if guarded candidate discovery itself did not
    # saturate and no exact hit exists beyond the retained K prefix.
    overflow_or_unknown = candidate_buffer_saturated | (exact_count > int(output_k))

    face = sorted_face[..., :output_k].astype(np.int32, copy=True)
    depth64 = sorted_depth[..., :output_k].copy()
    bary64 = sorted_bary[..., :output_k, :].copy()
    valid = face >= 0
    face[~valid] = -1
    depth64[~valid] = -1.0
    bary64[~valid] = 0.0
    return face, depth64.astype(np.float32), bary64.astype(np.float32), overflow_or_unknown.astype(np.bool_)


def _rasterize_k_guarded(
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
        raise RuntimeError("TRUTH139_LAYER_PEEL_V3_PYTORCH3D_REQUIRED") from exc

    candidate_k = _candidate_k(int(k))
    # CUDA remains candidate discovery only and intentionally consumes the same
    # float32 NDC convention as the original producer.
    ndc = v2.world_to_pytorch3d_ndc(vertices, camera)
    verts = torch.from_numpy(ndc).to(device=device, dtype=torch.float32)
    tri = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device=device)
    mesh = Meshes(verts=[verts], faces=[tri])
    pix_to_face, _zbuf, _bary, _dist = rasterize_meshes(
        mesh,
        image_size=v2.EXPECTED_RESOLUTION,
        blur_radius=0.0,
        faces_per_pixel=int(candidate_k),
        bin_size=None,
        max_faces_per_bin=None,
        perspective_correct=False,
        clip_barycentric_coords=False,
        cull_backfaces=False,
    )
    raw_face = pix_to_face[0].detach().cpu().numpy().astype(np.int32)
    return _float64_surface_address_guarded(
        raw_face,
        vertices,
        faces,
        camera,
        output_k=int(k),
    )


def build(**kwargs) -> dict:
    output_k = int(kwargs.get("k", 8))
    v2.PRODUCER_REVISION = PRODUCER_REVISION
    v2._rasterize_k = _rasterize_k_guarded
    payload = v2.build(**kwargs)

    # Correct the inherited V2 metadata so the manifest truthfully describes the
    # rescue implementation. View bytes were already produced by the guarded path.
    payload["rasterizer"] = {
        **dict(payload.get("rasterizer") or {}),
        "implementation": "PYTORCH3D_GUARDED_CANDIDATE_DISCOVERY__DIRECT_FLOAT64_CAMERA_COVERAGE_AND_DEPTH__K_PREFIX",
        "faces_per_pixel": output_k,
        "cuda_candidate_faces_per_pixel": _candidate_k(output_k),
        "stored_prefix_k": output_k,
        "direct_float64_world_camera_projection": True,
        "exact_pixel_center_coverage_filter": True,
        "coverage_intersection_epsilon": EXACT_INTERSECTION_EPSILON,
        "cuda_barycentric_is_truth_authority": False,
        "cuda_depth_is_truth_authority": False,
    }
    payload["policy"] = {
        **dict(payload.get("policy") or {}),
        "guard_candidate_capacity_saturation_preserves_overflow_unknown": True,
        "exact_survivor_count_above_k_preserves_overflow_unknown": True,
        "cpu_court_tolerance_relaxed": False,
    }
    payload.pop("layer_peel_teacher_identity_sha256", None)
    payload["layer_peel_teacher_identity_sha256"] = v2._digest(payload)
    manifest_out = Path(kwargs["output_dir"]).resolve() / "LAYER_PEEL_TEACHER_MANIFEST.json"
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "subject_id": result["subject_id"],
                "producer_revision": result["producer_revision"],
                "k": result["k"],
                "cuda_candidate_faces_per_pixel": result["rasterizer"]["cuda_candidate_faces_per_pixel"],
                "layer_peel_teacher_identity_sha256": result["layer_peel_teacher_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
