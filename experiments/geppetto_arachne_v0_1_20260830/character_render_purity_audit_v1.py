from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

VIEWS = 8
RENDER_SIZE = 512


@dataclass(frozen=True)
class RenderPurityAssetInputV1:
    canonical_asset_id: str
    source_registry_id: str
    asset_dir: Path
    arachne_structural_c0: bool


def _distribution(x: list[float]) -> dict[str, float | int | None]:
    if not x:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.asarray(x, dtype=np.float64)
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def _mask_metrics(mask: np.ndarray) -> dict[str, Any]:
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("mask must be 2D bool")
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    n = int(len(xs))
    if n == 0:
        raise ValueError("empty raster foreground")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    margins = (x0, W - 1 - x1, y0, H - 1 - y1)
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    labels, cc = ndimage.label(crop, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:]
    largest = int(counts.max()) if counts.size else 0
    sig = int(np.count_nonzero(counts >= max(4, int(round(n * 0.001)))))
    return {
        "foreground_pixels": n,
        "occupancy_fraction": float(n / (H * W)),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width_fraction": float(bw / W),
        "bbox_height_fraction": float(bh / H),
        "bbox_area_fraction": float((bw * bh) / (H * W)),
        "min_frame_margin_px": int(min(margins)),
        "min_frame_margin_fraction": float(min(margins) / min(H, W)),
        "border_touch": bool(min(margins) == 0),
        "centroid_offset_fraction": float(
            np.linalg.norm(np.asarray([xs.mean() - (W - 1) / 2, ys.mean() - (H - 1) / 2]))
            / np.linalg.norm(np.asarray([W / 2, H / 2]))
        ),
        "connected_component_count": int(cc),
        "significant_component_count": sig,
        "largest_component_fraction": float(largest / n),
    }


def _mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"bad vertices/faces shape: {V.shape}/{F.shape}")
    if len(V) == 0 or len(F) == 0 or not np.isfinite(V).all():
        raise ValueError("empty/nonfinite mesh")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError("face vertex index out of range")
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or float(area.sum()) <= 0:
        raise ValueError("invalid/zero mesh surface area")
    order = np.sort(area)[::-1]
    total = float(area.sum())
    bbox = V.max(0) - V.min(0)
    nz = bbox[bbox > 1e-12]
    aspect = float(nz.max() / nz.min()) if nz.size >= 2 else float("inf")
    return {
        "vertex_count": int(len(V)),
        "face_count": int(len(F)),
        "surface_area": total,
        "max_face_area_fraction": float(order[0] / total),
        "top2_face_area_fraction": float(order[:2].sum() / total),
        "mesh_bbox_extent": [float(x) for x in bbox],
        "mesh_bbox_aspect_nonzero": aspect,
    }, area


def _triangle_skin_fraction(g: Any, faces: np.ndarray) -> tuple[np.ndarray | None, str]:
    if "skin" not in g.files:
        return None, "SKIN_ARRAY_MISSING"
    skin = np.asarray(g["skin"])
    if skin.ndim != 2 or skin.shape[0] != int(np.max(faces)) + 1 and skin.shape[0] < int(np.max(faces)) + 1:
        return None, f"SKIN_VERTEX_AXIS_NOT_COMPATIBLE:{skin.shape}"
    if skin.shape[0] < int(np.max(faces)) + 1:
        return None, f"SKIN_VERTEX_AXIS_TOO_SHORT:{skin.shape}"
    if not np.isfinite(skin).all():
        return None, "SKIN_NONFINITE"
    row_mass = np.clip(skin.astype(np.float64), 0.0, None).sum(axis=1)
    vertex_skinned = row_mass > 1e-8
    return vertex_skinned[np.asarray(faces, dtype=np.int64)].mean(axis=1).astype(np.float64), "VALID_DIAGNOSTIC_ONLY"


def _raster_metrics(
    raster_path: Path,
    face_count: int,
    tri_skin_fraction: np.ndarray | None,
    expected_size: int = RENDER_SIZE,
) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(raster_path, allow_pickle=False) as z:
        required = ("pixel_linear_index", "triangle_id", "resolution")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"raster missing {missing}: {raster_path}")
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tid = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
        res = np.asarray(z["resolution"], dtype=np.int64).reshape(-1)
    if len(res) < 2:
        raise ValueError("raster resolution malformed")
    H, W = int(res[0]), int(res[1])
    if (H, W) != (expected_size, expected_size):
        raise ValueError(f"raster resolution mismatch: {(H, W)}")
    if pix.shape != tid.shape or len(pix) == 0:
        raise ValueError("raster pixel/triangle arrays malformed or empty")
    if pix.min() < 0 or pix.max() >= H * W:
        raise ValueError("raster pixel index out of range")
    if tid.min() < 0 or tid.max() >= face_count:
        raise ValueError("raster triangle index out of range")
    mask = np.zeros(H * W, dtype=bool)
    mask[pix] = True
    if int(np.count_nonzero(mask)) != len(pix):
        raise ValueError("raster contains duplicate pixel_linear_index")
    mask = mask.reshape(H, W)
    mm = _mask_metrics(mask)
    counts = np.bincount(tid, minlength=face_count)
    vis = counts[counts > 0]
    vis_sorted = np.sort(vis)[::-1]
    n = float(len(pix))
    out = {
        **mm,
        "visible_triangle_count": int(len(vis)),
        "max_visible_triangle_pixel_fraction": float(vis_sorted[0] / n),
        "top2_visible_triangle_pixel_fraction": float(vis_sorted[:2].sum() / n),
    }
    if tri_skin_fraction is None:
        out["visible_skinned_triangle_fraction_valid"] = False
    else:
        sf = tri_skin_fraction[tid]
        out.update({
            "visible_skinned_triangle_fraction_valid": True,
            "visible_pixel_mean_triangle_skin_fraction": float(np.mean(sf)),
            "visible_fully_unskinned_pixel_fraction": float(np.mean(sf <= 0.0)),
        })
    return out, mask


def audit_asset_v1(inp: RenderPurityAssetInputV1) -> dict[str, Any]:
    aid = inp.canonical_asset_id
    row: dict[str, Any] = {
        "canonical_asset_id": aid,
        "source_registry_id": inp.source_registry_id,
        "arachne_structural_c0": bool(inp.arachne_structural_c0),
        "status": "FAIL",
        "errors": [],
    }
    try:
        gp = inp.asset_dir / "primary_geometry.npz"
        if not gp.is_file():
            raise FileNotFoundError(gp)
        with np.load(gp, allow_pickle=False) as g:
            if "vertices" not in g.files or "faces" not in g.files:
                raise KeyError("primary_geometry missing vertices/faces")
            vertices = np.asarray(g["vertices"])
            faces = np.asarray(g["faces"], dtype=np.int64)
            mesh, _ = _mesh_metrics(vertices, faces)
            tri_skin, skin_diag = _triangle_skin_fraction(g, faces)
        row["mesh"] = mesh
        row["skin_visibility_diagnostic"] = skin_diag
        views = []
        for v in range(VIEWS):
            vd = inp.asset_dir / "renders" / f"V{v}"
            rp = vd / "raster_authority.npz"
            ip = vd / "cel_clean_512.png"
            cp = vd / "camera.json"
            if not rp.is_file() or not ip.is_file() or not cp.is_file():
                raise FileNotFoundError(f"missing V{v} raster/image/camera for {aid}")
            camera = json.loads(cp.read_text(encoding="utf-8"))
            yaw = float(camera["yaw_deg"])
            expected_yaw = float(v * 45)
            if abs(yaw - expected_yaw) > 1e-4:
                raise ValueError(f"noncanonical yaw V{v}: {yaw} != {expected_yaw}")
            rm, _ = _raster_metrics(rp, len(faces), tri_skin)
            rm.update({
                "view_index": v,
                "yaw_deg": yaw,
                "cel_clean_512_exists": True,
                "image_content_decoded": False,
            })
            views.append(rm)
        row["views"] = views
        def vals(k: str) -> list[float]:
            return [float(x[k]) for x in views if k in x]
        row["aggregate"] = {
            "occupancy_fraction": _distribution(vals("occupancy_fraction")),
            "bbox_area_fraction": _distribution(vals("bbox_area_fraction")),
            "min_frame_margin_fraction": _distribution(vals("min_frame_margin_fraction")),
            "centroid_offset_fraction": _distribution(vals("centroid_offset_fraction")),
            "connected_component_count": _distribution(vals("connected_component_count")),
            "significant_component_count": _distribution(vals("significant_component_count")),
            "largest_component_fraction": _distribution(vals("largest_component_fraction")),
            "visible_triangle_count": _distribution(vals("visible_triangle_count")),
            "max_visible_triangle_pixel_fraction": _distribution(vals("max_visible_triangle_pixel_fraction")),
            "top2_visible_triangle_pixel_fraction": _distribution(vals("top2_visible_triangle_pixel_fraction")),
            "visible_pixel_mean_triangle_skin_fraction": _distribution(vals("visible_pixel_mean_triangle_skin_fraction")),
            "visible_fully_unskinned_pixel_fraction": _distribution(vals("visible_fully_unskinned_pixel_fraction")),
            "border_touch_view_count": int(sum(bool(x["border_touch"]) for x in views)),
        }
        row["status"] = "PASS"
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    return row
