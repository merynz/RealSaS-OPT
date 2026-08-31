from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from gate0_geometry_v1 import OrthoCamera, grid_to_pixel, project_world

EXPECTED_MEMBERSHIP_RAW_SHA256 = "ae1edcaad47cf4867f11d51305eb57db06acb63ef13fc60ca3263c043ef8f71c"
EXPECTED_MEMBERSHIP_CONTENT_SHA256 = "305bcc2efc863c035ad7bce6e0afb29f2354ecc57abd1746e7b15b0485f71d24"
EXPECTED_TRAIN512_SET_SHA256 = "1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2"
EXPECTED_ASSET_COUNT = 512
RESOLUTION = 1024
PADDINGS = (0, 1, 2, 4, 8)
SPACINGS = (0.016, 0.008, 0.004)
UNIFORM_TRUTH_ROWS_PER_VIEW = 4096
BOUNDARY_TRUTH_ROWS_PER_VIEW = 2048
VOLUME_SAMPLES_PER_ASSET = 32768
VOLUME_MATERIAL_ACTIVE_FRACTION_MAX = 0.75
POINT_CHUNK = 65536


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def set_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in sorted(ids)).encode()).hexdigest()


def atomic_json(path: Path, o: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_name(path.name + ".tmp")
    q.write_text(json.dumps(o, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(path)


def seed_for(*parts: object) -> int:
    s = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(s).hexdigest()[:16], 16) & 0x7FFFFFFF


def normal_ci95(success: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = success / total
    se = math.sqrt(max(0.0, p * (1.0 - p) / total))
    return [float(max(0.0, p - 1.96 * se)), float(min(1.0, p + 1.96 * se))]


def image_foreground_mask(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.size != (RESOLUTION, RESOLUTION):
            raise ValueError(f"NATIVE_IMAGE_CONTRACT_DRIFT:{path}:{im.mode}:{im.size}")
        a = np.asarray(im, dtype=np.uint8)
    alpha = a[:, :, 3]
    if np.any(alpha < 255):
        return alpha >= 128, "ALPHA_THRESHOLD"
    green = np.asarray([0, 255, 0], dtype=np.uint8)
    return np.any(a[:, :, :3] != green, axis=2), "OPAQUE_EXACT_GREEN"


def raster_payload(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tri = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
        uv = np.asarray(z["barycentric_uv"], dtype=np.float64)
        rr = np.asarray(z["resolution"], dtype=np.int64).reshape(-1)
    good_res = (len(rr) >= 2 and tuple(map(int, rr[:2])) == (RESOLUTION, RESOLUTION)) or (len(rr) == 1 and int(rr[0]) == RESOLUTION)
    if not good_res:
        raise ValueError(f"RASTER_RESOLUTION_DRIFT:{path}:{rr.tolist()}")
    n = len(pix)
    if tri.shape != (n,) or uv.shape != (n, 2):
        raise ValueError(f"RASTER_ROW_SHAPE_DRIFT:{path}")
    if n == 0 or pix.min() < 0 or pix.max() >= RESOLUTION * RESOLUTION:
        raise ValueError(f"RASTER_PIXEL_ID_DRIFT:{path}")
    if len(np.unique(pix)) != n:
        raise ValueError(f"RASTER_DUPLICATE_PIXEL:{path}")
    return pix, tri, uv


def raster_mask_from_pix(pix: np.ndarray) -> np.ndarray:
    m = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
    m[np.asarray(pix, dtype=np.int64)] = True
    return m.reshape(RESOLUTION, RESOLUTION)


def four_neighbor_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
    interior4 = m & up & dn & lf & rt
    return m & ~interior4


def deterministic_truth_row_ids(aid: str, view: int, pix: np.ndarray, image_mask: np.ndarray) -> np.ndarray:
    n = len(pix)
    rng = np.random.default_rng(seed_for(aid, view, "gate0-real-truth-v1"))
    if n <= UNIFORM_TRUTH_ROWS_PER_VIEW:
        uniform = np.arange(n, dtype=np.int64)
    else:
        uniform = np.sort(rng.choice(n, size=UNIFORM_TRUTH_ROWS_PER_VIEW, replace=False)).astype(np.int64)

    boundary = four_neighbor_boundary(image_mask).reshape(-1)
    b_rows = np.flatnonzero(boundary[pix])
    if len(b_rows) > BOUNDARY_TRUTH_ROWS_PER_VIEW:
        b_rows = np.sort(rng.choice(b_rows, size=BOUNDARY_TRUTH_ROWS_PER_VIEW, replace=False)).astype(np.int64)
    ids = np.unique(np.concatenate([uniform, b_rows.astype(np.int64, copy=False)]))
    return ids


def reconstruct_rows(vertices: np.ndarray, faces: np.ndarray, tri: np.ndarray, uv: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    tt = tri[ids]
    if tt.min() < 0 or tt.max() >= len(faces):
        raise ValueError("TRIANGLE_ID_OUT_OF_RANGE")
    bc = np.stack([uv[ids, 0], uv[ids, 1], 1.0 - uv[ids, 0] - uv[ids, 1]], axis=1)
    return (vertices[faces[tt]] * bc[:, :, None]).sum(axis=1).astype(np.float64)


def padded_masks(mask: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for pad in PADDINGS:
        if pad == 0:
            out[pad] = mask.copy()
        else:
            out[pad] = maximum_filter(mask.astype(np.uint8), size=2 * pad + 1, mode="constant", cval=0) > 0
    return out


def project_pixels_all_views(points: np.ndarray, cameras: Sequence[OrthoCamera]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for cam in cameras:
        g, _ = project_world(points, cam)
        pix = grid_to_pixel(g, RESOLUTION)
        x, y = pix[:, 0], pix[:, 1]
        inside = (x >= 0) & (x < RESOLUTION) & (y >= 0) & (y < RESOLUTION)
        out.append((x, y, inside))
    return out


def membership_from_projected(projected: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], masks: Sequence[np.ndarray]) -> np.ndarray:
    if len(projected) != len(masks):
        raise ValueError("VIEW_COUNT")
    n = len(projected[0][0])
    keep = np.ones(n, dtype=bool)
    for (x, y, inside), mask in zip(projected, masks):
        ok = np.zeros(n, dtype=bool)
        ids = np.flatnonzero(inside)
        ok[ids] = mask[y[ids], x[ids]]
        keep &= ok
        if not keep.any():
            break
    return keep


def observed_medial_diameters_world(mask: np.ndarray, half_extent: float) -> np.ndarray:
    """Observed silhouette medial-thickness proxy frozen by prereg: 2*EDT at 3x3 local maxima."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        raise ValueError("EMPTY_FOREGROUND")
    edt = distance_transform_edt(m)
    locmax = m & (edt >= maximum_filter(edt, size=3, mode="constant", cval=0.0) - 1e-12)
    radii = edt[locmax]
    if radii.size == 0:
        raise ValueError("NO_MEDIAL_MAXIMA")
    px_to_world = 2.0 * float(half_extent) / float(RESOLUTION)
    return (2.0 * radii * px_to_world).astype(np.float32)


def lattice_axis_count(half_extent: float, spacing: float) -> int:
    return int(len(np.arange(-half_extent + spacing * 0.5, half_extent, spacing, dtype=np.float64)))


def _camera_from_file(path: Path, expected_view: int) -> OrthoCamera:
    o = json.loads(path.read_text(encoding="utf-8"))
    if abs(float(o.get("yaw_deg", -999.0)) - 45.0 * expected_view) > 1e-6:
        raise ValueError(f"CAMERA_YAW_DRIFT:{path}:{o.get('yaw_deg')}")
    return OrthoCamera.from_dict(o)


def _asset_worker(master_root: str, aid: str) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    root = Path(master_root)
    ad = root / "master" / "assets" / aid
    if not ad.is_dir():
        raise FileNotFoundError(f"ASSET_DIR_MISSING:{ad}")

    with np.load(ad / "primary_geometry.npz", allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("PRIMARY_GEOMETRY_SHAPE")

    cameras: list[OrthoCamera] = []
    masks: list[np.ndarray] = []
    padded_by_view: list[dict[int, np.ndarray]] = []
    truth_parts = []
    thickness_parts = []
    view_records = []

    for v in range(8):
        vd = ad / "renders" / f"V{v}"
        cam = _camera_from_file(vd / "camera.json", v)
        imask, fg_mode = image_foreground_mask(vd / "cel_clean.png")
        pix, tri, uv = raster_payload(vd / "raster_authority.npz")
        rmask = raster_mask_from_pix(pix)
        if not np.array_equal(imask, rmask):
            raise ValueError(f"IMAGE_RASTER_SUPPORT_MISMATCH:{aid}:V{v}")
        ids = deterministic_truth_row_ids(aid, v, pix, imask)
        truth_parts.append(reconstruct_rows(vertices, faces, tri, uv, ids))
        d = observed_medial_diameters_world(imask, cam.half_extent)
        thickness_parts.append(d)
        cameras.append(cam); masks.append(imask); padded_by_view.append(padded_masks(imask))
        view_records.append({
            "view": v,
            "yaw_deg": float(cam.yaw_deg),
            "half_extent": float(cam.half_extent),
            "foreground_mode": fg_mode,
            "foreground_pixels": int(imask.sum()),
            "sampled_truth_rows": int(len(ids)),
            "medial_proxy_count": int(len(d)),
            "medial_proxy_min_world": float(d.min()),
            "medial_proxy_p01_world": float(np.percentile(d, 1)),
            "medial_proxy_p05_world": float(np.percentile(d, 5)),
            "medial_proxy_p50_world": float(np.percentile(d, 50)),
        })

    h = float(cameras[0].half_extent)
    if any(abs(c.half_extent - h) > 1e-8 for c in cameras):
        raise ValueError(f"HALF_EXTENT_VIEW_DRIFT:{aid}")

    truth = np.concatenate(truth_parts, axis=0)
    truth_proj = project_pixels_all_views(truth, cameras)

    rng = np.random.default_rng(seed_for(aid, "gate0-real-volume-v1"))
    volume_points = rng.uniform(-h, h, size=(VOLUME_SAMPLES_PER_ASSET, 3)).astype(np.float64)
    volume_proj = project_pixels_all_views(volume_points, cameras)

    by_pad = {}
    for pad in PADDINGS:
        pm = [x[pad] for x in padded_by_view]
        truth_keep = membership_from_projected(truth_proj, pm)
        vol_keep = membership_from_projected(volume_proj, pm)
        success = int(vol_keep.sum())
        frac = float(success / len(vol_keep))
        spacing_counts = {}
        for s in SPACINGS:
            n_axis = lattice_axis_count(h, s)
            total = int(n_axis ** 3)
            spacing_counts[str(s)] = {
                "axis_count": n_axis,
                "total_dense_candidates": total,
                "estimated_active_candidates": int(round(total * frac)),
            }
        by_pad[str(pad)] = {
            "truth_sample_count": int(len(truth_keep)),
            "truth_miss_count": int((~truth_keep).sum()),
            "truth_containment": float(truth_keep.mean()),
            "volume_samples": int(len(vol_keep)),
            "active_volume_count": success,
            "active_volume_fraction": frac,
            "active_volume_fraction_ci95_normal": normal_ci95(success, len(vol_keep)),
            "spacing_candidate_accounting": spacing_counts,
        }

    diam = np.concatenate(thickness_parts).astype(np.float32, copy=False)
    strata = {}
    for s in SPACINGS:
        cells = diam.astype(np.float64) / s
        strata[str(s)] = {
            "lt_1": int(np.sum(cells < 1.0)),
            "ge_1_lt_2": int(np.sum((cells >= 1.0) & (cells < 2.0))),
            "ge_2_lt_4": int(np.sum((cells >= 2.0) & (cells < 4.0))),
            "ge_4": int(np.sum(cells >= 4.0)),
        }

    record = {
        "canonical_asset_id": aid,
        "status": "PASS_MEASURED",
        "half_extent": h,
        "sampled_truth_points": int(len(truth)),
        "views": view_records,
        "hull_frontier": by_pad,
        "thickness_proxy": {
            "definition": "2*EDT at 3x3 local maxima on inference foreground; observed silhouette proxy, not true 3D thickness",
            "count": int(len(diam)),
            "min_world": float(diam.min()),
            "p01_world": float(np.percentile(diam, 1)),
            "p05_world": float(np.percentile(diam, 5)),
            "p50_world": float(np.percentile(diam, 50)),
            "spacing_strata": strata,
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    return record, diam


def load_membership(path: Path, scientific: bool) -> tuple[dict, list[str]]:
    raw_sha = sha256_file(path)
    o = json.loads(path.read_text(encoding="utf-8"))
    ids = list(o.get("train_order_512", []))
    if scientific:
        if raw_sha != EXPECTED_MEMBERSHIP_RAW_SHA256:
            raise RuntimeError(f"MEMBERSHIP_RAW_SHA_DRIFT:{raw_sha}")
        if o.get("content_sha256") != EXPECTED_MEMBERSHIP_CONTENT_SHA256:
            raise RuntimeError("MEMBERSHIP_CONTENT_SHA_DRIFT")
        if len(ids) != EXPECTED_ASSET_COUNT or len(set(ids)) != EXPECTED_ASSET_COUNT:
            raise RuntimeError("TRAIN512_COUNT_OR_DUPLICATE_DRIFT")
        if set_sha(ids) != EXPECTED_TRAIN512_SET_SHA256:
            raise RuntimeError("TRAIN512_SET_SHA_DRIFT")
    if not ids:
        raise RuntimeError("EMPTY_MEMBERSHIP")
    return o, ids


def dist(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    return {
        "count": int(len(a)),
        "min": float(a.min()),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def run(master_root: Path, membership_path: Path, out_path: Path, workers: int, mode: str, smoke_count: int) -> dict:
    scientific = mode == "scientific"
    membership, ids = load_membership(membership_path, scientific=scientific)
    if not scientific:
        ids = ids[: max(1, int(smoke_count))]

    t0 = time.time(); records = []; failures = []; diameter_parts = []
    workers = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_asset_worker, str(master_root), aid): aid for aid in ids}
        for done, fut in enumerate(as_completed(futures), 1):
            aid = futures[fut]
            try:
                rec, diam = fut.result()
                records.append(rec); diameter_parts.append(diam)
            except Exception as exc:
                failures.append({"canonical_asset_id": aid, "error": f"{type(exc).__name__}: {exc}"})
            if done == 1 or done % 8 == 0 or done == len(ids):
                print(f"[real-gate0] {done}/{len(ids)} pass={len(records)} hard_fail={len(failures)} elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    records.sort(key=lambda x: x["canonical_asset_id"]); failures.sort(key=lambda x: x["canonical_asset_id"])

    total_truth_misses = {str(p): int(sum(r["hull_frontier"][str(p)]["truth_miss_count"] for r in records)) for p in PADDINGS}
    total_truth_samples = {str(p): int(sum(r["hull_frontier"][str(p)]["truth_sample_count"] for r in records)) for p in PADDINGS}
    volume_distributions = {str(p): dist([r["hull_frontier"][str(p)]["active_volume_fraction"] for r in records]) for p in PADDINGS}

    admissible_pad = None
    if scientific and not failures and len(records) == EXPECTED_ASSET_COUNT:
        for p in PADDINGS:
            if total_truth_misses[str(p)] == 0:
                admissible_pad = int(p); break
    material = None
    if admissible_pad is not None:
        material = bool(volume_distributions[str(admissible_pad)]["p50"] <= VOLUME_MATERIAL_ACTIVE_FRACTION_MAX)

    all_diam = np.concatenate(diameter_parts).astype(np.float64, copy=False) if diameter_parts else np.asarray([], dtype=np.float64)
    thickness_summary = {"world_diameter_distribution": dist(all_diam)}
    for s in SPACINGS:
        cells = all_diam / s if len(all_diam) else np.asarray([], dtype=np.float64)
        thickness_summary[str(s)] = {
            "cell_distribution": dist(cells),
            "bins": {
                "lt_1": int(np.sum(cells < 1.0)),
                "ge_1_lt_2": int(np.sum((cells >= 1.0) & (cells < 2.0))),
                "ge_2_lt_4": int(np.sum((cells >= 2.0) & (cells < 4.0))),
                "ge_4": int(np.sum(cells >= 4.0)),
            },
        }

    if scientific:
        status = "PASS_REAL_GATE0_MEASUREMENT_COMPLETE__SPACING_POLICY_FREEZE_NEXT" if not failures and len(records) == EXPECTED_ASSET_COUNT and admissible_pad is not None else "FAIL_REAL_GATE0_HARD_BLOCK"
    else:
        status = "SMOKE_ONLY__NO_SCIENTIFIC_INTERPRETATION"

    out = {
        "schema": "RealSaS.IRIS.ReprojectionV2.RealCorpusGate0Measurement.v1",
        "status": status,
        "mode": mode,
        "scientific_optimizer_steps": 0,
        "training_authorized": False,
        "population": {
            "requested_count": len(ids),
            "measured_count": len(records),
            "hard_failure_count": len(failures),
            "membership_file_sha256": sha256_file(membership_path),
            "membership_content_sha256": membership.get("content_sha256"),
            "measured_set_sha256": set_sha([r["canonical_asset_id"] for r in records]) if records else None,
            "expected_train512_set_sha256": EXPECTED_TRAIN512_SET_SHA256 if scientific else None,
        },
        "preregistered_constants": {
            "paddings_px": list(PADDINGS),
            "spacings_world": list(SPACINGS),
            "uniform_truth_rows_per_view": UNIFORM_TRUTH_ROWS_PER_VIEW,
            "boundary_truth_rows_per_view": BOUNDARY_TRUTH_ROWS_PER_VIEW,
            "volume_samples_per_asset": VOLUME_SAMPLES_PER_ASSET,
            "material_compute_active_fraction_median_max": VOLUME_MATERIAL_ACTIVE_FRACTION_MAX,
        },
        "hard_failures": failures,
        "aggregate": {
            "truth_sample_count_by_padding": total_truth_samples,
            "truth_miss_count_by_padding": total_truth_misses,
            "active_volume_fraction_distribution_by_padding": volume_distributions,
            "smallest_zero_sampled_truth_miss_padding_px": admissible_pad,
            "hull_compute_reduction_classification": ("MATERIAL_COMPUTE_REDUCTION" if material else "GEOMETRICALLY_ADMISSIBLE_BUT_NOT_MATERIAL_COMPUTE_REDUCTION") if admissible_pad is not None else "NO_ADMISSIBLE_PADDING",
            "thickness_proxy": thickness_summary,
            "elapsed_minutes": float((time.time() - t0) / 60.0),
        },
        "records": records,
    }
    out["content_sha256"] = canonical_sha(out)
    atomic_json(out_path, out)
    print(json.dumps({k: out[k] for k in ("schema", "status", "population", "aggregate", "content_sha256")}, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-root", type=Path, required=True, help="Path containing master/assets/<asset_id>")
    ap.add_argument("--membership-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    ap.add_argument("--smoke-count", type=int, default=2)
    args = ap.parse_args()
    result = run(args.master_root, args.membership_json, args.out, args.workers, args.mode, args.smoke_count)
    if args.mode == "scientific" and not result["status"].startswith("PASS_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
