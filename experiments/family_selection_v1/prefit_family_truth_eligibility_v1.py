from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.family_selection_v1.prefit_observation_authority_v1 import (
    ObservationViewEvidenceV1,
    verify_prefit_observation_authority_v1,
)
from experiments.single_family_e2e_v1.data_manifest_v1 import (
    MasterFamilyManifestV1,
    build_master_family_manifest_v1,
)

SCHEMA = "RealSaS.PrefitFamilyTruthEligibility.v1"
POLICY_ID = "FIT8_TRIANGLE_RASTER_TRUTH_ELIGIBILITY_V1"
RESOLUTION = 1024
MIN_VISIBLE_NORMAL_POLARITY_PURITY = 0.90
RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2 = 1.0
HULL_QUANTIZATION_PADDING_PX = 1
_TRIANGLE_BARYCENTRIC_SAMPLES = np.asarray([
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.60, 0.20, 0.20],
    [0.20, 0.60, 0.20],
    [0.20, 0.20, 0.60],
], dtype=np.float64)


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class _Camera:
    forward: np.ndarray
    right: np.ndarray
    screen_up: np.ndarray
    half_extent: float
    yaw_deg: float


@dataclass(frozen=True)
class ViewTruthQualityV1:
    view_index: int
    visible_triangle_count: int
    stable_visible_triangle_count: int
    raster_unstable_visible_triangle_count: int
    triangle_weighted_normal_polarity_purity: float
    pixel_weighted_normal_polarity_purity: float
    dominant_triangle_polarity: int
    dominant_pixel_polarity: int


@dataclass(frozen=True)
class PrefitFamilyTruthEligibilityV1:
    schema: str
    policy_id: str
    asset_id: str
    master_manifest_sha256: str
    observation_authority_sha256: str
    image_filename: str
    pass_prefit_truth_eligibility: bool
    observed_triangle_count: int
    observed_degenerate_triangle_count: int
    stable_triangle_view_pair_count: int
    raster_unstable_excluded_triangle_view_pair_count: int
    hull_miss_stable_triangle_view_pair_count: int
    fully_raster_unstable_observed_triangle_count: int
    minimum_triangle_weighted_normal_polarity_purity: float
    minimum_pixel_weighted_normal_polarity_purity: float
    views: tuple[ViewTruthQualityV1, ...]
    policy: dict
    scientific_fit_steps: int
    postfit_information_consumed: bool
    source_mesh_used_for_model_input: bool
    teacher_truth_used_for_model_input: bool
    eligibility_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load_camera(path: Path, expected_view: int) -> _Camera:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("contract") != "realsas.level_orthographic_z_orbit.v1":
        raise ValueError(f"CAMERA_CONTRACT_DRIFT:{path}")
    yaw = float(obj.get("yaw_deg", float("nan")))
    if not np.isfinite(yaw) or abs(yaw - 45.0 * expected_view) > 1e-6:
        raise ValueError(f"CAMERA_VIEW_ORDER_DRIFT:{path}:{yaw}")
    vec = {}
    for key in ("forward", "right", "screen_up"):
        value = np.asarray(obj.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all() or abs(float(np.linalg.norm(value)) - 1.0) > 1e-8:
            raise ValueError(f"CAMERA_BASIS_INVALID:{path}:{key}")
        vec[key] = value
    basis = np.stack([vec["right"], vec["screen_up"], vec["forward"]])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"CAMERA_BASIS_NOT_ORTHONORMAL:{path}")
    half_extent = float(obj.get("half_extent", 0.0))
    if not np.isfinite(half_extent) or half_extent <= 0.0:
        raise ValueError(f"CAMERA_HALF_EXTENT_INVALID:{path}")
    return _Camera(vec["forward"], vec["right"], vec["screen_up"], half_extent, yaw)


def _load_raster(path: Path, face_count: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        pix = np.asarray(data["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tri = np.asarray(data["triangle_id"], dtype=np.int64).reshape(-1)
        raw_res = np.asarray(data["resolution"], dtype=np.int64).reshape(-1)
    resolution = (int(raw_res[0]), int(raw_res[0])) if len(raw_res) == 1 else tuple(map(int, raw_res[:2])) if len(raw_res) >= 2 else ()
    if resolution != (RESOLUTION, RESOLUTION):
        raise ValueError(f"RASTER_RESOLUTION_DRIFT:{path}:{resolution}")
    if tri.shape != pix.shape or len(pix) == 0:
        raise ValueError(f"RASTER_SHAPE_OR_EMPTY:{path}")
    if pix.min() < 0 or pix.max() >= RESOLUTION * RESOLUTION or len(np.unique(pix)) != len(pix):
        raise ValueError(f"RASTER_PIXEL_AUTHORITY_INVALID:{path}")
    if tri.min() < 0 or tri.max() >= int(face_count):
        raise ValueError(f"RASTER_TRIANGLE_ID_INVALID:{path}")
    return pix, tri


def _raster_mask(pix: np.ndarray) -> np.ndarray:
    out = np.zeros(RESOLUTION * RESOLUTION, dtype=bool)
    out[pix] = True
    return out.reshape(RESOLUTION, RESOLUTION)


def _dilate_one(mask: np.ndarray) -> np.ndarray:
    if HULL_QUANTIZATION_PADDING_PX != 1:
        raise AssertionError("eligibility implementation sealed to one-pixel padding")
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, ((1, 1), (1, 1)), constant_values=False)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def _project_grid(points: np.ndarray, camera: _Camera) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    gx = np.tensordot(p, camera.right, axes=([-1], [0])) / camera.half_extent
    gy = -np.tensordot(p, camera.screen_up, axes=([-1], [0])) / camera.half_extent
    return np.stack([gx, gy], axis=-1)


def _grid_to_pixel_index(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    return np.stack([
        np.floor((g[..., 0] + 1.0) * 0.5 * RESOLUTION).astype(np.int64),
        np.floor((g[..., 1] + 1.0) * 0.5 * RESOLUTION).astype(np.int64),
    ], axis=-1)


def projected_triangle_area_px2(vertices_xyz: np.ndarray, camera: _Camera) -> np.ndarray:
    tri = np.asarray(vertices_xyz, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangle vertices must be [T,3,3]")
    xy = (_project_grid(tri, camera) + 1.0) * (0.5 * RESOLUTION)
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def normal_polarity_purity(signs: Iterable[float]) -> tuple[float, int]:
    a = np.asarray(tuple(float(x) for x in signs), dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        return 0.0, 0
    positive, negative = int(np.sum(a >= 0.0)), int(np.sum(a < 0.0))
    return (float(positive / len(a)), 1) if positive >= negative else (float(negative / len(a)), -1)


def _triangle_sample_points(vertices: np.ndarray, faces: np.ndarray, tid: int) -> np.ndarray:
    return _TRIANGLE_BARYCENTRIC_SAMPLES @ np.asarray(vertices[faces[int(tid)]], dtype=np.float64)


def _samples_inside_mask(points: np.ndarray, camera: _Camera, mask: np.ndarray) -> bool:
    pix = _grid_to_pixel_index(_project_grid(points, camera))
    x, y = pix[:, 0], pix[:, 1]
    inside = (x >= 0) & (x < RESOLUTION) & (y >= 0) & (y < RESOLUTION)
    return bool(np.all(inside) and np.all(mask[y, x]))


def evaluate_prefit_family_truth_eligibility_v1(asset_dir: str | Path, manifest: MasterFamilyManifestV1) -> PrefitFamilyTruthEligibilityV1:
    """Qualify one exact textured Master datum before model selection or fitting."""
    asset_dir = Path(asset_dir).resolve()
    rebound = build_master_family_manifest_v1(
        asset_dir,
        image_filename=manifest.image_filename,
        expected_file_hashes=dict(manifest.file_hashes),
        expected_manifest_hash=manifest.manifest_hash,
    )
    if rebound.asset_id != manifest.asset_id:
        raise ValueError("MASTER_MANIFEST_ASSET_ID_DRIFT")
    obs = verify_prefit_observation_authority_v1(
        asset_id=rebound.asset_id,
        raster_authority="MASTER_SOURCE_TEXTURED_RGBA",
        views=tuple(ObservationViewEvidenceV1(
            view_index=v.view_index,
            yaw_deg=int(round(v.yaw_deg)),
            rgba_path=str(asset_dir / v.image_relpath),
            rgba_sha256=v.image_sha256,
            camera_path=str(asset_dir / v.camera_relpath),
            camera_sha256=v.camera_sha256,
        ) for v in rebound.views),
    )

    with np.load(asset_dir / rebound.geometry_relpath, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("PRIMARY_GEOMETRY_SHAPE")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("PRIMARY_GEOMETRY_INVALID")

    tri_vertices = vertices[faces]
    cross = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    degenerate = cross_norm <= max(1e-15, scale * scale * 1e-12)
    unit_normal = np.zeros_like(cross)
    unit_normal[~degenerate] = cross[~degenerate] / cross_norm[~degenerate, None]

    cameras: list[_Camera] = []
    masks: list[np.ndarray] = []
    views: list[ViewTruthQualityV1] = []
    observed: set[int] = set()
    for binding in rebound.views:
        camera = _load_camera(asset_dir / binding.camera_relpath, binding.view_index)
        pix, tri = _load_raster(asset_dir / binding.raster_relpath, len(faces))
        mask = _dilate_one(_raster_mask(pix))
        cameras.append(camera); masks.append(mask)
        visible = np.unique(tri); observed.update(map(int, visible.tolist()))
        areas = projected_triangle_area_px2(tri_vertices[visible], camera)
        stable_mask = (~degenerate[visible]) & (areas > RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2)
        stable, unstable = visible[stable_mask], visible[~stable_mask]
        t_purity, t_pol = normal_polarity_purity(unit_normal[stable] @ camera.forward if len(stable) else ())
        stable_set = set(map(int, stable.tolist()))
        row_mask = np.asarray([int(tid) in stable_set for tid in tri], dtype=bool)
        p_purity, p_pol = normal_polarity_purity(unit_normal[tri[row_mask]] @ camera.forward if row_mask.any() else ())
        views.append(ViewTruthQualityV1(binding.view_index, int(len(visible)), int(len(stable)), int(len(unstable)), t_purity, p_purity, t_pol, p_pol))

    observed_ids = np.asarray(sorted(observed), dtype=np.int64)
    observed_degenerate = int(np.sum(degenerate[observed_ids])) if len(observed_ids) else 0
    stable_pairs = excluded_pairs = hull_miss_pairs = 0
    stable_per_triangle = {int(tid): 0 for tid in observed_ids.tolist()}
    for tid in observed_ids.tolist():
        samples = _triangle_sample_points(vertices, faces, tid)
        tri_batch = tri_vertices[[tid]]
        for camera, mask in zip(cameras, masks):
            area = float(projected_triangle_area_px2(tri_batch, camera)[0])
            if bool(degenerate[tid]) or area <= RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2:
                excluded_pairs += 1
                continue
            stable_pairs += 1; stable_per_triangle[int(tid)] += 1
            if not _samples_inside_mask(samples, camera, mask):
                hull_miss_pairs += 1
    fully_unstable = int(sum(v == 0 for v in stable_per_triangle.values()))
    min_t = min((v.triangle_weighted_normal_polarity_purity for v in views), default=0.0)
    min_p = min((v.pixel_weighted_normal_polarity_purity for v in views), default=0.0)
    passed = bool(
        len(observed_ids) > 0
        and observed_degenerate == 0
        and all(v.stable_visible_triangle_count > 0 for v in views)
        and hull_miss_pairs == 0
        and min_t >= MIN_VISIBLE_NORMAL_POLARITY_PURITY
        and min_p >= MIN_VISIBLE_NORMAL_POLARITY_PURITY
    )
    policy = {
        "normal_polarity_purity_min": MIN_VISIBLE_NORMAL_POLARITY_PURITY,
        "normal_polarity_gate": "PER_VIEW_MIN_OF_TRIANGLE_WEIGHTED_AND_PIXEL_WEIGHTED",
        "raster_unstable_projected_triangle_area_max_px2": RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2,
        "edge_on_policy": "EXCLUDE_TRIANGLE_VIEW_PAIR_WHEN_PROJECTED_AREA_LE_1PX2",
        "hull_quantization_padding_px": HULL_QUANTIZATION_PADDING_PX,
        "hull_requirement": "ZERO_MISSES_ON_ALL_STABLE_OBSERVED_TRIANGLE_VIEW_PAIRS",
        "triangle_samples_barycentric": _TRIANGLE_BARYCENTRIC_SAMPLES.tolist(),
        "degenerate_cross_norm_policy": "max(1e-15,bbox_max_extent^2*1e-12)",
        "fully_raster_unstable_observed_triangles": "TELEMETRY_ONLY_NOT_A_PASS_BLOCKER",
        "source_mesh_role": "PREFIT_DATA_QUALITY_ONLY__FORBIDDEN_AS_MODEL_INPUT",
    }
    hash_payload = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "asset_id": rebound.asset_id,
        "master_manifest_sha256": rebound.manifest_hash,
        "observation_authority_sha256": obs.observation_authority_sha256,
        "image_filename": rebound.image_filename,
        "pass_prefit_truth_eligibility": passed,
        "observed_triangle_count": int(len(observed_ids)),
        "observed_degenerate_triangle_count": observed_degenerate,
        "stable_triangle_view_pair_count": int(stable_pairs),
        "raster_unstable_excluded_triangle_view_pair_count": int(excluded_pairs),
        "hull_miss_stable_triangle_view_pair_count": int(hull_miss_pairs),
        "fully_raster_unstable_observed_triangle_count": fully_unstable,
        "minimum_triangle_weighted_normal_polarity_purity": float(min_t),
        "minimum_pixel_weighted_normal_polarity_purity": float(min_p),
        "views": [asdict(v) for v in views],
        "policy": policy,
        "scientific_fit_steps": 0,
        "postfit_information_consumed": False,
        "source_mesh_used_for_model_input": False,
        "teacher_truth_used_for_model_input": False,
    }
    return PrefitFamilyTruthEligibilityV1(
        schema=SCHEMA,
        policy_id=POLICY_ID,
        asset_id=rebound.asset_id,
        master_manifest_sha256=rebound.manifest_hash,
        observation_authority_sha256=obs.observation_authority_sha256,
        image_filename=rebound.image_filename,
        pass_prefit_truth_eligibility=passed,
        observed_triangle_count=int(len(observed_ids)),
        observed_degenerate_triangle_count=observed_degenerate,
        stable_triangle_view_pair_count=int(stable_pairs),
        raster_unstable_excluded_triangle_view_pair_count=int(excluded_pairs),
        hull_miss_stable_triangle_view_pair_count=int(hull_miss_pairs),
        fully_raster_unstable_observed_triangle_count=fully_unstable,
        minimum_triangle_weighted_normal_polarity_purity=float(min_t),
        minimum_pixel_weighted_normal_polarity_purity=float(min_p),
        views=tuple(views),
        policy=policy,
        scientific_fit_steps=0,
        postfit_information_consumed=False,
        source_mesh_used_for_model_input=False,
        teacher_truth_used_for_model_input=False,
        eligibility_sha256=_canonical_hash(hash_payload),
    )


__all__ = [
    "SCHEMA", "POLICY_ID", "MIN_VISIBLE_NORMAL_POLARITY_PURITY",
    "RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2", "HULL_QUANTIZATION_PADDING_PX",
    "ViewTruthQualityV1", "PrefitFamilyTruthEligibilityV1", "projected_triangle_area_px2",
    "normal_polarity_purity", "evaluate_prefit_family_truth_eligibility_v1",
]
