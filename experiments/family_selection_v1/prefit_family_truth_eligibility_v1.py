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

# Interior-only samples avoid turning exact silhouette-edge pixel-center convention
# into a geometry-quality failure. The policy is frozen before FIT8 selection.
_TRIANGLE_BARYCENTRIC_SAMPLES = np.asarray(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
    ],
    dtype=np.float64,
)


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


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
    vectors = {}
    for key in ("forward", "right", "screen_up"):
        v = np.asarray(obj.get(key), dtype=np.float64)
        if v.shape != (3,) or not np.isfinite(v).all() or abs(float(np.linalg.norm(v)) - 1.0) > 1e-8:
            raise ValueError(f"CAMERA_BASIS_INVALID:{path}:{key}")
        vectors[key] = v
    basis = np.stack([vectors["right"], vectors["screen_up"], vectors["forward"]], axis=0)
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"CAMERA_BASIS_NOT_ORTHONORMAL:{path}")
    half_extent = float(obj.get("half_extent", 0.0))
    if not np.isfinite(half_extent) or half_extent <= 0.0:
        raise ValueError(f"CAMERA_HALF_EXTENT_INVALID:{path}")
    return _Camera(vectors["forward"], vectors["right"], vectors["screen_up"], half_extent, yaw)


def _load_raster(path: Path, face_count: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        pix = np.asarray(data["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tri = np.asarray(data["triangle_id"], dtype=np.int64).reshape(-1)
        raw_res = np.asarray(data["resolution"], dtype=np.int64).reshape(-1)
    if len(raw_res) == 1:
        resolution = (int(raw_res[0]), int(raw_res[0]))
    elif len(raw_res) >= 2:
        resolution = (int(raw_res[0]), int(raw_res[1]))
    else:
        raise ValueError(f"RASTER_RESOLUTION_MISSING:{path}")
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
    out[np.asarray(pix, dtype=np.int64)] = True
    return out.reshape(RESOLUTION, RESOLUTION)


def _dilate_one(mask: np.ndarray) -> np.ndarray:
    if HULL_QUANTIZATION_PADDING_PX != 1:
        raise AssertionError("current eligibility implementation is sealed to one-pixel quantization padding")
    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def _project_grid(points: np.ndarray, camera: _Camera) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    gx = np.tensordot(p, camera.right, axes=([-1], [0])) / camera.half_extent
    gy = -np.tensordot(p, camera.screen_up, axes=([-1], [0])) / camera.half_extent
    return np.stack([gx, gy], axis=-1)


def _grid_to_pixel_index(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    x = np.floor((g[..., 0] + 1.0) * 0.5 * RESOLUTION).astype(np.int64)
    y = np.floor((g[..., 1] + 1.0) * 0.5 * RESOLUTION).astype(np.int64)
    return np.stack([x, y], axis=-1)


def projected_triangle_area_px2(vertices_xyz: np.ndarray, camera: _Camera) -> np.ndarray:
    tri = np.asarray(vertices_xyz, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangle vertices must be [T,3,3]")
    g = _project_grid(tri, camera)
    xy = (g + 1.0) * (0.5 * RESOLUTION)
    a = xy[:, 1] - xy[:, 0]
    b = xy[:, 2] - xy[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def normal_polarity_purity(signs: Iterable[float]) -> tuple[float, int]:
    a = np.asarray(tuple(float(x) for x in signs), dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        return 0.0, 0
    positive = int(np.sum(a >= 0.0))
    negative = int(np.sum(a < 0.0))
    if positive >= negative:
        return float(positive / len(a)), 1
    return float(negative / len(a)), -1


def _triangle_sample_points(vertices: np.ndarray, faces: np.ndarray, triangle_id: int) -> np.ndarray:
    tri = np.asarray(vertices[faces[int(triangle_id)]], dtype=np.float64)
    return _TRIANGLE_BARYCENTRIC_SAMPLES @ tri


def _samples_inside_mask(points: np.ndarray, camera: _Camera, mask: np.ndarray) -> bool:
    grid = _project_grid(points, camera)
    pix = _grid_to_pixel_index(grid)
    x = pix[:, 0]
    y = pix[:, 1]
    inside = (x >= 0) & (x < RESOLUTION) & (y >= 0) & (y < RESOLUTION)
    if not bool(np.all(inside)):
        return False
    return bool(np.all(mask[y, x]))


def evaluate_prefit_family_truth_eligibility_v1(
    asset_dir: str | Path,
    manifest: MasterFamilyManifestV1,
) -> PrefitFamilyTruthEligibilityV1:
    """Measure pre-fit truth/raster suitability for one exact textured Master family.

    Source geometry and raster triangle IDs are used only to qualify the training
    datum before selection. They are not emitted as learned-model inputs and do not
    authorize source-mesh topology at inference.
    """
    asset_dir = Path(asset_dir).resolve()
    rebound = build_master_family_manifest_v1(
        asset_dir,
        image_filename=manifest.image_filename,
        expected_file_hashes=dict(manifest.file_hashes),
        expected_manifest_hash=manifest.manifest_hash,
    )
    if rebound.asset_id != manifest.asset_id:
        raise ValueError("MASTER_MANIFEST_ASSET_ID_DRIFT")

    observation_rows = tuple(
        ObservationViewEvidenceV1(
            view_index=view.view_index,
            yaw_deg=int(round(view.yaw_deg)),
            rgba_path=str(asset_dir / view.image_relpath),
            rgba_sha256=view.image_sha256,
            camera_path=str(asset_dir / view.camera_relpath),
            camera_sha256=view.camera_sha256,
        )
        for view in rebound.views
    )
    observation_authority = verify_prefit_observation_authority_v1(
        asset_id=rebound.asset_id,
        raster_authority="MASTER_SOURCE_TEXTURED_RGBA",
        views=observation_rows,
    )

    with np.load(asset_dir / rebound.geometry_relpath, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("PRIMARY_GEOMETRY_SHAPE")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError("PRIMARY_GEOMETRY_EMPTY_OR_NONFINITE")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("PRIMARY_GEOMETRY_FACE_INDEX_INVALID")

    tri_vertices = vertices[faces]
    cross = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    degenerate_cross_norm_max = max(1e-15, (scale * scale) * 1e-12)
    degenerate = cross_norm <= degenerate_cross_norm_max
    unit_normal = np.zeros_like(cross)
    good = ~degenerate
    unit_normal[good] = cross[good] / cross_norm[good, None]

    cameras: list[_Camera] = []
    raster_masks: list[np.ndarray] = []
    raster_triangles: list[np.ndarray] = []
    view_quality: list[ViewTruthQualityV1] = []
    observed: set[int] = set()

    for binding in rebound.views:
        camera = _load_camera(asset_dir / binding.camera_relpath, binding.view_index)
        pix, tri = _load_raster(asset_dir / binding.raster_relpath, len(faces))
        cameras.append(camera)
        raster_masks.append(_dilate_one(_raster_mask(pix)))
        raster_triangles.append(tri)
        visible = np.unique(tri)
        observed.update(map(int, visible.tolist()))
        areas = projected_triangle_area_px2(tri_vertices[visible], camera)
        stable_mask = (~degenerate[visible]) & (areas > RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2)
        stable = visible[stable_mask]
        unstable = visible[~stable_mask]
        tri_sign = unit_normal[stable] @ camera.forward if len(stable) else np.asarray([], dtype=np.float64)
        triangle_purity, triangle_polarity = normal_polarity_purity(tri_sign)

        stable_set = set(map(int, stable.tolist()))
        pixel_ids = np.asarray([i for i, tid in enumerate(tri.tolist()) if int(tid) in stable_set], dtype=np.int64)
        pixel_sign = unit_normal[tri[pixel_ids]] @ camera.forward if len(pixel_ids) else np.asarray([], dtype=np.float64)
        pixel_purity, pixel_polarity = normal_polarity_purity(pixel_sign)
        view_quality.append(ViewTruthQualityV1(
            view_index=binding.view_index,
            visible_triangle_count=int(len(visible)),
            stable_visible_triangle_count=int(len(stable)),
            raster_unstable_visible_triangle_count=int(len(unstable)),
            triangle_weighted_normal_polarity_purity=triangle_purity,
            pixel_weighted_normal_polarity_purity=pixel_purity,
            dominant_triangle_polarity=triangle_polarity,
            dominant_pixel_polarity=pixel_polarity,
        ))

    observed_ids = np.asarray(sorted(observed), dtype=np.int64)
    observed_degenerate = int(np.sum(degenerate[observed_ids])) if len(observed_ids) else 0
    stable_pairs = 0
    excluded_pairs = 0
    hull_miss_pairs = 0
    stable_count_by_triangle = {int(tid): 0 for tid in observed_ids.tolist()}
    for tid in observed_ids.tolist():
        samples = _triangle_sample_points(vertices, faces, tid)
        tri_batch = tri_vertices[np.asarray([tid], dtype=np.int64)]
        for camera, mask in zip(cameras, raster_masks):
            area = float(projected_triangle_area_px2(tri_batch, camera)[0])
            if bool(degenerate[tid]) or area <= RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2:
                excluded_pairs += 1
                continue
            stable_pairs += 1
            stable_count_by_triangle[int(tid)] += 1
            if not _samples_inside_mask(samples, camera, mask):
                hull_miss_pairs += 1
    fully_unstable = int(sum(count == 0 for count in stable_count_by_triangle.values()))

    min_triangle_purity = min((v.triangle_weighted_normal_polarity_purity for v in view_quality), default=0.0)
    min_pixel_purity = min((v.pixel_weighted_normal_polarity_purity for v in view_quality), default=0.0)
    every_view_has_stable_visible_truth = all(v.stable_visible_triangle_count > 0 for v in view_quality)
    passed = bool(
        len(observed_ids) > 0
        and observed_degenerate == 0
        and every_view_has_stable_visible_truth
        and hull_miss_pairs == 0
        and min_triangle_purity >= MIN_VISIBLE_NORMAL_POLARITY_PURITY
        and min_pixel_purity >= MIN_VISIBLE_NORMAL_POLARITY_PURITY
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
    payload = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "asset_id": rebound.asset_id,
        "master_manifest_sha256": rebound.manifest_hash,
        "observation_authority_sha256": observation_authority.observation_authority_sha256,
        "image_filename": rebound.image_filename,
        "pass_prefit_truth_eligibility": passed,
        "observed_triangle_count": int(len(observed_ids)),
        "observed_degenerate_triangle_count": observed_degenerate,
        "stable_triangle_view_pair_count": int(stable_pairs),
        "raster_unstable_excluded_triangle_view_pair_count": int(excluded_pairs),
        "hull_miss_stable_triangle_view_pair_count": int(hull_miss_pairs),
        "fully_raster_unstable_observed_triangle_count": fully_unstable,
        "minimum_triangle_weighted_normal_polarity_purity": float(min_triangle_purity),
        "minimum_pixel_weighted_normal_polarity_purity": float(min_pixel_purity),
        "views": [asdict(v) for v in view_quality],
        "policy": policy,
        "scientific_fit_steps": 0,
        "postfit_information_consumed": False,
        "source_mesh_used_for_model_input": False,
        "teacher_truth_used_for_model_input": False,
    }
    return PrefitFamilyTruthEligibilityV1(
        **payload,
        views=tuple(view_quality),
        eligibility_sha256=_canonical_hash(payload),
    )


__all__ = [
    "SCHEMA",
    "POLICY_ID",
    "MIN_VISIBLE_NORMAL_POLARITY_PURITY",
    "RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2",
    "HULL_QUANTIZATION_PADDING_PX",
    "ViewTruthQualityV1",
    "PrefitFamilyTruthEligibilityV1",
    "projected_triangle_area_px2",
    "normal_polarity_purity",
    "evaluate_prefit_family_truth_eligibility_v1",
]
