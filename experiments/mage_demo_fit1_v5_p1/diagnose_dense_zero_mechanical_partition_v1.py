from __future__ import annotations

"""Exact self-hosted diagnostic for dense zero-surface mechanical partition coverage.

This experiment is intentionally product-input clean. It consumes only the sealed dense
zero-surface, current qualified S/G/W, exact cameras and exact source observations.
Source-component owner rasters, teacher partitions, P1/P1Q topology and semantic
filenames are forbidden. The report is diagnostic authority only; it does not mutate
or promote the product state.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.draw import polygon

from compiler.realsas_compiler_core.mechanical_component_partition import (
    derive_mechanical_component_partition,
)
from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import (
    project_dense_vertices,
    replay_dense_zero_surface_compaction,
)
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
)

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io


SCHEMA = "RealSaS.DenseZeroMechanicalPartitionCoverageDiagnostic.v1"
EXPECTED_ZERO_SURFACE_SHA256 = (
    "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
)


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_zero_surface(path: Path):
    actual = _sha(path)
    if actual != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError(f"DENSE_MECH_ZERO_SURFACE_SHA_DRIFT:{actual}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(
                f"DENSE_MECH_ZERO_SURFACE_PAYLOAD_DRIFT:{sorted(z.files)}"
            )
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise RuntimeError("DENSE_MECH_ZERO_SURFACE_SHAPE_INVALID")
    return vertices, faces


def _camera(path: Path, view: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("view_index", -1)) != int(view):
        raise RuntimeError(f"DENSE_MECH_CAMERA_VIEW_DRIFT:V{view}")
    return value


def _alpha(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if rgba.shape != (resolution, resolution, 4):
        raise RuntimeError(
            f"DENSE_MECH_OBSERVATION_SHAPE_DRIFT:{path}:{rgba.shape}"
        )
    return rgba[..., 3] >= 8


def _quality_mask(tri: np.ndarray) -> np.ndarray:
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )

    def angle(x, y, z):
        cosine = np.clip(
            (x * x + y * y - z * z) / np.maximum(2.0 * x * y, 1.0e-18),
            -1.0,
            1.0,
        )
        return np.degrees(np.arccos(cosine))

    min_angle = np.minimum.reduce(
        (angle(ab, ca, bc), angle(ab, bc, ca), angle(bc, ca, ab))
    )
    longest = np.maximum.reduce((ab, bc, ca))
    altitude = (2.0 * area) / np.maximum(longest, 1.0e-18)
    aspect = longest / np.maximum(altitude, 1.0e-18)
    return (
        (area > 1.0e-12)
        & (
            min_angle
            >= float(policy.min_raster_triangle_angle_deg) - 1.0e-12
        )
        & (
            aspect
            <= float(policy.max_raster_triangle_aspect_ratio) + 1.0e-12
        )
    )


def _coverage(authority: np.ndarray, predicted: np.ndarray) -> dict:
    foreground = int(np.count_nonzero(authority))
    predicted_count = int(np.count_nonzero(predicted))
    tp = int(np.count_nonzero(authority & predicted))
    fp = int(np.count_nonzero(predicted & ~authority))
    fn = int(np.count_nonzero(authority & ~predicted))
    recall = float(tp / foreground) if foreground else 1.0
    precision = float(tp / predicted_count) if predicted_count else 1.0
    denom = tp + fp + fn
    iou = float(tp / denom) if denom else 1.0

    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    missing = authority & ~predicted
    labels, count = ndimage.label(missing, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        largest_missing = int(sizes.max(initial=0))
    else:
        largest_missing = 0

    alpha_labels, alpha_count = ndimage.label(authority, structure=structure)
    large_threshold = (
        float(FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.large_alpha_component_min_fraction)
        * float(foreground)
    )
    large_recalls = []
    for label_id in range(1, int(alpha_count) + 1):
        mask = alpha_labels == label_id
        size = int(np.count_nonzero(mask))
        if size < large_threshold:
            continue
        large_recalls.append(
            float(np.count_nonzero(predicted & mask)) / float(size)
        )

    return {
        "foreground_pixel_count": foreground,
        "predicted_pixel_count": predicted_count,
        "true_positive_pixel_count": tp,
        "false_positive_pixel_count": fp,
        "false_negative_pixel_count": fn,
        "source_alpha_recall": recall,
        "precision_inside_alpha": precision,
        "alpha_iou": iou,
        "largest_uncovered_component_pixel_count": largest_missing,
        "largest_uncovered_component_fraction": (
            float(largest_missing) / float(foreground) if foreground else 0.0
        ),
        "large_alpha_component_recall": (
            min(large_recalls) if large_recalls else 1.0
        ),
        "large_alpha_component_count": len(large_recalls),
    }


def _policy_failures(report: dict) -> tuple[str, ...]:
    p = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    failures = []
    if report["source_alpha_recall"] + 1.0e-12 < p.min_source_alpha_recall:
        failures.append("source_alpha_recall")
    if report["precision_inside_alpha"] + 1.0e-12 < p.min_precision_inside_alpha:
        failures.append("precision_inside_alpha")
    if report["alpha_iou"] + 1.0e-12 < p.min_alpha_iou:
        failures.append("alpha_iou")
    if (
        report["largest_uncovered_component_fraction"]
        > p.max_largest_uncovered_component_fraction + 1.0e-12
    ):
        failures.append("largest_uncovered_component_fraction")
    if (
        report["large_alpha_component_recall"] + 1.0e-12
        < p.min_large_alpha_component_recall
    ):
        failures.append("large_alpha_component_recall")
    return tuple(failures)


def _rasterize_admitted(
    tri_xy: np.ndarray,
    face_ids: np.ndarray,
    component_index_by_dense: np.ndarray,
    dense_faces: np.ndarray,
    alpha: np.ndarray,
    component_names: tuple[str, ...],
):
    pure = np.zeros(alpha.shape, dtype=bool)
    full = np.zeros(alpha.shape, dtype=bool)
    component_masks = {
        name: np.zeros(alpha.shape, dtype=bool)
        for name in component_names
    }
    pure_counts = Counter()
    mixed_counts = Counter()
    alpha_rejected = 0
    admitted = 0
    admitted_pure = 0
    admitted_mixed = 0

    for raw_face_id in face_ids.tolist():
        face_id = int(raw_face_id)
        q = tri_xy[face_id]
        rr, cc = polygon(q[:, 1], q[:, 0], shape=alpha.shape)
        if len(rr) == 0 or not bool(np.all(alpha[rr, cc])):
            alpha_rejected += 1
            continue
        components = tuple(
            sorted(
                {
                    component_names[int(component_index_by_dense[int(vertex)])]
                    for vertex in dense_faces[face_id]
                }
            )
        )
        full[rr, cc] = True
        admitted += 1
        if len(components) == 1:
            pure[rr, cc] = True
            component_masks[components[0]][rr, cc] = True
            pure_counts[components[0]] += 1
            admitted_pure += 1
        else:
            mixed_counts[components] += 1
            admitted_mixed += 1

    return pure, full, component_masks, {
        "admitted_face_count": admitted,
        "admitted_pure_face_count": admitted_pure,
        "admitted_mixed_face_count": admitted_mixed,
        "alpha_rejected_face_count": alpha_rejected,
        "pure_component_face_counts": dict(sorted(pure_counts.items())),
        "mixed_component_face_counts": {
            "|".join(key): int(value)
            for key, value in sorted(mixed_counts.items())
        },
    }


def _largest_hole_attribution(
    *,
    alpha: np.ndarray,
    predicted: np.ndarray,
    projected_dense_xy: np.ndarray,
    component_index_by_dense: np.ndarray,
    component_names: tuple[str, ...],
    component_masks: dict[str, np.ndarray],
) -> dict:
    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    missing = alpha & ~predicted
    labels, count = ndimage.label(missing, structure=structure)
    if not count:
        return {
            "pixel_count": 0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "dense_vertex_component_counts": {},
            "boundary_component_pixel_counts": {},
        }
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(np.argmax(sizes))
    hole = labels == label_id
    ys, xs = np.nonzero(hole)
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )
    centroid = (float(xs.mean()), float(ys.mean()))

    rounded = np.rint(projected_dense_xy).astype(np.int64)
    valid = (
        np.isfinite(projected_dense_xy).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < alpha.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < alpha.shape[0])
    )
    dense_counts = Counter()
    for dense_index in np.flatnonzero(valid):
        x, y = rounded[int(dense_index)]
        if hole[int(y), int(x)]:
            dense_counts[
                component_names[
                    int(component_index_by_dense[int(dense_index)])
                ]
            ] += 1

    ring = ndimage.binary_dilation(
        hole,
        structure=structure,
        iterations=3,
    ) & (~hole)
    boundary_counts = {
        name: int(np.count_nonzero(ring & mask))
        for name, mask in component_masks.items()
    }
    boundary_counts = {
        key: value
        for key, value in boundary_counts.items()
        if value > 0
    }

    return {
        "pixel_count": int(sizes[label_id]),
        "bbox_xyxy": bbox,
        "centroid_xy": centroid,
        "dense_vertex_component_counts": dict(
            sorted(
                dense_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "boundary_component_pixel_counts": dict(
            sorted(
                boundary_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def _write_largest_hole_overlay(
    *,
    observation_path: Path,
    alpha: np.ndarray,
    predicted: np.ndarray,
    attribution: dict,
    output_path: Path,
) -> None:
    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    missing = alpha & ~predicted
    labels, count = ndimage.label(missing, structure=structure)
    with Image.open(observation_path) as source:
        image = source.convert("RGBA")
    rgba = np.asarray(image, dtype=np.uint8).copy()
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        label_id = int(np.argmax(sizes))
        hole = labels == label_id
        # Red tint = exact largest missing connected component.
        rgba[hole, 0] = 255
        rgba[hole, 1] = (rgba[hole, 1].astype(np.uint16) // 4).astype(np.uint8)
        rgba[hole, 2] = (rgba[hole, 2].astype(np.uint16) // 4).astype(np.uint8)
        rgba[hole, 3] = 255
    out = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(out)
    bbox = attribution.get("bbox_xyxy")
    if bbox:
        x0, y0, x1, y1 = map(int, bbox)
        for pad in range(3):
            draw.rectangle(
                (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                outline=(255, 255, 0, 255),
            )
        label = (
            f"LARGEST MISSING: {int(attribution.get('pixel_count', 0))} px | "
            f"{attribution.get('dense_vertex_component_counts', {})}"
        )
        tx = max(0, min(x0, out.width - 1))
        ty = max(0, y0 - 18)
        draw.rectangle(
            (tx, ty, min(out.width - 1, tx + 560), min(out.height - 1, ty + 16)),
            fill=(0, 0, 0, 220),
        )
        draw.text((tx + 3, ty + 2), label, fill=(255, 255, 0, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def run(args) -> dict:
    surface, skeleton, skin, _mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface),
        Path(args.skeleton),
        Path(args.fit2_skin),
    )
    partition = derive_mechanical_component_partition(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    dense_vertices, dense_faces = _load_zero_surface(Path(args.zero_surface))
    replay = replay_dense_zero_surface_compaction(
        surface,
        dense_vertices,
        source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256,
    )

    component_names = tuple(sorted(partition.component_surface_ids))
    component_index = {name: i for i, name in enumerate(component_names)}
    owner_by_surface = {
        str(row.surface_id): str(row.component_id)
        for row in partition.assignments
    }
    compact_component_index = np.asarray(
        [
            component_index[owner_by_surface[str(surface_id)]]
            for surface_id in replay.compact_surface_ids
        ],
        dtype=np.int16,
    )
    component_index_by_dense = compact_component_index[
        replay.dense_vertex_to_compact_index
    ]

    requested_views = tuple(int(x) for x in args.views)
    if not requested_views or any(view < 0 or view > 7 for view in requested_views):
        raise RuntimeError("DENSE_MECH_VIEW_SET_INVALID")

    rows = []
    resolution = int(dict(surface.metadata or {}).get("resolution", 0))
    for view in requested_views:
        camera = _camera(Path(args.cameras[view]), view)
        alpha = _alpha(Path(args.observations[view]), resolution)
        xy = project_dense_vertices(
            dense_vertices,
            camera,
            view_index=view,
        )
        tri_xy = xy[dense_faces]
        finite = np.isfinite(tri_xy).all(axis=(1, 2))
        in_frame = (
            (tri_xy[..., 0] >= 0.0)
            & (tri_xy[..., 0] <= float(resolution - 1))
            & (tri_xy[..., 1] >= 0.0)
            & (tri_xy[..., 1] <= float(resolution - 1))
        ).all(axis=1)
        quality = _quality_mask(tri_xy)
        pre_alpha = np.flatnonzero(finite & in_frame & quality)
        pre_alpha_clipped_frame = np.flatnonzero(finite & quality)

        pure_mask, full_mask, component_masks, counts = _rasterize_admitted(
            tri_xy,
            pre_alpha,
            component_index_by_dense,
            dense_faces,
            alpha,
            component_names,
        )
        pure_coverage = _coverage(alpha, pure_mask)
        full_coverage = _coverage(alpha, full_mask)

        (
            _clipped_pure_mask,
            clipped_full_mask,
            _clipped_component_masks,
            clipped_counts,
        ) = _rasterize_admitted(
            tri_xy,
            pre_alpha_clipped_frame,
            component_index_by_dense,
            dense_faces,
            alpha,
            component_names,
        )
        clipped_full_coverage = _coverage(alpha, clipped_full_mask)
        clipped_full_failures = _policy_failures(clipped_full_coverage)

        largest_hole = _largest_hole_attribution(
            alpha=alpha,
            predicted=full_mask,
            projected_dense_xy=xy,
            component_index_by_dense=component_index_by_dense,
            component_names=component_names,
            component_masks=component_masks,
        )
        overlay_path = Path(args.output).parent / (
            f"V{view}_LARGEST_HOLE_OVERLAY.png"
        )
        _write_largest_hole_overlay(
            observation_path=Path(args.observations[view]),
            alpha=alpha,
            predicted=full_mask,
            attribution=largest_hole,
            output_path=overlay_path,
        )
        largest_hole = {
            **largest_hole,
            "overlay_path": str(overlay_path),
        }
        pure_failures = _policy_failures(pure_coverage)
        full_failures = _policy_failures(full_coverage)
        row = {
            "view": view,
            "dense_face_universe_count": int(len(dense_faces)),
            "pre_alpha_quality_face_count": int(len(pre_alpha)),
            **counts,
            "pure_component_union_coverage": pure_coverage,
            "pure_component_union_failures": pure_failures,
            "pure_plus_mixed_union_coverage": full_coverage,
            "pure_plus_mixed_union_failures": full_failures,
            "pure_plus_mixed_full_frozen_coverage_pass": not full_failures,
            "largest_hole_attribution": largest_hole,
            "frame_clip_diagnostic": {
                "strict_in_frame_candidate_count": int(len(pre_alpha)),
                "clipped_raster_candidate_count": int(
                    len(pre_alpha_clipped_frame)
                ),
                "out_of_frame_quality_candidate_count": int(
                    len(pre_alpha_clipped_frame) - len(pre_alpha)
                ),
                "strict_in_frame_coverage": full_coverage,
                "clipped_raster_coverage": clipped_full_coverage,
                "clipped_raster_failures": clipped_full_failures,
                "clipped_raster_counts": clipped_counts,
            },
        }
        rows.append(row)
        print(
            "DENSE_MECH_V"
            + str(view)
            + "="
            + json.dumps(
                {
                    "pure_recall": pure_coverage["source_alpha_recall"],
                    "full_recall": full_coverage["source_alpha_recall"],
                    "full_precision": full_coverage["precision_inside_alpha"],
                    "full_iou": full_coverage["alpha_iou"],
                    "full_largest_hole": full_coverage[
                        "largest_uncovered_component_fraction"
                    ],
                    "mixed_faces": counts["admitted_mixed_face_count"],
                    "largest_hole_attribution": largest_hole,
                    "strict_failures": full_failures,
                    "frame_clip": {
                        "out_of_frame_quality_candidates": int(
                            len(pre_alpha_clipped_frame) - len(pre_alpha)
                        ),
                        "clipped_recall": clipped_full_coverage[
                            "source_alpha_recall"
                        ],
                        "clipped_precision": clipped_full_coverage[
                            "precision_inside_alpha"
                        ],
                        "clipped_iou": clipped_full_coverage["alpha_iou"],
                        "clipped_largest_hole": clipped_full_coverage[
                            "largest_uncovered_component_fraction"
                        ],
                        "clipped_failures": clipped_full_failures,
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema": SCHEMA,
        "status": "PASS__DENSE_ZERO_MECHANICAL_PARTITION_DIAGNOSTIC_COMPLETED",
        "zero_surface_sha256": EXPECTED_ZERO_SURFACE_SHA256,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "component_partition_hash": partition.partition_hash,
        "compaction_replay": replay.summary(),
        "policy": FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.to_dict(),
        "views": rows,
        "teacher_truth_used": False,
        "source_component_truth_used": False,
        "source_owner_raster_used": False,
        "p1_or_p1q_topology_used": False,
        "historical_full_subject_mesh_used": False,
        "dense_zero_surface_topology_used": True,
        "product_pass_claimed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DENSE_MECH_REPORT={output}", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--views", nargs="+", default=("2",))
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
