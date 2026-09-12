from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.draw import polygon

from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    _connected_components,
    _contract_boundary_recovery_vertices,
    _dedupe_projected_ids,
    _convex_hull_ids,
    _match_kernel_vertex,
    _oriented_triangle,
    _point_on_original_hull_segment,
    _safe_surface_graph,
    _signed_area2,
    _visible_subsets_of_full_components,
    build_mwb2_observation_cdt_candidate,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.quality import FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
from compiler.realsas_compiler_core.mesh.types import QualificationError, RiggingSurfaceIR

from experiments.mage_full_subject_reclosure_v1.run_frozen_geppetto_replay_v2 import (
    ACTIVE_GSA_EDGE_COUNT,
    ACTIVE_GSA_LINEAGE,
    ACTIVE_GSA_NODE_COUNT,
    _build_active_surface,
    _load_cameras,
    _load_observation_masks,
)

ZERO_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
IRIS_RUN_ID = "20260912T074348Z"
IRIS_CHECKPOINT_SHA256 = "76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5"
CAMERA_SHA256 = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)
OBSERVATION_SHA256 = (
    "8e9875c16bba3047c8f2fc211b984f2399d1145f5720c7427c6fc03a3fec0616",
    "fa94283780d5e5f3ad3943bbffc1f0592a70fc362fdd03b94a1d1cb46889dc30",
    "777bf4f7c505b405d3a5d2a111b2f297949454f77cae7c33064bf02479d384a5",
    "2a771c91c1d1c0b75dab14f6e98b7905a8429bbf0c0a8dd690261f93f6476d86",
    "354bb239feb6c1a515917fee4efb42fb1e0cb9f173d0eb3191fb0901a02a6228",
    "158fc14a75aa69f6133f2ba3ec5df2ad146afc62f477d7d3b4a41ca2cb0e42f2",
    "e3b08836c187863819d8b8aaa53aef79eddfee167e1fabf3fb1dd8ec0484563a",
    "87d4ad46edffec3c6bff034d305194820288d3cc6ef9a9480ac2234fb56451bc",
)


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _assert_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label}_MISSING:{path}")
    got = _sha_file(path)
    if got != expected:
        raise RuntimeError(f"{label}_SHA_DRIFT:{got}:{expected}")


def _mask_from_triangles(triangles, *, width: int, height: int) -> np.ndarray:
    out = np.zeros((int(height), int(width)), dtype=np.bool_)
    for tri in triangles:
        rr, cc = polygon(
            [float(p[1]) for p in tri],
            [float(p[0]) for p in tri],
            shape=out.shape,
        )
        out[rr, cc] = True
    return out


def _candidate_triangles(candidate) -> tuple:
    by_id = {
        str(vertex.candidate_vertex_id): tuple(map(float, vertex.metadata["raster_xy"]))
        for vertex in candidate.vertices
    }
    return tuple(tuple(by_id[str(vertex_id)] for vertex_id in face) for face in candidate.faces)


def _mask_metrics(authority: np.ndarray, predicted: np.ndarray) -> dict:
    authority = np.asarray(authority, dtype=np.bool_)
    predicted = np.asarray(predicted, dtype=np.bool_)
    if authority.shape != predicted.shape:
        raise ValueError("mask shape mismatch")

    inside = int(np.count_nonzero(authority & predicted))
    foreground = int(np.count_nonzero(authority))
    predicted_count = int(np.count_nonzero(predicted))
    precision = 1.0 if predicted_count == 0 else float(inside) / float(predicted_count)
    recall = 1.0 if foreground == 0 else float(inside) / float(foreground)
    union = foreground + predicted_count - inside
    iou = 1.0 if union == 0 else float(inside) / float(union)
    f1 = 1.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(authority, structure=structure)
    component_rows = []
    for component_id in range(1, int(count) + 1):
        mask = labels == component_id
        size = int(np.count_nonzero(mask))
        covered = int(np.count_nonzero(mask & predicted))
        component_rows.append((size, covered))
    component_rows.sort(key=lambda row: (-row[0], -row[1]))
    foreground_components = tuple(
        {
            "component_index": int(i),
            "foreground_pixel_count": int(size),
            "covered_pixel_count": int(covered),
            "foreground_fraction": 0.0 if foreground == 0 else float(size) / float(foreground),
            "recall": 1.0 if size == 0 else float(covered) / float(size),
        }
        for i, (size, covered) in enumerate(component_rows[:64])
    )

    uncovered = authority & ~predicted
    uncovered_labels, uncovered_count = ndimage.label(uncovered, structure=structure)
    if int(uncovered_count):
        sizes = np.bincount(uncovered_labels.ravel())[1:]
        largest = int(sizes.max()) if sizes.size else 0
    else:
        largest = 0

    return {
        "predicted_pixel_count": predicted_count,
        "inside_alpha_pixel_count": inside,
        "foreground_pixel_count": foreground,
        "precision_inside_alpha": float(precision),
        "source_alpha_recall": float(recall),
        "alpha_iou": float(iou),
        "alpha_f1": float(f1),
        "foreground_component_count": int(count),
        "foreground_component_recalls": foreground_components,
        "uncovered_component_count": int(uncovered_count),
        "largest_uncovered_component_pixels": int(largest),
        "largest_uncovered_component_fraction": (
            0.0 if foreground == 0 else float(largest) / float(foreground)
        ),
    }


def _coverage_failures(coverage: dict) -> tuple[str, ...]:
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    failures = []
    if float(coverage["source_alpha_recall"]) < float(policy.min_source_alpha_recall):
        failures.append("source_alpha_recall")
    if float(coverage["precision_inside_alpha"]) < float(policy.min_precision_inside_alpha):
        failures.append("precision_inside_alpha")
    if float(coverage["alpha_iou"]) < float(policy.min_alpha_iou):
        failures.append("alpha_iou")
    if float(coverage["largest_uncovered_component_fraction"]) > float(
        policy.max_largest_uncovered_component_fraction
    ):
        failures.append("largest_uncovered_component_fraction")
    eligible = [
        row
        for row in coverage["foreground_component_recalls"]
        if float(row["foreground_fraction"]) >= float(policy.large_alpha_component_min_fraction)
    ]
    if eligible and min(float(row["recall"]) for row in eligible) < float(
        policy.min_large_alpha_component_recall
    ):
        failures.append("large_alpha_component_recall")
    return tuple(failures)


@dataclass(frozen=True)
class SupportedKernelResult:
    triangles: tuple
    component_count: int
    kernel_triangle_count: int
    post_contraction_triangle_count: int
    partial_alpha_triangle_count: int
    fully_inside_alpha_triangle_count: int
    fully_outside_alpha_triangle_count: int


def _supported_pre_alpha_triangles(
    surface: RiggingSurfaceIR,
    *,
    view_index: int,
    observation_domain: ObservationRasterDomain,
    min_face_area_grid: float = 1.0e-8,
    min_component_nodes: int = 3,
) -> SupportedKernelResult:
    nodes, visible, safe_edges, _ = _safe_surface_graph(surface, int(view_index))
    full_components = _connected_components(set(nodes), safe_edges)
    components = _visible_subsets_of_full_components(full_components, visible)

    triangles = []
    kernel_triangle_count = 0
    post_contraction_triangle_count = 0
    cdt_component_count = 0

    for component in components:
        deduped = _dedupe_projected_ids(component, visible)
        if len(deduped) < int(min_component_nodes):
            continue
        hull_ids = _convex_hull_ids(deduped, visible)
        if len(hull_ids) < 3:
            continue
        hull_set = set(hull_ids)
        support_ids = tuple(sid for sid in deduped if sid not in hull_set)
        result = triangulate_production_cdt(
            [visible[sid] for sid in hull_ids],
            support_points=[visible[sid] for sid in support_ids],
            target_min_angle_deg=0.0,
            max_boundary_vertices=max(256, len(hull_ids) + 16),
            max_support_vertices=max(128, len(support_ids) + 16),
            max_constraint_recovery_iterations=96,
            max_quality_iterations=0,
            min_feature_spacing=1.0e-8,
        )
        if not bool(result.success):
            raise QualificationError(f"LEGAL_STEINER_CEILING_CDT_KERNEL_FAILURE:{result.reason}")
        if int(getattr(result, "quality_insert_count", 0)) != 0 or int(
            getattr(result, "inserted_steiner_count", 0)
        ) != 0:
            raise QualificationError("LEGAL_STEINER_CEILING_UNBOUND_QUALITY_STEINER_PRESENT")

        kernel_triangle_count += len(result.triangles)
        cdt_component_count += 1
        kernel_ids = []
        generated_indices = []
        for vertex_index, point in enumerate(result.vertices):
            sid = _match_kernel_vertex(point, deduped, visible)
            if sid is None:
                if _point_on_original_hull_segment(point, hull_ids, visible) is None:
                    raise QualificationError("LEGAL_STEINER_CEILING_UNSUPPORTED_GENERATED_VERTEX")
                generated_indices.append(int(vertex_index))
            kernel_ids.append(sid)
        if generated_indices:
            if len(generated_indices) != int(result.constraint_split_count):
                raise QualificationError("LEGAL_STEINER_CEILING_BOUNDARY_SPLIT_COUNT_MISMATCH")
            kernel_triangles = _contract_boundary_recovery_vertices(result, tuple(generated_indices))
        else:
            kernel_triangles = tuple(result.triangles)
        post_contraction_triangle_count += len(kernel_triangles)

        matched = [sid for sid in kernel_ids if sid is not None]
        if len(set(matched)) != len(matched):
            raise QualificationError("LEGAL_STEINER_CEILING_VERTEX_COLLAPSE_AFTER_BINDING")

        for ia, ib, ic in kernel_triangles:
            tri_raw = (kernel_ids[int(ia)], kernel_ids[int(ib)], kernel_ids[int(ic)])
            if any(sid is None for sid in tri_raw):
                raise QualificationError("LEGAL_STEINER_CEILING_UNSUPPORTED_VERTEX_SURVIVED_CONTRACTION")
            tri_ids = tuple(map(str, tri_raw))
            if len(set(tri_ids)) != 3:
                continue
            tri_ids = _oriented_triangle(tri_ids, visible)
            tri = tuple(visible[sid] for sid in tri_ids)
            if 0.5 * abs(_signed_area2(*tri)) < float(min_face_area_grid):
                continue
            triangles.append(tuple(tuple(map(float, p)) for p in tri))

    unique = []
    seen = set()
    for tri in triangles:
        key = tuple(sorted((round(p[0], 10), round(p[1], 10)) for p in tri))
        if key in seen:
            continue
        seen.add(key)
        unique.append(tri)

    authority = np.frombuffer(observation_domain.mask_bytes, dtype=np.uint8).reshape(
        observation_domain.height, observation_domain.width
    ) != 0
    fully_inside = partial = fully_outside = 0
    for tri in unique:
        rr, cc = polygon(
            [float(p[1]) for p in tri],
            [float(p[0]) for p in tri],
            shape=authority.shape,
        )
        if len(rr) == 0:
            fully_outside += 1
            continue
        inside_count = int(np.count_nonzero(authority[rr, cc]))
        if inside_count == 0:
            fully_outside += 1
        elif inside_count == len(rr):
            fully_inside += 1
        else:
            partial += 1

    return SupportedKernelResult(
        triangles=tuple(unique),
        component_count=int(cdt_component_count),
        kernel_triangle_count=int(kernel_triangle_count),
        post_contraction_triangle_count=int(post_contraction_triangle_count),
        partial_alpha_triangle_count=int(partial),
        fully_inside_alpha_triangle_count=int(fully_inside),
        fully_outside_alpha_triangle_count=int(fully_outside),
    )


def _checker_background(width: int, height: int, cell: int = 32) -> Image.Image:
    yy, xx = np.indices((height, width))
    blocks = ((xx // int(cell)) + (yy // int(cell))) % 2
    arr = np.where(blocks[..., None] == 0, 54, 72).astype(np.uint8)
    arr = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr
    return Image.fromarray(arr, "RGB")


def _masked_source(source: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    rgba[..., 3] = np.where(mask, rgba[..., 3], 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _on_checker(image: Image.Image) -> Image.Image:
    checker = _checker_background(image.width, image.height).convert("RGBA")
    return Image.alpha_composite(checker, image.convert("RGBA")).convert("RGB")


def _label(image: Image.Image, text: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0))
    draw.text((10, 11), text, fill=(255, 255, 255))
    return image


def _residual_panel(source: Image.Image, recoverable: np.ndarray, unreachable: np.ndarray, view: int) -> Image.Image:
    base = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    # Dim real source so spatial context remains visible; amber=recoverable, magenta=unreachable.
    base[..., :3] = (base[..., :3].astype(np.float32) * 0.22).astype(np.uint8)
    overlay = np.zeros_like(base)
    overlay[recoverable] = np.array([255, 170, 0, 235], dtype=np.uint8)
    overlay[unreachable] = np.array([255, 0, 220, 245], dtype=np.uint8)
    fg = Image.alpha_composite(Image.fromarray(base, "RGBA"), Image.fromarray(overlay, "RGBA"))
    return _label(fg, f"V{view} AMBER=recoverable | MAGENTA=unreachable")


def _exact_baseline_checks(candidate, expected: dict) -> dict:
    residual = candidate.residual_report
    checks = {
        "vertex_count": len(candidate.vertices) == int(expected["vertex_count"]),
        "edge_count": len(candidate.edges) == int(expected["edge_count"]),
        "face_count": len(candidate.faces) == int(expected["face_count"]),
        "kernel_triangle_count": int(residual["kernel_triangle_count"]) == int(expected["kernel_triangle_count"]),
        "alpha_rejected_face_count": int(residual["alpha_rejected_face_count"]) == int(expected["alpha_rejected_face_count"]),
        "kernel_constraint_split_count": int(residual["kernel_constraint_split_count"]) == int(expected["kernel_constraint_split_count"]),
        "contracted_boundary_recovery_vertex_count": int(residual["contracted_boundary_recovery_vertex_count"]) == int(expected["contracted_boundary_recovery_vertex_count"]),
        "post_contraction_triangle_count": int(residual["post_contraction_triangle_count"]) == int(expected["post_contraction_triangle_count"]),
        "predicted_pixel_count": int(residual["predicted_pixel_count"]) == int(expected["predicted_pixel_count"]),
        "foreground_pixel_count": int(residual["foreground_pixel_count"]) == int(expected["foreground_pixel_count"]),
        "visible_surface_node_count": int(residual["visible_surface_node_count"]) == int(expected["visible_surface_node_count"]),
        "recall": abs(float(residual["source_alpha_recall"]) - float(expected["source_alpha_recall"])) <= 1.0e-12,
        "precision": abs(float(residual["precision_inside_alpha"]) - float(expected["precision_inside_alpha"])) <= 1.0e-12,
    }
    if not all(checks.values()):
        raise RuntimeError("LEGAL_STEINER_BASELINE_REPLAY_DRIFT:" + json.dumps(checks, sort_keys=True))
    return checks


def run(args) -> dict:
    zero = Path(args.zero_surface)
    cameras = tuple(Path(p) for p in args.cameras)
    observations = tuple(Path(p) for p in args.observations)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(cameras) != 8 or len(observations) != 8:
        raise ValueError("exactly 8 cameras and 8 observations are required")

    _assert_sha(zero, ZERO_SHA256, "ZERO_SURFACE")
    for view, path in enumerate(cameras):
        _assert_sha(path, CAMERA_SHA256[view], f"CAMERA_V{view}")
    for view, path in enumerate(observations):
        _assert_sha(path, OBSERVATION_SHA256[view], f"OBSERVATION_V{view}")

    camera_ir = _load_cameras(cameras)
    observation_masks = _load_observation_masks(observations)
    surface_args = SimpleNamespace(
        zero_surface=str(zero),
        iris_run_id=IRIS_RUN_ID,
        iris_checkpoint_sha256=IRIS_CHECKPOINT_SHA256,
    )
    surface, tensor = _build_active_surface(surface_args, camera_ir, observation_masks)
    if surface.geometry_lineage_hash != ACTIVE_GSA_LINEAGE:
        raise RuntimeError("LEGAL_STEINER_GSA_LINEAGE_DRIFT")
    if int(tensor.node_count) != int(ACTIVE_GSA_NODE_COUNT) or int(tensor.edge_count) != int(ACTIVE_GSA_EDGE_COUNT):
        raise RuntimeError("LEGAL_STEINER_GSA_TENSOR_SHAPE_DRIFT")

    report_path = Path(__file__).with_name("MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json")
    sealed = json.loads(report_path.read_text(encoding="utf-8"))
    expected_by_view = {int(row["view"]): row for row in sealed["views"]}

    rows = []
    contact_rows = []
    for view, obs_path in enumerate(observations):
        with Image.open(obs_path) as im:
            source = im.convert("RGBA")
        rgba = np.asarray(source, dtype=np.uint8)
        authority = np.ascontiguousarray(rgba[..., 3] >= 8)
        domain = ObservationRasterDomain.from_rows(
            authority.tolist(),
            view_index=view,
            source_alpha_sha256=OBSERVATION_SHA256[view],
        )
        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view,
            camera_binding_hash=CAMERA_SHA256[view],
            observation_domain=domain,
        )
        baseline_checks = _exact_baseline_checks(candidate, expected_by_view[view])
        baseline_triangles = _candidate_triangles(candidate)
        baseline_mask = _mask_from_triangles(
            baseline_triangles, width=domain.width, height=domain.height
        )
        baseline_metrics = _mask_metrics(authority, baseline_mask)
        if abs(
            float(baseline_metrics["source_alpha_recall"])
            - float(candidate.residual_report["source_alpha_recall"])
        ) > 1.0e-12:
            raise RuntimeError(f"LEGAL_STEINER_BASELINE_RASTER_REPLAY_DRIFT_V{view}")

        kernel = _supported_pre_alpha_triangles(
            surface,
            view_index=view,
            observation_domain=domain,
        )
        if int(kernel.kernel_triangle_count) != int(candidate.residual_report["kernel_triangle_count"]):
            raise RuntimeError(f"LEGAL_STEINER_KERNEL_TRIANGLE_COUNT_DRIFT_V{view}")
        if int(kernel.post_contraction_triangle_count) != int(
            candidate.residual_report["post_contraction_triangle_count"]
        ):
            raise RuntimeError(f"LEGAL_STEINER_POST_CONTRACTION_COUNT_DRIFT_V{view}")

        support_mask = _mask_from_triangles(
            kernel.triangles, width=domain.width, height=domain.height
        )
        ceiling_mask = authority & support_mask
        if np.any(ceiling_mask & ~authority):
            raise RuntimeError(f"LEGAL_STEINER_CEILING_ALPHA_SPILL_V{view}")
        if np.any(baseline_mask & ~support_mask):
            raise RuntimeError(f"LEGAL_STEINER_BASELINE_NOT_SUBSET_OF_SUPPORTED_KERNEL_V{view}")
        if np.any((baseline_mask & authority) & ~ceiling_mask):
            raise RuntimeError(f"LEGAL_STEINER_CEILING_LOST_BASELINE_COVERAGE_V{view}")

        ceiling_metrics = _mask_metrics(authority, ceiling_mask)
        ceiling_failures = _coverage_failures(ceiling_metrics)
        recoverable = ceiling_mask & ~baseline_mask
        unreachable = authority & ~ceiling_mask

        baseline_render = _masked_source(source, baseline_mask)
        ceiling_render = _masked_source(source, ceiling_mask)
        baseline_path = out / f"V{view}_BASELINE_MESH_ONLY.png"
        ceiling_path = out / f"V{view}_LEGAL_STEINER_CEILING_MESH_ONLY.png"
        baseline_render.save(baseline_path)
        ceiling_render.save(ceiling_path)

        source_panel = _label(source, f"V{view} REAL SOURCE")
        baseline_panel = _label(
            _on_checker(baseline_render),
            f"V{view} BASELINE MESH-ONLY | recall={baseline_metrics['source_alpha_recall']*100:.2f}%",
        )
        ceiling_panel = _label(
            _on_checker(ceiling_render),
            f"V{view} LEGAL-STEINER CEILING | recall={ceiling_metrics['source_alpha_recall']*100:.2f}%",
        )
        residual_panel = _residual_panel(source, recoverable, unreachable, view)
        strip = Image.new("RGB", (4096, 1024))
        for i, panel in enumerate((source_panel, baseline_panel, ceiling_panel, residual_panel)):
            strip.paste(panel.convert("RGB"), (i * 1024, 0))
        strip_path = out / f"V{view}_LEGAL_STEINER_CEILING_4PANEL.png"
        strip.save(strip_path)
        contact_rows.append(strip.resize((2048, 512), Image.Resampling.LANCZOS))

        row = {
            "view": int(view),
            "baseline": baseline_metrics,
            "ceiling": ceiling_metrics,
            "baseline_replay_checks": baseline_checks,
            "kernel_supported_triangle_count": len(kernel.triangles),
            "kernel_component_count": int(kernel.component_count),
            "kernel_triangle_count": int(kernel.kernel_triangle_count),
            "post_contraction_triangle_count": int(kernel.post_contraction_triangle_count),
            "fully_inside_alpha_triangle_count": int(kernel.fully_inside_alpha_triangle_count),
            "partial_alpha_triangle_count": int(kernel.partial_alpha_triangle_count),
            "fully_outside_alpha_triangle_count": int(kernel.fully_outside_alpha_triangle_count),
            "recoverable_pixel_count": int(np.count_nonzero(recoverable)),
            "unreachable_pixel_count": int(np.count_nonzero(unreachable)),
            "recoverable_foreground_fraction": float(np.count_nonzero(recoverable)) / max(1, int(np.count_nonzero(authority))),
            "unreachable_foreground_fraction": float(np.count_nonzero(unreachable)) / max(1, int(np.count_nonzero(authority))),
            "recall_gain": float(ceiling_metrics["source_alpha_recall"] - baseline_metrics["source_alpha_recall"]),
            "ceiling_coverage_failures": list(ceiling_failures),
            "coverage_route_feasible": not ceiling_failures,
            "baseline_mesh_only_png": baseline_path.name,
            "ceiling_mesh_only_png": ceiling_path.name,
            "four_panel_png": strip_path.name,
        }
        rows.append(row)
        print(
            f"V{view} | baseline={baseline_metrics['source_alpha_recall']*100:7.3f}% "
            f"ceiling={ceiling_metrics['source_alpha_recall']*100:7.3f}% "
            f"gain={row['recall_gain']*100:+6.3f}pp "
            f"largest={ceiling_metrics['largest_uncovered_component_fraction']*100:6.3f}% "
            f"recoverable={row['recoverable_pixel_count']:6d} "
            f"unreachable={row['unreachable_pixel_count']:6d} "
            f"{'PASS' if row['coverage_route_feasible'] else 'FAIL'} "
            f"{','.join(row['ceiling_coverage_failures']) or '-'}",
            flush=True,
        )

    contact = Image.new("RGB", (2048, 512 * 8))
    for view, row_image in enumerate(contact_rows):
        contact.paste(row_image, (0, view * 512))
    contact_path = out / "FIT2_LEGAL_STEINER_CEILING_8VIEW_4PANEL.png"
    contact.save(contact_path)

    all_feasible = all(bool(row["coverage_route_feasible"]) for row in rows)
    decision = (
        "BOUNDARY_RECOVERY_GEOMETRICALLY_SUFFICIENT"
        if all_feasible
        else "BOUNDARY_RECOVERY_GEOMETRICALLY_INSUFFICIENT"
    )
    manifest = {
        "schema": "RealSaS.MageFIT2.LegalSteinerCeilingDiagnostic.v1",
        "status": "PASS__DIAGNOSTIC_COMPLETED",
        "decision": decision,
        "scope": "PIXELWISE_SUPPORT_CEILING_ONLY__NOT_EMITTED_MESH__NOT_PRODUCT_PASS",
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "surface_node_count": int(tensor.node_count),
        "surface_edge_count": int(tensor.edge_count),
        "zero_surface_sha256": ZERO_SHA256,
        "camera_sha256": list(CAMERA_SHA256),
        "observation_sha256": list(OBSERVATION_SHA256),
        "policy": FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.to_dict(),
        "ceiling_definition": "EXACT_ALPHA_INTERSECTION_WITH_POST_CONTRACTION_SUPPORTED_CDT_KERNEL_TRIANGLE_UNION",
        "cross_component_bridge_used": False,
        "unknown_relation_bridge_used": False,
        "source_mesh_used": False,
        "teacher_topology_used": False,
        "product_mesh_pass_claimed": False,
        "views": rows,
        "all_views_coverage_route_feasible": bool(all_feasible),
        "contact_sheet_png": contact_path.name,
    }
    manifest_path = out / "FIT2_LEGAL_STEINER_CEILING_RESULT.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 132, flush=True)
    print(f"DECISION: {decision}", flush=True)
    print(f"MANIFEST: {manifest_path}", flush=True)
    print(f"CONTACT : {contact_path}", flush=True)
    print("PRODUCT_MESH_PASS: NOT CLAIMED", flush=True)
    print("=" * 132, flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-surface", required=True)
    parser.add_argument("--cameras", nargs=8, required=True)
    parser.add_argument("--observations", nargs=8, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
