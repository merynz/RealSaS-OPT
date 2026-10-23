from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
import tools.demo.knight_oracle_drawable_court_v1 as v1


MAX_RENDER_WORKERS = 4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _linear_pm_alpha_composite_layers(
    layer_rgba: dict[int, np.ndarray], order: list[int]
) -> np.ndarray:
    if not order:
        raise RuntimeError("DRAWABLE_ORACLE_EMPTY_ORDER")
    first = np.asarray(layer_rgba[order[0]], dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] != 4:
        raise RuntimeError("DRAWABLE_ORACLE_LAYER_RGBA_SHAPE")
    canvas = np.zeros(first.shape, dtype=np.float64)
    for group in order:
        src_u8 = np.asarray(layer_rgba[group], dtype=np.uint8)
        if src_u8.shape != first.shape:
            raise RuntimeError("DRAWABLE_ORACLE_LAYER_SHAPE_DRIFT")
        src = straight_srgb_rgba_u8_to_premultiplied_linear(src_u8)
        src_a = src[..., 3:4]
        canvas[..., :3] = src[..., :3] + canvas[..., :3] * (1.0 - src_a)
        canvas[..., 3:4] = src_a + canvas[..., 3:4] * (1.0 - src_a)
    return premultiplied_linear_to_straight_srgb_u8(canvas)


def _install_parallel_subset_renderer() -> dict:
    original_subset = v1._subset_candidate
    original_render = v1.render_caa_reference
    registry: dict[int, dict] = {}
    full_context: dict[tuple[int, int], dict] = {}
    batch_cache: dict[tuple[int, int], dict[int, SimpleNamespace]] = {}
    stats = {
        "parallel_batches": 0,
        "parallel_subset_renders": 0,
        "max_workers": MAX_RENDER_WORKERS,
    }

    def registered_subset(candidate, face_indices):
        mesh = original_subset(candidate, face_indices)
        registry[id(mesh)] = {
            "mesh": mesh,
            "indices": np.asarray(face_indices, dtype=np.int64).copy(),
        }
        return mesh

    def render(*, mesh, camera, face_uv, texture_rgba_u8, provenance_atlas, face_page_index, positions):
        mesh_id = id(mesh)
        key = (id(camera), id(positions))
        if mesh_id not in registry:
            # Full-carrier render. Preserve the complete result because the court
            # needs owner/second-owner depth evidence for drawable ordering.
            result = original_render(
                mesh=mesh,
                camera=camera,
                face_uv=face_uv,
                texture_rgba_u8=texture_rgba_u8,
                provenance_atlas=provenance_atlas,
                face_page_index=face_page_index,
                positions=positions,
            )
            full_context[key] = {
                "camera": camera,
                "face_uv": np.asarray(face_uv),
                "texture_rgba_u8": texture_rgba_u8,
                "provenance_atlas": provenance_atlas,
                "face_page_index": np.asarray(face_page_index),
                "positions": positions,
            }
            return result

        if key not in batch_cache:
            ctx = full_context.get(key)
            if ctx is None:
                raise RuntimeError("DRAWABLE_ORACLE_PARALLEL_FULL_CONTEXT_MISSING")

            def worker(item):
                sub_id, spec = item
                idx = spec["indices"]
                result = original_render(
                    mesh=spec["mesh"],
                    camera=ctx["camera"],
                    face_uv=ctx["face_uv"][idx],
                    texture_rgba_u8=ctx["texture_rgba_u8"],
                    provenance_atlas=ctx["provenance_atlas"],
                    face_page_index=ctx["face_page_index"][idx],
                    positions=ctx["positions"],
                )
                # Subset callers only consume straight_rgba_u8. Copy it and drop
                # heavy depth/owner buffers immediately after each worker returns.
                rgba = np.asarray(result.straight_rgba_u8, dtype=np.uint8).copy()
                return sub_id, SimpleNamespace(straight_rgba_u8=rgba)

            items = list(registry.items())
            workers = min(MAX_RENDER_WORKERS, max(1, len(items)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rendered = dict(pool.map(worker, items))
            batch_cache[key] = rendered
            stats["parallel_batches"] += 1
            stats["parallel_subset_renders"] += len(items)

        rendered = batch_cache[key]
        if mesh_id not in rendered:
            raise RuntimeError("DRAWABLE_ORACLE_PARALLEL_SUBSET_CACHE_MISS")
        result = rendered.pop(mesh_id)
        if not rendered:
            batch_cache.pop(key, None)
            full_context.pop(key, None)
        return result

    v1._subset_candidate = registered_subset
    v1.render_caa_reference = render
    return stats


def run(*, authority_root: Path, run_id: str, out_dir: Path) -> dict:
    v1._alpha_composite_layers = _linear_pm_alpha_composite_layers
    parallel_stats = _install_parallel_subset_renderer()
    report = v1.run(
        authority_root=authority_root,
        run_id=run_id,
        out_dir=out_dir,
    )
    report = dict(report)
    report["evaluator_revision"] = "KNIGHT_ORACLE_DRAWABLE_COURT_V3_LINEAR_PM_PARALLEL"
    report["layer_compositor"] = "CAA_LINEAR_LIGHT_PREMULTIPLIED_OVER"
    report["subset_render_execution"] = {
        "mode": "PARALLEL_INDEPENDENT_DRAWABLE_BATCH",
        **parallel_stats,
    }
    report["scientific_design_changed_from_prereg_v1"] = False
    report["baseline_gate_reused_from_sealed_receipt"] = True
    report_path = out_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_ORACLE_DRAWABLE_COURT_V3="
        + json.dumps(
            {
                "status": report.get("status"),
                "evaluator_revision": report["evaluator_revision"],
                "layer_compositor": report["layer_compositor"],
                "subset_render_execution": report["subset_render_execution"],
                "report_sha256": _sha256(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        authority_root=a.authority_root,
        run_id=a.run_id,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    main()
