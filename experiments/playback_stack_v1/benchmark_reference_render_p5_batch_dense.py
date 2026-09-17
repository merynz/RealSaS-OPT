from __future__ import annotations

"""P5 scalar-vs-batch native reference-render measurement at dense BODY scale."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import tempfile

from compiler.realsas_compiler_services.cache.reference_render import NativeReferenceRenderCache
from compiler.realsas_compiler_services.cache.reference_render_batch import render_many_v4
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from experiments.playback_stack_v1.benchmark_runtime_v4_p0_p1 import (
    FACE_COUNT,
    FRAME_COUNT,
    VERTEX_COUNT,
    VIEW_COUNT,
    _fixture,
)

TIMES = (0.0, 0.5, 1.0)
EXPECTED_NODES = VIEW_COUNT * len(TIMES)
BATCH_TARGET_SECONDS = 8.0
WARM_TARGET_SECONDS = 2.0


def _requests(root: Path):
    rows = []
    for view_index in range(VIEW_COUNT):
        view = f"V{view_index}"
        for sample_index, time_seconds in enumerate(TIMES):
            rows.append({
                "clip": "dense_probe",
                "view": view,
                "time_seconds": time_seconds,
                "out_png": root / view / f"F{sample_index:04d}.png",
            })
    return rows


def _scalar(cache: NativeReferenceRenderCache, requests):
    out = []
    started = perf_counter()
    for row in requests:
        out.append(cache.render(
            clip=row["clip"],
            view=row["view"],
            time_seconds=row["time_seconds"],
            out_png=row["out_png"],
        ))
    return out, perf_counter() - started


def _hashes(rows):
    return [str(row["sha256"]) for row in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-demo", required=True)
    args = ap.parse_args()
    runtime_demo = Path(args.runtime_demo).resolve()
    if not runtime_demo.is_file():
        raise SystemExit(f"runtime demo missing: {runtime_demo}")

    with tempfile.TemporaryDirectory(prefix="realsas_p5_dense_") as temp:
        root = Path(temp)
        texture_root = root / "texture_root"
        contract, textures, clips = _fixture(texture_root)
        package = root / "dense_probe.rss"
        package_result = materialize_runtime_v4_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )

        scalar_cache = NativeReferenceRenderCache(
            runtime_package=package,
            runtime_demo=runtime_demo,
            cache_root=root / "scalar_cache",
        )
        scalar_requests = _requests(root / "scalar")
        scalar_rows, scalar_seconds = _scalar(scalar_cache, scalar_requests)

        batch_cache = NativeReferenceRenderCache(
            runtime_package=package,
            runtime_demo=runtime_demo,
            cache_root=root / "batch_cache",
        )
        batch_requests = _requests(root / "batch")
        batch_started = perf_counter()
        batch_rows = render_many_v4(batch_cache, batch_requests)
        batch_seconds = perf_counter() - batch_started

        warm_started = perf_counter()
        warm_rows = render_many_v4(batch_cache, batch_requests)
        warm_seconds = perf_counter() - warm_started

        exact = _hashes(scalar_rows) == _hashes(batch_rows) == _hashes(warm_rows)
        scalar_misses = sum(not bool(row["render_cache_hit"]) for row in scalar_rows)
        batch_misses = sum(not bool(row["render_cache_hit"]) for row in batch_rows)
        warm_hits = sum(bool(row["render_cache_hit"]) for row in warm_rows)
        batch_native_nodes = {int(row.get("native_batch_node_count", 0)) for row in batch_rows}
        speedup = scalar_seconds / batch_seconds if batch_seconds > 0 else float("inf")
        estimated_full_18_frame_batch_seconds = batch_seconds * (FRAME_COUNT / len(TIMES))

        gate = (
            len(scalar_rows) == len(batch_rows) == len(warm_rows) == EXPECTED_NODES
            and scalar_misses == EXPECTED_NODES
            and batch_misses == EXPECTED_NODES
            and warm_hits == EXPECTED_NODES
            and batch_native_nodes == {EXPECTED_NODES}
            and exact
            and batch_seconds < scalar_seconds
            and batch_seconds < BATCH_TARGET_SECONDS
            and warm_seconds < WARM_TARGET_SECONDS
        )
        report = {
            "schema": "RealSaS.RuntimeReferenceRenderP5BatchDenseMeasurement.v1",
            "claim_scope": "SYNTHETIC_DENSE_SCALE_BATCH_TRANSPORT_ONLY",
            "vertex_count": VERTEX_COUNT,
            "face_count": FACE_COUNT,
            "source_frame_count": FRAME_COUNT,
            "view_count": VIEW_COUNT,
            "measured_nodes": EXPECTED_NODES,
            "runtime_package_bytes": int(package_result["package_total_bytes"]),
            "scalar_cold_seconds": scalar_seconds,
            "batch_cold_seconds": batch_seconds,
            "scalar_to_batch_speedup": speedup,
            "batch_warm_seconds": warm_seconds,
            "scalar_misses": scalar_misses,
            "batch_misses": batch_misses,
            "batch_warm_hits": warm_hits,
            "batch_native_node_count": sorted(batch_native_nodes),
            "scalar_batch_png_hashes_exact": exact,
            "estimated_full_18_frame_batch_seconds_linear": estimated_full_18_frame_batch_seconds,
            "batch_target_seconds": BATCH_TARGET_SECONDS,
            "warm_target_seconds": WARM_TARGET_SECONDS,
            "performance_gate_pass": gate,
            "performance_pass_claimed": False,
            "product_pass_claimed": False,
        }
        print("RUNTIME_REFERENCE_RENDER_P5_BATCH_DENSE_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if not gate:
            raise SystemExit("P5 dense batch reference-render gate failed")


if __name__ == "__main__":
    main()
