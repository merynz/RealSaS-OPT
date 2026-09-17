from __future__ import annotations

"""P3 cache measurement at the preregistered dense BODY representation scale.

This is performance/representation evidence only. It uses the exact benchmark scale
(257,505 vertices, 500,000 faces, 18 source frames, 8 views) but synthetic geometry.
No product appearance or semantic-motion claim is made.
"""

import argparse
from argparse import Namespace
import json
from pathlib import Path
from time import perf_counter
import tempfile

from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from experiments.playback_stack_v1.benchmark_runtime_v4_p0_p1 import (
    FACE_COUNT,
    FRAME_COUNT,
    VERTEX_COUNT,
    VIEW_COUNT,
    _fixture,
)
from experiments.playback_stack_v1.run_runtime_reference_e2e_v1 import run as run_reference_e2e

VIEWS = ",".join(f"V{i}" for i in range(VIEW_COUNT))
TIMES = "0.0,0.5,1.0"
EXPECTED_NODES = VIEW_COUNT * 3
WARM_TARGET_SECONDS = 2.0


def _run_once(*, package: Path, runtime_demo: Path, out: Path, cache_root: Path):
    args = Namespace(
        rss=str(package),
        runtime_demo=str(runtime_demo),
        out=str(out),
        cache_root=str(cache_root),
        subject_id="P3_DENSE_SCALE_SYNTHETIC_BODY",
        clip="dense_probe",
        views=VIEWS,
        times=TIMES,
        require_exact_8_views=True,
        require_depth_v3=True,
    )
    started = perf_counter()
    report = run_reference_e2e(args)
    return report, perf_counter() - started


def _hashes(report: dict) -> list[str]:
    return [str(row["sha256"]) for row in report["frames"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-demo", required=True)
    args = ap.parse_args()
    runtime_demo = Path(args.runtime_demo).resolve()
    if not runtime_demo.is_file():
        raise SystemExit(f"runtime demo missing: {runtime_demo}")

    with tempfile.TemporaryDirectory(prefix="realsas_p3_dense_") as temp:
        root = Path(temp)
        texture_root = root / "texture_root"
        contract, textures, clips = _fixture(texture_root)
        package = root / "dense_probe.rss"

        materialize_started = perf_counter()
        package_result = materialize_runtime_v4_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="c" * 64,
            source_proof_bundle_hash="d" * 64,
        )
        materialize_seconds = perf_counter() - materialize_started
        cache_root = root / "persistent_cache"

        cold, cold_seconds = _run_once(
            package=package,
            runtime_demo=runtime_demo,
            out=root / "cold",
            cache_root=cache_root,
        )
        warm, warm_seconds = _run_once(
            package=package,
            runtime_demo=runtime_demo,
            out=root / "warm",
            cache_root=cache_root,
        )

        cold_cache = cold["reference_render_cache"]
        warm_cache = warm["reference_render_cache"]
        exact = _hashes(cold) == _hashes(warm)
        estimated_full_18_frame_cold_seconds = cold_seconds * (FRAME_COUNT / 3.0)
        gate = (
            len(cold["frames"]) == EXPECTED_NODES
            and cold_cache["hit_count"] == 0
            and cold_cache["miss_count"] == EXPECTED_NODES
            and warm_cache["hit_count"] == EXPECTED_NODES
            and warm_cache["miss_count"] == 0
            and exact
            and warm_seconds < WARM_TARGET_SECONDS
        )
        report = {
            "schema": "RealSaS.RuntimeReferenceRenderP3DenseScaleMeasurement.v1",
            "claim_scope": "SYNTHETIC_DENSE_SCALE_NATIVE_REFERENCE_RENDER_CACHE_ONLY",
            "vertex_count": VERTEX_COUNT,
            "face_count": FACE_COUNT,
            "source_frame_count": FRAME_COUNT,
            "view_count": VIEW_COUNT,
            "measured_time_samples": 3,
            "measured_render_nodes": EXPECTED_NODES,
            "runtime_package_bytes": int(package_result["package_total_bytes"]),
            "runtime_materialize_seconds": materialize_seconds,
            "cold_render_seconds": cold_seconds,
            "cold_hits": cold_cache["hit_count"],
            "cold_misses": cold_cache["miss_count"],
            "warm_restore_seconds": warm_seconds,
            "warm_hits": warm_cache["hit_count"],
            "estimated_full_18_frame_cold_seconds_linear": estimated_full_18_frame_cold_seconds,
            "restored_png_hashes_exact": exact,
            "warm_target_seconds": WARM_TARGET_SECONDS,
            "cache_performance_gate_pass": gate,
            "performance_pass_claimed": False,
            "product_pass_claimed": False,
        }
        print("RUNTIME_REFERENCE_RENDER_P3_DENSE_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if not gate:
            raise SystemExit("P3 dense-scale cache gate failed")


if __name__ == "__main__":
    main()
