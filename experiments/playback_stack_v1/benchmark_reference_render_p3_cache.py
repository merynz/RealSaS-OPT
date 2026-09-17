from __future__ import annotations

"""Reference-workstation P3 native reference-render cache measurement."""

import argparse
from argparse import Namespace
import json
from pathlib import Path
from time import perf_counter
import tempfile

from compiler.realsas_compiler_services.export.runtime_v3 import materialize_runtime_v3_archive
from experiments.playback_stack_v1.run_runtime_reference_e2e_v1 import run as run_reference_e2e
from runtime.realsas_cpp.tests.runtime_v3_writer_native_e2e import _build_fixture

WARM_TARGET_SECONDS = 2.0
VIEWS = ",".join(f"V{i}" for i in range(8))
TIMES = "0.0,0.5,1.0"
EXPECTED_NODES = 24


def _run_once(*, package: Path, runtime_demo: Path, out: Path, cache_root: Path):
    args = Namespace(
        rss=str(package),
        runtime_demo=str(runtime_demo),
        out=str(out),
        cache_root=str(cache_root),
        subject_id="P3_SYNTHETIC_RUNTIME_V3_FIXTURE",
        clip="idle",
        views=VIEWS,
        times=TIMES,
        require_exact_8_views=True,
        require_depth_v3=True,
    )
    started = perf_counter()
    report = run_reference_e2e(args)
    return report, perf_counter() - started


def _frame_hashes(report: dict) -> list[str]:
    return [str(row["sha256"]) for row in report["frames"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-demo", required=True)
    args = ap.parse_args()
    runtime_demo = Path(args.runtime_demo).resolve()
    if not runtime_demo.is_file():
        raise SystemExit(f"runtime demo missing: {runtime_demo}")

    with tempfile.TemporaryDirectory(prefix="realsas_p3_reference_cache_") as temp:
        root = Path(temp)
        texture_root = root / "texture_root"
        contract, textures, clips = _build_fixture(texture_root)
        package = root / "fixture.rss"
        materialize_runtime_v3_archive(
            out_path=package,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_product_state_hash="a" * 64,
            source_proof_bundle_hash="b" * 64,
        )
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
        same, same_seconds = _run_once(
            package=package,
            runtime_demo=runtime_demo,
            out=root / "warm",
            cache_root=cache_root,
        )

        cold_cache = cold["reference_render_cache"]
        warm_cache = warm["reference_render_cache"]
        same_cache = same["reference_render_cache"]
        byte_exact = _frame_hashes(cold) == _frame_hashes(warm) == _frame_hashes(same)
        gate = (
            len(cold["frames"]) == EXPECTED_NODES
            and cold_cache["hit_count"] == 0
            and cold_cache["miss_count"] == EXPECTED_NODES
            and warm_cache["hit_count"] == EXPECTED_NODES
            and warm_cache["miss_count"] == 0
            and same_cache["hit_count"] == EXPECTED_NODES
            and same_cache["miss_count"] == 0
            and byte_exact
            and warm_seconds < WARM_TARGET_SECONDS
            and same_seconds < WARM_TARGET_SECONDS
        )
        report = {
            "schema": "RealSaS.RuntimeReferenceRenderP3CacheMeasurement.v1",
            "claim_scope": "SYNTHETIC_NATIVE_REFERENCE_RENDER_CACHE_ONLY",
            "node_count": EXPECTED_NODES,
            "cold_seconds": cold_seconds,
            "cold_hits": cold_cache["hit_count"],
            "cold_misses": cold_cache["miss_count"],
            "warm_restore_seconds": warm_seconds,
            "warm_restore_hits": warm_cache["hit_count"],
            "warm_same_output_seconds": same_seconds,
            "warm_same_output_hits": same_cache["hit_count"],
            "restored_png_hashes_exact": byte_exact,
            "runtime_package_content_sha256": cold["runtime_package_content_sha256"],
            "native_reference_renderer_sha256": cold["native_reference_renderer_sha256"],
            "warm_target_seconds": WARM_TARGET_SECONDS,
            "cache_performance_gate_pass": gate,
            "product_pass_claimed": False,
        }
        print("RUNTIME_REFERENCE_RENDER_P3_CACHE_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if not gate:
            raise SystemExit("P3 native reference-render cache gate failed")


if __name__ == "__main__":
    main()
