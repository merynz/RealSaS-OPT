from __future__ import annotations

"""Reference-workstation P2 persistent cache measurement at the dense BODY scale."""

import json
from pathlib import Path
from time import perf_counter
import tempfile

from benchmark_runtime_v4_p0_p1 import _fixture
from compiler.realsas_compiler_services.export.runtime_v4_cache import (
    materialize_runtime_v4_archive_cached,
)

WARM_TARGET_SECONDS = 2.0


def _run_call(*, out: Path, cache_root: Path, texture_root: Path, contract, textures, clips, source_hash: str):
    started = perf_counter()
    result = materialize_runtime_v4_archive_cached(
        out_path=out,
        texture_root=texture_root,
        contract=contract,
        textures=textures,
        clips=clips,
        source_product_state_hash=source_hash,
        source_proof_bundle_hash="d" * 64,
        cache_root=cache_root,
    )
    measured = perf_counter() - started
    return result, measured


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="realsas_v4_p2_cache_bench_") as temp:
        root = Path(temp)
        texture_root = root / "texture_root"
        cache_root = root / "persistent_cache"
        contract, textures, clips = _fixture(texture_root)

        cold, cold_seconds = _run_call(
            out=root / "cold.rss",
            cache_root=cache_root,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_hash="c" * 64,
        )
        warm_copy, warm_copy_seconds = _run_call(
            out=root / "warm_copy.rss",
            cache_root=cache_root,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_hash="c" * 64,
        )
        warm_same, warm_same_seconds = _run_call(
            out=root / "warm_copy.rss",
            cache_root=cache_root,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_hash="c" * 64,
        )
        invalidated, invalidated_seconds = _run_call(
            out=root / "invalidated.rss",
            cache_root=cache_root,
            texture_root=texture_root,
            contract=contract,
            textures=textures,
            clips=clips,
            source_hash="e" * 64,
        )

        exact_bytes = (
            (root / "cold.rss").read_bytes() == (root / "warm_copy.rss").read_bytes()
        )
        performance_gate = (
            cold["cache_hit"] is False
            and warm_copy["cache_hit"] is True
            and warm_same["cache_hit"] is True
            and warm_same["cache_output_already_current"] is True
            and invalidated["cache_hit"] is False
            and invalidated["cache_key"] != cold["cache_key"]
            and exact_bytes
            and warm_copy_seconds < WARM_TARGET_SECONDS
            and warm_same_seconds < WARM_TARGET_SECONDS
        )
        report = {
            "schema": "RealSaS.RuntimeV4P2CacheMeasurement.v1",
            "claim_scope": "SYNTHETIC_CONTENT_ADDRESSED_RUNTIME_CACHE_ONLY",
            "cache_key": cold["cache_key"],
            "cold_cache_hit": cold["cache_hit"],
            "cold_seconds": cold_seconds,
            "warm_copy_cache_hit": warm_copy["cache_hit"],
            "warm_copy_seconds": warm_copy_seconds,
            "warm_copy_reason": warm_copy["cache_reason"],
            "warm_same_cache_hit": warm_same["cache_hit"],
            "warm_same_seconds": warm_same_seconds,
            "warm_same_reason": warm_same["cache_reason"],
            "invalidation_cache_hit": invalidated["cache_hit"],
            "invalidation_seconds": invalidated_seconds,
            "invalidation_key_changed": invalidated["cache_key"] != cold["cache_key"],
            "archive_sha256": cold["archive_sha256"],
            "package_total_bytes": cold["package_total_bytes"],
            "restored_archive_byte_exact": exact_bytes,
            "warm_target_seconds": WARM_TARGET_SECONDS,
            "cache_performance_gate_pass": performance_gate,
            "product_pass_claimed": False,
        }
        print("RUNTIME_V4_P2_CACHE_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if not performance_gate:
            raise SystemExit("Runtime-v4 P2 cache performance/invalidation gate failed")


if __name__ == "__main__":
    main()
