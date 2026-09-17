from __future__ import annotations

"""Batch transport for Runtime-v4 reference-render cache misses.

Node identity is intentionally owned by NativeReferenceRenderCache.  Batch mode
changes only how cache misses are executed: one native Runtime-v4 process opens the
package once and renders many exact `(clip, view, time)` requests.  Every PNG is
validated and stored independently under the same P3 node key used by scalar mode.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable, Mapping
import hashlib
import subprocess

from .content_addressed import sha256_file
from .reference_render import NativeReferenceRenderCache, _valid_png


def _safe_field(value: str, *, label: str) -> str:
    text = str(value)
    if not text or any(ch in text for ch in "\t\r\n"):
        raise ValueError(f"REFERENCE_RENDER_BATCH_INVALID_{label}")
    return text


def render_many_v4(
    cache: NativeReferenceRenderCache,
    requests: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore exact hits and execute all misses in one Runtime-v4 process."""

    rows = tuple(dict(row) for row in requests)
    if not rows:
        return []

    results: list[dict[str, Any] | None] = [None] * len(rows)
    misses: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        started = perf_counter()
        clip = _safe_field(row["clip"], label="CLIP")
        view = _safe_field(row["view"], label="VIEW")
        time_seconds = float(row["time_seconds"])
        target = Path(row["out_png"]).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        key = cache.node_key(clip=clip, view=view, time_seconds=time_seconds)

        restore_started = perf_counter()
        restored = cache.cache.restore(key, target)
        restore_seconds = perf_counter() - restore_started
        if restored.hit and _valid_png(target):
            metadata = dict(restored.metadata or {})
            results[index] = {
                "view": view,
                "time_seconds": time_seconds,
                "path": str(target),
                "sha256": str(restored.artifact_sha256),
                "stdout": str(metadata.get("stdout", "")),
                "render_cache_hit": True,
                "render_cache_key": key,
                "render_cache_reason": restored.reason,
                "render_cache_output_already_current": bool(restored.output_already_current),
                "render_cache_restore_seconds": restore_seconds,
                "render_wall_seconds": perf_counter() - started,
                "native_batch": False,
            }
            continue

        misses.append({
            "index": index,
            "clip": clip,
            "view": view,
            "time_seconds": time_seconds,
            "time_argument": f"{time_seconds:.9f}",
            "target": target,
            "key": key,
            "miss_reason": "MISS_INVALID_CACHED_PNG" if restored.hit else restored.reason,
            "restore_seconds": restore_seconds,
            "started": started,
        })

    if misses:
        with TemporaryDirectory(prefix="realsas_reference_render_batch_") as temp_dir:
            plan = Path(temp_dir) / "requests.tsv"
            lines = []
            for row in misses:
                target_text = _safe_field(str(row["target"]), label="OUTPUT_PATH")
                lines.append(
                    "\t".join((
                        row["clip"],
                        row["view"],
                        row["time_argument"],
                        target_text,
                    ))
                )
            plan.write_text("\n".join(lines) + "\n", encoding="utf-8")

            native_started = perf_counter()
            proc = subprocess.run(
                [str(cache.runtime_demo), str(cache.runtime_package), "--batch-file", str(plan)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            native_batch_seconds = perf_counter() - native_started
            if proc.returncode != 0:
                raise RuntimeError(
                    "E2E_NATIVE_REFERENCE_BATCH_RENDER_FAILED:"
                    f"nodes={len(misses)}:rc={proc.returncode}\n{proc.stdout}"
                )
            marker = f"batch_rendered={len(misses)} renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE"
            if marker not in proc.stdout:
                raise RuntimeError("E2E_NATIVE_REFERENCE_BATCH_MARKER_MISSING")
            stdout_sha256 = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()

            for row in misses:
                target = row["target"]
                if not target.is_file():
                    raise RuntimeError(f"E2E_REFERENCE_RENDER_DID_NOT_CREATE_PNG:{target}")
                if not _valid_png(target):
                    raise RuntimeError(f"E2E_REFERENCE_RENDER_OUTPUT_NOT_PNG:{target}")
                digest = sha256_file(target)
                store_started = perf_counter()
                stored = cache.cache.store(
                    row["key"],
                    target,
                    metadata={
                        "stdout": marker,
                        "batch_stdout_sha256": stdout_sha256,
                        "png_sha256": digest,
                        "native_batch_seconds": native_batch_seconds,
                        "native_batch_node_count": len(misses),
                    },
                )
                store_seconds = perf_counter() - store_started
                results[row["index"]] = {
                    "view": row["view"],
                    "time_seconds": row["time_seconds"],
                    "path": str(target),
                    "sha256": digest,
                    "stdout": marker,
                    "render_cache_hit": False,
                    "render_cache_key": row["key"],
                    "render_cache_reason": row["miss_reason"],
                    "render_cache_output_already_current": False,
                    "render_cache_restore_seconds": row["restore_seconds"],
                    "render_cache_store_seconds": store_seconds,
                    "native_batch_seconds": native_batch_seconds,
                    "native_batch_node_count": len(misses),
                    "native_batch": True,
                    "render_cache_entry_path": stored.entry_path,
                    "render_wall_seconds": perf_counter() - row["started"],
                }

    if any(row is None for row in results):
        raise RuntimeError("REFERENCE_RENDER_BATCH_INTERNAL_RESULT_GAP")
    return [dict(row) for row in results if row is not None]
