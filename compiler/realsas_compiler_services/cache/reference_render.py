from __future__ import annotations

"""Persistent, fail-closed cache for native reference-render PNG nodes.

A render node is keyed by exact runtime-package content, exact renderer binary,
and the effective command arguments. Cache reuse never bypasses output validation:
restored bytes must still be a PNG and must have passed the generic artifact
size/SHA-256 verification performed by ContentAddressedStageCache.
"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
import os
import struct
import subprocess

from .content_addressed import ContentAddressedStageCache, sha256_file

REFERENCE_RENDER_CACHE_STAGE = "native_reference_render_png"
REFERENCE_RENDER_CACHE_PRODUCER = "RealSaS.NativeReferenceRenderCache.p3.v1"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _default_cache_root() -> Path:
    override = os.environ.get("REALSAS_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "realsas").resolve()


def _hash_directory_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    h = sha256()
    h.update(b"RealSaS.DirectoryContentIdentity.v1\0")
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"REFERENCE_RENDER_PACKAGE_SYMLINK_FORBIDDEN:{path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest = sha256_file(path)
        h.update(struct.pack("<I", len(rel)))
        h.update(rel)
        h.update(struct.pack("<Q", size))
        h.update(bytes.fromhex(digest))
        file_count += 1
        total_bytes += size
    return {
        "kind": "directory_tree_v1",
        "sha256": h.hexdigest(),
        "file_count": file_count,
        "bytes": total_bytes,
    }


def runtime_package_identity(path: str | Path) -> dict[str, Any]:
    package = Path(path).expanduser().resolve()
    if package.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(package),
            "bytes": package.stat().st_size,
        }
    if package.is_dir():
        return _hash_directory_tree(package)
    raise FileNotFoundError(package)


def _valid_png(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(PNG_MAGIC)) == PNG_MAGIC
    except OSError:
        return False


class NativeReferenceRenderCache:
    """Caches exact native `(package, renderer, clip, view, time)` render nodes."""

    def __init__(
        self,
        *,
        runtime_package: str | Path,
        runtime_demo: str | Path,
        cache_root: str | Path | None = None,
    ) -> None:
        self.runtime_package = Path(runtime_package).expanduser().resolve()
        self.runtime_demo = Path(runtime_demo).expanduser().resolve()
        if not self.runtime_demo.is_file():
            raise FileNotFoundError(self.runtime_demo)
        self.package_identity = runtime_package_identity(self.runtime_package)
        self.renderer_identity = {
            "sha256": sha256_file(self.runtime_demo),
            "bytes": self.runtime_demo.stat().st_size,
        }
        self.cache = ContentAddressedStageCache(
            cache_root if cache_root is not None else _default_cache_root(),
            stage=REFERENCE_RENDER_CACHE_STAGE,
            producer_fingerprint=REFERENCE_RENDER_CACHE_PRODUCER,
        )

    def node_inputs(self, *, clip: str, view: str, time_seconds: float) -> dict[str, Any]:
        # Bind the exact argument string consumed by std::stof in the native CLI.
        time_argument = f"{float(time_seconds):.9f}"
        return {
            "runtime_package": self.package_identity,
            "native_renderer": self.renderer_identity,
            "invocation_schema": "realsas_runtime_demo.clip_view_time_out.v1",
            "clip": str(clip),
            "view": str(view),
            "time_argument": time_argument,
            "post_process": False,
        }

    def node_key(self, *, clip: str, view: str, time_seconds: float) -> str:
        return self.cache.key(self.node_inputs(clip=clip, view=view, time_seconds=time_seconds))

    def render(
        self,
        *,
        clip: str,
        view: str,
        time_seconds: float,
        out_png: str | Path,
    ) -> dict[str, Any]:
        started = perf_counter()
        target = Path(out_png).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        key = self.node_key(clip=clip, view=view, time_seconds=time_seconds)

        restore_started = perf_counter()
        restored = self.cache.restore(key, target)
        restore_seconds = perf_counter() - restore_started
        if restored.hit and _valid_png(target):
            metadata = dict(restored.metadata or {})
            return {
                "view": str(view),
                "time_seconds": float(time_seconds),
                "path": str(target),
                "sha256": str(restored.artifact_sha256),
                "stdout": str(metadata.get("stdout", "")),
                "render_cache_hit": True,
                "render_cache_key": key,
                "render_cache_reason": restored.reason,
                "render_cache_output_already_current": bool(restored.output_already_current),
                "render_cache_restore_seconds": restore_seconds,
                "render_wall_seconds": perf_counter() - started,
            }

        miss_reason = restored.reason
        if restored.hit:
            miss_reason = "MISS_INVALID_CACHED_PNG"

        time_argument = f"{float(time_seconds):.9f}"
        cmd = [
            str(self.runtime_demo), str(self.runtime_package),
            "--clip", str(clip),
            "--view", str(view),
            "--time", time_argument,
            "--out", str(target),
        ]
        render_started = perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        native_seconds = perf_counter() - render_started
        if proc.returncode != 0:
            raise RuntimeError(
                "E2E_NATIVE_REFERENCE_RENDER_FAILED:"
                f"view={view}:time={time_seconds}:rc={proc.returncode}\n{proc.stdout}"
            )
        if not target.is_file():
            raise RuntimeError(f"E2E_REFERENCE_RENDER_DID_NOT_CREATE_PNG:{target}")
        if not _valid_png(target):
            raise RuntimeError(f"E2E_REFERENCE_RENDER_OUTPUT_NOT_PNG:{target}")

        digest = sha256_file(target)
        store_started = perf_counter()
        stored = self.cache.store(
            key,
            target,
            metadata={
                "stdout": proc.stdout.strip(),
                "png_sha256": digest,
                "native_render_seconds": native_seconds,
            },
        )
        store_seconds = perf_counter() - store_started
        return {
            "view": str(view),
            "time_seconds": float(time_seconds),
            "path": str(target),
            "sha256": digest,
            "stdout": proc.stdout.strip(),
            "render_cache_hit": False,
            "render_cache_key": key,
            "render_cache_reason": miss_reason,
            "render_cache_output_already_current": False,
            "render_cache_restore_seconds": restore_seconds,
            "render_cache_store_seconds": store_seconds,
            "native_render_seconds": native_seconds,
            "render_cache_entry_path": stored.entry_path,
            "render_wall_seconds": perf_counter() - started,
        }
