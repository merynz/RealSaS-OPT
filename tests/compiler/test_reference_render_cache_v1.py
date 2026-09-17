from __future__ import annotations

from pathlib import Path

from compiler.realsas_compiler_services.cache.reference_render import (
    NativeReferenceRenderCache,
    runtime_package_identity,
)


_RENDERER = r'''#!/usr/bin/env python3
from pathlib import Path
import sys

count_path = Path(__file__).with_suffix(".count")
count = int(count_path.read_text() if count_path.exists() else "0") + 1
count_path.write_text(str(count))
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"reference-render-payload")
print("renderer=TEST_NATIVE_REFERENCE count=" + str(count))
'''


def _renderer(tmp_path: Path) -> Path:
    path = tmp_path / "renderer.py"
    path.write_text(_RENDERER, encoding="utf-8")
    path.chmod(0o755)
    return path


def _count(renderer: Path) -> int:
    path = renderer.with_suffix(".count")
    return int(path.read_text()) if path.exists() else 0


def test_reference_render_cache_cold_warm_and_static_input_invalidation(tmp_path: Path):
    package = tmp_path / "fixture.rss"
    package.write_bytes(b"runtime-package-v1")
    renderer = _renderer(tmp_path)
    root = tmp_path / "cache"

    cache = NativeReferenceRenderCache(runtime_package=package, runtime_demo=renderer, cache_root=root)
    cold = cache.render(clip="idle", view="V0", time_seconds=0.5, out_png=tmp_path / "cold.png")
    assert cold["render_cache_hit"] is False
    assert _count(renderer) == 1

    warm = cache.render(clip="idle", view="V0", time_seconds=0.5, out_png=tmp_path / "warm.png")
    assert warm["render_cache_hit"] is True
    assert warm["render_cache_reason"] == "HIT_RESTORED"
    assert warm["sha256"] == cold["sha256"]
    assert _count(renderer) == 1

    same = cache.render(clip="idle", view="V0", time_seconds=0.5, out_png=tmp_path / "warm.png")
    assert same["render_cache_hit"] is True
    assert same["render_cache_output_already_current"] is True
    assert same["render_cache_reason"] == "HIT_OUTPUT_ALREADY_CURRENT"
    assert _count(renderer) == 1

    package.write_bytes(b"runtime-package-v2")
    changed_package = NativeReferenceRenderCache(runtime_package=package, runtime_demo=renderer, cache_root=root)
    after_package_change = changed_package.render(
        clip="idle", view="V0", time_seconds=0.5, out_png=tmp_path / "package-changed.png"
    )
    assert after_package_change["render_cache_hit"] is False
    assert after_package_change["render_cache_key"] != cold["render_cache_key"]
    assert _count(renderer) == 2

    renderer.write_text(_RENDERER + "\n# renderer-byte-change\n", encoding="utf-8")
    renderer.chmod(0o755)
    changed_renderer = NativeReferenceRenderCache(runtime_package=package, runtime_demo=renderer, cache_root=root)
    after_renderer_change = changed_renderer.render(
        clip="idle", view="V0", time_seconds=0.5, out_png=tmp_path / "renderer-changed.png"
    )
    assert after_renderer_change["render_cache_hit"] is False
    assert after_renderer_change["render_cache_key"] != after_package_change["render_cache_key"]
    assert _count(renderer) == 3


def test_corrupt_cached_png_is_never_reused(tmp_path: Path):
    package = tmp_path / "fixture.rss"
    package.write_bytes(b"runtime-package")
    renderer = _renderer(tmp_path)
    cache = NativeReferenceRenderCache(
        runtime_package=package,
        runtime_demo=renderer,
        cache_root=tmp_path / "cache",
    )
    cold = cache.render(clip="run", view="V7", time_seconds=1.0, out_png=tmp_path / "cold.png")
    assert _count(renderer) == 1

    cached_artifact = cache.cache.entry_path(cold["render_cache_key"]) / "artifact.bin"
    cached_artifact.write_bytes(b"corrupt")
    repaired = cache.render(clip="run", view="V7", time_seconds=1.0, out_png=tmp_path / "restored.png")
    assert repaired["render_cache_hit"] is False
    assert repaired["render_cache_reason"] == "MISS_CORRUPT_ARTIFACT"
    assert _count(renderer) == 2
    assert (tmp_path / "restored.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_nodes_are_independent_by_clip_view_and_effective_time(tmp_path: Path):
    package = tmp_path / "fixture.rss"
    package.write_bytes(b"runtime-package")
    renderer = _renderer(tmp_path)
    cache = NativeReferenceRenderCache(
        runtime_package=package,
        runtime_demo=renderer,
        cache_root=tmp_path / "cache",
    )
    keys = {
        cache.node_key(clip="idle", view="V0", time_seconds=0.0),
        cache.node_key(clip="idle", view="V1", time_seconds=0.0),
        cache.node_key(clip="run", view="V0", time_seconds=0.0),
        cache.node_key(clip="idle", view="V0", time_seconds=0.1),
    }
    assert len(keys) == 4
    # Values beyond the native CLI's 9-decimal command precision intentionally
    # collapse to the same node because the renderer receives identical bytes.
    assert cache.node_key(clip="idle", view="V0", time_seconds=0.12345678901) == cache.node_key(
        clip="idle", view="V0", time_seconds=0.12345678904
    )


def test_directory_runtime_identity_is_content_based_not_absolute_path(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "manifest.json").write_text('{"schema":"x"}', encoding="utf-8")
        (root / "nested" / "asset.bin").write_bytes(b"same")
    assert runtime_package_identity(left)["sha256"] == runtime_package_identity(right)["sha256"]
    (right / "nested" / "asset.bin").write_bytes(b"different")
    assert runtime_package_identity(left)["sha256"] != runtime_package_identity(right)["sha256"]
