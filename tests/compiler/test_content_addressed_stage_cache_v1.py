from __future__ import annotations

from pathlib import Path

from compiler.realsas_compiler_services.cache.content_addressed import (
    ContentAddressedStageCache,
    stage_cache_key,
)


def test_stage_key_is_canonical_and_order_independent():
    a = stage_cache_key(stage="runtime", producer_fingerprint="p1", inputs={"b": 2, "a": 1})
    b = stage_cache_key(stage="runtime", producer_fingerprint="p1", inputs={"a": 1, "b": 2})
    assert a == b
    assert len(a) == 64


def test_cache_store_restore_and_already_current(tmp_path: Path):
    cache = ContentAddressedStageCache(tmp_path / "cache", stage="runtime", producer_fingerprint="p1")
    key = cache.key({"input": "a" * 64})
    source = tmp_path / "source.rss"
    source.write_bytes(b"realsas-cache-test" * 1024)

    stored = cache.store(key, source, metadata={"answer": 42})
    assert not stored.reused_existing_entry

    target = tmp_path / "out" / "product.rss"
    restored = cache.restore(key, target)
    assert restored.hit
    assert restored.reason == "HIT_RESTORED"
    assert target.read_bytes() == source.read_bytes()
    assert restored.metadata == {"answer": 42}

    second = cache.restore(key, target)
    assert second.hit
    assert second.output_already_current
    assert second.reason == "HIT_OUTPUT_ALREADY_CURRENT"


def test_corrupt_cached_artifact_is_a_miss_not_trusted(tmp_path: Path):
    cache = ContentAddressedStageCache(tmp_path / "cache", stage="proof", producer_fingerprint="p1")
    key = cache.key({"proof_inputs": ["x", "y"]})
    source = tmp_path / "proof.bin"
    source.write_bytes(b"sealed-proof")
    cache.store(key, source)

    artifact = cache.entry_path(key) / "artifact.bin"
    artifact.write_bytes(b"corrupt")
    target = tmp_path / "restored.bin"
    restored = cache.restore(key, target)
    assert not restored.hit
    assert restored.reason == "MISS_CORRUPT_ARTIFACT"
    assert not target.exists()


def test_changed_inputs_change_key(tmp_path: Path):
    cache = ContentAddressedStageCache(tmp_path / "cache", stage="render", producer_fingerprint="p1")
    assert cache.key({"frame": "a"}) != cache.key({"frame": "b"})
