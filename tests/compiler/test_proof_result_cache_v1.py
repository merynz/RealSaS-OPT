from __future__ import annotations

from pathlib import Path

import pytest

from compiler.realsas_compiler_core.directional_binding import DirectionalBindingPolicyV1, qualify_directional_joint_view_binding
from compiler.realsas_compiler_services.cache.content_addressed import ContentAddressedStageCache
from compiler.realsas_compiler_services.cache import proof_result as proof_cache
from compiler.realsas_compiler_services.proof.directional_motion_provider import make_qualified_directional_motion_provider
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import _deformation_fixture, run_complete_e2e_v1


def _provider(product):
    policy = DirectionalBindingPolicyV1(
        min_correspondences=4,
        required_affine_rank=3,
        max_p95_residual01=0.015,
        max_residual01=0.05,
        max_joint_affine_hull_residual01=0.02,
        min_raster_span=8.0,
    )
    binding = qualify_directional_joint_view_binding(product, policy=policy)
    return make_qualified_directional_motion_provider(binding)


def test_proof_result_cache_restores_exact_typed_bundle_and_bakes_fail_closed(tmp_path, monkeypatch):
    seed = run_complete_e2e_v1(tmp_path / "seed")
    product = seed["product"]
    expected_proof = seed["proof"]
    expected_bakes = tuple(seed["motion_bakes"])
    provider = _provider(product)
    fixture = _deformation_fixture(product)
    cache_root = tmp_path / "cache"

    calls = {"count": 0}

    def seeded_evaluator(_product, *, artifacts_out=None, **_kwargs):
        calls["count"] += 1
        if artifacts_out is not None:
            artifacts_out.setdefault("motion_bakes", {}).update({b.clip_id: b for b in expected_bakes})
        return expected_proof

    monkeypatch.setattr(proof_cache, "evaluate_product_proof", seeded_evaluator)
    cold_artifacts = {}
    cold = proof_cache.evaluate_product_proof_cached(
        product,
        cache_root=cache_root,
        deformation_fixture=fixture,
        motion_bake_provider=provider,
        artifacts_out=cold_artifacts,
    )
    assert cold.cache_hit is False
    assert calls["count"] == 1
    assert cold.proof_bundle.proof_bundle_hash == expected_proof.proof_bundle_hash
    assert {b.clip_id: b.bake_hash for b in cold.motion_bakes} == {b.clip_id: b.bake_hash for b in expected_bakes}
    assert {k: v.bake_hash for k, v in cold_artifacts["motion_bakes"].items()} == {b.clip_id: b.bake_hash for b in expected_bakes}

    def must_not_evaluate(*_args, **_kwargs):
        raise AssertionError("proof evaluator executed on exact cache hit")

    monkeypatch.setattr(proof_cache, "evaluate_product_proof", must_not_evaluate)
    warm_artifacts = {}
    warm = proof_cache.evaluate_product_proof_cached(
        product,
        cache_root=cache_root,
        deformation_fixture=fixture,
        motion_bake_provider=provider,
        artifacts_out=warm_artifacts,
    )
    assert warm.cache_hit is True
    assert warm.cache_key == cold.cache_key
    assert warm.proof_bundle.proof_bundle_hash == cold.proof_bundle.proof_bundle_hash
    assert [b.bake_hash for b in warm.motion_bakes] == [b.bake_hash for b in cold.motion_bakes]
    assert {k: v.bake_hash for k, v in warm_artifacts["motion_bakes"].items()} == {b.clip_id: b.bake_hash for b in expected_bakes}

    # Threshold/policy identity is part of the key. A changed policy must MISS,
    # which is proven here by the deliberately failing evaluator.
    with pytest.raises(AssertionError, match="proof evaluator executed"):
        proof_cache.evaluate_product_proof_cached(
            product,
            cache_root=cache_root,
            deformation_fixture=fixture,
            motion_bake_provider=provider,
            motion_policy={"max_edge_stretch_ratio": 999.0},
        )

    # Source/engine identity is also part of the producer fingerprint.
    original_source_fingerprint = proof_cache._source_fingerprint
    producer, rows = original_source_fingerprint()
    monkeypatch.setattr(
        proof_cache,
        "_source_fingerprint",
        lambda: ("f" * 64 if producer != "f" * 64 else "e" * 64, rows),
    )
    with pytest.raises(AssertionError, match="proof evaluator executed"):
        proof_cache.evaluate_product_proof_cached(
            product,
            cache_root=cache_root,
            deformation_fixture=fixture,
            motion_bake_provider=provider,
        )
    monkeypatch.setattr(proof_cache, "_source_fingerprint", original_source_fingerprint)

    # Exact byte corruption is never accepted as a HIT.
    producer, _ = proof_cache._source_fingerprint()
    cache = ContentAddressedStageCache(
        cache_root,
        stage=proof_cache.PROOF_RESULT_CACHE_STAGE,
        producer_fingerprint=producer,
    )
    artifact = cache.entry_path(cold.cache_key) / "artifact.bin"
    raw = artifact.read_bytes()
    artifact.write_bytes((b"X" + raw[1:]) if raw else b"X")
    with pytest.raises(AssertionError, match="proof evaluator executed"):
        proof_cache.evaluate_product_proof_cached(
            product,
            cache_root=cache_root,
            deformation_fixture=fixture,
            motion_bake_provider=provider,
        )
