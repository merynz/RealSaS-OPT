from __future__ import annotations

"""P4 product-proof content-addressed cache measurement on current synthetic V4 product."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import tempfile

from compiler.realsas_compiler_core.directional_binding import DirectionalBindingPolicyV1, qualify_directional_joint_view_binding
from compiler.realsas_compiler_services.cache.proof_result import evaluate_product_proof_cached
from compiler.realsas_compiler_services.proof.directional_motion_provider import make_qualified_directional_motion_provider
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import _deformation_fixture, run_complete_e2e_v1

WARM_TARGET_SECONDS = 2.0


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default=None)
    args = ap.parse_args()

    owner = None
    if args.work_root:
        root = Path(args.work_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        owner = tempfile.TemporaryDirectory(prefix="realsas_p4_proof_cache_")
        root = Path(owner.name)

    try:
        seed = run_complete_e2e_v1(root / "seed")
        product = seed["product"]
        provider = _provider(product)
        fixture = _deformation_fixture(product)
        cache_root = root / "persistent_cache"

        cold_artifacts = {}
        start = perf_counter()
        cold = evaluate_product_proof_cached(
            product,
            cache_root=cache_root,
            deformation_fixture=fixture,
            motion_bake_provider=provider,
            artifacts_out=cold_artifacts,
        )
        cold_seconds = perf_counter() - start

        warm_artifacts = {}
        start = perf_counter()
        warm = evaluate_product_proof_cached(
            product,
            cache_root=cache_root,
            deformation_fixture=fixture,
            motion_bake_provider=provider,
            artifacts_out=warm_artifacts,
        )
        warm_seconds = perf_counter() - start

        cold_bakes = {b.clip_id: b.bake_hash for b in cold.motion_bakes}
        warm_bakes = {b.clip_id: b.bake_hash for b in warm.motion_bakes}
        seed_bakes = {b.clip_id: b.bake_hash for b in seed["motion_bakes"]}
        gate = (
            cold.cache_hit is False
            and warm.cache_hit is True
            and cold.cache_key == warm.cache_key
            and cold.proof_bundle.proof_bundle_hash == warm.proof_bundle.proof_bundle_hash == seed["proof"].proof_bundle_hash
            and cold_bakes == warm_bakes == seed_bakes
            and {k: v.bake_hash for k, v in warm_artifacts["motion_bakes"].items()} == seed_bakes
            and warm_seconds < WARM_TARGET_SECONDS
        )
        report = {
            "schema": "RealSaS.ProofResultP4CacheMeasurement.v1",
            "claim_scope": "SYNTHETIC_CURRENT_V4_PRODUCT_PROOF_CACHE_ONLY",
            "cold_seconds": cold_seconds,
            "cold_cache_hit": cold.cache_hit,
            "warm_seconds": warm_seconds,
            "warm_cache_hit": warm.cache_hit,
            "warm_target_seconds": WARM_TARGET_SECONDS,
            "proof_bundle_hash": warm.proof_bundle.proof_bundle_hash,
            "motion_bake_hashes": warm_bakes,
            "artifact_sha256": warm.artifact_sha256,
            "cache_key": warm.cache_key,
            "exact_proof_and_bake_identity": cold_bakes == warm_bakes == seed_bakes,
            "cache_performance_gate_pass": gate,
            "product_pass_claimed": False,
        }
        print("PROOF_RESULT_P4_CACHE_MEASUREMENT=" + json.dumps(report, sort_keys=True))
        if not gate:
            raise SystemExit("P4 product-proof cache gate failed")
    finally:
        if owner is not None:
            owner.cleanup()


if __name__ == "__main__":
    main()
