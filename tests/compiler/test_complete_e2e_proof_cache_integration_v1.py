from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from compiler.realsas_compiler_services.cache import proof_result as proof_cache_module
from compiler.realsas_compiler_services.cache.proof_result import _exact_identity, _source_fingerprint
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import _deformation_fixture, run_complete_e2e_v1


def _sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def test_complete_e2e_second_transaction_reuses_exact_typed_proof(monkeypatch, tmp_path):
    cache_root = tmp_path / "persistent-cache"

    cold = run_complete_e2e_v1(tmp_path / "cold-product", proof_cache_root=cache_root)
    assert cold["proof_cache_hit"] is False
    assert cold["proof"].overall_status == "PASS"

    cold_proof_hash = cold["proof"].proof_bundle_hash
    cold_bake_hashes = tuple(bake.bake_hash for bake in cold["motion_bakes"])
    cold_runtime_sha = _sha256(cold["native_runtime_archive"])
    cold_deformation_identity = _exact_identity(_deformation_fixture(cold["product"]))
    cold_source_fingerprint = _source_fingerprint()[0]

    def forbidden_evaluator(*_args, **_kwargs):
        raise AssertionError("authoritative proof evaluator ran on exact warm transaction")

    monkeypatch.setattr(proof_cache_module, "evaluate_product_proof", forbidden_evaluator)

    warm = run_complete_e2e_v1(tmp_path / "warm-product", proof_cache_root=cache_root)
    warm_deformation_identity = _exact_identity(_deformation_fixture(warm["product"]))
    warm_source_fingerprint = _source_fingerprint()[0]

    cold_product = cold["product"]
    warm_product = warm["product"]
    cold_mech = cold_product.mechanical_state
    warm_mech = warm_product.mechanical_state

    assert warm_mech.surface.geometry_lineage_hash == cold_mech.surface.geometry_lineage_hash
    assert warm_mech.skeleton.skeleton_lineage_hash == cold_mech.skeleton.skeleton_lineage_hash
    assert warm_mech.skin.skin_lineage_hash == cold_mech.skin.skin_lineage_hash
    assert warm_product.mechanical_state_hash == cold_product.mechanical_state_hash
    assert warm_product.directional_visual_state_hash == cold_product.directional_visual_state_hash
    assert warm_product.motion_state_hash == cold_product.motion_state_hash
    assert warm_product.capability_contract_hash == cold_product.capability_contract_hash
    assert warm_product.product_state_hash == cold_product.product_state_hash
    assert warm["directional_binding_hash"] == cold["directional_binding_hash"]
    assert warm["qualified_motion_provider_hash"] == cold["qualified_motion_provider_hash"]
    assert warm_source_fingerprint == cold_source_fingerprint
    assert warm_deformation_identity == cold_deformation_identity
    assert warm["proof_cache_key"] == cold["proof_cache_key"]
    assert warm["proof_cache_hit"] is True, warm["proof_cache_reason"]
    assert warm["proof_cache_artifact_sha256"] == cold["proof_cache_artifact_sha256"]
    assert warm["proof"].proof_bundle_hash == cold_proof_hash
    assert tuple(bake.bake_hash for bake in warm["motion_bakes"]) == cold_bake_hashes
    assert warm["proof"].overall_status == cold["proof"].overall_status == "PASS"
    assert warm["runtime_report"]["status"] == cold["runtime_report"]["status"]
    assert warm["bundle_manifest"] == cold["bundle_manifest"]
    assert _sha256(warm["native_runtime_archive"]) == cold_runtime_sha
