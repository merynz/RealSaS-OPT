from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from compiler.realsas_compiler_services.cache import proof_result as proof_cache_module
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import run_complete_e2e_v1


def _sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def test_complete_e2e_second_transaction_reuses_exact_typed_proof(monkeypatch, tmp_path):
    cache_root = tmp_path / "persistent-cache"

    cold = run_complete_e2e_v1(
        tmp_path / "cold-product",
        proof_cache_root=cache_root,
    )
    assert cold["proof_cache_hit"] is False
    assert cold["proof"].overall_status == "PASS"

    cold_proof_hash = cold["proof"].proof_bundle_hash
    cold_bake_hashes = tuple(bake.bake_hash for bake in cold["motion_bakes"])
    cold_runtime_sha = _sha256(cold["native_runtime_archive"])

    def forbidden_evaluator(*_args, **_kwargs):
        raise AssertionError("authoritative proof evaluator ran on exact warm transaction")

    monkeypatch.setattr(proof_cache_module, "evaluate_product_proof", forbidden_evaluator)

    warm = run_complete_e2e_v1(
        tmp_path / "warm-product",
        proof_cache_root=cache_root,
    )

    assert warm["proof_cache_hit"] is True
    assert warm["proof_cache_key"] == cold["proof_cache_key"]
    assert warm["proof_cache_artifact_sha256"] == cold["proof_cache_artifact_sha256"]
    assert warm["proof"].proof_bundle_hash == cold_proof_hash
    assert tuple(bake.bake_hash for bake in warm["motion_bakes"]) == cold_bake_hashes
    assert warm["proof"].overall_status == cold["proof"].overall_status == "PASS"
    assert warm["runtime_report"]["status"] == cold["runtime_report"]["status"]
    assert warm["bundle_manifest"] == cold["bundle_manifest"]
    assert _sha256(warm["native_runtime_archive"]) == cold_runtime_sha
