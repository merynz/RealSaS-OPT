from __future__ import annotations

from pathlib import Path

import pytest

from tools import platform_foundation_cache_v1 as cache


def test_foundation_contracts_are_read_from_scientific_authority_sources():
    contracts = cache.foundation_contracts()
    dino = contracts["dinov2"]
    flex = contracts["flexicubes"]
    assert dino["DINO_REPOSITORY"] == "facebookresearch/dinov2"
    assert dino["DINO_SOURCE_REVISION"] == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    assert dino["DINO_S_WEIGHT_SHA256"] == "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    assert dino["DINO_S_WEIGHT_SIZE_BYTES"] == 88283115
    assert flex["FLEXICUBES_REPOSITORY"] == "https://github.com/nv-tlabs/FlexiCubes.git"
    assert flex["FLEXICUBES_COMMIT"] == "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"


def test_runtime_foundation_paths_are_deployment_root_relative(tmp_path):
    paths = cache.runtime_foundation_paths(tmp_path / "platform")
    assert paths["dino_source"] == (tmp_path / "platform/foundations/dinov2/source").resolve()
    assert paths["dino_weight"] == (tmp_path / "platform/foundations/dinov2/dinov2_vits14_pretrain.pth").resolve()
    assert paths["flexicubes_source"] == (tmp_path / "platform/foundations/flexicubes/source").resolve()


def test_cache_mutation_scope_accepts_only_strict_foundation_children(tmp_path):
    foundation = (tmp_path / "platform/foundations").resolve()
    child = foundation / "dinov2/source"
    assert cache._managed_cache_path(child, foundation) == child
    for unsafe in (foundation, tmp_path / "platform", tmp_path / "outside"):
        with pytest.raises(RuntimeError, match="FOUNDATION_CACHE_MUTATION_SCOPE_INVALID"):
            cache._managed_cache_path(unsafe, foundation)


def test_exact_weight_verification_rejects_same_size_wrong_bytes(tmp_path):
    path = tmp_path / "weight.bin"
    path.write_bytes(b"wrong")
    assert not cache._verified_weight(
        path,
        expected_sha256="0" * 64,
        expected_size=5,
    )


def test_foundation_gate_runs_after_deploy_and_dispatches_reseal():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/platform_foundation_cache_v1.yml").read_text(encoding="utf-8")
    assert "workflows: [platform-developer-deployment]" in text
    assert "tools/platform_foundation_cache_v1.py --root \"$PLATFORM_ROOT\"" in text
    assert "verify_dinov2_authority_files_v2" in text
    assert "verify_pinned_flexicubes_root_v1" in text
    assert "actions: write" in text
    assert "knight_current_subject_input_reseal.yml/dispatches" in text
    assert "scientific_authority_minted'] is False" in text
