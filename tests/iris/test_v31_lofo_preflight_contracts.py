from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_stage48_blender_entrypoint_bootstraps_repo_before_tools_import():
    source = (ROOT / "tools/truth_foundry/blender_presentation_ray_measure_v1.py").read_text()
    root_bootstrap = source.index("ROOT = Path(__file__).resolve().parents[2]")
    path_insert = source.index("sys.path.insert(0, str(ROOT))")
    tools_import = source.index("from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as measured")
    assert root_bootstrap < path_insert < tools_import


def test_v31_lofo_sealed_manifest_identity_excludes_cache_hit_runtime_telemetry():
    source = (ROOT / "tools/training/prepare_iris_v31_lofo_inputs_v1.py").read_text()
    start = source.index('"frozen_feature_cache": {')
    end = source.index('"camera_features_contract"', start)
    sealed_cache_block = source[start:end]
    assert '"cache_hit"' not in sealed_cache_block
    assert "cache_hit={int(cache_hit)}" in source
