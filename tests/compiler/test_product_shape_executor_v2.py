from __future__ import annotations

from compiler.realsas_compiler_services.platform_worker import shape_executor_v2 as executor


def test_product_shape_flexicubes_locator_is_deployment_bound(monkeypatch, tmp_path):
    root = tmp_path / "platform"
    monkeypatch.setenv("REALSAS_DEPLOY_ROOT", str(root))
    monkeypatch.setenv("REALSAS_FLEXICUBES_ROOT", "/ambient/must/not/be/used")
    assert executor._flexicubes_root_from_deployment() == (
        root.resolve() / "foundations/flexicubes/source"
    )


def test_product_shape_flexicubes_locator_requires_deployment_root(monkeypatch):
    monkeypatch.delenv("REALSAS_DEPLOY_ROOT", raising=False)
    monkeypatch.setenv("REALSAS_FLEXICUBES_ROOT", "/ambient/source")
    try:
        executor._flexicubes_root_from_deployment()
    except RuntimeError as exc:
        assert str(exc) == "SHAPE_V2_DEPLOY_ROOT_REQUIRED"
    else:
        raise AssertionError("ambient FlexiCubes override must not become product authority")
