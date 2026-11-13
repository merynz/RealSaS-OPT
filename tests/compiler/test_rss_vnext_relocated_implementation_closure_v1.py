"""Source-identity proof for physical RSS.vNext compiler/model relocations."""
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.mainline import (
    _local_import_closure,
    _local_module_path,
)

ROOT = Path(__file__).resolve().parents[2]


def test_exact_product_worker_closure_resolves_relocated_source():
    names = {
        "compiler.realsas_compiler_services.platform_worker.product_downstream_executor_v1":
            "compiler/rss_vnext/workers/product_downstream_executor_v1.py",
        "compiler.realsas_compiler_services.platform_worker.rss_vnext_shape_executor_v3":
            "compiler/rss_vnext/workers/rss_vnext_shape_executor_v3.py",
        "compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1":
            "compiler/rss_vnext/core/rss_vnext_surface_decisions_v1.py",
        "models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1":
            "historical/model_sources/geppetto/reference_strength_v1/rigging_surface_tensorization_v1.py",
    }
    for module, rel in names.items():
        assert _local_module_path(module) == (ROOT / rel).resolve(), module
        assert module in {row[0] for row in _local_import_closure(module)}, module


def test_historical_model_identity_not_an_independent_release():
    from models import __path__ as model_roots

    assert (ROOT / "historical/model_sources").as_posix() in list(model_roots)
    assert not (ROOT / "models/arachne").exists()
    assert not (ROOT / "models/geppetto").exists()
