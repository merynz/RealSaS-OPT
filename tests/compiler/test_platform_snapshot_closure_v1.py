from collections import Counter
from copy import deepcopy

from compiler.realsas_compiler_services.orchestrator import mainline
from tools import realsas_architecture as architecture


def test_snapshot_shares_module_scan_but_preserves_every_stage_identity(monkeypatch):
    plan = mainline.load_json(mainline.PLAN_PATH)
    original = mainline._local_import_closure
    calls = Counter()

    def counted(module):
        calls[module] += 1
        return original(module)

    monkeypatch.setattr(mainline, "_local_import_closure", counted)
    value = architecture.lightweight_implementation_closure(plan)
    modules = {stage["adapter"].split(":", 1)[0] for stage in plan["stages"]}
    assert len(modules) < len(plan["stages"])
    assert calls == Counter({module: 1 for module in modules})
    for row in value["adapter_implementation_closures"]:
        closure = original(row["adapter"].split(":", 1)[0])
        expected = [{"module": name, "sha256": digest} for name, digest in closure]
        assert row["local_python_import_closure"] == expected
        assert row["implementation_hash"] == mainline.content_sha256({
            "schema": "RealSaS.AdapterImplementationClosure.v2",
            "adapter": row["adapter"],
            "local_python_import_closure": expected,
        })


def test_next_snapshot_rereads_transitive_source_changes(tmp_path, monkeypatch):
    plan = deepcopy(mainline.load_json(mainline.PLAN_PATH))
    plan["stages"] = plan["stages"][:2]
    plan["stage_count"] = 2
    module = "fixture_adapter"
    (tmp_path / (module + ".py")).write_text(
        "from fixture_dependency import value\n"
        "def stage0(): return value\n"
        "def stage1(): return value\n")
    dependency = tmp_path / "fixture_dependency.py"
    dependency.write_text("value = 1\n")
    for index, stage in enumerate(plan["stages"]):
        stage["adapter"] = module + ":stage" + str(index)
    original_path = mainline._local_module_path

    def fixture_path(name):
        if name in {module, "fixture_dependency"}:
            return tmp_path / (name + ".py")
        return original_path(name)

    monkeypatch.setattr(mainline, "_local_module_path", fixture_path)

    first = architecture.lightweight_implementation_closure(
        plan, require_product_pass_authority=False)
    dependency.write_text("value = 2\n")
    second = architecture.lightweight_implementation_closure(
        plan, require_product_pass_authority=False)
    for before, after in zip(first["adapter_implementation_closures"],
                             second["adapter_implementation_closures"]):
        assert before["implementation_hash"] != after["implementation_hash"]
        assert before["local_python_import_closure"] != after["local_python_import_closure"]
