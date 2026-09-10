from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "build_knowledge_artifact_catalog.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_knowledge_artifact_catalog", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_census_branch_fetch_force_refreshes_only_disposable_local_ref(monkeypatch):
    module = _load_module()
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(module, "run", fake_run)

    branch = "exp/arachne-skintokens-cleanroom-fit1-20260908"
    local_ref = module.fetch_branch_for_census(branch)

    assert local_ref.startswith("refs/context-census/")
    assert calls == [
        (
            "git",
            "fetch",
            "--no-tags",
            "--depth=1",
            "origin",
            f"+refs/heads/{branch}:{local_ref}",
        )
    ]
    assert calls[0][-1].startswith("+refs/heads/")
    assert calls[0][-1].endswith(f":{local_ref}")
