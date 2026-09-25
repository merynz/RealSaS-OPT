from __future__ import annotations

import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_services.orchestrator.mainline import (
    _fingerprint,
    _manifest_subset,
    _outputs_verify,
    _persist_failure_diagnostics,
    _reset_stale_failures,
    _seal_outputs,
    _target_closure,
    dependency_failure_ids,
    ready_stage_ids,
    topological_stage_ids,
)


def _stage(ordinal, stage_id, deps=()):
    return {
        "ordinal": ordinal,
        "id": stage_id,
        "depends_on": list(deps),
    }


def _row(ordinal, stage_id, status="PENDING"):
    return {
        "ordinal": ordinal,
        "id": stage_id,
        "status": status,
    }


def test_ready_set_preserves_independent_branch_after_failure():
    plan = {
        "stages": [
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
            _stage(3, "03_C", ("01_A",)),
            _stage(4, "04_D", ("02_B",)),
            _stage(5, "05_E", ("03_C",)),
        ]
    }
    ledger = {
        "stages": [
            _row(1, "01_A", "PASS"),
            _row(2, "02_B", "FAIL"),
            _row(3, "03_C", "PENDING"),
            _row(4, "04_D", "PENDING"),
            _row(5, "05_E", "PENDING"),
        ]
    }
    assert ready_stage_ids(plan, ledger) == ("03_C",)
    assert dependency_failure_ids(plan, ledger, "04_D") == ("02_B",)
    assert dependency_failure_ids(plan, ledger, "05_E") == ()


def test_target_closure_is_dependency_ancestry_not_ordinal_prefix():
    plan = {
        "stages": [
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
            _stage(3, "03_C", ("01_A",)),
            _stage(4, "04_D", ("02_B",)),
            _stage(5, "05_E", ("03_C",)),
        ]
    }
    assert _target_closure(plan, ("05_E",)) == {"01_A", "03_C", "05_E"}
    assert "02_B" not in _target_closure(plan, ("05_E",))
    assert "04_D" not in _target_closure(plan, ("05_E",))


def test_topological_order_detects_cycle():
    plan = {
        "stages": [
            _stage(1, "01_A", ("02_B",)),
            _stage(2, "02_B", ("01_A",)),
        ]
    }
    with pytest.raises(RuntimeError, match="MAINLINE_PLAN_CYCLE"):
        topological_stage_ids(plan)


def test_topological_order_uses_ordinal_only_as_stable_display_tiebreak():
    plan = {
        "stages": [
            _stage(3, "03_C", ("01_A",)),
            _stage(1, "01_A"),
            _stage(2, "02_B", ("01_A",)),
        ]
    }
    assert topological_stage_ids(plan) == ("01_A", "02_B", "03_C")


def test_stage_fingerprint_binds_dependency_schema_and_authority_identity():
    plan = {
        "stages": [
            {
                **_stage(1, "01_A"),
                "manifest_keys": [],
                "policy": {"fail_closed": True, "output_hash_required": True},
            },
            {
                **_stage(2, "02_B", ("01_A",)),
                "manifest_keys": [],
                "policy": {"fail_closed": True, "output_hash_required": True},
            },
        ]
    }
    ledger = {
        "run_id": "R",
        "pipeline_plan_sha256": "p" * 64,
        "stages": [
            {
                **_row(1, "01_A", "PASS"),
                "outputs": [
                    {
                        "path": "/tmp/a",
                        "sha256": "a" * 64,
                        "bytes": 10,
                        "schema": "Schema.A",
                        "authority_class": "AUTH_A",
                    }
                ],
            },
            {**_row(2, "02_B", "PENDING"), "outputs": []},
        ],
    }
    stage = plan["stages"][1]
    first, _ = _fingerprint(plan, ledger, {}, stage, "i" * 64)
    ledger["stages"][0]["outputs"][0]["schema"] = "Schema.B"
    second, _ = _fingerprint(plan, ledger, {}, stage, "i" * 64)
    assert second != first
    ledger["stages"][0]["outputs"][0]["schema"] = "Schema.A"
    ledger["stages"][0]["outputs"][0]["authority_class"] = "AUTH_B"
    third, _ = _fingerprint(plan, ledger, {}, stage, "i" * 64)
    assert third != first


def test_sealed_output_requires_typed_authority_metadata(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="STAGE_OUTPUT_AUTHORITY_CLASS_REQUIRED"):
        _seal_outputs(
            [{"path": str(path), "schema": "Schema.A"}],
            allowed_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="STAGE_OUTPUT_SCHEMA_REQUIRED"):
        _seal_outputs(
            [{"path": str(path), "authority_class": "AUTH_A"}],
            allowed_root=tmp_path,
        )


def test_existing_output_verification_rejects_size_and_schema_drift(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "outputs": [
            {
                "path": str(path),
                "sha256": digest,
                "bytes": 3,
                "schema": "Schema.A",
                "authority_class": "AUTH_A",
            }
        ]
    }
    assert _outputs_verify(row, allowed_root=tmp_path)
    row["outputs"][0]["bytes"] = 4
    assert not _outputs_verify(row, allowed_root=tmp_path)
    row["outputs"][0]["bytes"] = 3
    row["outputs"][0]["schema"] = "UNSPECIFIED"
    assert not _outputs_verify(row, allowed_root=tmp_path)


def test_geppetto_prereg_fingerprint_subset_ignores_future_execution_refs():
    stage = {
        "id": "26_GEPPETTO_FIT_PREREGISTERED",
        "manifest_keys": ["geppetto_fit"],
    }
    manifest = {
        "geppetto_fit": {
            "preregistration": {"path": "/authority/prereg.json", "sha256": "a" * 64},
            "execution_receipt": None,
            "checkpoint": None,
            "result": None,
            "proposal": None,
        }
    }
    before = _manifest_subset(manifest, stage)
    manifest["geppetto_fit"].update(
        execution_receipt={"path": "/authority/receipt.json", "sha256": "b" * 64},
        checkpoint={"path": "/authority/model.pt", "sha256": "c" * 64},
        result={"path": "/authority/result.json", "sha256": "d" * 64},
        proposal={"path": "/authority/proposal.json", "sha256": "e" * 64},
    )
    after = _manifest_subset(manifest, stage)
    assert before == after
    assert before == {
        "geppetto_fit": {
            "preregistration": {"path": "/authority/prereg.json", "sha256": "a" * 64}
        }
    }


def test_geppetto_execution_fingerprint_subset_binds_execution_refs():
    stage = {"id": "27_GEPPETTO_FIT", "manifest_keys": ["geppetto_fit"]}
    manifest = {
        "geppetto_fit": {
            "preregistration": {"path": "/authority/prereg.json", "sha256": "a" * 64},
            "checkpoint": {"path": "/authority/model.pt", "sha256": "c" * 64},
        }
    }
    before = _manifest_subset(manifest, stage)
    manifest["geppetto_fit"]["checkpoint"]["sha256"] = "f" * 64
    after = _manifest_subset(manifest, stage)
    assert before != after


def test_manifest_subset_snapshot_is_detached_from_nested_source_mutation():
    stage = {"id": "27_GEPPETTO_FIT", "manifest_keys": ["geppetto_fit"]}
    manifest = {
        "geppetto_fit": {
            "preregistration": {
                "path": "/authority/prereg.json",
                "sha256": "a" * 64,
                "metadata": {"nested": {"token": "before"}},
            },
            "checkpoint": {
                "path": "/authority/model.pt",
                "sha256": "c" * 64,
            },
        }
    }
    snapshot = _manifest_subset(manifest, stage)
    manifest["geppetto_fit"]["preregistration"]["metadata"]["nested"]["token"] = "after"
    manifest["geppetto_fit"]["checkpoint"]["sha256"] = "f" * 64
    assert snapshot["geppetto_fit"]["preregistration"]["metadata"]["nested"]["token"] == "before"
    assert snapshot["geppetto_fit"]["checkpoint"]["sha256"] == "c" * 64


def test_arachne_prereg_fingerprint_subset_ignores_future_execution_refs():
    stage = {
        "id": "30_ARACHNE_FIT_PREREGISTERED",
        "manifest_keys": ["arachne_fit"],
    }
    manifest = {
        "arachne_fit": {
            "preregistration": {"path": "/authority/prereg.json", "sha256": "1" * 64},
            "checkpoint": None,
        }
    }
    before = _manifest_subset(manifest, stage)
    manifest["arachne_fit"]["checkpoint"] = {
        "path": "/authority/skin.pt",
        "sha256": "2" * 64,
    }
    assert _manifest_subset(manifest, stage) == before


def test_stale_failure_identity_reopens_failed_subgraph(monkeypatch):
    plan = {
        "stages": [
            {
                "ordinal": 1,
                "id": "01_A",
                "depends_on": [],
                "manifest_keys": [],
                "policy": {},
                "adapter": "pkg.mod:stage_a",
            },
            {
                "ordinal": 2,
                "id": "02_B",
                "depends_on": ["01_A"],
                "manifest_keys": ["cfg"],
                "policy": {"gate": 1},
                "adapter": "pkg.mod:stage_b",
            },
            {
                "ordinal": 3,
                "id": "03_C",
                "depends_on": ["02_B"],
                "manifest_keys": [],
                "policy": {},
                "adapter": "pkg.mod:stage_c",
            },
        ]
    }
    ledger = {
        "run_id": "RUN",
        "execution_class": "WITNESS",
        "pipeline_plan_sha256": "p" * 64,
        "stages": [
            {
                **_row(1, "01_A", "PASS"),
                "outputs": [],
                "input_fingerprint": "a" * 64,
                "implementation_hash": "impl-a",
                "policy_hash": "pa",
            },
            {
                **_row(2, "02_B", "FAIL"),
                "outputs": [],
                "input_fingerprint": "old-fingerprint",
                "implementation_hash": "old-impl",
                "policy_hash": "old-policy",
                "blockers": ["OLD_FAIL"],
            },
            {
                **_row(3, "03_C", "PENDING"),
                "outputs": [],
                "input_fingerprint": "",
                "implementation_hash": "",
                "policy_hash": "",
            },
        ],
    }
    manifest = {"cfg": {"version": 2}}

    monkeypatch.setattr(
        "compiler.realsas_compiler_services.orchestrator.mainline._adapter_impl_hash",
        lambda adapter: {"pkg.mod:stage_b": "new-impl"}.get(adapter, "stable"),
    )
    monkeypatch.setattr(
        "compiler.realsas_compiler_services.orchestrator.mainline._fingerprint",
        lambda plan, ledger, manifest, stage, implementation_hash: (
            "new-fingerprint",
            "new-policy",
        ),
    )

    assert _reset_stale_failures(plan, ledger, manifest) is True
    by = {row["id"]: row for row in ledger["stages"]}
    assert by["02_B"]["status"] == "PENDING"
    assert by["02_B"]["blockers"] == []
    assert by["03_C"]["status"] == "PENDING"
    assert ledger["history"][-1]["reason"] == "STALE_FAILURE_IDENTITY"


def test_stable_failure_identity_remains_sealed(monkeypatch):
    plan = {
        "stages": [
            {
                "ordinal": 1,
                "id": "01_A",
                "depends_on": [],
                "manifest_keys": [],
                "policy": {},
                "adapter": "pkg.mod:stage_a",
            }
        ]
    }
    ledger = {
        "run_id": "RUN",
        "execution_class": "WITNESS",
        "pipeline_plan_sha256": "p" * 64,
        "stages": [
            {
                **_row(1, "01_A", "FAIL"),
                "outputs": [],
                "input_fingerprint": "same-fingerprint",
                "implementation_hash": "same-impl",
                "policy_hash": "same-policy",
                "blockers": ["SEALED_FAIL"],
            }
        ],
    }
    monkeypatch.setattr(
        "compiler.realsas_compiler_services.orchestrator.mainline._adapter_impl_hash",
        lambda adapter: "same-impl",
    )
    monkeypatch.setattr(
        "compiler.realsas_compiler_services.orchestrator.mainline._fingerprint",
        lambda plan, ledger, manifest, stage, implementation_hash: (
            "same-fingerprint",
            "same-policy",
        ),
    )
    assert _reset_stale_failures(plan, ledger, {}) is False
    assert ledger["stages"][0]["status"] == "FAIL"
    assert ledger["stages"][0]["blockers"] == ["SEALED_FAIL"]


def test_failure_diagnostics_are_persisted_with_hash_bound_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    diagnostics = {
        "status": "FAIL",
        "evaluated_candidates": [
            {"cap": 128, "passed": False, "p95": 0.12},
            {"cap": 256, "passed": False, "p95": 0.08},
        ],
    }
    ref = _persist_failure_diagnostics(
        run_id="RUN",
        stage_id="14_GSA_BUILD",
        diagnostics=diagnostics,
    )
    path = Path(ref["path"])
    assert path == (
        tmp_path / "runs" / "RUN" / "artifacts" / "14_GSA_BUILD"
        / "failure_diagnostics.json"
    ).resolve()
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "RealSaS.StageFailureDiagnostics.v1"
    assert payload["run_id"] == "RUN"
    assert payload["stage_id"] == "14_GSA_BUILD"
    assert payload["diagnostics"] == diagnostics
    assert ref["bytes"] == path.stat().st_size
    assert len(ref["sha256"]) == 64
    assert len(ref["diagnostics_hash"]) == 64
