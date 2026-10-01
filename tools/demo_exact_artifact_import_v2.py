from __future__ import annotations

"""Fail-closed importer for exact historical artifacts into DEMO_WITNESS runs.

This tool never claims that the current adapter reproduced imported bytes. Imported
rows are persisted as PASS_DEMO_ONLY with an explicit execution_mode and provenance
hash. Normal WITNESS / product runs are rejected.
"""

import argparse
import json
import shutil
from pathlib import Path

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.orchestrator.mainline import (
    PLAN_PATH,
    _adapter_impl_hash,
    _fingerprint,
    _ledger_map,
    _refresh,
    _seal_outputs,
    _stage_map,
    atomic_json,
    load_json,
    run_ledger_path,
    run_manifest_path,
    sha256_file,
    topological_stage_ids,
    validate_demo_witness_authorization,
    validate_ledger,
)
from compiler.realsas_compiler_services.orchestrator.status_semantics import (
    DEMO_ONLY_STATUS,
    dependency_status_admissible,
)


_SPEC_SCHEMA = "RealSaS.DemoExactStageArtifactImportSpec.v1"
_EXECUTION_MODE = "DEMO_EXACT_ARTIFACT_IMPORT"


def _load_spec(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != _SPEC_SCHEMA:
        raise RuntimeError("DEMO_IMPORT_SPEC_SCHEMA_INVALID")
    if payload.get("product_authority_claimed") is not False:
        raise RuntimeError("DEMO_IMPORT_PRODUCT_AUTHORITY_FORBIDDEN")
    stages = dict(payload.get("stages") or {})
    if not stages:
        raise RuntimeError("DEMO_IMPORT_STAGE_SET_EMPTY")
    return payload


def _reset_import_subgraph(plan: dict, ledger: dict, imported_ids: set[str]) -> None:
    stages = _stage_map(plan)
    affected = set(imported_ids)
    changed = True
    while changed:
        changed = False
        for stage in plan["stages"]:
            sid = str(stage["id"])
            if sid in affected:
                continue
            if any(str(dep) in affected for dep in stage.get("depends_on", ())):
                affected.add(sid)
                changed = True
    for row in ledger["stages"]:
        if str(row["id"]) not in affected:
            continue
        row.update(
            status="PENDING",
            input_fingerprint="",
            implementation_hash="",
            policy_hash="",
            outputs=[],
            diagnostics_hash="",
            blockers=[],
            wall_seconds=0.0,
            performance={},
        )
        row.pop("execution_mode", None)
        row.pop("import_provenance", None)
        row.pop("import_provenance_hash", None)
    ledger.setdefault("history", []).append(
        {
            "event": "DEMO_EXACT_IMPORT_SUBGRAPH_RESET",
            "imported_stage_ids": sorted(imported_ids),
            "affected_stage_ids": sorted(affected),
        }
    )
    _refresh(plan, ledger)


def import_exact_demo_artifacts(*, run_id: str, spec_path: Path) -> dict:
    plan = load_json(PLAN_PATH)
    ledger_path = run_ledger_path(run_id)
    manifest_path = run_manifest_path(run_id)
    ledger = load_json(ledger_path)
    manifest = load_json(manifest_path)
    validate_ledger(plan, ledger)
    if str(ledger.get("execution_class") or "") != "DEMO_WITNESS":
        raise RuntimeError("DEMO_IMPORT_REQUIRES_DEMO_WITNESS")
    validate_demo_witness_authorization(
        manifest,
        subject_id=str(ledger.get("subject_id") or ""),
    )

    spec = _load_spec(spec_path)
    if str(spec.get("run_id") or "") != str(run_id):
        raise RuntimeError("DEMO_IMPORT_RUN_ID_DRIFT")
    if str(spec.get("subject_id") or "") != str(ledger.get("subject_id") or ""):
        raise RuntimeError("DEMO_IMPORT_SUBJECT_ID_DRIFT")

    declared = dict(spec.get("stages") or {})
    unknown = set(declared) - set(_stage_map(plan))
    if unknown:
        raise RuntimeError("DEMO_IMPORT_STAGE_UNKNOWN:" + ",".join(sorted(unknown)))

    _reset_import_subgraph(plan, ledger, set(declared))
    atomic_json(ledger_path, ledger)

    source_artifact = dict(spec.get("source_artifact") or {})
    source_artifact_hash = content_sha256(source_artifact)
    imported = []

    for stage_id in topological_stage_ids(plan):
        if stage_id not in declared:
            continue
        stage = _stage_map(plan)[stage_id]
        row = _ledger_map(ledger)[stage_id]
        for dep in stage.get("depends_on", ()):
            dep_row = _ledger_map(ledger)[str(dep)]
            if not dependency_status_admissible(
                ledger, str(dep_row.get("status") or "")
            ):
                raise RuntimeError(
                    f"DEMO_IMPORT_DEPENDENCY_NOT_ADMISSIBLE:{stage_id}:{dep}"
                )

        stage_spec = dict(declared[stage_id] or {})
        outputs_spec = list(stage_spec.get("outputs") or ())
        if not outputs_spec:
            raise RuntimeError("DEMO_IMPORT_OUTPUT_SET_EMPTY:" + stage_id)

        stage_root = (
            Path.home()
            / "realsas_authority"
            / "runs"
            / str(run_id)
            / "artifacts"
            / stage_id
        ).resolve()
        stage_root.mkdir(parents=True, exist_ok=True)
        for old in stage_root.iterdir():
            if old.is_file():
                old.unlink()
            elif old.is_dir():
                shutil.rmtree(old)

        outputs = []
        copied = []
        for item in outputs_spec:
            item = dict(item)
            source = Path(str(item.get("source_path") or "")).expanduser().resolve()
            expected = str(item.get("expected_sha256") or "")
            if not source.is_file() or len(expected) != 64:
                raise RuntimeError("DEMO_IMPORT_SOURCE_REF_INVALID:" + stage_id)
            actual = sha256_file(source)
            if actual != expected:
                raise RuntimeError(
                    f"DEMO_IMPORT_SOURCE_SHA_DRIFT:{stage_id}:{source.name}"
                )
            target_name = str(item.get("target_name") or source.name)
            target = (stage_root / target_name).resolve()
            if target.parent != stage_root:
                raise RuntimeError("DEMO_IMPORT_TARGET_NAME_INVALID:" + target_name)
            shutil.copyfile(source, target)
            if sha256_file(target) != expected:
                raise RuntimeError("DEMO_IMPORT_COPY_SHA_DRIFT:" + stage_id)
            outputs.append(
                {
                    "path": str(target),
                    "sha256": expected,
                    "schema": str(item.get("schema") or ""),
                    "authority_class": str(item.get("authority_class") or ""),
                }
            )
            copied.append(
                {
                    "source_path": str(source),
                    "target_path": str(target),
                    "sha256": expected,
                }
            )

        sealed = _seal_outputs(outputs, allowed_root=stage_root)
        implementation_hash = _adapter_impl_hash(stage["adapter"])
        fingerprint, policy_hash = _fingerprint(
            plan, ledger, manifest, stage, implementation_hash
        )
        provenance = {
            "schema": "RealSaS.DemoExactStageArtifactImportProvenance.v1",
            "stage_id": stage_id,
            "source_artifact": source_artifact,
            "source_artifact_hash": source_artifact_hash,
            "copied_outputs": copied,
            "evidence": dict(stage_spec.get("evidence") or {}),
            "product_authority_claimed": False,
            "adapter_execution_claimed": False,
        }
        provenance_hash = content_sha256(provenance)
        row.update(
            status=DEMO_ONLY_STATUS,
            attempts=int(row.get("attempts", 0)) + 1,
            input_fingerprint=fingerprint,
            implementation_hash=implementation_hash,
            policy_hash=policy_hash,
            outputs=sealed,
            diagnostics_hash=provenance_hash,
            blockers=[],
            wall_seconds=0.0,
            performance={"execution_mode": _EXECUTION_MODE},
            execution_mode=_EXECUTION_MODE,
            import_provenance=provenance,
            import_provenance_hash=provenance_hash,
        )
        ledger.setdefault("history", []).append(
            {
                "event": "DEMO_EXACT_ARTIFACT_IMPORTED",
                "stage_id": stage_id,
                "source_artifact_hash": source_artifact_hash,
                "import_provenance_hash": provenance_hash,
                "product_authority_claimed": False,
                "adapter_execution_claimed": False,
            }
        )
        _refresh(plan, ledger)
        atomic_json(ledger_path, ledger)
        validate_ledger(plan, ledger)
        imported.append(stage_id)

    return {
        "schema": "RealSaS.DemoExactStageArtifactImportReceipt.v1",
        "status": "PASS",
        "run_id": run_id,
        "imported_stage_ids": imported,
        "source_artifact_hash": source_artifact_hash,
        "product_authority_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--import-spec", required=True)
    args = parser.parse_args()
    receipt = import_exact_demo_artifacts(
        run_id=str(args.run_id),
        spec_path=Path(args.import_spec).expanduser().resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
