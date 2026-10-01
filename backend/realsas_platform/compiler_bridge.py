from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from compiler.realsas_compiler_services.orchestrator import mainline

from .persistence.models import AttemptRow, CompilerRunBindingRow
from .stage_graph import StageGraph


class CompilerBridgeRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class CompilerRunBinding:
    attempt_id: UUID
    compiler_run_id: str
    run_manifest_path: str
    run_ledger_path: str
    pipeline_plan_sha256: str


@dataclass(frozen=True)
class CompilerStageOutcome:
    stage_id: str
    status: str
    executed_stage_ids: tuple[str, ...]
    outputs: tuple[dict, ...]
    blockers: tuple[str, ...]
    diagnostics_hash: str


def _stage_attempt_counts(ledger: dict) -> dict[str, int]:
    return {
        str(row["id"]): int(row.get("attempts", 0))
        for row in ledger.get("stages") or ()
    }


def classify_execution_delta(
    *,
    before: dict[str, int],
    after: dict[str, int],
    allowed_stage_ids: tuple[str, ...],
) -> tuple[str, ...]:
    executed = tuple(
        stage_id
        for stage_id, count in after.items()
        if count > int(before.get(stage_id, 0))
    )
    forbidden = sorted(set(executed) - set(allowed_stage_ids))
    if forbidden:
        raise CompilerBridgeRejected(
            "COMPILER_PLATFORM_PLAN_DRIFT:" + ",".join(forbidden)
        )
    return executed


def bind_existing_compiler_run(
    session: Session,
    *,
    graph: StageGraph,
    attempt_id: UUID,
    compiler_run_id: str,
) -> CompilerRunBinding:
    if session.in_transaction():
        raise RuntimeError("COMPILER_RUN_BINDING_REQUIRES_CLEAN_SESSION")
    plan = mainline.load_json(mainline.PLAN_PATH)
    plan_sha = mainline.validate_plan(plan)
    if tuple(str(row["id"]) for row in plan["stages"]) != tuple(s.stage_id for s in graph.stages):
        raise CompilerBridgeRejected("COMPILER_PLATFORM_STAGE_GRAPH_DRIFT")

    manifest_path = mainline.run_manifest_path(compiler_run_id)
    ledger_path = mainline.run_ledger_path(compiler_run_id)
    if not manifest_path.is_file():
        raise CompilerBridgeRejected("COMPILER_RUN_MANIFEST_MISSING")
    if not ledger_path.is_file():
        raise CompilerBridgeRejected("COMPILER_RUN_LEDGER_MISSING")
    ledger = mainline.load_json(ledger_path)
    mainline.validate_ledger(plan, ledger)
    if str(ledger.get("run_id") or "") != compiler_run_id:
        raise CompilerBridgeRejected("COMPILER_RUN_ID_DRIFT")

    with session.begin():
        attempt = session.execute(
            select(AttemptRow).where(AttemptRow.id == attempt_id)
        ).scalar_one_or_none()
        if attempt is None:
            raise CompilerBridgeRejected("COMPILER_RUN_ATTEMPT_NOT_FOUND")
        existing = session.execute(
            select(CompilerRunBindingRow).where(
                CompilerRunBindingRow.attempt_id == attempt_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.compiler_run_id != compiler_run_id or existing.pipeline_plan_sha256 != plan_sha:
                raise CompilerBridgeRejected("COMPILER_RUN_BINDING_IMMUTABILITY_VIOLATION")
            return CompilerRunBinding(
                existing.attempt_id,
                existing.compiler_run_id,
                existing.run_manifest_path,
                existing.run_ledger_path,
                existing.pipeline_plan_sha256,
            )
        row = CompilerRunBindingRow(
            attempt_id=attempt_id,
            compiler_run_id=compiler_run_id,
            run_manifest_path=str(manifest_path),
            run_ledger_path=str(ledger_path),
            pipeline_plan_sha256=plan_sha,
        )
        session.add(row)
        return CompilerRunBinding(
            attempt_id, compiler_run_id, str(manifest_path), str(ledger_path), plan_sha
        )


def execute_bound_compiler_stage(
    session_factory: sessionmaker[Session],
    *,
    graph: StageGraph,
    attempt_id: UUID,
    stage_id: str,
    allowed_stage_ids: tuple[str, ...],
) -> CompilerStageOutcome:
    graph.get(stage_id)
    with session_factory() as session:
        binding = session.execute(
            select(CompilerRunBindingRow).where(
                CompilerRunBindingRow.attempt_id == attempt_id
            )
        ).scalar_one_or_none()
        if binding is None:
            raise CompilerBridgeRejected("COMPILER_RUN_BINDING_REQUIRED")

    plan = mainline.load_json(mainline.PLAN_PATH)
    plan_sha = mainline.validate_plan(plan)
    if plan_sha != binding.pipeline_plan_sha256:
        raise CompilerBridgeRejected("COMPILER_RUN_PLAN_SHA_DRIFT")

    ledger_path = Path(binding.run_ledger_path)
    before_ledger = mainline.load_json(ledger_path)
    before = _stage_attempt_counts(before_ledger)

    rc = mainline.execute(
        binding.compiler_run_id,
        targets=(stage_id,),
        resume=True,
    )

    after_ledger = mainline.load_json(ledger_path)
    mainline.validate_ledger(plan, after_ledger)
    after = _stage_attempt_counts(after_ledger)
    executed = classify_execution_delta(
        before=before,
        after=after,
        allowed_stage_ids=allowed_stage_ids,
    )
    row = next(
        row for row in after_ledger["stages"] if str(row["id"]) == stage_id
    )
    status = str(row.get("status") or "PENDING")
    if rc != 0 and status not in mainline.FAIL_STATUSES:
        raise CompilerBridgeRejected(
            f"COMPILER_RETURN_STATUS_DRIFT:{stage_id}:rc={rc}:status={status}"
        )
    return CompilerStageOutcome(
        stage_id=stage_id,
        status=status,
        executed_stage_ids=executed,
        outputs=tuple(dict(x) for x in row.get("outputs") or ()),
        blockers=tuple(map(str, row.get("blockers") or ())),
        diagnostics_hash=str(row.get("diagnostics_hash") or ""),
    )
