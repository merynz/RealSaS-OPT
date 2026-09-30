from __future__ import annotations

"""Execution-status semantics shared by the V2 scheduler and adapters.

PASS_DEMO_ONLY is intentionally not a generic success state. It is admissible only
inside an explicitly authorized DEMO_WITNESS run, and once a dependency path is
demo-only the status taint must remain visible downstream.
"""

NORMAL_PASS_STATUSES = frozenset({"PASS", "CACHE_HIT"})
DEMO_ONLY_STATUS = "PASS_DEMO_ONLY"
PASS_STATUSES = frozenset(set(NORMAL_PASS_STATUSES) | {DEMO_ONLY_STATUS})
FAIL_STATUSES = frozenset({"FAIL", "ABSTAIN", "BLOCKED"})


def is_demo_witness(ledger: dict) -> bool:
    return str(ledger.get("execution_class") or "WITNESS") == "DEMO_WITNESS"


def dependency_status_admissible(ledger: dict, status: str) -> bool:
    status = str(status or "")
    if status in NORMAL_PASS_STATUSES:
        return True
    return status == DEMO_ONLY_STATUS and is_demo_witness(ledger)


def assert_demo_only_scope(ledger: dict) -> None:
    if is_demo_witness(ledger):
        return
    leaked = [
        str(row.get("id") or "")
        for row in ledger.get("stages") or ()
        if str(row.get("status") or "") == DEMO_ONLY_STATUS
    ]
    if leaked:
        raise RuntimeError(
            "PASS_DEMO_ONLY_OUTSIDE_DEMO_WITNESS:" + ",".join(sorted(leaked))
        )


def normalize_success_status(
    *,
    plan: dict,
    ledger: dict,
    stage_id: str,
    reported_status: str,
) -> str:
    """Return the persisted success status while preserving demo-only lineage."""
    status = str(reported_status or "").upper()
    if status == DEMO_ONLY_STATUS:
        if not is_demo_witness(ledger):
            raise RuntimeError("PASS_DEMO_ONLY_OUTSIDE_DEMO_WITNESS:" + str(stage_id))
        return status
    if status not in NORMAL_PASS_STATUSES:
        return status
    if not is_demo_witness(ledger):
        return status

    stage = next(
        (row for row in plan.get("stages") or () if str(row.get("id")) == str(stage_id)),
        None,
    )
    if stage is None:
        raise RuntimeError("DEMO_TAINT_STAGE_UNKNOWN:" + str(stage_id))
    by_id = {
        str(row.get("id")): row
        for row in ledger.get("stages") or ()
    }
    for dependency in stage.get("depends_on", ()):
        dep_id = str(dependency)
        dep = by_id.get(dep_id)
        if dep is None:
            raise RuntimeError(
                "DEMO_TAINT_DEPENDENCY_UNKNOWN:" + str(stage_id) + ":" + dep_id
            )
        if str(dep.get("status") or "") == DEMO_ONLY_STATUS:
            return DEMO_ONLY_STATUS
    return status
