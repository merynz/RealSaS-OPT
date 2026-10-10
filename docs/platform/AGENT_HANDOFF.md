# Canonical agent entry and handoff

Source authority is `main`; the Go Attempt/Registry owns execution state.
Session/handoff records use the existing `attempt_events` journal. There is no
second ledger, migration or branch-based experiment state.

## Normal operation

1. Read `CURRENT_STATE.md` and this subject's `agent-context --id SUBJECT_UUID`.
2. Seal the intended release and SubjectInput. Use `research-start` for the first
   bootstrap or an explicit parent plus mandatory `intervention` for continuation.
3. `agent-enter --repo CANONICAL_CHECKOUT --request entry.json` pins the Attempt,
   release/spec hashes, SubjectInput/hash, target, current main SHA and actor.
4. `research-run` supplies the returned `agent_session_id` and the same tuple.
5. Inspect the Attempt until its command is terminal. `agent-exit` seals its state,
   exact artifact bindings and the next action. Keep the returned handoff hash.

An entry request is:

```json
{
  "attempt_id": "ATTEMPT_UUID",
  "subject_input_id": "SEALED_INPUT_UUID",
  "target_stage_id": "EXPLICIT_RELEASED_STAGE_ID",
  "canonical_code_sha": "EXACT_40_CHARACTER_MAIN_SHA",
  "expected_handoff_sha256": "PREVIOUS_HANDOFF_SHA_OR_EMPTY_FOR_FIRST_ENTRY",
  "created_by": "agent-name"
}
```

An exit request is:

```json
{
  "session_id": "TICKET_SESSION_UUID",
  "attempt_id": "TICKET_ATTEMPT_UUID",
  "scope_sha256": "TICKET_SCOPE_SHA256",
  "next_action": "Concrete continuation using the stored artifacts",
  "summary": "Observed result and remaining gap",
  "created_by": "agent-name"
}
```

The endpoints are `GET /v1/agents/context/{subject_id}`,
`POST /v1/agents/enter` and `POST /v1/agents/exit`. Public
`POST /v1/research/compile` rejects missing sessions. New commands validate the
active session and the API binary's embedded deployment SHA transactionally.
An exact idempotent command retry returns its existing receipt, including after
session closure; it cannot initiate more execution or change session identity.

## Recovery and source review

There is one active execution session per subject. Independent subjects can
execute concurrently; within a subject the existing workflow schedules its DAG.
This guard does not introduce a new scientific scheduler. Agents may inspect
read-only state without acquiring a session.

After interruption, entry must include `resume_session_id` from `agent-context`
and the same Attempt/input/target/code. A different actor can resume the exact
scope without rewriting it. A pending command prevents exit: inspect and reconcile
its stored workflow diagnostics rather than launching another run. Exit retries
are idempotent only with the same scope/action/summary/actor.

The next entry must acknowledge the latest handoff SHA. Its Attempt must be the
same one or an explicit child. An intended target/input/release change opens a
new scope after handoff; child interventions declare all actual changed nodes
and input roles. Stale handoff, unrelated lineage and silent scope changes fail.

CLI execution/release commands verify live remote main, clean tracked source
and absence of untracked source overlays. A detached exact-main checkout is
allowed. A short review branch remains source review and merges before execution;
it does not change the entry/exit contract. Exit is still possible after main
advances because it records the original immutable scope without executing code.

The deployed API SHA is embedded by `platform_deploy.py`; a development API must
also be built with `-ldflags '-X main.sourceCodeSHA=MAIN_SHA'`. An unbound binary
cannot accept a research session.

## Scope and evidence

No model name is special in this contract. Geometry, tree, skin, appearance,
new heads and unseen evaluations use the same released DAG semantics. A new
independent output node preserves unrelated artifacts; an actual shared trunk
dependency invalidates its consumers. Dataset size changes evaluation apparatus,
not the identity of an unrelated model artifact.

Unit contracts cover generic scope changes and dirty/stale/review checkouts;
PostgreSQL tests cover entry/resume/exit, pending commands, exact bindings, stale
handoffs and child lineage. Deployment smoke exercises real session -> API ->
Temporal -> Engine -> handoff, then exact artifact reuse after an independent
node removal. These are engineering checks, not model/generalization PASS.

Legacy direct scripts and historical workflow-dispatch paths remain a known
bypass surface restricted by `AGENTS.md` to sealed replay/CI. This API guard
does not claim filesystem/administrator isolation or complete removal of those
paths. See [architecture audit](MODULARITY_AUDIT_20261010.md) for the remaining
producer/consumer boundaries.
