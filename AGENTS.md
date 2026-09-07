# RealSaS-OPT Agent Entry Contract

This repository is intentionally structured so a new AI agent/chat/session can reconstruct current scientific context without relying on conversational memory.

## Mandatory first-read order

Before making any architecture, experiment, branch, or scientific-state claim:

1. `canonical/REHYDRATION_PACKET.md`
2. `CURRENT_STATE.md`
3. `canonical/CONTEXT_COVERAGE_AUDIT.md`
4. `canonical/BOOTSTRAP_AUDIT_QUEUE.md` when bootstrap is incomplete
5. `canonical/LIVE_AUTHORITY_MAP.md`
6. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
7. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
8. `canonical/EXPERIMENT_REGISTRY_V1.json`
9. `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` when searching for prior evidence not yet semantically indexed
10. `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` when historical reasoning/decision sequence matters

If generated views are missing or stale, regenerate locally:

```bash
python3 tools/build_knowledge_artifact_catalog.py
python3 tools/audit_context_coverage.py
python3 tools/render_authority_map.py --write
python3 tools/render_rehydration_packet.py --write
```

## Census vs semantic memory

Never confuse **discoverability** with **understanding**.

- `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` is the complete automated census of high-signal tracked knowledge artifacts. It proves that an artifact was discovered, not what it means.
- `canonical/CONTEXT_COVERAGE_AUDIT.md` measures semantic reconciliation into the continuity spine.
- An artifact may therefore be `CATALOGUED_UNREVIEWED`: known to exist, but not safe to summarize from filename/memory alone.

If a historical claim is not semantically indexed, locate candidate evidence in the catalog, inspect the source artifact/commit/workflow, then backfill the registry/ledger/journal. Do not guess.

## Bootstrap honesty rule

Read `canonical/BOOTSTRAP_COVERAGE_STATE_V1.json`.

While it says `BOOTSTRAP_AUDIT_INCOMPLETE`:

- an unindexed historical experiment/mechanism/report is **UNKNOWN / NEEDS AUDIT**, not absent;
- do not infer completeness from the experiment registry;
- do not reconstruct numerical historical claims from memory alone;
- use the artifact catalog + bootstrap audit queue to find the exact evidence cluster that needs backfill.

After bootstrap closure, newly unindexed high-signal artifacts are continuity regressions.

## Scientific claim discipline

Never conflate these states:

- source exists;
- mechanism implemented;
- mechanism experimentally tested;
- full formulation tested;
- canonical architecture promoted;
- FIT1 witness closed;
- generalization established.

Every experiment claim should answer:

1. What was the **local question**?
2. What **global program goal** did the experiment serve?
3. What exact arms/controls were compared?
4. Which mechanisms were present?
5. Which mechanisms were intentionally absent/held fixed?
6. What could the experiment falsify?
7. What could it **not** prove?
8. What result occurred under which evidence/metric/gate?
9. Did that move the product/scientific target closer, reveal a dead end, or merely isolate a variable?
10. Which next dependency/fork follows and why?

These fields belong in `canonical/EXPERIMENT_REGISTRY_V1.json`, not only in prose reports.

## Chronology rule

Scientific chronology is append-only in `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`.

- New events require exact RFC3339 UTC timestamps.
- Historical backfill must preserve the strongest verified time precision; never invent clock times.
- Corrections/retractions are new journal events. Do not silently rewrite history.

## Current known context guard

Before claiming that RealSaS still needs a RigAnything-equivalent Geppetto challenger, inspect:

`models/geppetto/challengers/riganything_mechanisms_v1.py`

A fuller research challenger already exists. Current AR-01 is a narrower minimal mechanical-feedback isolation experiment and must not be widened into a full RigAnything-formulation verdict.

## Branch rule

Only `main/CURRENT_STATE.md` is continuation authority.

- Active experiment branches are named in `canonical/AUTHORITY_MAP_V1.json` and `CURRENT_STATE.md`.
- Unregistered non-main branches are evidence-only by safe default.
- Branch recency does not imply authority.
- Do not delete evidence branches automatically; classify/dispose them first.

## Execution environment

Routine RealSaS scientific/mainline GitHub Actions run on the user's local self-hosted runner:

- labels: `self-hosted, linux, x64, realsas`
- known runner: `realsas-wsl-1660ti`
- operator path: `~/actions-runner`

Do not migrate routine jobs to GitHub-hosted runners or create branch-push fan-out without an explicit reason. Prior hosted Actions volume triggered a quota/usage warning.

## Completion transaction

An experiment is not complete merely because a notebook/report exists. Closing a scientific gate requires reconciling, as applicable:

- preregistration;
- exact result/evidence/provenance;
- implementation/source commit;
- `canonical/EXPERIMENT_REGISTRY_V1.json`;
- `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`;
- experiment authority ledger;
- architecture authority ledger when an architecture belief changed;
- context state / bootstrap coverage;
- `CURRENT_STATE.md` when stop/go changes;
- explicit supersession/retraction of any old interpretation.

The goal is not more documentation. The goal is **deterministic context reconstruction**.
