# IRIS/TESSA apparatus audit — Platform execution authority migration V1

**Date:** 2026-10-26  
**Status:** `ACTIVE_EXECUTION_AUTHORITY_RULE`

## Rule

Git branches and GitHub Actions are source-review / engineering-validation surfaces. They are not the durable scientific execution authority for this audit.

Canonical operational authority follows the existing Platform model:

- reviewed executable source: Git `main`;
- research question / preregistration: Platform `Experiment`;
- one execution: Platform `Attempt`;
- exact executable snapshot: `EngineRelease` / released research plan;
- immutable reports and evidence bytes: Artifact Registry / CAS;
- execution chronology and handoff: existing `attempt_events` / agent handoff;
- affected product/capability ownership: Responsibility / capability terrain.

No second branch-local provenance ledger is permitted.

## Existing 2026-10-25 courts

The already-recorded GitHub Actions courts remain useful **exploratory engineering evidence** and historical apparatus diagnostics. They are not erased or relabeled.

However, a result that materially selects or rejects the architecture used for Causal Truth / FIT8 must be either:

1. a proof-level deterministic/source-contract fact that does not depend on learned execution; or
2. rerun as a Platform-linked `Experiment -> Attempt -> Artifact` execution from reviewed `main`.

Therefore old learned/tiny courts may nominate treatments and eliminate obviously bad apparatus designs, but they do not by themselves freeze the final architecture.

## First Platform-native court

`ORACLE_Z_REPRESENTATION_PREREG_V2.md` is the first court created under this rule from the start. Its scientific mode fails closed without Platform Experiment/Attempt/EngineRelease/code-SHA context.

## Authority boundary

This migration changes no ProductRevision, ModelRelease, model channel, Truth139 authority or FIT8 claim. It changes only how future research execution identity and evidence are recorded.
