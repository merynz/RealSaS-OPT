# Evidence and Lineage Rules

> Alfred's reasoning discipline for RealSaS repo reconstruction. Non-authoritative.

## Evidence precedence

Use the strongest applicable evidence, in this order:

1. **Live canonical authority + exact current source/test agreement**
2. **Exact current main implementation with passing, relevant mechanical proof**
3. **Sealed historical proof whose scope still matches unchanged semantics**
4. **Current documentation/navigation summaries**
5. **Historical canonical documents explicitly superseded for current topology**
6. **Non-main branch artifacts/commits**
7. **Conversation/agent claims without repository evidence**

A newer timestamp does not automatically outrank a stronger authority class.

## Confidence classes

- **A — CURRENTLY PROVEN:** current authority, current source, relevant tests/CI and bindings agree.
- **B — CURRENTLY SUPPORTED:** current source/authority agree but final end-to-end mechanical proof is absent or stale.
- **C — HISTORICAL GREEN:** once mechanically proven, but current readiness or source identity has changed.
- **D — UNRESOLVED:** conflicting evidence, open blocker, incomplete lineage, or timeout-era work with unclear closure.
- **F — FALSIFIED / SUPERSEDED:** explicitly rejected, replaced, or prohibited by newer authority.

Never silently promote C/D to A.

## Lineage record

For every disputed subsystem, reconstruct:

```text
question/hypothesis
  -> implementation
  -> measurement
  -> failure or success
  -> authority decision
  -> supersession/repair
  -> current consumer
```

A surviving source file does not imply a surviving product claim.

## Dependency reasoning

The 46-stage DAG is the primary execution map. For any stage:
- upstream authority = exact transitive `depends_on` closure;
- implementation = bound adapter + imported core modules;
- downstream meaning = direct and transitive consumers;
- repair owner must be explicit;
- downstream PASS cannot repair an invalid upstream producer.

## Frozen-identity rules

- Do not use `latest` aliases as evidence.
- Cache/PASS reuse requires exact input fingerprint and output SHA.
- Frozen Stage18 mesh cannot be silently edited after Stage35 failure; repair creates a new Stage18 lineage.
- Appearance assets are invalid if bound mesh/addressing/camera/output-direction identity changes.
- Run-local execution ledgers are subject authority; repository `ACTIVE_RUN_V2.json` is implementation governance only.

## Historical-code handling

V1 / Mage / FIT / donor-era code can be:
- provenance,
- a generic numerical helper,
- a source of a falsified idea,
- or dead compatibility surface.

It is **not current product authority** unless the current V2 authority/import closure explicitly admits its semantics.

## Timeout-era handling

For work performed during the recovery-polling timeout period:
- default classification is **D — unresolved** if closure is not mechanically established;
- non-main branches remain evidence-only under repository policy;
- never reconstruct intent from branch name alone;
- inspect diff, tests, CI, artifacts, and later superseding commits;
- prefer preserving uncertainty over “finishing” an ambiguous abandoned repair.

## Conflict rule

If docs disagree:
1. check current readiness/current state;
2. check exact source implementation;
3. check tests and run evidence;
4. inspect the commit that changed the claim;
5. record the conflict explicitly.

Do not average contradictory documents into a synthetic truth.

## Mutation rule during reconstruction

Until the map is trustworthy:
- no architecture rewrite;
- no broad cleanup;
- no main promotion;
- no “fix everything” patches;
- only memory-map changes and narrowly justified forensic instrumentation are allowed.

Production changes begin only after the affected lineage reaches at least confidence B and its repair owner is identified.
