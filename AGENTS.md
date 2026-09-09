# RealSaS-OPT Agent Entry Contract

This repository is intentionally structured so a new AI agent/chat/session can reconstruct current scientific context without relying on conversational memory.

## Mandatory first-read order

Before making any architecture, experiment, branch, or scientific-state claim:

1. `canonical/REHYDRATION_PACKET.md`
2. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl` — inspect the recent tail first so the ordered sequence of current decisions/requests and the reason for the current task are recovered, not only the final state
3. `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`
4. `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`
5. `CURRENT_STATE.md`
6. `canonical/FIT1_COMMIT_LINEAGE_V1.md` when exact FIT1-to-now chronology/commit provenance matters
7. `canonical/CONTEXT_COVERAGE_AUDIT.md`
8. `canonical/LIVE_AUTHORITY_MAP.md`
9. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
10. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
11. `canonical/EXPERIMENT_REGISTRY_V2.json`
12. `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` when locating exact older evidence

`canonical/EXPERIMENT_REGISTRY_V1.json` and `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` remain historical continuity/provenance inputs. They are not the current live registry/journal after the 2026-09-09 Geppetto promotion transaction. The current machine pointers are also declared in `canonical/AUTHORITY_MAP_V1.json` and `canonical/CONTEXT_STATE_V1.json`; if a future schema supersedes V2, follow those machine pointers rather than guessing from filenames.

If `canonical/BOOTSTRAP_COVERAGE_STATE_V1.json` says `BOOTSTRAP_AUDIT_CLOSED`, also read `canonical/AUDIT_OF_AUDITS_CLOSURE_20260907.md` before interpreting coverage semantics.

If generated views are missing or stale, regenerate locally:

```bash
python3 tools/render_fit1_commit_lineage.py
python3 tools/build_knowledge_artifact_catalog.py
python3 tools/audit_context_coverage.py
python3 tools/render_authority_map.py --write
python3 tools/render_rehydration_packet.py --write
```

## Learned model shorthand expansion — mandatory

A RealSaS learned model is **not** the whole subsystem around that model. Unless the discussion is explicitly restricted to model internals, expand subsystem shorthand before reasoning about responsibility:

- **IRIS** means: observation/camera contract -> learned IRIS evidence -> deterministic GSA/RiggingSurfaceIR assembly, validation and provenance -> learned consumers. Current Mage FIT1 signed-geometry evidence is the promoted scene-first V3 line; V2 remains the promoted underlying observation/foundation/evidence package where referenced by that lineage.
- **Geppetto** means: lossless RiggingSurfaceIR -> learned skeleton/control/parent/root/mechanical-salience proposal evidence -> Compiler exact graph qualification -> QualifiedSkeletonIR/canonical IDs.
- **Arachne** means: qualified surface+skeleton -> learned skin/deformation proposal -> Compiler skin/mesh/reference/simplex qualification -> qualified editable deformation state.
- **Compiler** is deterministic qualification/canonicalization/proof/routing authority; it is **not permission to invent missing learned semantics**.

Binding memory guard:

> **Geppetto is proposal, not canonical rig authority.**

If a future statement says “Geppetto made the skeleton,” restate it more precisely: Geppetto proposed controls/relations and the Compiler selected/qualified the legal canonical skeleton. Apply the equivalent distinction to IRIS/GSA and Arachne/Compiler.

Before moving a responsibility across layers, inspect `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` and ask whether the move creates a second semantic owner, hides model failure with deterministic repair, or violates fail-close.

## FIT1 continuity rule

The scientific FIT1 gate began at commit:

`f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee`

The first executable FIT base is:

`de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3`

`canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md` is the semantic epoch map. `canonical/FIT1_COMMIT_LINEAGE_V1.md/.json` is the exhaustive discovery/provenance ledger for commits descended from the FIT1 gate across live refs.

Every FIT1-descendant commit is contextually important. A commit subject proves only that a change exists; it does not prove that a mechanism was run, passed, promoted, or generalized. Use exact prereg/result/source/authority evidence for those claims.

## Census vs semantic memory

Never confuse **discoverability** with **understanding**.

- `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` is the automated census of high-signal tracked artifacts.
- `canonical/FIT1_COMMIT_LINEAGE_V1.json` makes FIT1-to-now changes chronologically recoverable.
- `canonical/CONTEXT_COVERAGE_AUDIT.md` applies the AOA disposition policy.
- `FIT1_COMMIT_LEDGER_COVERED` means exact change provenance is recoverable, **not** that its scientific claim is promoted.
- `HISTORICAL_PROVENANCE_ACCEPTED_RESIDUAL` means the artifact already existed at the FIT1 gate and remains discoverable historical evidence unless explicitly promoted elsewhere.

For any numerical/mechanistic historical claim, locate and inspect the exact source artifact rather than reconstructing it from a filename or memory.

## Bootstrap / AOA honesty rule

Read `canonical/BOOTSTRAP_COVERAGE_STATE_V1.json`.

While it says `BOOTSTRAP_AUDIT_INCOMPLETE`:

- an unindexed historical experiment/mechanism/report is **UNKNOWN / NEEDS AUDIT**, not absent;
- do not infer completeness from the experiment registry;
- do not reconstruct numerical historical claims from memory alone.

After `BOOTSTRAP_AUDIT_CLOSED`:

- closure means **context/provenance coverage**, not retroactive scientific validation of every historical file;
- FIT1-to-now changes must remain recoverable through the exhaustive commit ledger;
- pre-FIT high-signal residuals remain historical provenance by explicit disposition;
- any new high-signal artifact that is neither explicitly indexed nor ancestry/ledger-covered is a continuity regression and CI must fail.

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

Current experiment records belong in `canonical/EXPERIMENT_REGISTRY_V2.json`; V1 is retained as historical structure/provenance.

## Chronology rule — live continuation memory

Current scientific/project chronology is append-only in `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`. The V1 journal is historical and must not be rewritten.

- New events require exact RFC3339 UTC timestamps.
- Historical backfill must preserve the strongest verified time precision; never invent clock times.
- Corrections/retractions are new journal events. Do not silently rewrite history.
- **Before ending a session, append every substantive user/project decision that changes the next work item, architecture constraint, experiment selection, demo artifact queue, promotion interpretation, or stop/go state.**
- Record requests and priorities as requests/priorities; never upgrade them into experimental PASS, promotion or scientific evidence merely because the user chose the next direction.
- A new chat/agent must inspect the recent current-journal tail and be able to reconstruct the causal sequence: what was learned, what was decided next, what artifact/work was queued, and why the project moved on.
- The journal is not a miscellaneous diary. Only continuity-relevant decisions, corrections, experiments, promotions, architecture changes and engineering-queue changes belong there.

The continuity requirement is stronger than “know the latest state”: the agent should be able to answer questions such as “what did we decide immediately after IRIS FIT1?” from repository chronology without asking the user to repeat it.

## Current known context guards

Before claiming RealSaS needs a RigAnything-equivalent Geppetto challenger from scratch, inspect:

- `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md`
- `models/geppetto/challengers/riganything_mechanisms_v1.py`

A fuller research challenger already exists. AR-01 was a narrower minimal mechanical-feedback isolation experiment and must not be widened into a full RigAnything-formulation verdict.

Before citing an old IRIS learned result as clean current observation-only evidence, inspect:

- `canonical/IRIS_LEAK_SCOPE_20260903.md`
- `canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`
- `models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json` for the current promoted Mage signed-geometry witness.

Affected historical learned IRIS results remain quarantined; the repaired source firewall does not retroactively cleanse them.

## Branch rule

Only `main/CURRENT_STATE.md` is continuation authority.

- Active experiment branches are named in `canonical/AUTHORITY_MAP_V1.json` and `CURRENT_STATE.md`.
- Zero active experiments is a valid state and must not be treated as a manifest error.
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
- current experiment registry (`canonical/EXPERIMENT_REGISTRY_V2.json`);
- current scientific journal (`canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`);
- experiment authority ledger;
- architecture authority ledger when an architecture belief changed;
- context state / AOA coverage;
- `CURRENT_STATE.md` when stop/go changes;
- explicit supersession/retraction of any old interpretation.

The goal is not more documentation. The goal is **deterministic context reconstruction with scientific responsibility boundaries intact**.
