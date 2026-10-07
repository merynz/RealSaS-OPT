# RealSaS-OPT Agent Entry Contract

## Mandatory current handoff — 2026-10-07

**First read:** `canonical/CANONICAL_HANDOFF_20261007.md`, then `CURRENT_STATE.md`.

If the user says only “continue / devam et”, resume from canonical `main` and the priority order in `CURRENT_STATE.md`. Do not resurrect an older runtime-audit, VF-11, recovery, research-branch or promotion-branch continuation merely because historical files still exist.

This repository must be resumable without conversational memory.

## Canonical code-line rule

`main` is the single canonical continuation line.

Long-lived research/promotion/integration branches must not be used as scientific, product or execution authority. Temporary review branches are allowed only as short-lived source-code collaboration; they must merge promptly and must never become a second continuation state.

**Git is source-code history, not product/research state.**

## Research / product / execution separation

The normative state model is:

```text
Git/main          = code truth
Artifact Registry = immutable typed artifacts + dependencies
Attempt           = research truth
ProductRevision   = production truth
Workflow Engine   = durable execution truth
Proof/Qualification = promotion eligibility
```

Research experiments belong in `Attempt`, not branch chronology. Product promotion belongs in `ProductRevision`, not Git merge semantics.

Research execution should reuse exact qualified upstream artifacts according to dependency identity. Product compile must execute the required model/Compiler construction path for a new subject unless an exact semantic cache hit is proven. Product render/playback consumes a qualified ProductRevision and must not rerun model inference merely to render it.

## Platform-first execution rule

The Go control plane is the intended sole owner of durable product/research state, Artifact Registry metadata, Attempt/ProductRevision lifecycle, promotion, dependency scheduling and workflow state.

Current known enforcement gap: legacy GitHub Actions and direct `$REALSAS_AUTHORITY_ROOT` scripts can still bypass Platform state. Until that gap is closed, such direct paths are allowed only for CI/reproducibility of already sealed evidence. They must not be treated as the normal way to mint new research/product authority.

The active platform program is to make this rule mechanically unavoidable.

## Current scientific boundary

The promoted TESSA mechanical line establishes a real static Knight carrier survivor under frozen policy. It does **not** establish generic TESSA dynamic carrier-field transport, actual-motion product PASS, unseen generalization or native attachment closure.

Stage19 may own a statically qualified TESSA carrier. Stage35 must remain fail-closed for the TESSA product lane until the missing subject-free carrier-field transport proof exists.

Canonical public model naming is `IRIS -> TESSA -> AXIS -> MIRA`; legacy Geppetto/Arachne/ATLAS names may remain only where compatibility requires them.

## Fresh render baseline

Run `37578120472` on exact `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` freshly rebuilt Stage23, requalified Stage24 and rerendered Knight IDLE/RUN/SLASH. All three GIF hashes matched the frozen reference.

That run reused sealed artifacts; IRIS/TESSA/AXIS/MIRA were not reinferred. Do not describe it as a fresh source-to-puppet compile.

## Active priority order

1. Repository currentness closure.
2. Go platform enforcement: stateful work must flow through Artifact/Attempt/ProductRevision/workflow authority.
3. Performance optimization using measured telemetry and dependency-aware reuse/native hot paths.
4. Only then return to runtime/presentation image diagnosis if the image remains wrong.

Do not preempt priorities 1–3 by reopening visual/runtime forensics unless new evidence falsifies the sealed static topology/geometry/mechanical baseline.

## Product-quality rule

Geometry, Mechanics and Appearance are co-equal product authorities; Presentation is first-class editable addressing authority. A visually incorrect puppet is not accepted merely because mechanics are valid.

## Runtime invariant

Runtime consumes qualified product artifacts. It may not create a second topology, silently re-solve skin, perform donor search/generative correction, or mutate scientific authority during playback/render.

## Corpus status

`canonical/TRUTH_CORPUS_V1_STATUS_20261004.md` remains corpus authority. Current status is `SEALED_RAW__STRUCTURALLY_AUDITED__ADMISSION_PENDING`. Do not overclaim it as training-ready canonical truth.

## Claim discipline

A FIT/witness PASS is scoped evidence for the exact subject/apparatus. Engineering CI is not unseen/generalization evidence. Static TESSA survivor evidence is not dynamic product authority.

## Read order

1. `canonical/CANONICAL_HANDOFF_20261007.md`
2. `CURRENT_STATE.md`
3. `SYSTEM_INDEX.md`
4. `canonical/TESSA_MIRA_MECHANICAL_SURVIVOR_PROMOTION_LEDGER_V1_20261006.json`
5. `canonical/TESSA_MIRA_FINAL_DAG_PROMOTION_AUDIT_V1_20261007.json`
6. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
7. `docs/platform/adr/0002-compiler-platform-boundary.md`
8. `docs/platform/adr/0003-go-control-plane.md`
9. `canonical/TRUTH_CORPUS_V1_STATUS_20261004.md`
10. older recovery/scientific material only as needed for provenance
