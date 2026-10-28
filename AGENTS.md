# RealSaS-OPT Agent Entry Contract

## Mandatory current handoff — 2026-10-11

**First read:** `CURRENT_STATE.md`, then `docs/platform/EXECUTION_LANES.md`.
Use `canonical/CANONICAL_HANDOFF_20261007.md` for the sealed mechanics handoff.

If the user says only “continue / devam et”, resume from canonical `main` and the priority order in `CURRENT_STATE.md`. Do not resurrect an older runtime-audit, VF-11, recovery, research-branch or promotion-branch continuation merely because historical files still exist.

This repository must be resumable without conversational memory.

Before research execution, read `realsasctl agent-context --id <subject UUID>`
and enter with `agent-enter`: exact deployed main SHA, Attempt, released DAG,
immutable SubjectInput and explicit target. A live session must be explicitly
resumed; never resubmit a timed-out run blindly. `research-run` requires that
session ID. Exit with `agent-exit`, its exact scope hash, and the next action;
the Go owner seals current state/artifact bindings and returns a handoff hash.
The next entry must acknowledge that hash and stay on the same Attempt or an
explicit child. Read `docs/platform/AGENT_HANDOFF.md` for the command contract.

## Required operating path

Read `docs/platform/EXECUTION_LANES.md` before stateful development. Use Go
`realsasctl research-start` and `research-run` for component replacement;
explicitly select the target and inspect the resolved reuse/execute scope.
Use `product-compile` for product construction and `product-render` only for an
exact qualified ProductRevision. Research results, including scoped FIT PASS,
must not silently mint product authority. Preserve independent artifacts and
report code/artifact identities with timings. Do not drive a new result by
searching old branch names or editing a historical execution ledger.

For Fit8 or later multi-subject genericization campaigns, do **not** create a
notebook-owned authority or eight ad-hoc agent continuations. Read
`canonical/FIT8_PLATFORM_CAMPAIGN_V1.json` and
`docs/platform/FIT8_PLATFORM_CAMPAIGN.md`; use `realsasctl campaign-start`,
`campaign-run` and `campaign --id`. Every Fit8 subject enters the released DAG
at Stage01 and targets the same final product-closure stage. Knight is a separate
regression member. Colab/A100 may execute preregistered learned work, but it is a
compute worker only; imported receipts/checkpoints remain non-authoritative until
the subject-bound Compiler stages validate them. A campaign failure is a signal
to fix the owning generic module, never permission for a subject-specific branch.

Finite source-apparatus measurement may use the same Platform campaign API with
an explicit released component target, preserving all source-entry roots, the
entire cohort and its separate regression. Its `goal_complete` closes only that
measurement target; `full_closure_complete` remains false. This bounded research
scope must not enter an unrelated downstream training wait or claim FIT8/product
closure. The system-development FIT8 campaign still targets product closure.

Targets include model inference and Compiler modules, not only render. Treat
DAG nodes/edges as versioned release data: add, remove or rewire in a new
EngineRelease, inspect graph-aware impact, and preserve unrelated artifacts.
Keep implementation and plan changes on canonical main through short reviewed
PRs; research continuation belongs to the Attempt, never a long-lived branch.

M/G/W preservation is a boundary of the current Knight witness, not a global
architecture rule. Models, heads, datasets and evaluations may change under a
declared intervention. Only their actual DAG consumers invalidate; independent
artifacts retain identity. A shared trunk change legitimately affects its heads.
Never hide a dependency hash to make a requested scope appear smaller.

Use standard GitHub-hosted `ubuntu-latest` for host-independent CPU CI while
the repository is public; private-repository jobs skip. Notebook/Colab is the
default research/GPU lane. Local WSL/GTX remains an explicit native-host or
selected inference option. Old frozen recovery is manual provenance only.
Absolute G3 remains an open gap without cancelling teacher-relative FIT1 PASS.

## Current machine authority spine

Read current authority before historical provenance:

1. `canonical/EXPERIMENT_REGISTRY_V3.json`
2. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
3. `canonical/ACTIVE_RUN_V2.json`
4. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
5. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
6. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`

Historical provenance only:

- `canonical/EXPERIMENT_REGISTRY_V2.json`
- `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`

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

The latest developed FIT1 mechanics line is carrier-native: Stage19 owns frozen mechanical carrier `M`; AXIS V5.4.1 owns the hard causal required tree plus deterministic 256-bin XYZ; MIRA V5.5 predicts `W_M` directly on that exact `M`; Compiler validates/materializes the required tree and exact `(M,G,W_M)` bindings without semantic skin transfer.

Coincident controls are handled subject-free after binding: a mechanically unobservable subtree may inherit its parent frame; any coincident subtree with non-zero final `W_M` influence fails closed and requires explicit AXIS orientation/tail evidence.

Knight FIT1 mechanics are closed relative to the sealed source/teacher motion envelope. The absolute frozen G3 threshold remains inconclusive because the sealed teacher itself is outside that absolute threshold. This is not FIT8/FITK/unseen/product/generalization evidence.

Canonical public model naming is `IRIS -> TESSA -> AXIS -> MIRA`; legacy Geppetto/Arachne/ATLAS names may remain only where compatibility requires them.

## Current render baseline and priority

The user approved the single-M source-wins + existing CAA cutover. Read
`CURRENT_STATE.md` and `tests/fixtures/single_m_caa_knight_v1/baseline.json`.
Use `tools/demo/render_knight_m_caa_v1.py --output <directory>` for exact sealed
RUN regression, never the old retarget trial. Geometry, topology and recorded
motion all use M. There is no independent visual mesh/rebind continuation.
The previous main is retained only at `archive/visual-v-before-m-caa-20261011`.

Priority after source cutover: grip/weapon relation, then semantic overlap/draw
order. Preserve the approved CPU pixels and recorded M poses while investigating.
Native exact parity at diagonal depth ties and linear-light composition remain
open; do not mint scientific/product PASS from engineering regression.
IRIS shared-head research remains a proposal awaiting user approval.

For source work, generate the current map and inspect edited files with
`tools/realsas_architecture.py map` and `impact`; see
`docs/platform/DYNAMIC_ARCHITECTURE.md`. This is discovery, not reuse authority.
An unknown impact must be investigated; it never means zero consumers. At exit,
record exact source/Attempt/artifacts and the pending decision in the handoff.

## Product-quality rule

Geometry, Mechanics and Appearance are co-equal product authorities; Presentation is first-class editable addressing authority. A visually incorrect puppet is not accepted merely because mechanics are valid.

## Runtime invariant

Runtime consumes qualified product artifacts. It may not create a second topology, silently re-solve skin, perform donor search/generative correction, or mutate scientific authority during playback/render.

## Corpus status

`canonical/TRUTH_CORPUS_V1_STATUS_20261004.md` remains corpus authority. Current status is `SEALED_RAW__STRUCTURALLY_AUDITED__ADMISSION_PENDING`. Do not overclaim it as training-ready canonical truth.

The user authorized diverse Fit8 as the next program. Read
`canonical/FIT8_DIVERSITY_SELECTION_V1.json`, `canonical/FIT8_PLATFORM_CAMPAIGN_V1.json`,
`docs/corpus/FIT8_ADMISSION.md`, `docs/platform/FIT8_PLATFORM_CAMPAIGN.md` and
`docs/platform/PRODUCTIZATION.md`. Eight source identities are verified; every
subject must independently close source/import/rest pose, camera, observation and
normalization before learned-stage evidence is consumed. Knight stays the separate
regression witness. Shared model parameters and common contracts are required; no
per-character production fix or silent substitution. New Fit8 scientific state is
owned by the platform campaign/Attempts, never by notebook completion alone.

## Claim discipline

A FIT/witness PASS is scoped evidence for the exact subject/apparatus. Engineering CI is not unseen/generalization evidence. Knight FIT1 teacher-relative mechanics closure is not unseen/generalization or product authority.

## Read order

1. `CURRENT_STATE.md`
2. `canonical/CANONICAL_HANDOFF_20261007.md`
3. `SYSTEM_INDEX.md`
4. `canonical/V2_IMPLEMENTATION_READINESS.json`
5. `canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json`
6. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
7. `docs/platform/adr/0002-compiler-platform-boundary.md`
8. `docs/platform/adr/0003-go-control-plane.md`
9. `canonical/TRUTH_CORPUS_V1_STATUS_20261004.md`
10. older recovery/scientific material only as needed for provenance
