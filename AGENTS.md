# RealSaS-OPT Agent Entry Contract

## Mandatory current handoff — 2026-10-21

Repository authority outranks conversational memory and historical notes.

### Read first

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V4.json`
4. `canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V1.json`
5. `canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json`
6. `canonical/STRUCTURAL_EVIDENCE_AB_COURT_METRIC_DEFINITIONS_V1.json`
7. `canonical/STRUCTURAL_EVIDENCE_INPUT_ADMISSION_V2.json`
8. `canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json`
9. `docs/research/IRIS_SHARED_STRUCTURAL_EVIDENCE_PLAN_20261020.md`
10. `docs/research/IRIS_H_GEO_P3_SCALE_XFER_RESULT_20261020.json`
11. `SYSTEM_INDEX.md`
12. `docs/platform/EXECUTION_LANES.md`

Historical continuation authority only:

- `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V3.json`
- `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V2.json`
- `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`
- `canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V1.json`
- `canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V1.json`

## Current continuation

Active lane: `SHARED_STRUCTURAL_EVIDENCE`.

Active workstream: `STRUCTURAL_EVIDENCE_AB_NOTEBOOK_BUILD_AND_HANDOFF`.

The actual chain is now:

```text
V2 tie-safe evidence admitted
    ↓
PR #137 inference-safe presentation producer/binding merged
    ↓
PR #138 matched-capacity A/B preregistration V2 locked
    ↓
PR #139 trainer + preflight + free-running + independent courts merged
    ↓
exact execution source bound post-merge
    ↓
BUILD ONE FAIL-CLOSED A100 RUN-ALL NOTEBOOK
    ↓
STOP FOR EXPLICIT USER HANDOFF BEFORE EXECUTION
```

Exact execution source:

`ebfe80fc9d0c43cc8075d3d38bdbc9231357b969`

Exact execution binding:

`canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V1.json`

No Structural-Evidence A/B optimizer step or scientific court has executed yet.

## Hard stop

If the user says only `continue / devam et`, continue source/artifact work **only until the fail-closed A100 run-all notebook is built and audited**.

Then explicitly report that the notebook boundary has been reached and stop before:

- notebook execution;
- optimizer steps;
- scientific court execution;
- observing a scientific A/B PASS/FAIL result.

Notebook build is authorized. Notebook execution is not.

## Locked A/B experiment

A — `A_NULL_PRESENTATION`

```text
shared scene + surface/mechanics + matched null presentation memory → TESSA V3
```

B — `B_ROLE_FREE_PRESENTATION`

```text
same shared scene + same surface/mechanics
+ inference-admissible role-free presentation memory → same TESSA V3
```

Both arms use the same architecture, initialization, seed, optimizer, schedule, update budget, subject/window policy and checkpoint schedule.

Locked training horizon: `2400` optimizer steps per arm, seed `26091991`.

V2 all-hit / teacher truth is supervision and independent evaluation authority only. It must never directly condition B inference.

## Notebook contract

The run-all notebook is orchestration, not a new source of scientific logic. It must call the merged repo tools at the exact execution SHA.

Required order:

1. mount/authenticate Drive and establish an output root;
2. clone repo and checkout `ebfe80fc9d0c43cc8075d3d38bdbc9231357b969`;
3. hydrate exact admitted V2 evidence and all bound Fit8 inputs;
4. verify every bound SHA256 before use;
5. require physical NVIDIA A100; no reduced-capacity fallback;
6. run `preflight_structural_evidence_ab_a100_v1.py` — real forward/backward, 0 optimizer steps;
7. train A exactly 2400 steps;
8. train B exactly 2400 steps;
9. run strict final-step free-running for both arms;
10. run topology/surface court;
11. run static presentation court;
12. run dynamic authored-motion court, which owns the final locked A/B decision;
13. persist checkpoints, receipts, court JSONs and hashes to Drive.

A valid scientific FAIL is a completed result and must be persisted. Infrastructure/integrity failures fail closed separately.

## Court law

Final PASS requires all three families:

```text
TOPOLOGY_SURFACE_NO_REGRESSION
AND PRESENTATION_PRIMARY
AND MOTION_PRIMARY
```

Presentation metrics:

- `GLOBAL_VISIBILITY_ACCURACY`
- `LOCAL_PARTIAL_ORDER_ACCURACY`
- `POSE_DEPENDENT_ORDER_FLIP_ACCURACY`
- `FRAGMENT_IDENTITY_STABILITY`
- `SOURCE_ART_SUPPORT_PRESERVATION`

Motion metrics:

- `OWNER_FRAGMENT_STABILITY_OVER_MOTION`
- `GAP_TEAR_RATE`
- `ATTACHMENT_PERSISTENCE_WHERE_AUTHORITATIVE`
- `G3_DYNAMIC_QUALIFICATION`

Numeric thresholds are already locked in preregistration V2 and may not be relaxed after result observation.

## Teacher/evaluator boundary

The authored teacher export may provide rest/posed geometry to the **independent evaluator only**.

For the isolated TESSA A/B motion court, candidate vertices are bound to the authored rest surface with exact closest-point barycentric coordinates and transported through the same authored pose field for both arms.

This does **not** mint or assume product AXIS/MIRA authority.

Forbidden in training/inference:

- teacher geometry as conditioning;
- teacher topology;
- teacher rig/skin;
- teacher motion;
- named semantic taxonomy as structure;
- material ID as drawable owner truth.

## Authority law

```text
main              = merged product code truth
review branch     = temporary source collaboration
Artifact Registry = immutable artifact/dependency identity
Attempt           = durable research truth
ProductRevision   = durable production truth
Workflow Engine   = durable execution state
```

Green CI is not scientific PASS. Engineering preflight is not scientific PASS. Evidence materialization is not scientific PASS. Drive paths are transport, not authority. Scientific claims require exact hashes and typed receipts/courts bound to exact source and preregistration.

Compiler alone may mint canonical M.

Unknown / ABSTAIN is first-class when source-defensible truth is absent. Pairwise front/order is not global visibility.

## Product invariant

```text
8-view observations
        ↓
IRIS continuous shared structural evidence
        +
qualified metric surface provider
        ↓
geometry + graphics + mechanics evidence
        ↓
role-free topology / structuralization
        ↓
Compiler qualification
        ↓
ONE canonical animation-ready M
        ↓
3D mechanics → source-preserved editable 2D presentation
```

No second deformable visual mesh.

## Immutable older authority

IRIS V8 remains immutable:

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- scene core sealed after M1.

`H-GEO-P3-SCALE-XFER` remains sealed `MIXED`; do not automatically resume P4 or treat the current endpoint diagnostic as a universal product gate.

## Fit8 / claim law

Fit8 is shared-system contract qualification, not unseen-generalization proof. A completed training run is not PASS; the preregistered independent court is the authority.

## Source-work exit rule

Every source change must leave exact source, owner, rationale, evidence, proved/not-proved scope, replaced/falsified authority and next action.

When interpretation changes, update:

- `CURRENT_STATE.md`;
- `CURRENT_PLAN.md`;
- the current machine-readable continuation ledger (`canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V4.json`);
- the active execution/preregistration binding where appropriate.
