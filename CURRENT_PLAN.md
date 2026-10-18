# RealSaS-OPT — Current Plan

**Date:** 2026-10-18  
**Status:** PRETRAINING MATERIALIZATION / FOCUSED POST-V8 RELATION PROGRAM  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Verified executable pretraining source:** `726768813c523081166cc1ac49f08453260b66b9`  
**Review vehicle:** PR #116  
**Production authority:** `main` until reviewed merge  
**Training policy:** reach artifact-bound readiness first; execute no optimizer step before readiness PASS

## 0. Non-negotiable rules

### One canonical carrier

```text
K x candidate M*
        ↓
static admission + candidate mechanics
        ↓
deterministic selection
        ↓
ONE canonical M/G/W_M
        ↓
Graphics + Motion + Runtime
```

A second deformable visual mesh remains forbidden.

### Post-M1 carrier seal

```text
Geometry + Mechanics -> shared H -> M1 seal -> A/R/T/J specialist learning
```

`scene_core` LR remains zero unless a separate preregistered carrier-falsification court authorizes reopening with non-regression gates.

### Truth / conditioning

- unknown is first-class;
- unidentifiable owner/amodal/presentation targets abstain;
- view-conditioned truth requires matching prediction conditioning;
- Compiler may reconcile admissible evidence but may not fabricate truth or hide model incapacity.

## 1. Completed source tranche

Exact source `726768813c523081166cc1ac49f08453260b66b9` passed:

- pretraining candidate source gate run `39137955380`;
- full vNext suite run `39137955347`;
- PR #116 broad gate set **15/15 success** on the same executable head.

Implemented source now includes:

- SHA-scoped candidate preparation writer;
- no-write quarantine separation;
- full remote round-trip verifier;
- Relation Truth V2 materializer;
- Boundary Causality V1 materializer;
- exact completed-V8 relation-head → HeadV2 zero-step migration;
- materialization seal;
- artifact-bound preregistration/readiness audit.

Training remains zero-step.

## 2. Quarantined remote — forensic result CLOSED

Remote:

`gdrive:RealSaS_Fit8_IRIS_vNext_20261016`

Read-only inspection result:

`INCOMPLETE_REMOTE_SNAPSHOT`

Observed facts:

- only `teachers/` exists at top level;
- `prepared/` and `receipts/` are missing;
- teacher batch SHA256 is `7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27` and matches stale run `39121291171`;
- George `.teacher.npz` is absent while its receipt exists.

Decision: **never consume or promote this remote**. Preserve only as forensic evidence.

Machine record: `docs/research/IRIS_VNEXT_QUARANTINE_CONNECTOR_AUDIT_20261018.json`.

## 3. Candidate materialization — NEXT EXECUTION

Use only the manual candidate workflow:

`.github/workflows/iris_vnext_fit8_pretraining_candidate_self_hosted_v1.yml`

For one exact final active-review SHA:

1. obtain a read-only quarantine audit artifact bound to the same SHA;
2. hydrate exact Stage01 and legacy admitted inputs by immutable file ids/hashes;
3. export exact authored-motion teacher batch;
4. build the V4 cohort + PreparationReceipt;
5. materialize Relation Truth V2;
6. materialize Boundary Causality V1;
7. hydrate exact completed V8 authority artifacts;
8. migrate only the V8 relation head to `GraphicsRelationHeadV2`, with **zero training steps**;
9. assemble candidate under `gdrive:RealSaS_Fit8_IRIS_vNext_candidates/<exact_sha>`;
10. refuse any non-empty candidate destination;
11. copy Drive candidate back locally and require complete fileset + SHA256 equality;
12. seal `PRETRAINING_MATERIALIZATION_RECEIPT.json` with `training_authorized=false`.

No automatic shared-remote promotion.

## 4. Exact V8 migration source

The only authorized completed-V8 migration source is selected by V8 `LATEST.json`:

```text
source commit:
  5083c9915533ec8ec6530c7d229c635887abfea7
checkpoint:
  step_00012000_ac23a9ea448e465cb668773a131981be.pt
step:
  12000
checkpoint sha256:
  0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5
binding sha256:
  735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2
```

A second step-12000 file is not authority because `LATEST.json` does not select it.

Migration requirements:

- historical V8 file unchanged;
- scene_core excluded from migrated artifact;
- optimizer state excluded;
- no training step;
- compatible V1 relation parameters copied exactly;
- new projected-boundary rows neutral-initialized and camera-conditioned.

## 5. Truth support audit after materialization

For each family record valid/abstained/positive/negative counts before defining held-out courts.

Current expected policy:

- `same_drawable_affinity`: abstain absent independent authority;
- `cut_probability`: abstain absent independent authority;
- `attachment_affinity`: abstain absent independent authority;
- `front_probability`: require positive + negative support;
- `visibility_probability_a/b`: require positive + negative support;
- `occlusion_probability`: current truth is positive-only; not a binary PASS court;
- projected alpha/appearance boundaries: camera-conditioned evidence, evaluate only under valid support.

No fabricated negatives.

## 6. Preregistration gate — FINAL STEP BEFORE TRAINING

Start from:

`docs/research/IRIS_VNEXT_FOCUSED_RELATION_PREREGISTRATION_TEMPLATE_V1.json`

Fill it only after materialization, because exact artifact hashes are part of the preregistration.

Required fields include:

- final source commit SHA;
- materialization receipt SHA256;
- migration receipt SHA256;
- Relation Truth V2 manifest SHA256;
- held-out split policy + seed;
- metric-specific held-out gates and thresholds;
- non-regression gates;
- relation-head LR, weight decay, maximum steps, eval/checkpoint cadence;
- `scene_core_learning_rate = 0.0`;
- `thresholds_locked_before_training = true`.

Then run:

`tools/iris_vnext/audit_fit8_pretraining_readiness_v1.py`

Required terminal state:

`READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED`

That exact status is the requested stopping point before training.

## 7. Work after the training boundary

Not execute yet:

- focused frozen-carrier relation/boundary optimization;
- independent TESSA topology court;
- remaining Compiler/runtime attachment/support-band behavioral closure;
- any capacity/LoRA/extraction change not justified by a failed court.

No broad 12k rerun is authorized by source readiness alone.

## Execution order from here

```text
A. make this final state/plan/docs head green
B. fast-forward PR #116 to the same green head
C. dispatch exact-SHA read-only quarantine audit
D. dispatch exact-SHA candidate materialization
E. inspect round-trip/materialization/truth/migration receipts
F. write artifact-bound preregistration
G. run readiness audit
H. STOP at READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED
```
