# RealSaS-OPT — Current State

**Date:** 2026-08-31  
**Canonical continuation branch:** `main`  
**Status:** `DTB_ND1_CLOSED__CONSUMER_INTERLOCK_CLOSED__DINO_S1_EARLY_TERMINATED_AFTER_S_B__SHARED_APPARATUS_AUDIT_REQUIRED__DEV32_CLOSED__NEW_DINO_TRAINING_NOT_AUTHORIZED`

This file is continuation authority only on `main`.

## One-line state

`consumer-sensitive depth tolerance is closed -> DINO S/B were run under one frozen SharedLearnerV1 -> S/B both primary-failed with essentially flat accessibility -> code audit exposed a material shared-apparatus confound -> L stopped partial, g not run -> frozen-component audit is now mandatory before any new DINO training`

## 1. Product/system contract — STABLE

North star:

`ONE NEUTRAL 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

Typed route remains:

```text
8 neutral views + known orthographic cameras
 -> IRIS forward depth d
 -> P = O + dF
 -> SurfaceBuilder / RiggingSurfaceIR
 -> Geppetto SkeletonProposalIR
 -> Compiler skeleton qualification
 -> Arachne SkinProposalIR
 -> Compiler skin qualification
 -> CanonicalPuppetGraph
 -> deformation/motion proof
 -> runtime
```

IRIS/Geppetto/Arachne emit evidence/proposals. Compiler owns canonical product state.

## 2. Geometry tolerance — CLOSED FOR CURRENT CONSUMER PROFILE

Historical frozen-D2 boundary:

`0.00225 <= epsilon_critical < 0.00250 RMS`

After DTB-ND1 robust local-plane normal derivation:

`0.00250 <= epsilon_critical < 0.00275 RMS`

Matched ell=0 depth absolute-P95 is approximately `.00490-.00539`.

Canonical authority:
`canonical/DTB_ND1_ROBUST_LOCAL_PLANE_CLOSURE_20260829.md`.

## 3. Downstream consumer validity — CLOSED FOR SACRIFICIAL G0/A0

The explicit Geppetto/Arachne coupling probe and Compiler qualification route are closed 6/6 for the sacrificial profile.

Canonical authority:
`canonical/CONSUMER_VALIDITY_INTERLOCK_CLOSURE_20260829.md`.

These sacrificial proxies are downstream-sensitive evaluation carriers, not product Geppetto/Arachne models.

## 4. DINO controlled ladder S1 — EARLY TERMINATED

Original preregistered order:

`S -> B -> L -> g`

Completed primary results:

| Rung | FIT_PROXY32 | TRAIN_DIAG32 | Hard route failures | Verdict |
|---|---:|---:|---:|---|
| S @32768 | 10/54 | 17/56 | 0 | `COMPLETE_PRIMARY_FAIL` |
| B @32768 | 10/54 | 18/56 | 0 | `COMPLETE_PRIMARY_FAIL` |
| L | no primary eval | no primary eval | n/a | `TERMINATED_PARTIAL_NOT_EVALUATED` |
| g | not run | not run | n/a | `NOT_RUN` |

L persisted training history only through step `192`; it is provenance and has no scientific interpretation.

DEV32 remained closed.

Canonical closure:
`canonical/DINO_LADDER_S1_EARLY_TERMINATION_CLOSURE_20260831.md`.

Machine-readable closure:
`experiments/iris_dino_controlled_20260829/DINO_LADDER_S1_EARLY_TERMINATION_V1.json`.

### Supported claim boundary

Supported:

> Under `SharedLearnerV1` and the S1 apparatus, S->B backbone scaling did not materially improve downstream accessibility at the fixed primary budget.

Not supported:

- L/g would fail;
- DINO-L/g are insufficient;
- backbone scale is globally irrelevant;
- S->B would remain flat under another decoder/loss/interface/native-detail path;
- partial L history predicts primary behavior.

## 5. Why the ladder stopped

A post-result code audit found that the fixed measuring apparatus itself may impose a shared accessibility ceiling:

- local DINO field is `37x37`;
- cross-view context uses `37 -> 16 -> 37`;
- all rung widths enter the same `1536 -> 256` stem;
- native-1024 detail access is a sparse 25-sample stencil + small MLP rather than a dense learned high-resolution field;
- final query prediction is pointwise;
- training uses uniform mean SmoothL1 with `beta=.01`.

None of these facts individually proves causality. Together they materially weaken the information value of spending the remaining compute on L/g without first testing the apparatus.

S1 is therefore recorded as **incomplete by explicit post-hoc early termination**, not as a four-rung capacity falsification.

## 6. Mandatory next gate — FROZEN APPARATUS CODE AUDIT

Authority:
`experiments/iris_dino_controlled_20260829/FROZEN_APPARATUS_AUDIT_GATE_V1.md`.

The audit must inspect actual code/config/runtime behavior for:

- representation extraction/cache;
- 4096-locus sampling and supervision masks;
- shared decoder/spatial/native access;
- loss;
- optimizer/LR/AMP/resume semantics;
- evaluator residual masks and eligibility;
- corpus substrate/membership.

Default rule:

> Every frozen component is suspect until audited.

**NEW DINO TRAINING: NOT AUTHORIZED.**

## 7. Planned causal sequence after audit

If the audit permits execution:

1. one B-rung decoder/access ceiling experiment with current loss fixed;
2. DINO ON/native ON, DINO OFF/native ON, DINO ON/native OFF ablations;
3. beta-only loss experiment;
4. tail/CVaR-only experiment with beta fixed;
5. only if single-rung accessibility becomes adequate, consider a new versioned S/B/L/g ladder.

A result where native-only largely solves the task is explicitly legitimate evidence.

## 8. Parallel corpus work

The Geppetto/Arachne clean-character audit remains a separate parallel workstream. Automatic audit results must receive random visual sanity checks before final clean-C0 membership is trusted.

Do not use corpus-purity observations to reinterpret the sealed S1 results post hoc. Any clean-C0 replay is a new versioned causal experiment.
