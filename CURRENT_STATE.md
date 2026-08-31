# RealSaS-OPT — Current State

**Date:** 2026-08-31  
**Canonical continuation branch:** `main`  
**Status:** `DTB_ND1_CLOSED__CONSUMER_INTERLOCK_CLOSED__DINO_S1_EARLY_TERMINATED_AFTER_S_B__FROZEN_APPARATUS_AUDIT_BLOCK_AND_FIX__DEV32_CLOSED__NEW_DINO_TRAINING_NOT_AUTHORIZED`

This file is continuation authority only on `main`.

## One-line state

`consumer-sensitive depth tolerance is closed -> DINO S/B primary-failed nearly flat -> S1 early-terminated after code audit exposed a material shared-access confound -> full frozen-apparatus audit now closes BLOCK_AND_FIX -> no new DINO training until a versioned spatial decoder/access apparatus passes preflight`

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

Supported:

> Under `SharedLearnerV1` and the S1 apparatus, S->B backbone scaling did not materially improve downstream accessibility at the fixed primary budget.

Not supported:

- L/g would fail;
- DINO-L/g are insufficient;
- backbone scale is globally irrelevant;
- S->B would remain flat under another decoder/loss/interface/native-detail path;
- partial L history predicts primary behavior.

## 5. Frozen apparatus code audit — CLOSED: BLOCK_AND_FIX

Canonical report:
`canonical/DINO_FROZEN_APPARATUS_CODE_AUDIT_CLOSURE_20260831.md`.

Machine-readable result:
`experiments/iris_dino_controlled_20260829/DINO_FROZEN_APPARATUS_CODE_AUDIT_V1.json`.

Final decision:

`BLOCK_AND_FIX`

Main findings:

- representation source/weights/cache: `PASS_WITH_NAMED_LIMITATIONS`;
- 4096-locus sampler: semantic `PASS`, but structure-sensitive coverage was not measured and sampling is uniform visible-raster;
- shared learner/access path: **BLOCK for another capacity ladder**;
- loss/optimizer: implementation pass, objective alignment limitation;
- runtime: RNG/resume are saved, but global deterministic algorithms are not forced; tiny one-run deltas are not reliable rankings;
- evaluator / DTB-ND1 direct replay: **PASS**;
- camera-scale feature: image-derived proxy matches a checked master witness but must be replaced by exact known camera authority in V2;
- corpus purity: separate H0 intervention; cannot retroactively reinterpret S1.

Critical correction retained:

> The local DINO field stays `37x37`. Only the cross-view context is pooled to `16x16` and bilinearly returned to `37x37`.

The blocked component is therefore not “DINO became 16x16”; it is the broader shared high-resolution accessibility path: common `1536->256` stem, 37x37 token ceiling, 16x16 cross-view context, sparse 25-sample native detail, and pointwise final query head.

## 6. New training authority

**NEW DINO TRAINING: NOT AUTHORIZED.**

Before optimizer step 1 of any successor:

1. version/freeze a high-resolution spatial decoder/native-access V2;
2. use exact `camera.json` authority for scale;
3. add repeatability calibration or an adequate deterministic substitute;
4. seal all frozen source/config/runtime hashes;
5. preserve the S1 membership and current loss for the first decoder-only causal experiment.

No S/B/L/g ladder may reopen yet.

## 7. Planned causal sequence

First experiment after V2 preflight:

`B + same S1 membership + same current SmoothL1(beta=.01)`

Only decoder/native-access changes.

Separately trained/evaluated ablations:

- DINO ON / native ON;
- DINO OFF / native ON;
- DINO ON / native OFF.

A native-only success is explicitly legitimate evidence.

Only after decoder causality:

1. beta-only loss ablation;
2. tail/CVaR-only ablation with beta fixed;
3. separate clean-C0 H0 intervention;
4. only if revised single-rung accessibility becomes adequate, consider a new versioned S/B/L/g ladder.

## 8. Parallel corpus work

The Geppetto/Arachne clean-character audit remains a separate parallel workstream. Automatic audit results must receive random/blind visual sanity checks before final clean-C0 membership is trusted.

Do not use corpus-purity observations to reinterpret the sealed S1 results post hoc.
