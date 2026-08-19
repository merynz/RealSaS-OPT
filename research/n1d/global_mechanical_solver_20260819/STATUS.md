# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_TYPED_EVIDENCE_SUPPORTED__DIRECT_LOGAMP_FAIL__CANDIDATE_CONDITIONED_AMPLITUDE_NEXT__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Immutable qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; primary F-activity `0.433160 < 0.50`.
- Stage-B/e00 truth is open-development evidence only; it cannot be reused as blind qualification.
- sealed21 / external10 remain **CLOSED**.
- Future qualification requires a new untouched preregistered panel.

## Representation / geometry authority

Representation Sufficiency Battery V1 remains supported for the revised set-valued factorized contract.

```text
P_A
bounded H_i / P_B_geom
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred + typed U_obs
```

Authority split:

```text
H_i       -> physical XYZ authority
p_active  -> absolute motion/non-motion evidence
log_amp   -> conditional magnitude/ranking evidence
dir       -> direction evidence
compiler  -> final constrained/global collapse
```

Fixed early top4 and free XYZ motion heads remain forbidden.

## Frozen bounded H research contract

Decision:

`FREEZE_F16_FOR_RESEARCH__PRODUCT_COMPRESSION_DEFERRED`

File/commit:

- `BOUNDED_H_RESEARCH_CONTRACT_V1.md`
- `b3cec959a9d593e0a2693c2bf894b9ea7673a4ac`

F16 open-development coverage:

```text
pooled primary-2x .98171
worst family       .94915
8/8 families >=    .90
best-worst gap     5.08pp
strict-1x pooled   .94919
```

Vectorized H-only execution diagnostic: ~`21.2 ms / 64-carrier episode`, mean H `2146`, median `1536`, p95 `4608`; this excludes model/search/scoring and is not a product latency claim.

Compact-retention attempts are closed for now to avoid overfitting the same truth-open panel:

- Adaptive H V1: FAIL.
- D8-C4: efficient but worst family `.91525`.
- D10-C4: pooled `.97967`, worst `.93220`, one 11032 carrier short of the `.94` gate.
- D12-C8: coverage PASS but outside compact K<=10 budget.
- score K10/K12: coverage FAIL.

Product compression is deferred to a later dedicated split/problem.

## Factorized Motion Head P0

Prereg:

- `FACTORIZED_MOTION_HEAD_P0_PREREG.md`
- commit `5ad75bbe5434af901f105f1654e1f8e2b074ff33`

Result:

`FAIL__PACTV_SUPPORTED__DIRECT_LOGAMP_NOT_SUPPORTED`

### p_active

Typed scalar frozen evidence B1:

```text
pooled AUROC             .95854
worst family AUROC       .89610
mean family bal-acc      .89176
worst family bal-acc     .77737
Brier                    .07252
```

P0 p_active gate: **PASS**.

Conclusion: the `p_active` factor has a supported typed scalar input contract for a later calibrated small head.

### direct log_amp

Direct global/normalized amplitude regression does **not** pass family-disjoint tail gates.

B1 typed scalar:

```text
pooled active Spearman   .57431
median family Spearman   .75250
worst family Spearman    .12609
worst pairwise accuracy  .54893
```

Hard tails:

```text
13203 rho .12609
15290 rho .18941
```

Therefore direct `log_amp` head execution is **not authorized**.

### naive descriptor Z fusion

B2 (`typed scalar + Z`) is RED:

```text
p_active worst AUROC       .83302
15290 log_amp Spearman    -.40326
worst pairwise accuracy    .32615
```

This recreates the historical held-out multifeature fusion collapse. Naive Z injection into the motion-factor head is forbidden by current evidence.

## Current scientific frontier

The intended architecture never required `log_amp` to create a free absolute displacement. H is already the XYZ authority.

Next required diagnostic:

### Candidate-conditioned amplitude factor

For each reliable truth-open carrier:

1. keep frozen F16 `H_i`;
2. form candidate displacement magnitudes `a_h = ||h - P_A||`;
3. identify the evaluator-only oracle-near candidate already inside H;
4. measure its magnitude rank/quantile inside the carrier's feasible H distribution;
5. test whether frozen current/typed evidence predicts **that candidate-conditioned rank/quantile** family-disjoint;
6. score candidate magnitudes inside H rather than regress a free truth amplitude.

If `13203/15290` tails recover under candidate-conditioned ranking, redefine `log_amp` as an H-conditioned energy/rank factor and preregister P1.

If they do not, the representation is missing a local differential amplitude evidence channel and must be extended before further training.

## Authorization

Supported:

- F16 bounded H research contract;
- typed-scalar `p_active` evidence contract;
- candidate-conditioned amplitude diagnostics.

Paused/forbidden:

- direct/global `log_amp` training;
- combined p_active+log_amp training;
- naive Z fusion into motion factors;
- large/end-to-end training;
- Stage-B requalification.

## Canonical current artifacts

- `REPRESENTATION_CONTRACT_V2.md`
- `BOUNDED_H_RESEARCH_CONTRACT_V1.md`
- `FACTORIZED_MOTION_HEAD_P0_PREREG.md`
- `FACTORIZED_MOTION_HEAD_P0_RESULT.json`
- `FACTORIZED_MOTION_HEAD_P0_REPORT.md`
- `factorized_motion_head_p0.py`
- `F16_H_CONSTRUCTION_BENCHMARK_V1.json`
- compression/audit artifacts preserved beside them.

**Current frontier:** keep H and p_active semantics; determine the correct candidate-conditioned formulation for amplitude before any further learning.
