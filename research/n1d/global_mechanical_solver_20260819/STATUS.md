# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_TYPED_EVIDENCE_SUPPORTED__AMPLITUDE_SCALAR_CONTRACT_FALSIFIED__LOCAL_CANDIDATE_MOTION_EVIDENCE_REQUIRED__BIG_TRAINING_FORBIDDEN`  
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
conditional motion compatibility evidence
dir
U_pred + typed U_obs
```

Authority split:

```text
H_i        -> physical XYZ authority
p_active   -> absolute motion/non-motion evidence
motion compatibility -> candidate-conditioned magnitude/local differential evidence
dir        -> direction evidence
compiler   -> final constrained/global collapse
```

Fixed early top4 and free XYZ motion heads remain forbidden.

## Frozen bounded H research contract

Decision: `FREEZE_F16_FOR_RESEARCH__PRODUCT_COMPRESSION_DEFERRED`

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

Vectorized H-only diagnostic: ~`21.2 ms / 64-carrier episode`, mean H `2146`, median `1536`, p95 `4608`; this excludes neural forward/search/scoring and is not a product latency claim.

Compact-retention optimization is deferred to a dedicated later split/problem to avoid overfitting the current truth-open 8-family panel.

## Motion factor P0 — p_active survives, direct amplitude fails

P0 typed scalar `p_active`:

```text
pooled AUROC             .95854
worst family AUROC       .89610
mean family bal-acc      .89176
worst family bal-acc     .77737
Brier                    .07252
```

`p_active` input evidence is supported.

Direct/global normalized amplitude regression is not family-robust:

```text
B1 pooled active Spearman .57431
median family             .75250
worst family              .12609
13203                     .12609
15290                     .18941
```

Naive descriptor-Z fusion is RED; it recreates held-out family collapse (`15290 rho = -.40326`).

## Candidate-Conditioned Amplitude P0.5 — FAIL

Prereg:
- `CANDIDATE_CONDITIONED_AMPLITUDE_P05_PREREG.md`
- commit `de4ca727956a96db095181f49613db8ff9e7ee24`

Result:
`NO_ARM_PASSES__AMPLITUDE_EVIDENCE_CHANNEL_INSUFFICIENT`

P0.5 restricted amplitude to the frozen F16 feasible distribution. For each active/reliable/F16-contained carrier, the target was the normalized amplitude coordinate of the oracle-near candidate already inside H; no free truth XYZ was introduced.

Evaluated carriers: `264`.

### R0 — raw current, H-relative

```text
pooled Spearman             .16383
median family               .22568
worst family               -.21811
negative family count       1
mean pairwise accuracy      .57927
worst pairwise              .36876
pooled within25%            .18939
```

### R1 — calibrated H-relative current

```text
pooled Spearman             .11780
worst family               -.21811
pooled within25%            .45076
worst within25%             .27273
```

Calibration moves estimates toward typical H amplitude regions but does not repair carrier ordering.

### R2 — typed scalar + H summaries

```text
pooled Spearman             .33528
median family               .39128
worst family                .05475
negative family count       0
mean pairwise accuracy      .54836
worst pairwise              .44319
pooled within25%            .46970
worst within25%             .39216
```

Tail examples:

```text
13203 rho .38336
15290 rho .05475
```

### Consequence

Candidate-conditioning does **not** rescue amplitude. Therefore the current failure is not just a badly chosen absolute target. The frozen carrier-level scalar/typed evidence lacks a family-robust local magnitude-ordering channel under the tested contracts.

This does not revoke:
- F16 geometry sufficiency;
- p_active evidence support;
- separate direction evidence.

It falsifies the current **standalone scalar `log_amp` contract**.

## Current scientific frontier — candidate-specific local motion evidence

Do not continue target reformulation or train through the amplitude failure.

The next required diagnostic keeps F16 H frozen and evaluates observation-native motion **per candidate**:

```text
predicted_motion_2d(i,h,v) = project(h,v) - project(P_A(i),v)
```

Compare each candidate's projected displacement to paired-raster local motion evidence at the Pose-A carrier, such as:

```text
DIS / optical-flow displacement
transport_offset_srcA
other frozen paired-view local motion fields
```

Keep causal terms separate:

```text
E_amp(i,h) -> view-wise magnitude compatibility
E_dir(i,h) -> view-wise direction compatibility
```

The immediate question is whether candidate-specific multiview amplitude compatibility ranks the truth-near members of H on hard families (`13203`, `15290`) even though carrier-level scalar amplitude does not.

If yes, replace standalone `log_amp` with candidate-conditioned local differential compatibility/energy.
If no, current paired-raster motion front door is missing a necessary local differential cue and representation must be extended before any amplitude learning.

## Authorization

Supported:
- F16 bounded H research contract;
- typed-scalar `p_active` evidence;
- deterministic candidate-specific local motion diagnostics.

Paused/forbidden:
- standalone/direct/H-relative `log_amp` training;
- combined factor-head training;
- naive Z fusion into motion factors;
- large/end-to-end training;
- Stage-B requalification.

## Canonical current artifacts

- `REPRESENTATION_CONTRACT_V2.md`
- `BOUNDED_H_RESEARCH_CONTRACT_V1.md`
- `FACTORIZED_MOTION_HEAD_P0_RESULT.json`
- `FACTORIZED_MOTION_HEAD_P0_REPORT.md`
- `CANDIDATE_CONDITIONED_AMPLITUDE_P05_PREREG.md`
- `CANDIDATE_CONDITIONED_AMPLITUDE_P05_RESULT.json`
- `CANDIDATE_CONDITIONED_AMPLITUDE_P05_REPORT.md`
- `candidate_conditioned_amplitude_p05.py`

**Current frontier:** preserve H and p_active; test candidate-specific local multiview motion compatibility before any further learning.
