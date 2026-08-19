# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_SUPPORTED__SINGLE_CARRIER_MOTION_FALSIFIED__LOCAL_RELATION_STRONG_PASS__R_REL_DIS_FROZEN_FOR_GLOBAL_SOLVER__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Immutable qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; primary F-activity `0.433160 < 0.50`.
- Stage-B/e00 truth is open-development evidence only; it cannot be reused as blind qualification.
- sealed21 / external10 remain **CLOSED**.
- Any later blind qualification requires a new untouched preregistered panel.

## Frozen geometry / representation authority

Representation Sufficiency Battery V1 remains supported for the revised set-valued contract.

```text
P_A
bounded set-valued H_i / P_B_geom
N_A, N_B
V_A, V_B
Z
p_active
local relational/differential mechanical evidence
U_pred + typed U_obs
compiler/global solver
```

Authority split:

```text
F16 H_i              -> physical Pose-B XYZ authority
p_active             -> absolute motion/non-motion evidence
local pair relations -> conditional mechanical compatibility between feasible candidates
compiler/global solve-> final candidate configuration
```

Fixed early top4, free XYZ motion heads, and truth-conditioned candidate retention remain forbidden.

Bounded research H is frozen by `BOUNDED_H_RESEARCH_CONTRACT_V1.md`, commit `b3cec959a9d593e0a2693c2bf894b9ea7673a4ac`.

F16 broad e00 coverage:

```text
pooled primary-2x .98171
worst family       .94915
8/8 families >=    .90
best-worst gap     5.08pp
strict-1x pooled   .94919
```

`K=16` is a research safety contract, not a permanent product constant. Compression is deferred to a dedicated later split/problem.

## p_active — supported

Factorized Motion Head P0 typed scalar evidence:

```text
pooled AUROC             .95854
worst family AUROC       .89610
mean family bal-acc      .89176
worst family bal-acc     .77737
Brier                    .07252
```

`p_active` remains a supported separate factor.

## Per-carrier conditional motion — falsified under tested contracts

The following increasingly permissive single-carrier formulations failed family-robust hard-tail selection:

1. direct/global `log_amp` regression;
2. H-relative candidate-conditioned amplitude;
3. candidate-specific amplitude-only local DIS/transport/delta3D compatibility;
4. candidate-specific full projected 2D motion-vector compatibility.

Representative hard-tail failures:

```text
P0 direct typed log_amp:
13203 rho .12609
15290 rho .18941

P0.5 H-conditioned typed amplitude:
worst family rho .05475

V1 amplitude-only raw DIS:
pooled within25 .40152
13203 .35135
15290 .49020

V1.5 full-vector raw DIS:
pooled contain2 .60985
11032 .23077
13203 .45946
15290 .68627
```

Naive descriptor Z fusion into the motion factor remains RED (`15290` direct log-amp rho `-.40326`).

Conclusion: the missing hard-tail information is not another independent per-carrier scalar/vector head.

## Local Differential Relation Separability V1 — STRONG PASS

Prereg:

- `LOCAL_DIFFERENTIAL_RELATION_SEPARABILITY_V1_PREREG.md`
- commit `2f94f42127e4efde0b0483e90e7887997c3332bf`

Canonical result:

- `LOCAL_DIFFERENTIAL_RELATION_SEPARABILITY_V1_RESULT.json`
- commit `14342e87c78b90d5e254607156ff1158d80902a6`

Report:

- `LOCAL_DIFFERENTIAL_RELATION_SEPARABILITY_V1_REPORT.md`
- commit `3fa61392967ff205ef4607b849a77e450dabaa7b`

### Selected relation — R_REL_DIS

For local graph edge `(i,j)`, candidate pair `(h_i,h_j)`:

```text
candidate relative motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed relative motion(v) = DIS_i(v) - DIS_j(v)

R_REL_DIS = median_v ||candidate_relative_motion - observed_relative_motion||
```

The relation is observation-native and truth-free. Truth is evaluator-only to identify the oracle H pair.

Oracle pair energy percentile among deterministic sampled alternative candidate pairs:

```text
edges                              603
pooled median percentile           .01465
fraction <= .25                    .93035
fraction <= .10                    .80597
worst-family median                .05249
worst-family fraction <= .25       .84615

11032 median                       .01367
13203 median                       .03711
15290 median                       .00195

active-active median               .01172
active-inactive median             .06445
```

All preregistered separability gates pass by large margins.

Other relations also pass:

```text
R_DISP pooled median       .01367
R_RIGID                     .05371
R_REL_TRANSPORT             .01953
R_REL_DELTA3D               .01465
R_REL_FUSION                .01270
```

The preregistered simplicity rule selects **R_REL_DIS** because fusion does not improve the worst-family median by the required `.05`.

## Scientific interpretation

The repeated hard tail is now localized at the correct abstraction level:

> A carrier's correct H candidate can remain ambiguous in isolation, yet the correct **candidate pair** becomes extremely easy to distinguish through local differential/mechanical relations.

Therefore standalone amplitude authority is removed from the current architecture target.

Current supported path:

```text
p_active
+ set-valued frozen H_i
+ frozen local R_REL_DIS pair factors
+ global compiler/solver
-> globally coherent feasible P_B configuration
```

Magnitude and direction may remain diagnostic/supervision decompositions, but not independent final candidate authorities.

## Current authorized next experiment

**Separately preregister a discrete/global solver V1 with R_REL_DIS frozen.**

Non-negotiable:

- do not tune relation and solver together;
- use observation-only candidate compression for tractability and measure its truth-open coverage before interpreting solver failure;
- preserve an observation-only unary baseline;
- compare unary-only vs graph relation causally;
- no post-result lambda/iteration/mode-count tuning;
- if graph passes endpoint gates, freeze solver behavior before unchanged GFDR evaluation;
- if graph fails despite strong relation separability, diagnose candidate compression/optimization rather than reopening the per-carrier amplitude head.

## Authorization

Supported:
- F16 bounded H research contract;
- typed-scalar `p_active`;
- frozen `R_REL_DIS` local relation;
- deterministic/global solver diagnostics.

Paused/forbidden:
- standalone/direct/H-relative `log_amp` training;
- single-carrier vector head as final motion authority;
- naive Z fusion into motion factors;
- large/end-to-end training;
- Stage-B requalification.

**Current frontier:** frozen H + p_active + frozen local relation -> preregistered global candidate solver.
