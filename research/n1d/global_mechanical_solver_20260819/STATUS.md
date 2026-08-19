# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_SUPPORTED__SINGLE_CARRIER_MOTION_FALSIFIED__R_REL_DIS_STRONG_PASS__GLOBAL_SOLVER_V1_M128_PREFLIGHT_FAIL__SOLVER_NOT_RUN__NEXT_DOMAIN_EXPERIMENT_MUST_BE_PREREGISTERED__BIG_TRAINING_FORBIDDEN`  
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

`K=16` is a research safety contract, not a permanent product constant. Compression is a separate problem.

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

Direct/global `log_amp`, H-relative amplitude, candidate-specific amplitude-only compatibility, and candidate-specific full projected 2D motion-vector compatibility all fail family-robust hard-tail selection. Naive descriptor-Z fusion into the motion factor is also RED.

Representative failures:

```text
P0 direct typed log_amp:       13203 rho .12609, 15290 rho .18941
P0.5 H-conditioned amplitude:  worst family rho .05475
V1 raw-DIS amplitude:          pooled within25 .40152
V1.5 full-vector raw DIS:      pooled contain2 .60985
                               11032 .23077 / 13203 .45946 / 15290 .68627
```

Conclusion: the missing hard-tail information is not another independent per-carrier scalar/vector authority.

## Local Differential Relation Separability V1 — STRONG PASS

Selected frozen relation: `R_REL_DIS`.

```text
candidate relative motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed relative motion(v) = DIS_i(v) - DIS_j(v)

R_REL_DIS = median_v ||candidate_relative_motion - observed_relative_motion||
```

Oracle-pair separability across the eight open-development families:

```text
edges                            603
pooled median percentile         .01465
fraction <= .25                  .93035
fraction <= .10                  .80597
worst-family median              .05249
worst-family fraction <= .25     .84615
11032 median                     .01367
13203 median                     .03711
15290 median                     .00195
```

All preregistered relation-separability gates pass by large margins. `R_REL_DIS` remains frozen; relation and solver may not be co-tuned.

## Global Relational Candidate Solver V1 — PRE-FLIGHT FAIL / SOLVER NOT RUN

Preregistered file: `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_PREREG.md`.

V1 froze an observation-only M128 candidate domain before endpoint results:

```text
F16 H -> dedup 1e-5
best 32 by frozen unary reprojection score
+ deterministic XYZ farthest-point fill to 128
```

Before interpreting M128, execution replayed the full F16 source and required exact canonical parity. After correcting the evaluator implementation to the preregistered local-scale definition (median of the four nearest other surface samples), parity is exact on all eight families and the mapping-reliable denominator is exactly `492/512`.

Full-F16 per-family primary-2x parity:

```text
09908  1.0000000000
11032   .9491525424
12772  1.0000000000
13203   .9508196721
14404   .9682539683
14702  1.0000000000
14758   .9838709677
15290  1.0000000000
```

M128 preregistered preflight:

```text
pooled primary-2x   .9695121951   FAIL (< .970)
worst family        .9152542373   PASS (>= .900)
families >= .90     8/8           PASS
best-worst gap      8.4746pp       PASS (<=10pp)
```

Per-family M128 primary-2x:

```text
09908  1.0000000000
11032   .9152542373
12772  1.0000000000
13203   .9344262295
14404   .9682539683
14702  1.0000000000
14758   .9838709677
15290   .9508196721
```

Canonical V1 verdict:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`

The miss is numerically tiny but the preregistered threshold is exact; it is not rounded into a pass. Consequently **G_REL_DIS ICM was not run or interpreted**. This result does not falsify `R_REL_DIS`; it only rejects M128 as the authorized V1 solver domain.

Canonical V1 artifacts:

- `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_REPORT.md` — commit `7defd9c57e020d823932595e047e27c4f3f5c5ca`
- `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_RESULT.json` — commit `20afc4e9b51e144c72ad0a556ba5f1f2adaa09e9`
- `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_SOURCE_MANIFEST.json` — commit `0501f5ccfaef4e2191702238ab3d6578d028bbb8`
- exact execution source is persisted in 5 base64 parts under `source_bundle_b64/`

## Current scientific interpretation

The current supported path remains:

```text
p_active
+ frozen F16 H_i
+ frozen R_REL_DIS local pair factors
+ deterministic/global solver
-> globally coherent feasible P_B configuration
```

V1 did **not** test whether the global relation can solve the endpoint problem because its compressed domain narrowly failed the frozen preflight.

Therefore the next scientific question is candidate-domain capacity, not a new motion head and not a new relation.

## Authorized next experiment

A new candidate-domain/global-solver experiment may be run only under a **separate preregistration**. It must keep frozen:

- F16 source H;
- unary definition and normalization;
- graph definition;
- `R_REL_DIS` formula and normalization;
- edge weights;
- deterministic ICM mechanics;
- endpoint gates.

Only the candidate-domain capacity/selection contract may change, and it must be fixed before its preflight is inspected. V1 M128 may not be retroactively modified.

## Authorization

Supported:
- F16 bounded H research contract;
- typed-scalar `p_active`;
- frozen `R_REL_DIS` local relation;
- separately preregistered deterministic/global solver diagnostics.

Paused/forbidden:
- modifying V1 M128 post hoc;
- standalone/direct/H-relative `log_amp` training;
- single-carrier vector head as final motion authority;
- new relation tuning on this panel;
- naive Z fusion into motion factors;
- large/end-to-end training;
- Stage-B requalification.

**Current frontier:** freeze a new observation-only candidate domain under a separate preregistration; only if its preflight passes may the unchanged frozen global solver be evaluated.
