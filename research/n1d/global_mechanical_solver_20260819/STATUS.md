# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_SUPPORTED__R_REL_DIS_STRONG_PASS__M256_DOMAIN_PASS__GLOBAL_SOLVER_V2_CAUSAL_BUT_ABSOLUTE_FAIL__LOCAL_OBJECTIVE_CONTEXT_FAIL__RANK_CALIBRATION_INSUFFICIENT__MIXED_MULTI_EDGE_FAILURE__NO_SIMPLE_EDGE_RELIABILITY_SIGNAL__INCIDENT_SUMMARY_LOFO_FAIL__FULL_PAIR_MIN_SUM_CAUSAL_SAFE_BUT_ABSOLUTE_FAIL__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Immutable qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; primary F-activity `0.433160 < 0.50`.
- Stage-B/e00 is truth-open development evidence only and may not be reused as blind qualification.
- `sealed21` / `external10` remain **CLOSED**.
- Any future blind qualification requires a new untouched preregistered panel.

## Frozen representation / geometry authority

Supported representation:

```text
P_A
F16 bounded H_i / P_B_geom
N_A, N_B
V_A, V_B
Z
p_active
local relational/differential evidence
U_pred + typed U_obs
compiler/global solver
```

Authority split:

```text
F16/M256 candidates -> physical Pose-B XYZ authority
p_active            -> absolute motion/non-motion evidence
R_REL_DIS           -> pairwise differential compatibility
compiler/solver     -> final feasible candidate configuration
```

No free XYZ final motion head. Fixed early top4 and truth-conditioned candidate retention remain forbidden.

F16 broad-e00 development coverage:

```text
pooled primary-2x .98171
worst family       .94915
8/8 families >=    .90
strict-1x pooled   .94919
```

M256 deterministic solver domain passes preregistered coverage:

```text
reliable denominator 492/512
pooled primary-2x     .9756097561
worst family          .9322033898
8/8 families >=       .90
```

## p_active — supported

```text
pooled AUROC             .95854
worst-family AUROC       .89610
mean family bal-acc      .89176
worst family bal-acc     .77737
Brier                    .07252
```

Direct/global log_amp, H-relative amplitude, candidate-specific amplitude/vector heads and naive Z fusion have been falsified under tested hard-tail contracts. The endpoint authority problem is not another independent per-carrier scalar/vector head.

## Frozen local relation — R_REL_DIS STRONG PASS

`R_REL_DIS` compares candidate relative projected motion to observed DIS relative motion across common views.

Oracle-pair separability on the eight open-development families:

```text
edges                            603
pooled median percentile         .01465
fraction <= .25                  .93035
fraction <= .10                  .80597
worst-family median              .05249
worst-family fraction <= .25     .84615
```

Complete M256 transfer also passes (`pooled median oracle percentile .05733`, worst-family `.15030`). Relation definition remains frozen.

## Global Solver V2 — relation causal, absolute FAIL

Primary n=261:

```text
                    U_ONLY      G_REL_DIS
contain1             .37165       .45211
contain2             .67816       .80843
worst-family c2      .28000       .44000
median norm error   1.31793      1.14222
```

Hard-tail gains: `11032 +.16`, `13203 +.0833`, `15290 +.12`. Secondary contain2 `.79583 -> .87917`. Causal/safety PASS, absolute FAIL.

Canonical verdict: `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`.

## Failure localization

The following simple explanations have been ruled out on the open panel:

- M256 candidate capacity failure;
- R_REL_DIS transfer failure;
- pure ICM immobility/local initialization only;
- dominant common family translation;
- simple median-DIS common anchor;
- safe fixed unary-anchor subset;
- percentile/rank calibration alone.

Even an evaluator-perfect initialization is pulled by the frozen V2 objective toward worse endpoint truth. Correct-neighbor pair-only evidence is strong (`contain2 .85441`, worst `.68`) but incident aggregation is not uniformly reliable.

Local Objective Component Rank Audit classification: `MULTIEDGE_RELATION_AGGREGATION_FAIL`.

## Global Solver V3 — rank calibration insufficient

V3 changes only z-score factor calibration to empirical mid-rank percentile costs. It preserves all raw factor orderings.

```text
                    U_RANK_ONLY   G_RANK_REL
contain1                .37165       .45594
contain2                .67816       .75862
worst-family c2         .28000       .36000
median norm error      1.31793      1.11417
```

Causal/safety PASS, absolute FAIL. Verdict: `RANK_CALIBRATION_INSUFFICIENT_V3`.

## Multi-edge incident consensus audit — MIXED failure

Prereg commit `39c80ce3a636307ba9414d14a3f9099ed6d10ef8`.

Exact parent behavioral parity reproduced. Among pair-sum baseline-bad nodes:

```text
bad nodes                              46
best one-edge deletion rescue <=.25   .4565217391
rescue <=.10                           .0652173913
median individually-supporting frac   .25
```

Observation-only robust aggregators fail hard-tail promotion:

```text
                    baseline   TRIM_MAX1_Z   MEDIAN_EDGE_RANK
pooled contain2       .85441       .83525          .84291
worst-family c2       .68000       .64000          .64000
11032 c2              .68000       .64000          .64000
```

Canonical verdict: `MIXED_MULTI_EDGE_FAILURE`.

Artifacts:
- report commit `0ea98ec54f7c58cb97e19978e87b6e84a8be8f4d`
- result summary commit `567fed1b05ed1aea718b4bf68d68bfa52b0f16a5`
- readable reproducer commit `e191f24f6571d8a9c6904eb9d5450d72201aafd9`

## Observation-native edge reliability audit — no simple scalar signal

Prereg commit `f08f788301eef559abf245da1e703bd52b50d2c4`.

Three truth-free damaging-edge predictors were frozen and tested on the 46 parent bad nodes:

```text
                         top1     top2    median norm rank   mean norm rank
rank-vector disagree     .2609    .5000       .4500             .4457
endpoint-vote outlier    .3043    .5652       .3333             .4504
relative-flow noise      .1957    .3913       .6667             .5739
```

None passes preregistered support gates. Verdict: `NO_SIMPLE_OBS_EDGE_RELIABILITY_SIGNAL_V1`.

Artifacts:
- report `65dcb09d18cc37f9f05c2c7640494eb4a271413d`
- summary `0598eeef1ed422d6987047c94dc6113781c82ddd`
- exact source `0768ad3fa4c91552aae60d68d57372b0eb564aae`

## Incident-set learned candidate ranker P0 — family-LOFO FAIL

Prereg commit `37a6ba7b975176fa8d549223166578707dfa2329`.

A tiny logistic ranker used only U percentile plus symmetric incident-edge percentile statistics and set-valued 2x-positive labels under 8-way family LOFO.

```text
pooled contain2       .68966
worst-family c2       .16000
11032 c2              .16000
median norm error    1.39859
```

This is only `+.01149` pooled contain2 over U_ONLY and catastrophically under-generalizes on 11032. Verdict: `INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_FAIL`.

Artifacts:
- report `ca1ed9d40965ece32d83ec20230f9ba21e7360a4`
- summary `f0e6abe6745f6cac79f655a9222ee3ee74777d0e`
- exact source `dfe7ad9fc2f52ac20035d47967bd69ce96e2a8bb`

## Full pair-structure one-pass min-sum P0 — causal/safe, absolute FAIL

Prereg commit `a1169be16dea0b706a179c59c979770b0ae06347`.

Instead of conditioning every edge on one hard U_ONLY neighbor, preserve the complete M256×M256 relation matrix and send one synchronous min-sum message:

```text
m_{j->i}(k) = min_l [U_j(l) + w_ij R_ij(k,l)]
B_i(k)      = U_i(k) + sum_j m_{j->i}(k)
```

Primary:

```text
                    U_ONLY   MIN_SUM_P0
contain1             .37165      .47126
contain2             .67816      .77778
worst-family c2      .28000      .40000
median norm error   1.31793     1.07155
```

Hard-tail gains:

```text
11032 +.12
13203 +.02778
15290 +.10
```

Secondary all-reliable M256-contained:

```text
contain1   .46667 -> .61042
contain2   .79583 -> .86667
worst c2   .58182 -> .67273
median err 1.04946 -> .86270
```

Frozen gates:

```text
causal       PASS
hard-tail    PASS
safety       PASS
absolute     FAIL
```

Verdict: `ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL`.

This is positive structural evidence despite the fail: **hard single-neighbor conditioning is demonstrably lossy, and retaining complete candidate-to-candidate pair structure recovers substantial family-robust signal.** One pass is insufficient for final endpoint authority.

Artifacts:
- report `52e6de038ec6469dcce6bd9c3a7dfef938475139`
- summary `229c044d2773e9ade4110f292bdded92316a3047`
- exact source `911b841fd9ac96765141d2f6756bd18276d3388a`

## Current scientific interpretation

Supported path is now more specific:

```text
p_active
+ frozen F16/M256 feasible geometry
+ frozen R_REL_DIS full pair compatibility
+ solver that preserves neighbor candidate uncertainty / pair structure
```

Ruled out as sufficient tested fixes:

- independent per-carrier motion heads;
- M128 compression;
- V2 hard-state ICM objective;
- simple gauge anchor;
- fixed unary confidence anchors;
- factor rank calibration;
- one-edge trimming / median edge ranks;
- simple scalar edge reliability prediction;
- symmetric incident-statistic logistic aggregation;
- one synchronous min-sum pass.

The frontier is no longer “does relation information exist?” It does. The unresolved question is **how much structured multi-node inference is required to convert full pair compatibility into robust endpoint authority without letting the frozen objective collapse toward a wrong coherent basin**.

## Authorization / next experiment

Large/end-to-end training remains forbidden. sealed21/external10 remain closed.

Any next solver experiment must be separately preregistered. It may build directly on the causal/safe one-pass min-sum result, but may not post-hoc tune iteration count, damping, unary weight, pair weight, graph degree or relation definition on this panel.

A safe next step is a diagnostic that distinguishes:

1. whether another bounded round of full-pair message propagation adds genuinely new useful context; versus
2. whether the remaining hard-tail error is an objective/factor insufficiency that repeated propagation only amplifies.

If message depth is tested, the depth and update schedule must be fixed before viewing endpoint results; no sweep.

**Current frontier:** full pair candidate-to-candidate structure is causally useful and must be preserved; one-pass min-sum is insufficient, so the next preregistered question is bounded structured message depth versus factor insufficiency.
