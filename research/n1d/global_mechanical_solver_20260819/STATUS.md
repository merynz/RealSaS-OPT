# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_H_FROZEN__PACTV_SUPPORTED__R_REL_DIS_STRONG_PASS__M256_DOMAIN_PASS__GLOBAL_SOLVER_V2_CAUSAL_BUT_ABSOLUTE_FAIL__FAILURE_LOCALIZED_TO_LOCAL_OBJECTIVE_CONTEXT__COMMON_MODE_NOT_DOMINANT__NO_SAFE_UNARY_ANCHOR__RANK_CALIBRATION_INSUFFICIENT_V3__NEXT_MULTI_EDGE_CONSENSUS_DIAGNOSTIC__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Immutable qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; primary F-activity `0.433160 < 0.50`.
- Stage-B/e00 truth is open-development evidence only and cannot be reused as blind qualification.
- `sealed21` / `external10` remain **CLOSED**.
- Any future blind qualification requires a new untouched preregistered panel.

## Frozen representation / geometry authority

Representation Sufficiency Battery V1 supports the revised set-valued factorized contract.

```text
P_A
F16 bounded set-valued H_i / P_B_geom
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
p_active             -> absolute motion / non-motion evidence
R_REL_DIS            -> local pairwise differential mechanical compatibility
compiler/global solve-> final feasible configuration
```

Fixed early top4, free XYZ motion heads and truth-conditioned retention remain forbidden.

`BOUNDED_H_RESEARCH_CONTRACT_V1.md` remains the frozen research geometry contract.

F16 broad-e00 coverage:

```text
pooled primary-2x .98171
worst family       .94915
8/8 families >=    .90
best-worst gap     5.08 pp
strict-1x pooled   .94919
```

`K=16` remains a research safety contract, not a permanent product constant. Adaptive compression attempts are a separate efficiency problem and are not allowed to weaken geometry authority.

## p_active — supported

Typed scalar motion-state evidence remains supported:

```text
pooled AUROC             .95854
worst-family AUROC       .89610
mean family bal-acc      .89176
worst family bal-acc     .77737
Brier                    .07252
```

Direct/global `log_amp`, H-relative amplitude, candidate-specific amplitude and candidate-specific full projected motion-vector heads were previously falsified on hard-tail family robustness. The missing endpoint authority is not another independent per-carrier scalar/vector head under those tested contracts.

## Frozen local relation — R_REL_DIS STRONG PASS

```text
candidate relative motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed relative motion(v) = DIS_i(v)-DIS_j(v)

R_REL_DIS = median_v ||candidate_relative_motion-observed_relative_motion||
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

Relation definition remains frozen; relation and solver may not be co-tuned.

## Global Solver V1 — M128 preflight FAIL, solver not run

M128 narrowly failed the exact frozen candidate-domain preflight:

```text
pooled primary-2x   .9695121951 < .970
worst family        .9152542373
families >= .90     8/8
best-worst gap      8.4746 pp
```

Canonical V1 verdict remains:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`

No rounding into a pass is permitted.

## Global Solver V2 — M256 PASS, relation causal, absolute solver FAIL

Canonical V2 prereg commit: `d44911b78dcf18f92a599310d17769f240dd296e`.

M256 changes only deterministic observation-only candidate capacity. Full F16 parity and reliable denominator `492/512` reproduce exactly.

M256 preflight:

```text
pooled primary-2x   .9756097561  PASS
worst family        .9322033898  PASS
families >= .90     8/8           PASS
best-worst gap      6.7797 pp      PASS
```

Primary endpoint result:

```text
                    U_ONLY      G_REL_DIS
contain1             .37165       .45211
contain2             .67816       .80843
worst-family c2      .28000       .44000
median norm error   1.31793      1.14222
```

Hard-tail contain2 gains:

```text
11032 +.16000
13203 +.08333
15290 +.12000
```

Secondary all-reliable M256-contained contain2 improves `.79583 -> .87917`.

Thus V2 relation is **causal and safe**, but absolute hard-tail quality fails. Canonical verdict:

`RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`

## V2 failure localization — FROZEN_LOCAL_OBJECTIVE_CONTEXT_FAIL

The complete M256 pair matrices preserve strong oracle-pair separability:

```text
pooled median oracle percentile   .05733
fraction <= .25                   .83284
worst-family median               .15030
```

So relation transfer is not the primary failure.

With evaluator-correct neighbor context, the unchanged V2 local objective still gives only:

```text
pooled contain2       .87739
worst-family contain2 .52000
11032 contain2        .52000
```

Even an evaluator-perfect M256 initialization is pulled away by frozen ICM toward a poor basin:

```text
oracle-hybrid init contain2 1.00000
oracle-init final contain2   .82759
worst final contain2         .48000
```

The objective decreases while endpoint truth quality degrades. Therefore the dominant unresolved layer is objective/context alignment, not merely an optimizer failing to move.

## Common-mode gauge audit — not dominant

R_REL_DIS has an exact common-translation null direction numerically (`max diff ~4.97e-14 px`), but the best evaluator-fitted family common translation explains only about `12.5%` of squared endpoint error and does not rescue hard tails. A simple observation-native median-DIS common anchor materially worsens results.

Canonical decision:

`COMMON_MODE_NOT_DOMINANT__SIMPLE_MEDIAN_DIS_ANCHOR_NOT_SUPPORTED`

## Local objective component rank audit — MULTIEDGE_RELATION_AGGREGATION_FAIL

Under correct neighbor context, oracle candidate ranks are:

```text
                    median rank   frac rank<=.25
unary-only             .19336          .58238
pair-sum-only          .07227          .82375
total unary+pair       .10742          .77011
```

Pair-only local endpoint quality is substantially stronger than unary:

```text
                    contain1   contain2   worst-family c2
unary-only           .37165     .67816       .280
pair-only            .48276     .85441       .680
total                .52107     .87739       .520
```

However the frozen pair-aggregation support gate misses exactly at worst-family median oracle rank `.201171875 > .20`, and unary interaction is heterogeneous: it damages `11032`/`13203` but helps `15290`. A global lambda interpretation is therefore not supported.

## Unary-anchor confidence audit — no safe fixed anchor set

Simple observation-only confidence signals do not safely identify a fixed subset on which unary may be frozen.

Top-25% examples:

```text
C_MARGIN contain2 .67826, worst .09091  # actively misleading on 11032
C_ABS    contain2 .86667, worst .75000
C_AGREE  contain2 .90984, worst .64286
```

Canonical decision:

`NO_SAFE_UNARY_ANCHOR_CONFIDENCE_FOUND_V1`

## Global Solver V3 — rank calibration insufficient

V3 preregistered exactly one objective change: replace median/IQR z-score calibration by deterministic empirical mid-rank percentile costs independently for each unary vector and each pair matrix. Raw evidence, factor ordering, M256, graph, degree weights and deterministic ICM remain frozen.

Behavioral source guard reproduces exact full-F16 and M256 parity before endpoint interpretation.

`U_RANK_ONLY` is endpoint-identical to V2 U_ONLY, as required.

Primary result:

```text
                    U_RANK_ONLY   G_RANK_REL
contain1                .37165       .45594
contain2                .67816       .75862
worst-family c2         .28000       .36000
median norm error      1.31793      1.11417
best-worst c2 gap       .55333       .64000
```

Per-family G_RANK_REL contain2:

```text
09908  .93333
11032  .36000
12772  .81818
13203  .69444
14404  .53125
14702  .93182
14758 1.00000
15290  .72000
```

Hard-tail gains vs unary:

```text
11032 +.08000 exactly (2/25)
13203 +.08333
15290 +.02000
```

Frozen V3 gates:

```text
causal improvement PASS
secondary safety   PASS
absolute quality   FAIL
```

V3 is worse than V2 on pooled contain2 (`.75862 < .80843`), worst-family contain2 (`.36 < .44`) and family spread, despite a small median-error improvement.

Canonical V3 verdict:

`RANK_CALIBRATION_INSUFFICIENT_V3`

Canonical V3 artifacts:

- `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V3_RANK_CALIBRATED_REPORT.md` — commit `cb8dc9387fa3805ccc5c47717085397cc116c4b2`
- `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V3_RANK_CALIBRATED_RESULT_SUMMARY.json` — commit `e760eafc1850f93d46a786083c52ee26c32616f4`
- `global_relational_candidate_solver_v3_rank_reproducer.py` — commit `28fddd7016fcfb08d88159970026123f2ab7159b`

The local V3 execution was a semantic reconstruction, not claimed byte-identical to a previously existing source. It was accepted for development interpretation only after exact F16/M256 behavioral parity and exact V2 unary endpoint parity.

## Superseded / non-frontier artifacts from reconnect recovery

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_CAPACITY_LADDER_PREREG.md` commit `030c925c85d56a689eae713aebfa4c9d5dea3a92` was written during reconnect recovery before discovering that canonical V2 M256 had already been executed. It is **not** the canonical frontier and must not override canonical V2 evidence.

`ADAPTIVE_HYPOTHESIS_RETENTION_V1_DECISIVE_FAIL.md` commit `10a474f581f0aef6bbce8f8c87acce41d318cc89` records a valid open-development adaptive-compression failure but is also not the current scientific frontier.

## Current scientific interpretation

The evidence now supports:

```text
p_active
+ frozen F16/M256 feasible candidate geometry
+ frozen R_REL_DIS local relation
```

but does **not** yet support the tested global objective conversions as final endpoint authority.

Already ruled out as simple explanations/fixes on this panel:

- candidate-domain capacity at M256;
- missing R_REL_DIS signal;
- pure ICM immobility;
- dominant family common translation;
- simple median-DIS absolute anchor;
- fixed high-confidence unary anchoring;
- pure order-preserving percentile/rank calibration.

The unresolved layer is now specifically **multi-edge relational context composition / incident-edge consensus**: strong local pair orderings do not combine safely and uniformly across nodes/families, and neighbor errors can propagate through the graph.

## Authorized next experiment

Next experiment must be separately preregistered and diagnostic-only before any objective change.

It should test whether hard-tail local failure is explained by a minority of inconsistent incident edges versus a coherent but wrong multi-edge consensus. Observation/evaluator separation must remain strict.

Candidate diagnostics may include, under evaluator-correct neighbor context:

```text
per-incident-edge oracle percentile ranks
fraction of incident edges supporting oracle candidate in top25/top10
edge-rank dispersion / disagreement
leave-one-edge-out pair-sum oracle rank sensitivity
fixed robust diagnostic aggregators (e.g. median / preregistered trim) only as diagnostic arms
```

No graph edit, edge deletion, relation retuning, lambda sweep, learned arbitration or optimizer change is authorized until that audit localizes the failure.

## Authorization

Supported/frozen:
- Representation V2 set-valued contract;
- F16 bounded H research geometry;
- M256 as a coverage-valid solver-domain reference;
- typed-scalar `p_active`;
- raw `R_REL_DIS` local relation;
- diagnostic-only solver/factor audits under preregistration.

Paused/forbidden:
- free XYZ;
- fixed early top4;
- standalone/direct/H-relative motion heads as final endpoint authority;
- post-hoc M128 repair;
- relation retuning on this panel;
- lambda/temperature/rank-power sweeps;
- graph/optimizer edits before localization;
- large/end-to-end training;
- Stage-B requalification;
- sealed21/external10 access.

**Current frontier:** preregister and run a multi-edge incident-consensus / outlier localization audit. If it identifies a stable observation-independent structural aggregation failure, only then preregister one corresponding solver-objective change.
