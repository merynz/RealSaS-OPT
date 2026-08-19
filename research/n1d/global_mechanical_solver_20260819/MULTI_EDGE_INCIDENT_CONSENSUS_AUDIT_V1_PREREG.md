# RealSaS N1D — Multi-Edge Incident Consensus / Outlier Audit V1 — Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DIAGNOSTIC_ONLY__NO_SOLVER_CHANGE_AUTHORIZED`  
**Parent state:** `RANK_CALIBRATION_INSUFFICIENT_V3`  
**Qualification:** unchanged; Stage-B/e00 is truth-open development evidence only.

## Purpose

The current evidence chain shows:

- M256 candidate-domain coverage PASS;
- R_REL_DIS pair signal transfers strongly into full M256 pair matrices;
- pair-only local selection under evaluator-correct neighbor context is substantially stronger than unary;
- V2 z-score global objective and V3 rank-calibrated global objective both fail absolute hard-tail gates;
- common-mode translation and fixed high-confidence unary anchoring are not sufficient explanations/fixes.

The remaining localization question is:

> When multiple R_REL_DIS edges meet at a node, is hard-tail failure driven mainly by a minority of inconsistent/outlier incident edges that poison the aggregate, or do the incident edges coherently prefer a wrong local state?

This audit is diagnostic-only. It does not change the deployed/global solver and cannot promote a new objective directly.

## Frozen source / panel

Use the same open-development e00 panel and exact V2 M256 domains:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Require exact source parity before interpretation:

```text
mapping reliable denominator 492/512
M256 pooled primary2x .9756097561
M256 worst family      .9322033898
8/8 families >= .90
```

Frozen quantities:

- F16 construction;
- M256 compression;
- 4-nearest-neighbor symmetrized P_A graph;
- edge eligibility;
- raw R_REL_DIS;
- V2 per-edge median/IQR normalization;
- degree weights `w_ij = 1/deg_i + 1/deg_j`;
- evaluator primary set;
- local surface scale.

No unary term is included in this audit because the question is specifically incident-relation aggregation.

## Evaluator-correct neighbor context

For every node, define its evaluator oracle candidate as the M256 candidate with minimum 3D distance to its mapped truth endpoint. Neighbor states are fixed to these oracle candidates for the local diagnostic.

Truth is used only to:

- choose evaluator oracle candidates;
- score oracle candidate rank and endpoint quality;
- classify diagnostic outcomes.

Truth may not define graph edges, relation costs or observation-only robust arms.

## Per-edge conditional diagnostics

For primary node `i` and each incident edge `e=(i,j)`:

1. compute the exact frozen V2 normalized pair matrix `R_z`;
2. fix neighbor `j` to its evaluator oracle candidate;
3. obtain the conditional candidate-cost vector `c_e(h_i)`;
4. measure the evaluator oracle candidate's empirical mid-rank percentile inside `c_e`.

Report per node:

```text
incident edge count
median single-edge oracle rank
max single-edge oracle rank
IQR single-edge oracle rank
fraction edges oracle-rank <= .25
fraction edges oracle-rank <= .10
```

Report pooled/per-family distributions and hard-tail families `11032/13203/15290`.

## Frozen pair-sum baseline

Reproduce the prior correct-neighbor pair-only local cost:

```text
C_SUM(h_i) = sum_e w_e c_e(h_i)
```

Report oracle rank and argmin endpoint quality. The prior audit reference is approximately:

```text
pooled contain2 .85441
worst-family contain2 .680
pooled median oracle rank .07227
worst-family median oracle rank .201171875
```

Behavioral consistency with that prior result is required before interpreting new arms.

## Diagnostic A — evaluator-only leave-one-edge-out ceiling

For every primary node with degree >=2, form `C_SUM_-e` for each single incident edge removed. Evaluator truth may choose the removal that gives the best oracle candidate rank.

This arm is explicitly **not deployable** and cannot be promoted. It answers only whether one bad incident edge is sufficient to explain a material fraction of failures.

Report:

```text
best leave-one-edge-out oracle rank
rank improvement over C_SUM
fraction of C_SUM-bad nodes (oracle rank >.25) rescued to <=.25
fraction rescued to <=.10
```

## Diagnostic B — observation-only TRIM_MAX1_Z

For each candidate `h_i`, compute weighted conditional edge contributions:

```text
a_e(h_i) = w_e c_e(h_i)
```

and define:

```text
C_TRIM_MAX1(h_i) = sum_e a_e(h_i) - max_e a_e(h_i)
```

For degree <=1, use the untrimmed available relation cost.

This rule is observation-only and deterministic, but remains diagnostic-only in V1.

Report oracle rank and argmin endpoint contain1/contain2/median normalized error pooled and per family.

## Diagnostic C — observation-only MEDIAN_EDGE_RANK

For each incident conditional vector `c_e`, independently convert candidate costs to empirical mid-rank percentiles across `h_i`:

```text
p_e(h_i) = empirical_percentile_h c_e(h_i)
```

Then:

```text
C_MEDIAN_EDGE_RANK(h_i) = median_e p_e(h_i)
```

This suppresses edge scale and a minority of high-cost incident edges without truth-conditioned deletion. It is diagnostic-only.

Report oracle rank and argmin endpoint quality pooled/per family.

## Predeclared localization rules

Define `C_SUM-bad` nodes as primary nodes with pair-sum oracle rank `> .25`.

### `SINGLE_EDGE_OUTLIER_MATERIAL`

Classify as material single-edge/outlier failure if all hold:

```text
C_SUM-bad node count >= 10
LOEO rescue-to-<=.25 fraction >= .50
and at least one observation-only robust arm satisfies:
  pooled contain2 >= C_SUM contain2 - .03
  worst-family contain2 >= C_SUM worst-family contain2 + .08
  11032 contain2 >= C_SUM 11032 contain2 + .08
```

### `COHERENT_MULTI_EDGE_WRONG`

Classify as coherent/inlier consensus failure if:

```text
C_SUM-bad node count >= 10
LOEO rescue-to-<=.25 fraction < .30
median edge-support-fraction<=.25 among C_SUM-bad nodes < .50
```

### `MIXED_MULTI_EDGE_FAILURE`

Otherwise classify as mixed/heterogeneous multi-edge failure.

No threshold may be changed after results are inspected.

## Boundaries

This audit does **not** authorize:

- dropping edges in the global solver;
- replacing sum by trim/median in the global solver;
- changing graph degree;
- changing R_REL_DIS;
- changing M256;
- adding unary/p_active factors;
- lambda/temperature tuning;
- learned edge gating/arbitration;
- optimizer changes/restarts;
- large/end-to-end training;
- Stage-B requalification;
- sealed21/external10 access.

If a stable structural aggregation pathology is localized, exactly one corresponding solver-objective experiment must be preregistered separately before execution.
