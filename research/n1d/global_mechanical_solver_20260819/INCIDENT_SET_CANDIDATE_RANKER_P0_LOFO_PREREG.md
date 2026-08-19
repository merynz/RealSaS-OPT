# RealSaS N1D — Incident-Set Candidate Ranker P0 — 8-Way Family-LOFO Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_SMALL_LEARNED_AGGREGATION_PROBE__NO_GEOMETRY_CHANGE`

## Parent evidence

The canonical chain now establishes:

- F16/M256 feasible geometry coverage is sufficient for development (`M256` preflight PASS);
- raw `R_REL_DIS` edge ordering is strongly informative;
- V2 fixed global objective is causal/safe but absolute hard-tail quality fails;
- simple rank calibration does not repair it;
- multi-edge failure is mixed rather than pure one-edge poisoning or coherent consensus;
- three frozen observation-only per-edge reliability scores fail to identify damaging edges robustly.

This P0 asks a narrower question:

> Is the **joint incident-edge evidence pattern around each candidate** learnably predictive of a functionally acceptable endpoint under family-disjoint holdout, without creating new XYZ authority or changing the frozen relation?

This is a small development probe, not qualification and not authorization for large/end-to-end training.

## Frozen geometry and evidence

Unchanged:

- F16 H construction;
- M256 observation-only candidate compression and ordering;
- `M_i` is the only Pose-B XYZ authority;
- frozen U_raw definition;
- frozen graph and edge eligibility;
- raw `R_REL_DIS` relation;
- no new raster encoder/model inference;
- no candidate expansion, interpolation or free XYZ output.

The ranker may only choose an existing `M256` candidate.

## Evaluation population

Use exactly the existing V2 primary evaluator population:

- mapping reliable;
- truth active;
- full-F16 contained2x;
- M256 contained2x.

Expected pooled n = 261 with the same per-family denominators as prior audits.

No node from the held-out family may participate in fitting feature normalization or model parameters.

## Observation-only candidate features

For node `i`, define frozen U_ONLY neighbor state for every neighbor `j`:

```text
x_j^U = argmin U_raw(j)
```

For each incident edge `e=(i,j)`, condition the frozen V2 `R_REL_DIS` matrix on `x_j^U`, then convert the resulting candidate vector to empirical mid-rank percentile:

```text
p_e(k) = percentile_h R_z(i,j,k,x_j^U)
```

For every candidate `k in M_i`, use exactly these 8 scalar features:

```text
f1 = percentile(U_raw_i(k))
f2 = mean_e   p_e(k)
f3 = median_e p_e(k)
f4 = min_e    p_e(k)
f5 = max_e    p_e(k)
f6 = std_e    p_e(k)
f7 = fraction_e[p_e(k) <= .25]
f8 = fraction_e[p_e(k) <= .10]
```

No evaluator truth, target distance, B-scale, oracle neighbor state, family ID, node ID, descriptor Z, p_active or raw image feature may enter these features.

## Training label

The label is set-valued rather than exact-teacher singleton:

```text
y(i,k) = 1  iff  ||M_i[k] - truth_P_B_i|| <= 2 * evaluator_local_scale_i
```

Thus every functionally acceptable candidate inside the frozen primary-2x basin is positive. The model is not trained to reproduce the unique nearest teacher endpoint.

Truth/local scale are labels only and are absent from inference features.

## 8-way family LOFO protocol

For each held-out family `F`:

1. training data = all primary nodes/candidates from the other seven families;
2. test data = primary nodes of `F` only;
3. fit `StandardScaler` on training feature rows only;
4. fit exactly one logistic classifier:

```text
LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    random_state=0
)
```

5. choose the M256 candidate with maximum predicted positive probability for each held-out node;
6. exact probability ties choose the lowest candidate index.

No hyperparameter sweep, early stopping, feature selection, family weighting or threshold tuning.

## Baseline / parity guards

Before interpreting LOFO:

```text
M256 preflight denominator       492
M256 pooled primary2x           .9756097561
M256 worst family primary2x     .9322033898
primary evaluation n            261
U_ONLY pooled contain2          .6781609195
```

If source reconstruction is inconsistent, stop and repair parity without changing P0.

## Primary held-out gates

Aggregate the eight held-out folds exactly once.

Absolute gates, inherited from V2:

```text
pooled contain2 >= .75
worst-family contain2 >= .60
11032/13203/15290 each contain2 >= .60
pooled contain1 >= .50
worst-family contain1 >= .35
pooled median normalized error <= 1.00
best-worst contain2 gap <= .30
```

Causal gate versus U_ONLY:

```text
pooled contain2 gain >= +.08
```

Hard-tail causal condition requires either:

```text
worst-family contain2 gain >= +.08
```

or

```text
at least 2 of {11032,13203,15290} gain >= +.10
and none degrades by >.05
```

## Decision

If parity, all absolute gates and causal gates pass:

`INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_PASS`

If parity passes but endpoint gates fail:

`INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_FAIL`

A pass supports the proposition that higher-order incident relation patterns are learnably aggregatable under family-disjoint generalization. It does not yet authorize blind qualification or large training.

## Boundaries

Forbidden:

- adding more features after viewing P0 results;
- tuning C/class weights/solver;
- using held-out-family statistics for normalization;
- changing M256/F16/R_REL_DIS/graph;
- outputting free XYZ;
- adding image/descriptor features;
- adding p_active in this probe;
- changing the label to nearest-teacher singleton;
- accessing sealed21/external10;
- large/end-to-end training.

If P0 fails, this exact simple learned incident-statistic route is falsified on the open-development panel. Any richer relational model requires a new preregistration justified by the failure localization.
