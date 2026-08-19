# RealSaS N1D — Observation-Native Edge Reliability / Cross-Consensus Audit V1 — Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DIAGNOSTIC_PREREG__NO_EDGE_WEIGHT_CHANGE`

## Parent result

Canonical parent: `MIXED_MULTI_EDGE_FAILURE` from `MULTI_EDGE_INCIDENT_CONSENSUS_AUDIT_V1`.

The parent established, under exact behavioral parity, that:

- frozen correct-neighbor pair-sum baseline is `contain2=.8544061303`, worst-family `.68`;
- 46 primary nodes have pair-sum oracle rank `>.25`;
- evaluator-only best single-edge deletion rescues `.4565217391` of those bad nodes to rank `<=.25`;
- fixed `TRIM_MAX1_Z` and `MEDIAN_EDGE_RANK` both preserve pooled quality but degrade worst-family/11032 `.68 -> .64`;
- therefore neither pure single-edge poisoning nor uniformly coherent wrong consensus explains the full failure.

This audit asks whether the damaging incident edge can be predicted from observation-native consistency evidence, without evaluator truth.

## Frozen state

Unchanged from V2 / parent audit:

- F16 H and M256 candidate domains;
- candidate ordering;
- 4-NN symmetrized graph and edge eligibility;
- raw `R_REL_DIS`;
- per-edge median/IQR z calibration;
- degree weights;
- raw DIS observations and camera projection basis;
- frozen U_ONLY candidate index as the only observation-side neighbor state used by predictors.

No relation, graph, weight, candidate, optimizer or learned parameter may change.

## Evaluator-only target label

Only for the 46 parent baseline-bad primary nodes, reproduce the parent correct-neighbor evaluator context and leave-one-edge-out oracle ranks.

Define the damaging edge label `e*` as the incident edge whose deletion gives the minimum oracle candidate rank. Ties use the stable incident-edge ordering already induced by the frozen sorted graph.

This label is evaluator-only. It may never enter a predictor feature.

Behavioral guards before interpretation:

```text
primary n                         261
bad n                              46
pair-sum pooled contain2         .8544061302681992
pair-sum worst-family contain2   .68
pooled median oracle rank        .072265625
worst-family median oracle rank  .201171875
LOEO rescue<=.25                 .45652173913043476
```

Any material mismatch invalidates the audit until parity is repaired without changing this preregistration.

## Observation-only predictor context

For every node, neighbor candidate state for predictor construction is the frozen U_ONLY argmin only. Evaluator oracle neighbor states are forbidden in all predictor scores.

For each incident edge `e=(i,j)`, construct the conditional candidate cost vector for node i using frozen U_ONLY state of j:

```text
c_e^U(h_i) = w_ij * R_z(i,j,h_i,x_j^U)
```

Convert it to empirical mid-rank candidate percentile vector `p_e^U(h_i)` using the same tie definition as prior audits.

## Frozen predictor S1 — RANK_VECTOR_DISAGREE

For node i, form the candidate-wise incident median rank vector:

```text
p_med(h) = median_e p_e^U(h)
```

Score edge unreliability by mean absolute disagreement over the complete M256 candidate domain:

```text
S1(e) = mean_h |p_e^U(h) - p_med(h)|
```

Higher means more cross-edge ranking disagreement.

## Frozen predictor S2 — ENDPOINT_VOTE_OUTLIER

For every incident edge independently choose:

```text
q_e = argmin_h c_e^U(h)
P_e = M_i[q_e]
```

Define the geometric incident consensus point as the componentwise median of `{P_e}`. Score:

```text
S2(e) = ||P_e - median_e(P_e)|| / local_scale_i
```

Higher means the edge's independently preferred feasible endpoint is geometrically more isolated from the other incident edge votes.

No evaluator target participates.

## Frozen predictor S3 — REL_FLOW_VIEW_INCONSISTENCY

For the common visible views of edge `(i,j)`, let observed relative 2D motion be:

```text
r_v = DIS_i(v) - DIS_j(v)
```

Fit one 3D relative displacement vector `g_e` by ordinary least squares against the known orthographic right/up projection rows over all common views. Compute per-view 2D residual norms. Define:

```text
S3(e) = median_v ||Proj_v(g_e)-r_v||
        / (median_v ||r_v|| + 1e-9)
```

Higher means the observed relative flow itself is less compatible with a single common 3D relative displacement across views.

No candidate target or evaluator truth participates.

## Prediction evaluation

For each parent bad node and each S1/S2/S3:

1. rank incident edges by descending unreliability score;
2. stable graph order breaks exact ties;
3. measure whether `e*` is top1 and top2;
4. measure normalized rank of `e*`:

```text
normalized_rank = position(e*) / max(degree-1,1)
```

where best position is zero.

Report pooled and per-family results, plus degree strata `<=4`, `5`, `>=6`.

## Frozen support gate

A predictor is `SUPPORTED` only if all hold on the 46 parent bad nodes:

```text
top1 hit rate                >= .35
top2 hit rate                >= .60
median normalized rank       <= .35
mean normalized rank         <= .40
```

and at least 5 families with bad nodes have mean normalized rank `<=.50`.

No best-arm metric averaging or threshold relaxation is allowed.

## Decision

If one or more preregistered predictors pass all support gates:

`OBS_EDGE_RELIABILITY_SIGNAL_FOUND_V1`

Report every passing predictor; do not choose a solver modification on this panel.

If none pass:

`NO_SIMPLE_OBS_EDGE_RELIABILITY_SIGNAL_V1`

If behavioral parity fails:

`INVALID_PARENT_PARITY_FAIL`

## Boundaries

This audit is diagnostic only. It does **not** authorize:

- deleting or downweighting a predicted edge;
- choosing a best score by endpoint truth;
- combining S1/S2/S3 after seeing results;
- fitting a classifier or learned gate;
- lambda/temperature/threshold sweeps;
- changing graph topology or degree;
- changing R_REL_DIS;
- adding p_active/unary anchors;
- changing optimizer or M256;
- accessing sealed21/external10;
- large/end-to-end training.

If a score passes, a separate preregistered solver intervention must use that frozen score and test causal endpoint improvement. If none pass, the next hypothesis must move beyond simple per-edge reliability ranking toward higher-order relational structure rather than post-hoc edge trimming.
