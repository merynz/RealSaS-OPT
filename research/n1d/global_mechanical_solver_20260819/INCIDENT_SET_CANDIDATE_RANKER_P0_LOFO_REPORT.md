# RealSaS N1D — Incident-Set Candidate Ranker P0 — Family-LOFO Canonical Result

**Date:** 2026-08-19  
**Verdict:** `INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_FAIL`  
**Prereg commit:** `37a6ba7b975176fa8d549223166578707dfa2329`  
**Qualification:** truth-open small learned aggregation probe only.

## Parity

All frozen guards pass exactly before LOFO interpretation:

```text
M256 reliable denominator   492
M256 pooled primary2x       .9756097561
M256 worst-family primary2x .9322033898
primary evaluator n         261
U_ONLY contain1             .3716475096
U_ONLY contain2             .6781609195
```

The model selects only existing M256 candidates. No free XYZ, image feature, descriptor feature, family ID, p_active or evaluator quantity enters inference features.

## Frozen P0

Each candidate receives only:

```text
U percentile
incident R_REL_DIS percentile mean / median / min / max / std
fraction incident edge percentiles <= .25
fraction incident edge percentiles <= .10
```

Neighbor context is frozen U_ONLY. Labels are set-valued: every M256 candidate within evaluator 2x local scale is positive. Eight folds hold out one entire family from both feature scaling and logistic fitting.

## Held-out result

```text
family   contain1   contain2
09908      .3000      .8000
11032      .0800      .1600
12772      .2727      .8182
13203      .3056      .6667
14404      .2500      .5938
14702      .5227      .8864
14758      .3939      .8788
15290      .2800      .6400
```

Aggregate:

```text
contain1                 .3180076628
contain2                 .6896551724
worst-family contain1    .0800000000
worst-family contain2    .1600000000
median normalized error  1.3985875171
p90 normalized error     4.0052816581
best-worst c2 gap        .7263636364
```

Relative to U_ONLY:

```text
pooled contain2 gain   +.0114942529
worst-family gain      -.1200000000
11032 gain             -.1200000000
13203 gain             +.0555555556
15290 gain             -.0600000000
```

Frozen gates:

```text
absolute quality FAIL
pooled causal    FAIL
hard-tail causal FAIL
```

Therefore:

`INCIDENT_SET_CANDIDATE_RANKER_P0_LOFO_FAIL`

## Interpretation

The existing unary percentile plus simple symmetric statistics of U-conditioned incident `R_REL_DIS` rank vectors are **not family-robust sufficient statistics** for endpoint selection. The failure is not a small miss: held-out family 11032 collapses to contain2 `.16`, below even U_ONLY `.28`.

This rules out the tested idea that higher-order context can be recovered merely by learning a fixed logistic weighting over mean/median/min/max/std/support-count summaries of incident pair ranks.

It does not falsify relational mechanical evidence itself: raw R_REL_DIS edge separability remains strong and V2 global relation remains causally useful. It falsifies this compressed learned aggregation representation under family-LOFO.

A next diagnostic should inspect **what information is destroyed by symmetric incident summary compression**, especially whether candidate-to-candidate relational structure / neighbor identity geometry must be retained instead of reducing each incident edge to scalar summary statistics.

## Boundaries

Not authorized:

- tuning logistic hyperparameters on these folds;
- adding features post hoc to P0;
- using family ID or held-out statistics;
- promoting this ranker;
- changing M256/F16/R_REL_DIS;
- free XYZ output;
- large/end-to-end training;
- sealed21/external10 access.

## Provenance

- execution source SHA256: `df23180d93532668cdad5a2f8ac32f552a87b4d7c98c6338c2d910c007f7237f`
- full local result SHA256: `aa7a2b041a1157a9a5931051869bb1e8be3465cc4f5def261ad18961edb377df`
