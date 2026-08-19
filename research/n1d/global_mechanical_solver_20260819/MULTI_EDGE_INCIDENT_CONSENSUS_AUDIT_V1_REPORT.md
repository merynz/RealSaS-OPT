# RealSaS N1D — Multi-Edge Incident Consensus / Outlier Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `MIXED_MULTI_EDGE_FAILURE`  
**Qualification:** truth-open diagnostic only; frozen Stage-B qualification unchanged.  
**Prereg commit:** `39c80ce3a636307ba9414d14a3f9099ed6d10ef8`

## Behavioral parity / validity

Execution rebuilt the exact V2/F16/M256 domain from the frozen cache and applied the preregistered oracle-neighbor context:

- evaluator oracle index is used only for `mapping_reliable && M256-contained2x` neighbors;
- otherwise the frozen U_ONLY argmin is used;
- raw `R_REL_DIS`, per-edge median/IQR normalization, graph, and degree weights are unchanged.

All preregistered behavioral guards reproduce the prior Local Objective Component Rank Audit exactly:

```text
primary n                              261      PASS
pair-only pooled contain2              .8544061302681992 PASS
pair-only worst-family contain2        .6800000000000000 PASS
pooled median oracle pair-sum rank     .072265625        PASS
worst-family median oracle rank        .201171875        PASS
```

Therefore this audit is valid and its diagnostics may be interpreted.

## Frozen pair-sum baseline

Under evaluator-correct neighbor context, frozen pair-only aggregation gives:

```text
contain1                  .4827586207
contain2                  .8544061303
worst-family contain1     .1600000000
worst-family contain2     .6800000000
median normalized error   1.0231690472
best-worst contain2 gap   .2533333333
```

Per-family contain2:

```text
09908  .933333
11032  .680000
12772  .818182
13203  .861111
14404  .843750
14702  .931818
14758  .848485
15290  .840000
```

## Diagnostic A — evaluator-only leave-one-edge-out ceiling

A baseline-bad node is defined preregistered as pair-sum oracle rank `> .25`.

```text
bad primary nodes                         46
best single-edge deletion rescue <=.25   .4565217391
best single-edge deletion rescue <=.10   .0652173913
median individually-supporting edge frac .2500000000
```

Deleting the single most damaging incident edge can rescue about 45.65% of bad nodes to oracle rank <=.25. This is substantial, but below the preregistered 50% threshold required for `SINGLE_EDGE_OUTLIER_MATERIAL`.

The rescue rate is also above the `<30%` condition required for `COHERENT_MULTI_EDGE_WRONG`. Thus the bad nodes are not explained by a uniformly coherent wrong consensus either.

## Diagnostic B — observation-only TRIM_MAX1_Z

Frozen rule:

```text
C_trim(h) = sum_e c_e(h) - max_e c_e(h)
```

Result:

```text
contain1                  .4597701149
contain2                  .8352490421
worst-family contain2     .6400000000
median normalized error   1.0664711176
```

Relative to frozen pair-sum baseline:

```text
pooled contain2     -.0191570881
worst-family c2     -.0400000000
11032 c2            .680000 -> .640000
```

Pooled preservation passes, but both required hard-tail improvement checks fail. This rule is not supported for promotion.

## Diagnostic C — observation-only MEDIAN_EDGE_RANK

Each incident edge is independently converted to candidate percentile rank and the node cost is their median.

Result:

```text
contain1                  .4712643678
contain2                  .8429118774
worst-family contain2     .6400000000
median normalized error   1.0392852347
```

Relative to frozen pair-sum baseline:

```text
pooled contain2     -.0114942529
worst-family c2     -.0400000000
11032 c2            .680000 -> .640000
```

Again pooled preservation passes but the hard-tail improvement gates fail. This rule is not supported for promotion.

## Preregistered decision

`SINGLE_EDGE_OUTLIER_MATERIAL` requires both >=50% one-edge rescue among bad nodes and a robust observation-only arm satisfying the hard-tail gates. Observed rescue is 45.65% and neither robust arm passes.

`COHERENT_MULTI_EDGE_WRONG` requires <30% one-edge rescue and median individually-supporting edge fraction <.50. The support fraction is low (.25), but rescue is 45.65%, so this classification also does not apply.

Therefore the preregistered decision tree selects:

`MIXED_MULTI_EDGE_FAILURE`

## Scientific interpretation

The failure is genuinely mixed:

1. a nontrivial subset of bad nodes is sensitive to one damaging incident edge;
2. a similarly important remainder cannot be repaired by deleting only one edge;
3. simple observation-only trimming or median-of-edge-ranks does not generalize safely to the hard tail, especially family 11032.

Together with the prior audits, this narrows the unresolved layer further. Candidate-domain coverage is adequate, individual `R_REL_DIS` edge ordering is strong, and correct-neighbor pair aggregation is generally useful; however **incident relations do not carry uniform reliability and their reliability cannot be recovered by the two simple fixed robust aggregators tested here**.

A next experiment should therefore diagnose observation-native **edge reliability / cross-edge consensus evidence** before changing relation weights, graph topology, or optimizer. It must not use evaluator truth to remove edges at inference.

## Boundaries

This result does not authorize:

- permanent leave-one-edge-out deletion;
- `TRIM_MAX1_Z` or `MEDIAN_EDGE_RANK` promotion;
- lambda/weight sweeps;
- graph-degree edits;
- changing `R_REL_DIS`;
- adding p_active to this objective;
- new optimizer/restarts;
- opening sealed21/external10;
- large/end-to-end retraining.

## Provenance

- prereg commit: `39c80ce3a636307ba9414d14a3f9099ed6d10ef8`
- execution source SHA256: `54b7b113976f730bbd4524cdcc0d6e71b0c7538f7e0f0f40ab80ac4da426969f`
- full local result SHA256: `d151ede430aa587a42f767e0a93968c86e58c32a2e3baf6f81f33774bb88b70a`
- compact result summary SHA256: `71791a147197757328bbd2c4fbf67369b790cb462522da799c33582650e1286e`
