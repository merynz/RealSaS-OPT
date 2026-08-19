# RealSaS N1D — Global Solver Common-Mode Gauge Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Decision:** `COMMON_MODE_NOT_DOMINANT__SIMPLE_MEDIAN_DIS_ANCHOR_NOT_SUPPORTED`  
**Prereg commit:** `f8117f2f5e6491f83ae3c2f32bb56e449497a69a`

## A — algebraic gauge invariance: PASS

R_REL_DIS is numerically invariant to a common 3D translation applied to both endpoints of every pair:

```text
max absolute raw relation-energy difference = 4.97e-14 pixels
```

Frozen numerical gate `<=1e-9` passes by a large margin. The common-translation gauge/null direction is therefore real as a mathematical property of the pair factor.

## B — truth-fitted best common translation: NOT DOMINANT

V2 G_REL_DIS primary endpoint quality before common correction:

```text
n                       261
contain1                 .452107
contain2                 .808429
median norm error        1.142215
p90 norm error           3.045158
SSE                       .830250
```

For each family separately, fit the least-squares evaluator-only translation:

```text
g* = -mean(selected_P_B - truth_P_B)
```

After applying this best possible family common translation continuously:

```text
contain1                 .429119
contain2                 .793103
median norm error        1.093749
p90 norm error           2.806517
SSE                       .726101
pooled SSE explained      .125443
```

Hard-tail contain2 gains:

```text
11032  -.080000
13203  +.055556
15290  -.120000
```

Preregistered `COMMON_MODE_MATERIAL` criterion fails. A single family-level translation explains only about 12.5% of squared endpoint error and makes two of the three hard tails worse under containment.

Per-family before -> after contain2 / SSE explained:

```text
09908  .900 -> .967 / .082
11032  .440 -> .360 / .096
12772  .727 -> .818 / .208
13203  .694 -> .750 / .191
14404  .719 -> .688 / .058
14702  .977 -> .977 / .022
14758 1.000 ->1.000 / .060
15290  .820 -> .700 / .078
```

Therefore the mathematical gauge freedom is not the dominant explanation for the observed hard-tail endpoint failure.

## C — simple observation-native median-DIS common anchor: NOT SUPPORTED

Frozen construction:

1. componentwise median raw DIS per view over all Pose-A-visible finite carriers;
2. least-squares 3D common displacement from the known orthographic view bases;
3. subtract current solution mean displacement to obtain diagnostic `g_obs`;
4. apply continuously for evaluation only.

Result:

```text
before contain2          .808429
after contain2           .643678
gain                    -.164751
before contain1          .452107
after contain1           .252874
mean cosine(g_obs,g*)   -.368900
```

Hard-tail gains:

```text
11032 -.320000
13203 -.027778
15290 -.140000
```

Canonical classification:

`NOT_SUPPORTED_UNDER_SIMPLE_MEDIAN_DIS_ESTIMATOR`

This explicitly forbids promoting the simple median-DIS common anchor from this diagnostic.

## Interpretation

Two facts now coexist:

1. pairwise R_REL_DIS has an exact common-translation null direction;
2. actual solver error is not primarily a common translation.

The remaining objective failure is therefore more structured. The most direct next diagnostic is candidate-rank decomposition under evaluator-correct neighbor context:

```text
oracle candidate rank under unary-only cost
oracle candidate rank under pair-sum-only cost
oracle candidate rank under frozen unary + pair total cost
```

If pair-only ranks the oracle well but total does not, the current unary/relative scaling is damaging relation evidence. If pair-only itself fails, strong per-edge separability is being lost when multiple incident relations are aggregated around a node.

No weight, graph, relation, optimizer or anchor change is authorized by this audit.

## Provenance

- execution source SHA256: `4b8e53cf76f77cc28b586a64a8f45ac0e1ddc48f81a1079d338e1a69d0d3bc0b`
- full local result SHA256: `825d150fff5dd18f36904228cdd6f201d179d3f1bde6ef9d4674a213c9effd9b`
