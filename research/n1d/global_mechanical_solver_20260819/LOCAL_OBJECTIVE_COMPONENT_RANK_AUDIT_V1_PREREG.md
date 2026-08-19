# RealSaS N1D — Local Objective Component Rank Audit V1 Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DIAGNOSTIC_PREREG__NO_WEIGHT_TUNING`

## Parent evidence

- M256 candidate domain passes preflight.
- R_REL_DIS transfers strongly to complete M256 pair matrices.
- Oracle-neighbor frozen total local cost still fails endpoint gates.
- Oracle-perfect initialization is pulled toward a bad basin.
- Common-mode translation is not the dominant error.

This audit asks exactly where the oracle candidate loses rank under correct neighbor context.

## Frozen state

Use exact V2 M256, graph, U normalization, R_REL_DIS normalization and degree weights. No changes.

For each primary node, define evaluator oracle candidate `o_i` as nearest M256 endpoint to truth. Fix every neighbor to its oracle index when available; otherwise frozen U_ONLY index, exactly as Failure Localization Diagnostic B.

## Three candidate costs

For every primary node and candidate `k`:

```text
C_unary(k) = U_i(k)

C_pair(k)  = sum_{j in N(i)} w_ij R_ij(k, x_j^oracle-context)

C_total(k) = C_unary(k) + C_pair(k)
```

Compute percentile rank of oracle candidate under each vector:

```text
rank_pct = [# cost < oracle_cost + .5 * # cost == oracle_cost] / # candidates
```

Also select argmin candidate independently under unary-only, pair-only and total and evaluate endpoint contain1/contain2/normalized error. Pair-only is diagnostic only and is not a promoted solver arm.

## Diagnostic classification

### Pair aggregation support

`PAIR_AGGREGATION_SUPPORTED` iff:

```text
pooled median oracle pair-only rank <= .10
worst-family median pair-only rank  <= .20
pooled fraction pair-only rank<=.25 >= .70
```

### Unary damage

If pair aggregation is supported, classify `UNARY_DOMINATION_SUPPORTED` iff both:

```text
median oracle total-rank - median oracle pair-rank >= .10
pair-only pooled contain2 - total pooled contain2 >= .05
```

and no more than one hard-tail family loses contain2 under pair-only relative to total by >.05.

### Multi-edge relation aggregation failure

If pair aggregation support fails, classify `MULTIEDGE_RELATION_AGGREGATION_FAIL`.

Otherwise classify `MIXED_LOCAL_OBJECTIVE_FAILURE`.

## Report strata

Report pooled/per-family:

- median oracle rank for unary / pair / total;
- fraction oracle rank<=.25 for all three;
- argmin contain1 / contain2 / median normalized error;
- degree strata `<=4`, `5`, `>=6`;
- hard tails 11032, 13203, 15290.

## Boundaries

Forbidden:

- changing unary/pair weights;
- trying lambda values;
- changing graph degree;
- changing R_REL_DIS;
- using pair-only as an inference promotion based on this same panel;
- changing M256/ICM;
- adding p_active or any new anchor;
- accessing sealed21/external10.

Any objective modification requires a separate preregistered experiment after this audit is frozen.
