# RealSaS N1D — Global Solver V2 Failure Localization V1 Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DIAGNOSTIC_PREREG__NO_SOLVER_TUNING`  
**Parent result:** `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`

## Purpose

Global Relational Candidate Solver V2 established:

- M256 candidate-domain preflight PASS;
- frozen R_REL_DIS graph causally improves endpoint selection over U_ONLY;
- safety PASS;
- absolute endpoint gate FAIL.

This diagnostic must distinguish three failure levels **without changing any production-side solver quantity**:

1. relation transfer failure: the previously strong R_REL_DIS oracle-pair separability does not survive the full M256 candidate domain;
2. objective/context failure: R_REL_DIS remains separable, but the frozen unary + degree-weighted pair objective does not locally prefer the correct candidate even under correct neighboring context;
3. optimization-basin failure: the frozen objective supports a good configuration, but unary-initialized deterministic ICM converges to a worse local basin.

No result from the diagnostics below has been inspected before this preregistration.

## Frozen quantities

Identical to Global Relational Candidate Solver V2:

- F16 H construction;
- M256 candidate domain;
- U_raw and robust unary normalization;
- 4-NN symmetrized graph and >=2 common-visible-view edge rule;
- R_REL_DIS raw relation;
- per-edge median/IQR normalization;
- degree weight `1/deg_i + 1/deg_j`;
- no lambda;
- deterministic ICM update mechanics and maximum 20 sweeps.

No new feature, relation, head, graph, weight, restart or candidate is allowed.

Truth is evaluator-only for diagnostic oracle indices and labels.

## Oracle candidate definition

For every mapping-reliable M256-contained carrier:

```text
o_i = argmin_{h in M_i} ||h - truth_P_B_i||
```

Tie: lowest candidate index.

This index is diagnostic only and never becomes an authorized inference input.

Primary nodes are exactly the V2 primary evaluator set:

- mapping reliable;
- truth-active;
- full F16 contained2x;
- M256 contained2x.

## Diagnostic A — M256 relation-transfer audit

For every frozen graph edge `(i,j)` where both endpoints have diagnostic oracle candidates, compute the complete frozen normalized R_REL_DIS matrix over `M_i x M_j` and the percentile rank of `R(o_i,o_j)` among all matrix entries:

```text
percentile = [# values < oracle + 0.5 * # values == oracle] / # values
```

Report pooled and per-family median percentile and fraction `<= .25`, plus active-active and active-inactive strata.

Relation transfer is `PASS` iff:

```text
pooled median oracle-pair percentile <= .10
worst-family median percentile       <= .20
pooled fraction <= .25               >= .70
```

If this fails, classification is `M256_RELATION_TRANSFER_FAIL` and later objective/optimizer diagnostics remain descriptive only.

## Diagnostic B — oracle-neighbor local recoverability

Construct one frozen neighbor state for every node:

- if neighbor has a diagnostic oracle candidate, use that oracle index;
- otherwise use frozen U_ONLY index.

For each primary node `i`, hold all neighbors at those fixed states and choose the candidate minimizing the **unchanged V2 local cost**:

```text
U(i,k) + sum_{j neighbor i} w_ij R_ij(k,x_j)
```

No ICM sweep is run in this diagnostic.

Evaluate the selected endpoint with the same contain1/contain2/normalized-error metrics.

Local objective recoverability passes iff the same V2 absolute endpoint gates pass:

```text
pooled contain2 >= .75
worst-family contain2 >= .60
11032/13203/15290 each contain2 >= .60
pooled contain1 >= .50
worst-family contain1 >= .35
pooled median normalized error <= 1.00
best-worst contain2 gap <= .30
```

If A passes but B fails, classification includes `FROZEN_LOCAL_OBJECTIVE_CONTEXT_FAIL`.

## Diagnostic C — oracle-initialized ICM stability / basin test

Construct diagnostic initialization:

- mapping-reliable M256-contained node -> oracle index;
- all other nodes -> frozen unary argmin.

Run the **same frozen deterministic ICM** from this initialization. No other change.

Report:

1. primary endpoint quality before and after ICM;
2. objective of oracle-hybrid initialization;
3. objective after oracle-initialized ICM;
4. objective of the already-defined unary-initialized V2 ICM solution.

Oracle-init endpoint stability passes iff the post-ICM configuration passes the same V2 absolute endpoint gates.

### Failure classification

After A/B:

- if A passes, B passes, oracle-init post-ICM passes absolute endpoint gates, and summed oracle-init-final objective is **strictly lower** than summed unary-init-final objective, classify `OPTIMIZATION_BASIN_FAILURE_SUPPORTED`;
- if oracle-init has materially better endpoint quality but its final objective is >= unary-init-final objective, classify `OBJECTIVE_ALIGNMENT_FAILURE_SUPPORTED`;
- if oracle-init itself is pulled below absolute endpoint gates, classify `OBJECTIVE_STABILITY_FAILURE_SUPPORTED`;
- if evidence is mixed, classify `MIXED_OBJECTIVE_AND_OPTIMIZATION_FAILURE` and do not tune.

`strictly lower` uses raw summed frozen objective with tolerance only for floating equality (`1e-9`); no empirical margin is introduced.

## Diagnostic D — component-energy comparison

For primary nodes/edges report, without promotion gates:

- unary component at U_ONLY, V2 G_REL_DIS, and oracle-hybrid configurations;
- pair component at those configurations;
- total frozen objective;
- per-family objective differences.

This is explanatory only and may identify unary domination, pair domination or cross-family heterogeneity. It does not authorize weight changes.

## Boundaries

This diagnostic cannot:

- change R_REL_DIS;
- tune pair weight/lambda;
- change graph degree;
- add p_active gating to the objective;
- add restarts or alternative optimizer;
- change M256;
- reopen standalone amplitude/vector heads;
- access sealed21/external10;
- alter frozen Stage-B qualification.

Any solver modification suggested by the localization requires a **new preregistered experiment** after this diagnostic is frozen.
