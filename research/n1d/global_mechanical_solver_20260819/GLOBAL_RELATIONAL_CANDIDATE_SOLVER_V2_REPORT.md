# RealSaS N1D — Global Relational Candidate Solver V2 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`  
**Qualification:** unchanged; truth-open development only.  
**Parent prereg:** `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_PREREG.md` commit `d44911b78dcf18f92a599310d17769f240dd296e`.

## Source parity

The exact F16 reconstruction reproduces the canonical broad-8-family source on all 8 families with mapping-reliable denominator `492/512`.

```text
09908 1.0000000000
11032 0.9491525424
12772 1.0000000000
13203 0.9508196721
14404 0.9682539683
14702 1.0000000000
14758 0.9838709677
15290 1.0000000000
```

## M256 preflight — PASS

V2 changes only the deterministic observation-only mode budget from M128 to M256. Best-32 unary seed, XYZ farthest-point fill, unary, graph, R_REL_DIS, normalization, edge weighting and deterministic ICM remain frozen.

M256 preflight:

```text
pooled primary-2x  0.9756097561   PASS >= .970
worst family       0.9322033898   PASS >= .900
families >= .90    8/8            PASS
best-worst gap     0.0677966102   PASS <= .10
```

Per-family M256 primary-2x:

```text
09908 1.0000000000
11032 0.9322033898
12772 1.0000000000
13203 0.9344262295
14404 0.9682539683
14702 1.0000000000
14758 0.9838709677
15290 0.9836065574
```

Strict-1x:

```text
09908 0.9841269841
11032 0.8135593220
12772 1.0000000000
13203 0.8524590164
14404 0.9047619048
14702 1.0000000000
14758 0.9516129032
15290 0.8852459016
```

Because all preflight gates pass, the preregistered frozen solver is authorized and was executed.

## Primary endpoint evaluation

Primary evaluator-only set requires mapping-reliable + truth-active + full-F16-contained2x + M256-contained2x. `n=261`.

### U_ONLY

```text
pooled contain1          0.3716475096
pooled contain2          0.6781609195
worst-family contain1    0.0000000000
worst-family contain2    0.2800000000
best-worst contain2 gap  0.5533333333
median norm error        1.3179303570
p90 norm error           4.1520870275
```

### G_REL_DIS

```text
pooled contain1          0.4521072797
pooled contain2          0.8084291188
worst-family contain1    0.0400000000
worst-family contain2    0.4400000000
best-worst contain2 gap  0.5600000000
median norm error        1.1422151401
p90 norm error           3.0451581388
```

Per-family primary contain2, U_ONLY -> G_REL_DIS:

```text
09908  .833333 -> .900000
11032  .280000 -> .440000
12772  .818182 -> .727273
13203  .611111 -> .694444
14404  .500000 -> .718750
14702  .818182 -> .977273
14758  .818182 -> 1.000000
15290  .700000 -> .820000
```

Hard-tail gains:

```text
11032 +0.160000
13203 +0.083333
15290 +0.120000
```

## Secondary safety

All mapping-reliable M256-contained carriers, `n=480`:

```text
U_ONLY pooled contain2    .7958333333
G_REL_DIS pooled contain2 .8791666667
```

Safety gate passes comfortably; the graph does not create a broad regression.

## Frozen gates

```text
causal improvement: PASS
safety:             PASS
absolute quality:   FAIL
```

The absolute gate fails for multiple independent reasons:

- worst-family primary contain2 `0.44 < 0.60`;
- hard-tail `11032 = 0.44 < 0.60`;
- worst-family primary contain1 `0.04 < 0.35`;
- pooled median normalized error `1.1422 > 1.00`;
- best-worst contain2 gap `0.56 > 0.30`.

Therefore the canonical verdict is:

`RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`

## Solver dynamics

All families reduce the frozen objective from unary initialization and converge deterministically in 4–12 sweeps. Examples:

```text
11032 objective -89.1872 -> -104.2964, sweeps 5
13203 objective -91.4412 -> -108.7661, sweeps 4
14404 objective -82.4709 -> -102.8473, sweeps 12
15290 objective -99.2309 -> -103.8111, sweeps 4
```

This shows ICM is not simply failing to move. It moves strongly and relation factors causally improve endpoint quality, but the frozen objective/optimization combination still prefers configurations with unacceptable absolute hard-tail error.

## Scientific conclusion

The candidate-domain bottleneck is no longer the explanation: M256 passes its preregistered coverage preflight.

The frozen R_REL_DIS signal is also not inert: it improves pooled primary contain2 by `+0.13027`, strongly improves several hard families, and passes causal/safety gates.

What remains unresolved is **solver conversion**: whether the limitation is primarily

1. transfer of oracle pair separability to the full M256 pair matrices;
2. the frozen unary/pair weighting objective preferring a wrong globally coherent basin; or
3. deterministic ICM becoming trapped in a local basin from unary initialization.

Those possibilities must be separated under a new preregistered failure-localization diagnostic before changing relation, weights, graph or optimizer.

## Provenance

- checkpoint SHA256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- V2 execution source SHA256: `047c264e5a59ff05386cb801e23f0fb8a083828817a329737c50b407093ae8ba`
- V2 preflight JSON SHA256: `a3a63efc3a6543c8663d13b1a9986bb4965f6cc6237300666c567bcd6e4b4a5b`
- full local result JSON SHA256: `9bccac5988f24c8a2f2fe900f60c10bad796a8367f6c3f37efaa24d37ce6a3df`
