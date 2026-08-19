# RealSaS N1D — Local Objective Component Rank Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Classification:** `MULTIEDGE_RELATION_AGGREGATION_FAIL`  
**Prereg commit:** `8642222e8a1011d25fdaf754ad0f7cf8efd98d20`

## Oracle candidate ranks under correct neighbor context

Pooled primary nodes `n=261`:

```text
                    median rank   frac rank<=.25
unary-only             .19336          .58238
pair-sum-only          .07227          .82375
total unary+pair       .10742          .77011
```

Worst-family median oracle rank:

```text
unary-only             .43164
pair-sum-only          .20117
total                  .18750
```

The preregistered pair-aggregation support gate required worst-family median `<= .20`. Observed `12772 = .201171875`, so the gate fails exactly; it is not rounded into a PASS.

Canonical classification under the frozen decision tree:

`MULTIEDGE_RELATION_AGGREGATION_FAIL`

## Argmin endpoint quality under oracle neighbor context

```text
                    contain1   contain2   median norm err   worst fam c2
unary-only           .37165     .67816       1.31793          .280
pair-only            .48276     .85441       1.02317          .680
total                .52107     .87739        .98261          .520
```

Pair-only is a strong diagnostic improvement over unary, but it is not promoted: its own frozen relation-aggregation rank gate narrowly fails and its contain1/median-error are still just outside the V2 absolute endpoint gates.

## Per-family median oracle rank / contain2

```text
family   unary rank  pair rank  total rank    unary c2  pair c2  total c2
09908      .248        .170       .188          .833      .933     .933
11032      .217        .072       .154          .280      .680     .520
12772      .275        .201       .166          .818      .818     .818
13203      .221        .068       .094          .611      .861     .806
14404      .432        .066       .105          .500      .844     .875
14702      .191        .095       .146          .818      .932     .977
14758      .236        .084       .107          .818      .848    1.000
15290      .135        .035       .061          .700      .840     .920
```

Hard-tail pair-minus-total contain2:

```text
11032 +.16000   # unary damages correct-neighbor pair selection
13203 +.05556   # unary damages
15290 -.08000   # unary helps
```

Therefore a fixed global interpretation such as “remove unary” or “increase pair weight” is not supported. Unary interaction is heterogeneous across families/nodes.

## Scientific interpretation

The pair relation remains substantially more informative than the unary under correct context, but multiple incident relations and the unary interact in a way that is not uniformly calibrated. The global ICM then compounds those local context errors as neighbors leave their truth-near states.

The next safe question is not a lambda sweep. It is:

> Do observation-only confidence/uncertainty diagnostics predict when unary should be trusted versus when relational context should dominate?

A separately preregistered confidence-arbitration audit should test unary margin/absolute score and pair-consensus margin against evaluator-only local correctness, without changing solver weights.

## Provenance

- execution source SHA256: `1a60efea7873583ced7461dfce190ff9cc9e205ab50b4d2fde6bb98417c91387`
- full local result SHA256: `a16a0c58976a321394376d2336c741fa4508a1961c7fa4d3d277705a5f28295d`
