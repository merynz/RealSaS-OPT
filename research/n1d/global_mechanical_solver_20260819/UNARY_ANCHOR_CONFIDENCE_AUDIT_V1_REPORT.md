# RealSaS N1D — Unary Anchor Confidence Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Decision:** `NO_SAFE_UNARY_ANCHOR_CONFIDENCE_FOUND_V1`  
**Prereg commit:** `61bd2c33ba33ef2c8fc1196133ac59b7fd439f75`

## Baseline

All mapping-reliable M256-contained U_ONLY endpoints:

```text
n                 480
contain1           .466667
contain2           .795833
worst family c2    .581818
```

## Top-25% confidence sets

Confidence ranking is observation-only across all 64 carriers within each family; truth is evaluator-only after selection.

### C_MARGIN — normalized unary winner margin

```text
n evaluable         115
contain1             .365217
contain2             .678261
worst-family c2      .090909
contain2 gain       -.117572
```

11032 top-quartile contain2 is only `.0909`. Unary margin is actively misleading on this hard tail.

### C_ABS — minimum raw unary reprojection residual

```text
n evaluable         120
contain1             .491667
contain2             .866667
worst-family c2      .750000
contain2 gain       +.070833
```

Useful enrichment exists, but it misses the frozen safe-anchor gates (`pooled c2 .90`, worst .80, c1 .65, gain .10).

Per-family top25 contain2:

```text
09908 .875
11032 .750
12772 .933
13203 .750
14404 .867
14702 .938
14758 .933
15290 .867
```

### C_AGREE — unary / pair-context agreement under U_ONLY neighbors

```text
n evaluable         122
contain1             .590164
contain2             .909836
worst-family c2      .642857
contain2 gain       +.114003
```

Pooled enrichment is strong, but hard-tail safety fails because 11032 remains only `.642857`.

Per-family top25 contain2:

```text
09908 1.000
11032 .643
12772 1.000
13203 .812
14404 1.000
14702 1.000
14758 .933
15290 .867
```

## Decision

No preregistered confidence arm passes all safe-anchor gates.

Therefore:

`NO_SAFE_UNARY_ANCHOR_CONFIDENCE_FOUND_V1`

Confidence-freezing / anchored propagation is not authorized from this audit.

## Interpretation

The current failure is not safely repaired by preserving a fixed high-confidence unary subset. Unary confidence itself is family-dependent, and the classic hard family `11032` specifically breaks both margin and agreement safety.

Combined with prior evidence:

- M256 candidate coverage PASS;
- R_REL_DIS full-matrix transfer PASS;
- pair-only correct-context ranking substantially stronger than unary;
- fixed z-score objective fails absolute endpoint gates;
- safe unary anchors cannot be identified by simple frozen confidence signals;

a principled next objective experiment is scale/outlier calibration rather than node freezing or lambda tuning.

A separately preregistered rank-calibrated objective may monotonically transform each unary vector and pair matrix to empirical percentile costs while keeping candidates, graph, factor ordering, weights and ICM unchanged.

## Provenance

- execution source SHA256: `c62f79b2c6e5f991f584a63b0c885930a3ef6adf58f76b458dd526a1eb795ec8`
- full local result SHA256: `7eb6d544e35a1d0fc49440c141de6ead79ae30ca9c3be69289b4dfcacc1dc7e7`
