# RealSaS N1D — Global Relational Candidate Solver V3 — Rank-Calibrated Objective — Canonical Development Result

**Date:** 2026-08-19  
**Verdict:** `RANK_CALIBRATION_INSUFFICIENT_V3`  
**Qualification:** unchanged; truth-open development only.  
**Parent prereg:** `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V3_RANK_CALIBRATED_PREREG.md`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Execution validity / parity

V3 was reconstructed from the frozen V2 semantics and guarded by exact behavioral parity before any endpoint result was interpreted.

The guard reproduced:

- mapping-reliable denominator: `492/512`;
- full-F16 primary-2x coverage on every family;
- M256 primary-2x coverage on every family;
- pooled M256 primary-2x `0.975609756097561`.

Full F16 / M256 per-family parity:

```text
family   F16          M256
09908   1.0000000000  1.0000000000
11032    .9491525424   .9322033898
12772   1.0000000000  1.0000000000
13203    .9508196721   .9344262295
14404    .9682539683   .9682539683
14702   1.0000000000  1.0000000000
14758    .9838709677   .9838709677
15290   1.0000000000   .9836065574
```

Therefore the V3 endpoint comparison is behaviorally source-valid relative to V2.

## Single preregistered change

All raw evidence, candidates, graph, degree weights, ICM mechanics and evaluator sets remain frozen from V2.

Only factor calibration changes:

```text
U_rank(i,h) = empirical mid-rank percentile of U_raw(i,:)
R_rank(i,j,h_i,h_j) = empirical mid-rank percentile of R_REL_DIS_raw(i,j,:,:)

E_rank = sum_i U_rank + sum_(i,j) (1/deg_i + 1/deg_j) R_rank
```

The transform preserves each unary/pair factor ordering and bounds factors to percentile scale.

## Baseline parity

Because unary percentile calibration is monotonic, `U_RANK_ONLY` is endpoint-identical to V2 `U_ONLY`:

```text
n                         261
pooled contain1           .3716475096
pooled contain2           .6781609195
worst-family contain1     .0000000000
worst-family contain2     .2800000000
best-worst contain2 gap   .5533333333
median norm error         1.3179303570
p90 norm error            4.1520870275
```

This exactly reproduces the V2 unary endpoint baseline.

## G_RANK_REL primary result

```text
n                         261
pooled contain1           .4559386973
pooled contain2           .7586206897
worst-family contain1     .1200000000
worst-family contain2     .3600000000
best-worst contain2 gap   .6400000000
median norm error         1.1141697069
p90 norm error            3.2071796487
```

Per-family primary contain2, `U_RANK_ONLY -> G_RANK_REL`:

```text
09908  .833333 -> .933333
11032  .280000 -> .360000
12772  .818182 -> .818182
13203  .611111 -> .694444
14404  .500000 -> .531250
14702  .818182 -> .931818
14758  .818182 -> 1.000000
15290  .700000 -> .720000
```

Hard-tail gains:

```text
11032 +.080000  # exactly 2/25
13203 +.083333
15290 +.020000
```

## Secondary safety

All mapping-reliable M256-contained carriers, `n=480`:

```text
U_RANK_ONLY contain2   .7958333333
G_RANK_REL contain2    .8562500000
```

Safety is PASS; rank-calibrated relations do not create a broad regression.

## Frozen gates

### Absolute quality — FAIL

Although pooled contain2 barely clears `.75`, multiple independent absolute gates fail:

```text
pooled contain2              .75862  PASS >= .75
worst-family contain2        .36000  FAIL < .60
11032 contain2               .36000  FAIL < .60
pooled contain1              .45594  FAIL < .50
worst-family contain1        .12000  FAIL < .35
median normalized error     1.11417  FAIL > 1.00
best-worst contain2 gap      .64000  FAIL > .30
```

### Causal improvement — PASS at exact boundary

Pooled contain2 gain is:

```text
.7586206897 - .6781609195 = .0804597701 >= .08
```

Worst-family gain is exactly:

```text
11032: 9/25 - 7/25 = 2/25 = .08
```

A binary-float representation displays this as `.07999999999999996`; the preregistered mathematical `>= .08` gate is therefore **PASS**, not FAIL.

### Safety — PASS

Secondary contain2 improves by about `+.06042`.

## Comparison to V2 z-score solver

V2 `G_REL_DIS`:

```text
contain1          .452107
contain2          .808429
worst contain2    .440000
median norm err   1.142215
best-worst gap    .560000
```

V3 rank calibration:

```text
contain1          .455939
contain2          .758621
worst contain2    .360000
median norm err   1.114170
best-worst gap    .640000
```

Rank calibration slightly improves pooled contain1 and median normalized error, but materially worsens pooled contain2, worst-family contain2 and family spread. In particular `11032` degrades `.44 -> .36` and `15290` `.82 -> .72` relative to V2.

Therefore the hypothesis that factor-scale/outlier mismatch can be repaired by a pure monotonic empirical-percentile calibration is falsified on this open-development panel.

## Scientific interpretation

The evidence chain now rules out several simple explanations:

1. candidate-domain insufficiency: M256 preflight PASS;
2. missing pair signal: R_REL_DIS full-matrix transfer PASS;
3. optimizer simply failing to move: V2 ICM strongly lowers its objective;
4. simple common-mode translation: NOT DOMINANT;
5. simple fixed high-confidence unary anchors: NOT SAFE;
6. z-score scale/outlier pathology fixable by order-preserving percentile normalization: V3 FAIL.

The unresolved layer is more specifically **multi-edge relational context composition**. Prior correct-neighbor diagnostics showed pair-only local selection is much stronger than unary, yet incident relations aggregate heterogeneously and errors compound once neighboring nodes leave truth-near states. V3 confirms that making every factor merely bounded/comparable by rank is insufficient.

The next safe diagnostic should localize whether endpoint failure is driven by a minority of incident edges / inconsistent relation neighborhoods versus coherent but wrong multi-edge consensus. It should remain diagnostic-only before changing graph, weights, relation, optimizer, or adding learned arbitration.

## Decision

`RANK_CALIBRATION_INSUFFICIENT_V3`

This result does **not** authorize:

- lambda sweeps;
- pair/unary mixing temperatures;
- rank powers/logits/clipping;
- changing R_REL_DIS;
- changing M256;
- graph edits;
- restarts/new optimizer;
- reopening per-carrier motion heads;
- large/end-to-end training;
- Stage-B requalification;
- sealed21/external10 access.

## Provenance

- checkpoint SHA256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- local reconstructed V3 execution source SHA256: `2bdbb6f0628a225c667f7687a69af2ad8a47211671f247b93269e324fb33aa09`
- local aggregate result JSON SHA256: `e98b3be23033248ecf9cb08509e5773bf9507c684e2fbed5c77a10417adb01e9`

The local runner is a semantic reconstruction, not claimed byte-identical to a previously existing canonical V3 source. Its validity is established by exact F16/M256 behavioral parity and exact V2 unary endpoint parity before applying the single preregistered rank-calibration change.
