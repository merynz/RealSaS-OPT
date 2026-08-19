# RealSaS N1D — Candidate-Specific Full-Vector Motion Energy V1.5

**Date:** 2026-08-19  
**Verdict:** `FAIL__SINGLE_CARRIER_MOTION_VECTOR_INSUFFICIENT`  
**Status:** `TRUTH_OPEN_DETERMINISTIC_DIAGNOSTIC__NOT_TRAINING__NOT_QUALIFICATION`  
**Prereg commit:** `52da162d25c598ee793a0cabac807baf75198e32`

## Question

Amplitude-only local motion compatibility failed in V1. V1.5 kept the same frozen F16 H and the same three motion fields, but restored the complete candidate projected 2D vector.

Arms tested:

```text
V_* = median native-pixel L2 vector residual
P_* = median equal-weight (log-magnitude residual + direction residual)
```

for raw DIS, learned transport, projected per-view delta3D, plus deterministic within-H robust fusion.

Evaluation: the same 264 reliable truth-active F16-contained carriers.

## Best aggregate arm — V_DIS

```text
pooled contain1               .28409
pooled contain2               .60985
worst-family contain2         .23077
best-worst gap                .55711
median normalized error       1.59479
13203 contain2                .45946
15290 contain2                .68627
```

This is far below the preregistered joint-motion evidence gates.

## Tail behavior

### 11032

Best tested contain2 is only `.50` (`P_DELTA3D`), with raw DIS/transport at `.23077`.

### 13203

Best tested contain2 is `.56757` (`P_TRANSPORT`), below the `.60` hard-tail floor.

### 15290

Raw DIS vector reaches `.68627`, but this does not generalize to the other hard families.

## Other aggregate arms

```text
V_TRANSPORT contain2  .57576
V_DELTA3D             .49621
V_FUSION              .56439
P_DIS                 .58333
P_TRANSPORT           .54545
P_DELTA3D             .46591
P_FUSION              .56439
```

No arm passes.

## Scientific conclusion

Restoring direction into a single-carrier candidate motion score **does not solve the hard tail**.

Therefore the amplitude failure cannot be explained solely by an overly independent `log_amp + dir` factorization. The stronger result is now:

> A single carrier's local A->B motion vector—whether raw DIS, learned transport, frozen per-view 3D differential, or deterministic fusion—is insufficient to select the correct member of a geometrically sufficient H set in a family-robust way.

This sharply narrows the missing representation:

- geometry set H is sufficient under oracle;
- p_active is learnable;
- carrier-local scalar amplitude is insufficient;
- carrier-local candidate magnitude compatibility is insufficient;
- carrier-local full-vector compatibility is insufficient.

The remaining evidence must therefore be **relational / neighborhood / global mechanical** rather than another independent per-carrier scalar or vector.

## Next required diagnostic — local differential relation separability

Build a kNN graph over observation-derived `P_A` carriers and test whether oracle-near candidate pairs are distinguished by local relation energies that are fully observation/computation-native.

Candidate pair `(h_i,h_j)` relation candidates include:

```text
1. local displacement coherence
   ||(h_i-P_A_i) - (h_j-P_A_j)||

2. local metric/rigidity residual
   | ||h_i-h_j|| - ||P_A_i-P_A_j|| |

3. observation-relative differential motion
   median_v ||
      [(proj(h_i)-proj(P_A_i)) - (proj(h_j)-proj(P_A_j))]
      - [obs_motion_i(v)-obs_motion_j(v)]
   ||
```

The third term is especially important: subtracting neighboring observed motion can remove family/global calibration bias and expose the local differential field.

The first diagnostic should not solve the whole graph. It should ask whether the evaluator-only oracle candidate pair has low energy percentile among deterministic sampled H_i x H_j alternatives, strict per-family. If it does, a graph/global solver is justified. If it does not, neighborhood/Jacobian evidence itself must be enriched before solver construction.

## Architecture consequence if relational signal exists

Do not train a standalone `log_amp` head. Use:

```text
p_active
+ set-valued H_i
+ pairwise/local differential mechanical factors
+ global compiler/solver
```

with amplitude emerging from the globally selected feasible configuration rather than from a free per-carrier amplitude prediction.

## Authorization boundary

- F16 H remains frozen.
- typed p_active remains supported.
- standalone/global/H-relative amplitude remains falsified.
- single-carrier full-vector scoring remains falsified.
- local relational/mechanical separability audit is authorized next.
- no head/backbone/large training.

## Reproducibility

Exact aggregate result SHA-256:
`95c639baacdb4a994626925e370f2f8961507a7a4a95f4030014b7b2eaad852e`

Exact source SHA-256:
`78755b8dbede9ff39f66abb18a9c9061c1b8e2928bd950e93e914ed2717e04e2`
