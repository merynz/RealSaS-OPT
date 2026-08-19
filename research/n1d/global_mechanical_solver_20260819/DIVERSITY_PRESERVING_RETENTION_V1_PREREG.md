# RealSaS N1D — Diversity-Preserving Hypothesis Retention V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DEVELOPMENT_TEST`  
**Parent representation:** `REPRESENTATION_CONTRACT_V2.md`  
**Parent adaptive result:** `FAIL__NO_COMPACT_ADAPTIVE_CONTRACT_PASSES__KEEP_F16_REFERENCE`

## Motivation fixed before evaluation

Adaptive H V1 showed that predicting the rare `expand/not-expand` event is not yet reliable: only `13/492` reliable carriers are F8 failures rescued by F16. Expansion-need diagnostics found stronger observation-only signal in **spatial candidate multimodality** (best single LOFO AUROC ~`.808`) than in score entropy/reprojection alone, but the 13-positive panel is too small to freeze a family-robust rare-event classifier.

This prereg therefore tests a structural alternative:

> preserve multiple spatial modes inside a small fixed candidate budget, instead of predicting exactly which rare carrier needs K=16.

## Frozen panel and evidence

Use the same open-development e00 panel and frozen lineage:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

For every carrier/view, use the **same frozen refined top16 descriptor candidates and their existing score order** from Representation Sufficiency Test C / Adaptive H V1.

No model update, new observable channel, truth-conditioned candidate selection or score retuning is allowed.

Truth is used only after retention and H construction to compute the existing reliable-mapping primary-2x containment metric.

## Deterministic diversity selector

For one view, let score-ordered refined candidates be `c1...c16`.

A diversity arm has `(core_k,total_k)`:

1. retain the first `core_k` score-ranked candidates unchanged;
2. while fewer than `total_k` candidates are retained, choose from the remaining top16 candidate with maximum

```text
min_{r in retained} ||candidate - r||_2
```

in native 256px image coordinates;
3. break exact distance ties by the original descriptor score rank (better rank first);
4. stop when `total_k` is reached or no candidate remains.

The selector is deterministic and uses no truth.

## Arms

### S8 — score-only K8 comparator

Historical fixed score top8 (`core=8,total=8`). Already known to fail, re-evaluated only for parity.

### D8-C4 — diversity K8

```text
core_k=4
total_k=8
```

Keep the strongest four descriptor candidates and use four slots to preserve spatially distinct modes from ranks 5–16.

### D8-C6 — diversity K8

```text
core_k=6
total_k=8
```

Keep six score candidates and two diversity slots from ranks 7–16.

### D10-C6 — diversity K10

```text
core_k=6
total_k=10
```

Keep six score candidates and four diversity slots from ranks 7–16.

### D12-C8 — diagnostic ceiling, not freeze-eligible

```text
core_k=8
total_k=12
```

Keep score top8 and four diversity slots from ranks 9–16. This arm tests whether four explicit alternate-mode slots approach F16 coverage. It is not freeze-eligible because its fixed K exceeds the V1 compact target.

### F16 — safety reference

Score top16, unchanged.

## Coverage gates for a freeze-eligible diversity arm

All must hold:

```text
pooled primary-2x containment >= 0.975
worst-family primary-2x containment >= 0.94
families with primary-2x >= 0.90 = 8/8
best-worst gap <= 7 percentage points
```

Report strict-1x containment as secondary.

## Efficiency gates

For freeze-eligible arms:

```text
mean retained K per usable view <= 10
median retained K per usable view <= 10
fraction usable views at K>10 = 0
mean H hypothesis count / F16 mean H count <= 0.45
```

These gates deliberately measure the actual combinatorial H reduction as well as candidate count. A fixed K10 arm is eligible because pairwise H cost is expected to be substantially below F16, while D12 remains diagnostic only.

## Decision rule

1. If either D8 arm passes all coverage + efficiency gates, prefer the lower-core D8-C4 only if its pooled coverage is no worse than D8-C6 by >0.5pp and worst-family no worse by >1pp; otherwise prefer the stronger D8 arm.
2. Else if D10-C6 passes all coverage + efficiency gates, freeze D10-C6 as the bounded candidate-retention contract.
3. If only D12/F16 passes coverage, diversity-preserving compact retention is **not solved**; keep F16 safety reference and do not start factor-head training.
4. No result here changes Stage-B qualification or authorizes large/end-to-end training.

## Scientific interpretation boundary

A pass would mean the top16 hard-tail benefit comes primarily from **mode diversity**, and can be retained with a much smaller bounded set without a rare-event expansion classifier.

A fail would mean simply adding diverse tail slots is insufficient; the next uncertainty mechanism must preserve richer cross-view/geometric state or use more data for calibrated expansion.
