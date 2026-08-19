# RealSaS N1D — Score-Rank Depth Retention V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_CAUSAL_TEST`  
**Parent representation:** `REPRESENTATION_CONTRACT_V2.md`

## Motivation fixed before evaluation

The K10 diversity follow-up localized the remaining `11032` gap to carrier 36. In two critical views the target-near candidate is descriptor rank 9. `D10-C4` drops those rank-9 candidates while keeping more spatially distant alternatives; `D12-C8` keeps them and recovers the carrier.

Therefore the missing causal comparator is **plain score-rank depth** between the already-tested K8 and K16 endpoints.

## Frozen evidence

Use the exact same frozen refined descriptor top16 pool, score order, carrier mapping, H constructor and primary-2x evaluator as prior Test C / Adaptive H / Diversity Retention work.

No score retuning, model update, truth-conditioned retention or new observable feature is allowed.

## Arms

### S8
Score top8, known comparator.

### S10
Score top10 per usable view.

### S12
Score top12 per usable view.

### F16
Score top16 safety reference.

All arms construct the same pairwise rank-sufficient H from the retained per-view candidates.

## Freeze gates

A compact score-depth arm is freeze-eligible only if:

```text
pooled primary-2x containment >= 0.975
worst-family primary-2x containment >= 0.94
8/8 families >= 0.90
best-worst gap <= 7 pp
mean K <= 10
median K <= 10
mean H / F16 mean H <= 0.45
```

Thus S10 is freeze-eligible; S12 is diagnostic only under the current compact budget.

## Decision rule

- If S10 passes all coverage + efficiency gates, freeze S10 as the bounded retention contract. Prefer it over more complex adaptive/diversity policies.
- If S10 fails but S12 passes coverage, retain S12 only as a diagnostic lower-cost safety reference; compact retention remains unsolved and factor-head execution stays paused.
- If S12 also fails coverage, F16 remains the safety reference.

This test does not change Stage-B qualification and cannot authorize large/end-to-end training.
