# RealSaS N1D — Diversity-Preserving Retention V1 K10 Follow-Up Prereg

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_CAUSAL_FOLLOWUP`  
**Parent prereg:** `ea7b4743838f6a03c4588d1a88b69cdaf321ca71`  
**Parent result:** `459d32423926b6384f626b900641eb588455325a`

## Known result before this prereg

- D8-C4 (top4 score + 4 diversity): pooled `.97764`, worst `11032=.91525`; efficiency PASS, tail coverage FAIL.
- D12-C8 (top8 score + 4 diversity): pooled `.97967`, worst `.94915`; coverage PASS, compact efficiency FAIL.
- D10-C6 (top6 score + 4 diversity): worst `.89831`; therefore simply keeping four diversity slots is not sufficient when the score core is only six.

The unresolved causal question is whether K12 succeeds because it preserves **the full score-top8 core**, or because it has **four total diversity slots**, or both.

## Frozen selector

Use the exact deterministic max-min spatial diversity selector from Diversity-Preserving Retention V1. Same frozen refined top16 candidate pool, no new model evidence and no truth-conditioned retention.

## Arms

### D10-C8

```text
score core = top8
diversity slots = 2 from ranks 9-16
total K = 10
```

Tests whether preserving the full score-top8 core plus only two alternate-mode slots is enough.

### D10-C4

```text
score core = top4
diversity slots = 6 from ranks 5-16
total K = 10
```

Tests whether greater diversity with a small score core is enough.

Comparators remain D8-C4, D12-C8 and F16 from the already frozen parent result; they are not retuned.

## Freeze gates

Same as the parent compact diversity prereg:

```text
pooled primary-2x >= 0.975
worst-family primary-2x >= 0.94
8/8 families >= 0.90
best-worst gap <= 7 pp
mean K <= 10
median K <= 10
no usable view K>10
mean H / F16 mean H <= 0.45
```

## Decision

- If D10-C8 passes all gates, freeze `D10-C8` as the bounded retention contract unless D10-C4 also passes and has higher worst-family coverage by >1pp.
- If D10-C4 alone passes, freeze D10-C4.
- If neither passes, compact diversity retention is not closed; F16 remains safety reference and factor-head execution remains paused.

No result here changes Stage-B qualification or authorizes large training.
