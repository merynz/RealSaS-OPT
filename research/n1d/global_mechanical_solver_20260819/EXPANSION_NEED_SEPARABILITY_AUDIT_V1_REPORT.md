# RealSaS N1D — Expansion-Need Separability Audit V1

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DEVELOPMENT_DIAGNOSTIC`  
**Parent failure:** Adaptive Hypothesis Retention V1  
**Qualification:** unchanged

## Diagnostic label

Truth is used only to define the open-development label:

```text
NEEDS_EXPANSION = F8 misses primary-2x AND F16 contains primary-2x
```

All candidate trigger features are inference-time observation evidence.

Across the broad 8-family e00 panel:

- reliable carriers: **492**
- F16-rescuable F8 failures: **13**
- prevalence: **2.64%**

Per family:

```text
9908   1
11032  4
12772  0
13203  4
14404  2
14702  0
14758  0
15290  2
```

The engineering implication is immediate: using F16 on every carrier is highly conservative, but predicting the rare expansion event is also statistically difficult on this small open panel.

## Audit 1 — score/reprojection features

Strict leave-one-family-out diagnostics tested descriptor score margins/entropy together with current F4/F8 reprojection and support quantities.

Best single feature:

- `score_span_q75`: LOFO AUROC **0.7161**

Bundles:

- descriptor-only: **0.6062**
- reprojection/support: **0.6738**
- combined: **0.6477**

The reprojection bundle is useful inside some hard families (`11032` and `13203`) but not stable enough across all held-out families to become the retention authority.

This explains Adaptive H V1: raw q75 reprojection escalation catches some hard cases but also expands many easy carriers.

## Audit 2 — spatial candidate multimodality

The frozen refined top16 candidate pool was then analyzed without truth for spatial mode structure between high-ranked candidates and ranks 9–16.

Strongest single signals:

- `all16_spread_q25`: LOFO AUROC **0.8079**
- `tail_near_head_maxpx_q25`: **0.7962**
- `tail_near_head_maxpx_min`: **0.7841**
- `tail_near_head_maxpx_median`: **0.7761**

A compact multimodality + reprojection bundle reaches LOFO AUROC **0.7525**.

Thus **candidate multimodality is materially more informative than descriptor score entropy alone**. The hard tail often corresponds to an alternate spatial mode living in lower-ranked but still plausible descriptor candidates.

## Why not freeze an expansion classifier now?

Only 13 positive carriers exist. Multi-feature logistic models vary substantially across held-out families even when pooled AUROC is respectable. This is insufficient evidence to freeze a learned rare-event `expand/not-expand` gate without creating another hard-tail calibration risk.

The safer interpretation is structural:

> If the correct mode is rare and difficult to predict as an event, preserve a small number of distinct modes by construction rather than asking a classifier to predict exactly when the rare mode matters.

## Next architecture test

Preregister a **diversity-preserving retention** counterfactual using the same frozen refined top16 descriptor candidates:

```text
high-score core
+
a small number of deterministic spatial-diversity slots
```

No new model evidence, no truth-conditioned candidate choice.

The target is to determine whether `K≈8–10` can retain top16 hard-tail coverage by avoiding redundant same-mode score candidates.

This is conceptually different from Adaptive V1:

```text
V1: predict which carrier needs a larger K
V2 direction: keep multiple plausible modes inside the small K itself
```

If diversity-preserving K passes, the bounded H contract can be frozen without a fragile rare-event expansion classifier. If it fails, F16 remains the safety reference and the next step must model uncertainty with more data or a richer geometric state.

## Reproducibility

Base audit exact result SHA-256:
`3d85262c6d2065d175ced80e70c66fd3f58e13930e9d9d34e9210e16c8729ed6`

Multimodality audit exact result SHA-256:
`78039a738f3581f3f5f68ba8c0e0f44486f921717d48722cbfba84ea846144b0`

Base audit source SHA-256:
`4f6ecae0c0e0ec09f7cc94ccd3848e7de8eab8b9426b52a4250aa18c02a73b58`

Multimodality audit source SHA-256:
`ad2aa440cdaac09315d66a1459f1ef384620c571f36c32519477471875f93462`
