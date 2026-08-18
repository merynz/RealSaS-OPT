# N1D Descriptor Global-Foreground Top-4 Multiview 3D — Open-DEV Canonical Report V1

**Date:** 2026-08-18  
**Status:** OPEN-DEV TREATMENT PASS  
**sealed21:** CLOSED  
**external10:** CLOSED

## Question

Can the strong frozen N1D persistent descriptor be consumed without making raw DIS the search authority, so that per-view B correspondences are found globally on the observable foreground and calibrated multi-view geometry can deterministically recover a world-space response?

## Frozen treatment

For each Pose-A Problem-A carrier and visible view:

1. sample the frozen N1D Pose-A descriptor at the projected carrier;
2. scan the full Pose-B observable foreground on a fixed 4-pixel coarse grid;
3. keep top-8 coarse descriptor matches;
4. refine each within a fixed ±4-pixel window at 2-pixel spacing;
5. keep final top-4 candidates per view;
6. generate rank-3 3D displacement hypotheses from view pairs;
7. score each 3D hypothesis by mean minimum reprojection distance to the per-view top-4 candidate sets;
8. refit the winning hypothesis against each view's nearest candidate;
9. fall back to frozen N1D current response only on abstain;
10. preserve the frozen N1D activity/silence gate.

No teacher identity, joint, weight, parent, surface correspondence or observation sidecar enters prediction.

The decisive change relative to the prior descriptor-top4 route is that **DIS no longer centers or bounds the search window**.

## Localization that authorized the treatment

Previous candidate-coverage diagnostic:

- true endpoint inside fixed DIS-centered ±8 px search square: **0.853416**;
- descriptor top-4 hit ≤2 px given endpoint is in-window: **0.847889**;
- total view-level top-4 hit ≤2 px: **0.727950**;
- raw DIS center error median: **0.0363 px**;
- raw DIS center error p90: **14.3438 px**.

A minority of large DIS failures were erasing otherwise observable descriptor evidence before multiview geometry could act.

## 16-family development qualification

Frozen development prereg SHA-256: `640d95233ec47a7640413b62832917897322024a781b517cefb88a46c35d7753`  
Frozen source SHA-256: `2110414ed63b06145a34642440dd80b09d47d23cce8ee1cf917aeae43a061825`

| Metric | Frozen N1D current | Global-foreground treatment |
|---|---:|---:|
| flow / zero | 0.740877 | **0.441086** |
| weighted direction cosine | 0.583855 | **0.899404** |
| family direction non-regress | — | **14 / 16** |
| carrier non-abstain | — | **0.999023** |
| false activation | 41 | **39** |

**Development gates: 5 / 5 PASS.**

## Untouched open-DEV qualification

Frozen families, episode e00: `10763, 11214, 12907, 14714`.

Untouched prereg SHA-256: `2cd7c07e3f863b3abbd8b996f7e68e807bb6a8d327c4301e575a7d17c3912606`

Before prediction all four families passed 8/8 Pose-A raster SHA checks, 8/8 Pose-B raster SHA checks and sidecar SHA checks. All four predictions were completed before evaluator sidecars were opened.

| Metric | Frozen N1D current | Frozen global-foreground treatment |
|---|---:|---:|
| flow / zero | 0.743900 | **0.671552** |
| weighted direction cosine | 0.547553 | **0.740279** |
| family direction non-regress | — | **3 / 4** |
| carrier non-abstain | — | **1.000000** |
| false activation | 26 | **23** |

**Untouched open-DEV gates: 5 / 5 PASS.**

Per-family direction:
- 10763: 0.923707 → 0.870466 (regress)
- 11214: 0.101154 → 0.706936 (recovery)
- 12907: 0.312688 → 0.636050 (recovery)
- 14714: 0.671031 → 0.729239 (recovery)

The preregistered 3/4 family robustness gate passes exactly; no post-hoc relaxation was used.

## Scientific conclusion

The post-N1D blocker was not calibrated-camera 3D inversion and was not absence of persistent raster evidence. The frozen N1D descriptor already contained strong persistence information, but the mechanics path consumed it through a **DIS-centered local search window**. Large-tail DIS errors erased correct correspondence candidates before multiview geometry could act.

Supported route:

```text
Pose A/B rasters
  -> frozen N1D persistent descriptor
  -> global observable B-foreground candidate search
  -> top-k set per view
  -> calibrated multiview 3D hypothesis solve
  -> frozen N1D activity/silence gate
  -> observable world-response evidence
```

This matches the IRIS-SEES doctrine: the learner measures observation-native evidence; deterministic geometry closes the structured world quantity.

## Authority

This report authorizes the global-foreground descriptor route as the **post-N1D open-DEV correspondence / world-response treatment**.

It does **not** authorize sealed21, external10, product/model-2 handoff, GFDR semantic rewriting, hidden rig/teacher targets, or new training solely because this route passed.

## Next gate

Re-run the canonical downstream mechanical diagnostics on open development with the repaired world-response evidence, preserving the original compiler/mechanics definitions. In particular determine whether the previously failing dynamic-flow and finite-articulation/G diagnostics close. Only after that causal closure should sealed/external access be considered.
