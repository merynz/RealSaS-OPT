# P-V5 R256 8×2 — Residual Localization After V1 FAIL

**Date:** 2026-08-25  
**Status:** `LOCALIZATION_EVIDENCE_RECORDED__NO_ARCHITECTURE_CHANGE`

## Frozen result being localized
`P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT` remains immutable. Selected `TAIL_0512`: 8/16 cells PASS, aggregate P95 `0.0051227101590484376`, worst-cell P95 `0.008320469176396726`.

## Cell pattern
Failure pairs by asset rather than style: four assets pass both styles and four fail both styles. Within-asset `cel_clean` and `ink_cel` P95 values are nearly paired, strongly disfavoring render-style interference as the primary blocker.

The prior one-asset×two-style control asset `asset_76313e4bd82b82fcd1659c70` previously passed at approximately `0.00373/0.00383`, but under the 8×2 shared model degrades to approximately `0.00679/0.00675`. This falsifies intrinsic non-extractability for that asset and supports a shared-fit effect.

## Optimizer-zero residual microscope
No optimizer step was run for localization. The frozen `TAIL_0512` weights were evaluated on the same deterministic 4096 truth loci/view.

### Worst asset `asset_36fb02305846592b1ecdf3d4`
CPU FP32 reproduction P95: `0.008299856912344682` versus canonical CUDA autocast result `0.008320469176396726`.

View P95s: `0.00817, 0.00873, 0.00750, 0.00809, 0.00792, 0.00853, 0.00873, 0.00859`. The failure is broad across all eight views rather than one failed view.

Aggregate boundary bins:
- <=2 px: P95 `0.0079072`;
- 2–5 px: `0.0081648`;
- 5–15 px: `0.0086959`;
- >15 px interior: `0.0072173`.

Only `16.48%` of the top-5% residual samples are within 2 px of the silhouette. The residual is not strongly enriched at the silhouette/occlusion boundary.

### Prior solo-PASS control `asset_76313e4bd82b82fcd1659c70`
8×2 checkpoint CPU FP32 P95: `0.006796193355694392`.  
Solo two-style checkpoint CPU FP32 P95 on the same asset/loci: `0.0037895937566645443`.

At the 8×2 checkpoint, top-5% residual boundary<=2px fraction is `15.38%`; again no boundary concentration. The shared model degradation is broad rather than a localized silhouette hard tail.

## Current localization judgment
- style interference: strongly disfavored;
- analytic screen-plane P reconstruction: disfavored as blocker;
- one bad view: disfavored;
- silhouette/occlusion-localized hard tail: disfavored as **primary** blocker;
- intrinsic extractability of `76313...`: falsified by prior solo PASS;
- shared multi-asset optimization/capacity burden: strongly supported;
- optimizer budget versus true shared capacity/interference: not yet separated.

Therefore PatchMatch-RL remains preserved as an important future *geometric* intervention, but current evidence does not justify inserting it before shared-optimization localization.
