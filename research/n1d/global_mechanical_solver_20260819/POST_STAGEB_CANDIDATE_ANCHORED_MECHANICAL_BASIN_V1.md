# RealSaS N1D — Post-Stage-B Candidate-Anchored Mechanical Basin V1

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DEVELOPMENT_EVIDENCE__NOT_QUALIFICATION`  
**Frozen Stage-B authority remains:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`  
**sealed21 / external10:** `CLOSED`

## Question

After frozen Hybrid V11 failed Stage-B primarily on F-activity (`0.433160 < 0.50`), the post-truth ceiling showed that scalar motion ordering could close the activity gap without replacing the frozen direction field. The missing causal test was whether raster-derived activity could be composed **inside the actual V11 feasible mechanical basin**, rather than by free XYZ/radial scaling.

This experiment reconstructs the seed-route candidate/mechanical basin from the frozen V11 execution lineage and uses the frozen N1D `delta_point_map_srcA` only as scalar activity evidence.

## Provenance guard

The replay lineage was hash-checked against the frozen V11 manifest before the development intervention: checkpoint, V8 source, V5 source, frozen runner, GFDR source, and canonical N1D source ZIP matched the frozen authority. Stage-B truth is already open and is used only for evaluation in this development line.

The historical Stage-B qualification is immutable. Nothing below rewrites it as PASS.

## Treatment

For each seed-basin carrier, keep the frozen V11 mechanical displacement basis `q_i` fixed and replace only the scalar basin coordinate:

```text
current_amp_i = ||delta_point_map_srcA(i)||
scale = q95(current_amp) + eps
w_i = sqrt(clip(current_amp_i / scale, 0, 1))
P_B(i) = P_A(i) + w_i * q_i
```

No truth enters prediction. No free XYZ displacement is learned or injected. Geometry remains anchored to the multiview/candidate-derived mechanical basin.

## Primary seed-route witnesses

Only three primary Stage-B episodes route through `SEED_BASIN_WSEED`: `10763/e04`, `11214/e04`, and `14714/e04`. The remaining primary episodes remain unchanged `V8_BASE` predictions.

### 10763/e04

- F activity: `0.1261 -> 0.6872`
- F kernel: `0.4355 -> 0.7077`
- R differential: `0.0762 -> 0.0617`
- G line: `0.10165 -> 0.08024`
- treated G direction remains strong: approximately `0.838`
- median distance from `w_current` to the nearest candidate-projection weight: approximately `0.00706`

### 11214/e04

- F activity: `0.4332 -> 0.68485`
- F kernel: `0.851 -> 0.722` (still well above gate)
- G line: `0.0592 -> 0.0375`
- median nearest candidate-projection weight distance: approximately `0.00701`
- raw current-amplitude p95: approximately `1.384e-3`

### 14714/e04

- F activity: `0.4000 -> 0.8333`
- median nearest candidate-projection weight distance: approximately `0.00431`

The very small candidate-projection distances are important: the scalar treatment is not inventing an unrelated displacement regime; it stays close to endpoints represented by the reconstructed feasible candidate geometry.

## Development aggregate

Combining the three treated seed-primary episodes with the unchanged primary `V8_BASE` episodes yields:

- **F activity:** `0.43316 -> 0.68718`
- **F kernel:** `0.70775`
- **D tangent action error:** `0.10163`
- **R differential body error:** `0.04274`
- **G direction:** `0.89947`
- **G line:** `0.08024`
- principal quality metrics: **6/6 PASS**
- family G robustness: **3/4**
- episode-index G robustness: **3/3** (frozen Stage-B was `2/3`)

This is **not** a new qualification PASS. The frozen qualification remains FAIL, and its separate primary G-family coverage defect remains: all three `12907` Stage-B episodes are `near_zero`, so the preregistered primary-only four-family coverage requirement was structurally unattainable.

## Silence / activity separation

The candidate-basin result also exposes why one normalized amplitude scalar is insufficient.

For `11214/e04` (primary finite motion), raw current-amplitude p95 is approximately `1.384e-3`. For near-zero `11214/e06`, it is approximately `3.068e-4`, a roughly **4.51x** absolute separation. But q95-normalization removes this absolute scale distinction. On the near-zero witness, normalized basin composition increases predicted moved fraction from approximately `0.1719` to `0.1875`.

Therefore the representation should not collapse activity/silence and conditional magnitude into one normalized scalar.

## Architecture decision supported by this experiment

```text
candidate/mechanical basin -> physical XYZ authority
p_active                  -> absolute motion / silence gate
log_amp                   -> conditional magnitude / ranking evidence
dir                       -> direction evidence
```

`p_active`, `log_amp`, and `dir` are evidence used to score/select within feasible geometry. They are not independent free-XYZ authority.

**Current evidence argues against adding a free XYZ motion head.**

## What is now demonstrated

The post-truth oracle/rank ceiling is no longer the only evidence for the missing activity factor. On real Stage-B rasters, using the frozen N1D `delta_point_map_srcA` amplitude inside the frozen candidate-anchored mechanical basin closes the primary F-activity gap while preserving the main geometry/mechanics quality gates.

This supports moving to a learned held-out factorized head rather than another geometry rewrite.

## Next development gate

Train/evaluate a small **family/episode-held-out** factorized probe:

```text
p_active + log_amp (+ existing direction evidence)
```

with explicit ranking loss for conditional amplitude and silence/activity supervision for `p_active` (e.g. BCE under the frozen activity semantics). Final endpoint selection must remain candidate/basin-constrained.

Any later qualification requires a **new untouched preregistered panel**. Stage-B cannot be recycled as blind qualification.

## Recovery note

A prior tool session reported ephemeral artifact hashes:

```text
report  cec872d613c4b8dcbd2f62c6c03e7d66095f27917d60d3c2db2f62b9fc126ad0
result  3925c55209b77ab5b7ea75b982bf18d0ba497f589724d2cc6aa20f92b3bcfda2
source  885de8ac38d3b834e0d446537d620a19f92184b35cfec9e658c15536fea7e215
```

Those exact bytes were not persisted across the tool-session boundary. This canonical recovery file preserves the experimentally reported values and conclusions and **does not claim byte identity** with the lost ephemeral files.
