# RealSaS-OPT — Current State

**Date:** 2026-08-21  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `G0/G1_STARTER_COMMITTED__NEXT_SPLIT_AND_HASH_FREEZE`

## Read this first

This file is the single continuation authority for a new session. Older N1D/GFDR reports remain historical evidence but do **not** define the active product roadmap.

## Canonical product architecture

```text
ONE neutral pose × 8 ordered views
 -> IRIS: P/N/V/U + required surface correspondence/persistence semantics
 -> Geppetto: clean skeleton/hierarchy proposal
 -> Arachne: skinning/weight proposal
 -> Compiler: canonicalize/verify/repair/export
 -> editable puppet + runtime animation
```

Shipping target is `1 pose × 8 views`. Pose B may remain research/training evidence but is not an inference dependency unless a later controlled end-to-end product ablation proves it indispensable.

### IRIS boundary

Required semantic output/capability:
- `P`: common/object-frame position or equivalent surface geometry;
- `N`: local surface orientation/normal evidence;
- `V`: per-view visibility/observational support;
- `U`: calibrated geometric uncertainty;
- geometric surface correspondence/persistence strong enough to form coherent multiview surface hypotheses;
- provenance/source-view support.

`Z` is optional as an explicit learned correspondence embedding. Correspondence itself is **not optional**.

IRIS does **not** own authored mechanical owner identity, parent/topology prediction, skeleton, weights or mandatory GFDR.

## D2 closure — preserved diagnostic

D2 fine spatial is sealed as `D2_NOT_SUFFICIENT__PROCEED_TO_D3_MATCHER` under its original prereg lineage, but that historical decision does not auto-authorize D3 in the new product roadmap.

Matched C2 → D2 highlights:
- PCK@2: `0.550313 -> 0.692652` (`+0.142338`);
- oracle hit: `0.308073 -> 0.427844` (`+0.119771`);
- oracle top-4: `0.644630 -> 0.765229` (`+0.120599`);
- PCK@4: `0.816959 -> 0.826706` (`+0.009747`);
- mean top-1 error: `3.265649 -> 4.556463 px` (worse);
- family mean-distance nonworse: `3/29`;
- primary gates `1/4`, safety `6/6`.

Interpretation: high-resolution local spatial evidence is real and useful, but using the fine descriptor as global ranking authority creates a hard-tail failure. D1/D2/D3 mechanisms are therefore preserved as `RESERVE_CALLABLE`, especially for later G5 geometry-aware local refinement.

## G0 — contract/evaluator freeze

Starter exists at `experiments/g0_g1_single_pose_geometry/`.

G0 policy:
- input exactly `A×8`;
- freeze SurfaceEvidenceSet semantics;
- no mean-only promotion: median + p90/p95 hard-tail metrics required;
- product thresholds are not invented at G0; G1 establishes baseline and G7 owns product qualification;
- sealed/external panels stay closed during architecture selection.

## G1 — current executable line

Scientific question:

> With Pose B and all mechanical/cross-pose objectives removed, can the retained IRIS multiview core produce a usable common-frame surface from `A×8` alone?

Causal intervention is intentionally minimal:

```text
A×8
 -> retained SharedImageEncoder
 -> retained 8-view GlobalMultiViewFusion
 -> retained DenseFusionDecoder
 -> retained geometry head
 -> P / N / V / U
```

No MV-TAP module, DPM/GGPT route, D3 matcher, bigger backbone or resolution increase is allowed in G1. Those are later causal interventions only if G1 evidence calls for them.

### Preservation/migration result

Real canonical N1D `BEST.pt` migration into the G1 active model:
- destination tensors: `141`;
- loaded tensors: `141/141 = 100%`;
- missing: `0`;
- shape mismatches: `0`;
- source-only tensors: `29`, all `differential.*`, archived/reserve-callable rather than deleted.

D1 coarse descriptor remains callable as optional geometric `Z` evidence with zero G1 loss authority. D2 fine descriptor source and D3 matcher concept are preserved for later use if common-frame geometry retains a localization tail.

## NEXT EXECUTABLE STEP

**Do not start optimizer steps yet.**

1. Build the exact family-disjoint G1 fit/dev split from the existing corpus.
2. Freeze family IDs, episode/sample selection, source hashes and truth-access policy.
3. Freeze G1 evaluator implementation against the G0 definitions.
4. Produce a final `G1_PREREG_V1.json` and preflight manifest with hashes.
5. Only then build/run the A100 G1 Run-All.

After G1 result:
- coherent but camera-limited -> G2 camera/ray-aware cross-view fusion;
- weak common-frame geometry -> G3 direct common-frame pointmap/surface route;
- learned geometry needing safety -> G4 explicit geometric grounding/refinement;
- remaining localization hard-tail after world seed -> G5, where D2/D3 reserve components may return.

## Non-negotiable process discipline

- GitHub is the canonical continuation surface.
- `CURRENT_STATE.md` must be updated at every closed gate or material architecture decision.
- No silent component deletion; update `PRESERVATION_LEDGER.json`.
- Every prereg is frozen before optimizer steps/truth opening.
- Keep sealed/external sets closed until explicitly authorized.
- Heavy corpus/cache/checkpoint data stays in Drive; GitHub stores code, hashes, manifests and compact results.
