# IMPORTANT External Precedent — PatchMatch-RL vs IRIS

**Date:** 2026-08-25  
**Status:** `IMPORTANT_RESEARCH_PRECEDENT__NOT_CURRENT_ARCHITECTURE_AUTHORITY`

## Why this note exists

PatchMatch-RL (Lee, DeGol, Zou, Hoiem; ICCV 2021), *Deep MVS with Pixelwise Depth, Normal, and Visibility*, is the closest working/open-code precedent found so far for the current IRIS problem formulation.

The important similarity is not generic "3D vision". It is the problem decomposition:

```text
calibrated multi-view raster observations
        +
known camera geometry
        ↓
per-pixel depth + normal + visibility/support
        ↓
reprojection / cross-view consistency
        ↓
common geometric surface / point cloud
```

Current IRIS is likewise no longer defined as `image -> hidden mechanical latent`. Its active geometric core is:

```text
8 ordered RGBA views
+ known yaw orbit
+ observable native scale
        ↓
learn camera-forward depth
        ↓
analytic canonical P
+ N / support-risk / correspondence evidence as justified
        ↓
rigging-sufficient observable 2/2.5D substrate
```

This precedent materially lowers the novelty/risk of the *geometry extraction* question: known-camera multi-view images can support a learned oriented-surface reconstruction pipeline. It does **not** prove RealSaS product-domain success, especially under stylized, texture-poor, symmetric or cross-view-inconsistent art.

## Code-level correspondence inspected

Official code inspected: `leejaeyong7/patchmatch-rl`.

### PatchMatch-RL

- `patchmatch_rl/feature_extractor.py` — FPN multi-scale image features.
- `patchmatch_rl/feature_scorer.py` — plane-conditioned cross-view warping and group correlation.
- `patchmatch_rl/view_scorer.py` — pixel-wise source-view weighting from correlation + geometric priors.
- `patchmatch_rl/patchmatch_rl.py` — iterative plane state `(depth, normal)`, propagation, perturbation, view selection and candidate scoring.
- `patchmatch_rl/recurrent_regularizer.py` — recurrent/GRU belief regularization across iterative updates.
- `bin/generate.py` — inference writes depth, normal, confidence and camera products.
- `bin/fuse_output.py` — backprojection/reprojection, point-distance + projection-distance + normal-angle consistency, visibility/support accumulation and final oriented point-cloud fusion.

### Current IRIS

- `experiments/iris_single_pose_v2/model_pv5_r256.py` — true full-resolution image-conditioned scalar-depth field and analytic canonical P reconstruction.
- `experiments/iris_single_pose_v2/pv5_depth_objective.py` — camera-forward depth supervision with full canonical P evaluation.
- `experiments/iris_single_pose_v2/matcher.py` — current explicit persistence/correspondence path: Z_coarse/P high-recall admission, Z_fine local-only refinement, set-valued top-k.

## Most important architectural lesson

The strongest transferable idea is **geometry-in-the-loop hypothesis verification**, not copying PatchMatch-RL wholesale.

PatchMatch-RL does not require a network to infer metric depth solely from an opaque image latent. A candidate surface hypothesis determines, through known cameras, exactly where evidence should appear in other views. The method then measures actual feature support, weights useful views, propagates/refines hypotheses, and repeats.

If future family-disjoint IRIS evidence shows a persistent P/N hard tail after the currently frozen direct-R256 ladder, the first serious intervention candidate should therefore be a bounded experiment of the form:

```text
IRIS direct depth proposal
        ↓
local depth / oriented-surface hypotheses
        ↓
analytic projection into other known-yaw views
        ↓
cross-view feature agreement + support weighting
        ↓
local refinement / rejection / uncertainty
```

This is preferable to reflexively increasing transformer width/depth because it injects the known physical geometry directly into evidence evaluation.

## Critical difference / product risk

PatchMatch-RL operates in calibrated photographic MVS. RealSaS ultimately operates on stylized 2D assets with possible:

- texture-poor flat regions;
- symmetry / repeated parts;
- line-art dominance;
- silhouette disagreement;
- missing or redrawn detail across views;
- mild violations of exact orthographic consistency.

Therefore PatchMatch-RL is a **feasibility precedent and intervention library**, not a proof that the current IRIS contract is sufficient for product art.

## Change-control status

**IMPORTANT:** this note does not authorize an architecture change.

The active R256 promotion ladder remains authoritative. PatchMatch-style geometry-in-the-loop refinement may be opened only if a preregistered controlled result localizes a residual failure to extractability/coherence that the current direct representation/learner cannot close.

References:
- Paper: *PatchMatch-RL: Deep MVS with Pixelwise Depth, Normal, and Visibility*, ICCV 2021, arXiv:2108.08943.
- Official code: `https://github.com/leejaeyong7/patchmatch-rl`
