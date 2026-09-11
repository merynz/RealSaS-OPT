# RealSaS — Arachne Information-Preservation Audit V1 — Phase 8 A0 Interface Closure

**Date:** 2026-09-11  
**Status:** `PHASE8_CLOSED__PRE_A1_INFORMATION_AUDIT_COMPLETE`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Product PASS:** not claimed.  
**Unseen/generalization PASS:** not claimed.

## 1. Frozen A0 interface

The manually selected parsimonious K=4 A0 interface was regenerated and closed successfully.

- checkpoint SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`
- A1 supervision bank SHA-256: `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`
- closure result SHA-256: `14043d1f2b638d576e86e25568ffa80e931f84807fafe0135f3f328e027abd6c`
- field tokens: `4`
- latent width: `512`
- condition tokens: `384`

Stable terminal GSA observations:

- step 3584: p95 `0.04870356093865557`, deformation `0.01905974932014942`
- step 3840: p95 `0.04750191859206554`, deformation `0.018206628039479256`
- step 4096: p95 `0.04491063521144626`, deformation `0.018186409026384354`

Therefore the frozen Mage/GSA A0 representation gate is closed.

## 2. Disjoint-surface result

Final same-character disjoint-surface diagnostic:

- p95 `0.10624249461034009`
- deformation `0.14126436412334442`

The completed 4/8/16/32 experiment did not establish a meaningful practical internal-token capacity gain beyond K=4. The nominal numerical winner was not treated as scientific evidence of a larger-K requirement; K=4 was manually selected for parsimony and interface continuity.

The remaining disjoint-surface gap is not an A1-transition gate and is not unseen-character generalization.

## 3. Token-set semantic audit

The initial post-fit guard observed BF16 permutation delta `0.0016632080078125` and failed a `5e-4` threshold. A no-training/no-optimizer rescue audit separated semantic behavior from finite-precision runtime drift:

- FP32 full-set permutation max absolute logit delta: `5.7220458984375e-06`
- FP32 semantic tolerance: `1e-05`
- BF16 full-set permutation max absolute logit delta: `0.0016632080078125`

Verdict:

`K4_FULL_SET_IS_SEMANTICALLY_PERMUTATION_INVARIANT_IN_FP32`.

No optimizer step or parameter mutation occurred after the original guard failure.

A1 consequence:

- ordered latent-slot regression is not semantic authority;
- behavior-first decoded-field and coupled normalized-row supervision is preferred;
- any later latent-set loss must be permutation-aware.

## 4. Holdout deformation signal finding retained

Phase 4 remains valid: the A0 training objective supplied no direct gradient from the post-normalization synthetic-LBS `hold_deform` metric. The diagnostic probe is a fixed synthetic per-joint translation bank, not an articulated physical proof.

Therefore flat holdout deformation is not evidence that K4 lacks representational capacity and is not a reason to increase internal token count blindly.

## 5. Pre-A1 audit closure

The primary actionable information-preservation finding remains the historical A1 consumer boundary:

- exact GSA graph must not be collapsed to degree/mean-score summaries;
- exact per-view raster/support evidence must not be collapsed to counts;
- exact sparse joint→surface support-anchor identity must not be collapsed to a constant support-count scalar;
- the useful deterministic 10D point↔joint/parent-segment geometry may be retained;
- teacher/source-only evidence remains forbidden from predictor inputs;
- current IRIS legacy `log_uncertainty` is not calibrated uncertainty and is excluded;
- Compiler-qualified skeleton topology remains sole canonical authority.

The audit therefore closes with the following downstream requirement:

`V7_NATIVE_A1_MUST_CONSUME_A_RICH_TYPED_S_PLUS_QUALIFIED_G_BOUNDARY_AND_KEEP_TEACHER_W_IN_THE_OBJECTIVE_LANE_ONLY`.

No remaining Phase-8 finding blocks V7-native A1 Mage FIT1 execution after an exact source/config/objective preregistration is sealed.
