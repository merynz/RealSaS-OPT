# Exact DINOv2-S/14 Feature-Access Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `DINOV2_FEATURE_ACCESS_PREREG_V1.md`  
**Workflow:** `iris-dinov2-feature-access-v1`  
**Run:** `40219693183`  
**Status:** `EARLIER_FEATURE_ACCESS_MATERIALLY_BETTER_TINY_V1`

## Authority / validity

The exact current frozen foundation authority was verified before inference:

- source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- constructor: `dinov2_vits14`
- weight SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- feature contract hash: `efcf9bf4ab025a64cbacf6983ebccc2d58570a49af6f90dedbb726b25acde21e`
- foundation frozen: true
- L11 vs `x_norm_patchtokens` max absolute error: `0.0`
- label balance: PASS
- all features finite: PASS

No product, ModelRelease, Truth139 or FIT8 authority changed.

## Results

Held-out local four-bit recovery accuracy from one 384-D patch token:

| native square width | L2 | L5 | L8 | final contract |
|---:|---:|---:|---:|---:|
| 2 px | 1.0000 | 0.9957 | 0.9762 | 0.8878 |
| 4 px | 1.0000 | 1.0000 | 0.9736 | 0.8605 |
| 8 px | 1.0000 | 0.9983 | 0.9745 | 0.8954 |
| 12 px | 1.0000 | 0.9923 | 0.9439 | 0.8325 |

Mean paired gain over the final contract:

- L2: `+0.1309524`
- L5: `+0.1275510`
- L8: `+0.0980017`

All three earlier layers satisfy the preregistered `MATERIALLY_BETTER` rule. L2 is strongest on this controlled local-access task and reaches 1.0 on all four width strata.

## Interpretation

This court rejects the premature interpretation that the current local-detail risk requires a new backbone.

The stronger current diagnosis is:

> the exact DINOv2-S/14 backbone contains substantially more linearly accessible local/subpatch information at earlier transformer depth than the current final-only `x_norm_patchtokens` contract exposes to IRIS V3.

The post-foundation IRIS subcell-routing court separately established that, when distinct subcell states are present in source tokens, the current-like scene-query -> 2x carrier path can route them perfectly on its controlled task. Taken together, the first treatment target should therefore be **feature access**, not immediate replacement of DINOv2 or the scene-query/upsample mechanism.

## What this does not establish

- L2 is not automatically the correct sole production feature.
- Earlier layers may trade local detail for high-level/global semantics useful to geometry, correspondence or morphology.
- A synthetic RGB probe does not prove real FIT8 superiority.
- DINOv3 remains a legitimate later candidate if a same-backbone multilevel contract cannot retain both local and final information sufficiently.

## Authorized next court

Preregister a same-backbone, same-patch-grid feature-contract treatment that preserves final DINO semantics while injecting earlier local detail **without increasing the patch-token count**.

Preferred candidate form:

`scene_token = W([L2_patch ; FINAL_patch]) + camera + xy`

with the fused image contribution projected directly to the existing `scene_dim=512`, rather than doubling scene-memory tokens.

The next mechanism court must verify both:

1. local-detail recoverability at least comparable to L2; and
2. retention/recoverability of the original final-contract representation.

Only if this treatment fails should the audit escalate to resolution/backbone alternatives such as DINOv3.
