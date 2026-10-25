# Matched DINOv2 Dual-Level View-Token Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `DINOV2_DUAL_LEVEL_VIEW_TOKEN_PREREG_V1.md`  
**Workflow:** `iris-dinov2-dual-level-view-token-v1`  
**Run:** `40222240883`  
**Status:** `DUAL_LEVEL_VIEW_TOKEN_SUPPORTED_TINY_V1`

## Question

Can the existing frozen DINOv2-S/14 foundation expose the early local detail found in the feature-access court while retaining final-layer semantics, without increasing patch-token count or image-adapter parameter count?

## Matched arms

All three image adapters have exactly **197,120 trainable parameters** and emit one 512-D token per existing 37x37 patch location:

- `CURRENT_FINAL_512`: final 384-D patch token -> 512;
- `EARLY_L2_512`: L2 384-D patch token -> 512;
- `DUAL_L2_FINAL_MATCHED_512`: L2 -> 128 and final -> 384, concatenated to 512.

Frozen DINO authority is unchanged:

- source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
- weight SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

## Result

| Arm | Mean local recovery | Minimum local recovery | Mean final-feature cosine | Minimum final-feature cosine |
|---|---:|---:|---:|---:|
| CURRENT_FINAL_512 | 0.864626 | 0.816327 | 0.998078 | 0.997072 |
| EARLY_L2_512 | 1.000000 | 1.000000 | 0.988064 | 0.984258 |
| DUAL_L2_FINAL_MATCHED_512 | **1.000000** | **1.000000** | **0.997598** | **0.996222** |

Paired dual-minus-current local-recovery gain by seed:

`+0.124787, +0.125425, +0.131378, +0.133716, +0.161565`

All preregistered gates passed:

- local-detail retention: PASS;
- final-feature retention: PASS;
- paired improvement: PASS;
- validity/authority: PASS.

## Interpretation

The current local-detail problem does **not** justify a backbone replacement. The frozen DINOv2-S/14 already contains the needed local signal and a same-token-count, same-parameter-budget L2+final representation can retain both the early local signal and almost all recoverable final-layer representation in this controlled court.

This supports a V3 treatment based on **dual-level same-patch fusion**, not immediate DINOv3 adoption.

The exact production fusion form is not frozen by this court. A direct/frozen final residual plus projected early-detail variant remains eligible if it preserves final semantics more strongly by construction. Real-data/reference-scale validation is still required before architecture freeze.

## Authority

This result does not authorize Truth139 training, FIT8 training, Product activation, ModelRelease promotion or a DINOv3 decision.
