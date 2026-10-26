# DINOv2 Alpha Residual Result V1

**Run:** `40315204922`  
**Head SHA:** `980b4bf502190cfebe59c38b04f150e496115507`  
**Artifact digest:** `sha256:3b2681f39ab5e0ae3919d9db59b277f84e74495a01c56be1325a62c0cb1a0ace`  
**Status:** `PATCH_ALIGNED_RAW_ALPHA64_SUPPORTED_TINY_V1`  
**Selected arm:** `DUAL_RAW_ALPHA64`

## Frozen foundation authority

- DINOv2 source head: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- DINOv2 S/14 weight SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`

## Court validity

- exact frozen foundation identity: PASS;
- all adapter arms remain within the current adapter parameter budget;
- alpha target positive fraction: `0.501284`;
- mean-alpha construction variance: exactly `0.0`, making the mean-only arm a valid negative control;
- no-alpha and mean-alpha arms remained at chance for alpha recovery;
- local-detail and final-feature preservation gates passed.

## Summary

| arm | adapter params | mean alpha accuracy | min alpha accuracy | mean final cosine | mean local |
|---|---:|---:|---:|---:|---:|
| DUAL_NO_ALPHA | 197120 | 0.49956 | 0.49245 | 0.99740 | 1.0 |
| DUAL_MEAN_ALPHA64 | 172608 | 0.49914 | 0.49298 | 0.99724 | 1.0 |
| **DUAL_RAW_ALPHA64** | **185088** | **1.0** | **1.0** | **0.99738** | **1.0** |
| DUAL_RAW_ALPHA96 | 179072 | 1.0 | 1.0 | 0.99728 | 1.0 |

Both raw spatial-alpha treatments passed. The preregistered selection order chooses the compact 64-D raw-alpha residual contract rather than adding more alpha output width.

## Decision

Carry into V3.1 observation ingestion:

`frozen DINOv2-S/14 aligned L2 + FINAL evidence + patch-aligned RAW_ALPHA64 residual`

The alpha/support signal must preserve its patch-local spatial pattern. A scalar/mean-alpha summary is not sufficient and discarding alpha is ineligible.

This result does not authorize DINOv3 adoption, product/model promotion, Truth Builder training or FIT8 training.
