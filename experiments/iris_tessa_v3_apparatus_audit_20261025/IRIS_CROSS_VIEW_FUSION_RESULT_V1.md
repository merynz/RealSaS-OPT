# IRIS Cross-View Fusion Placement Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `SYNTHETIC_IRIS_CROSS_VIEW_FUSION_PREREG_V1.md`  
**Workflow:** `iris-v3-cross-view-fusion-v1`  
**Run with emitted result:** `40220772357`  
**Status:** `CURRENT_QUERY_MEDIATED_CORRESPONDENCE_ESTABLISHED_TINY_V1`

## Provenance note

Run `40220772357` completed the sealed court and printed the complete result before a later PR synchronize event cancelled the job during the post-result/artifact phase. The executed probe blob SHA was `1d9326b4506a3707238b11f58b2d5648fe6f8a57`, which is byte-identical to the probe on the current audit head. The result is therefore usable as scientific output; cancellation did not occur during model optimization or evaluation.

## Question

Does cross-view correspondence require source-token self-attention before scene-query compression, or can the current query-mediated fusion mechanism recover the correspondence without it?

The controlled task used four rotated orthographic views. Every canonical locus carried an independent bit in each view; token order was shuffled, so successful prediction required the scene query to recover the four corresponding view observations.

## Result

| Arm | Mean accuracy | Minimum accuracy | Parameters |
|---|---:|---:|---:|
| CURRENT_QUERY_MEDIATED | **1.000000** | **1.000000** | 204,388 |
| SOURCE_SELF_ATTENTION_FIRST | **1.000000** | **1.000000** | 187,620 |
| RAW_CORRESPONDENCE_POSITIVE_CONTROL | **1.000000** | **1.000000** | 20 |
| RANDOM_LABEL_FIREWALL | 0.500586 | 0.499344 | — |

All five seeds reached 1.0 for both learned correspondence arms. Early source self-attention therefore supplied **no measurable gain** on the preregistered correspondence mechanism task.

## Interpretation

The audit does not support adding source-token cross-view self-attention merely because it appears earlier in the architecture. Current learned scene queries can perform the required cross-view reconciliation on this controlled task.

This keeps the current placement as the preferred low-compute baseline and avoids an expensive O(source-token^2) attention block over the full multi-view token bank without evidence that it is needed.

This court does **not** establish that the current camera/XY encoding is optimal. The synthetic source supplied already-composed view/position features. A separate court is still required for the real V3 question of factorized camera+XY embeddings versus deterministic orthographic ray/backprojection features.

## Authority

No Truth139, FIT8, product or ModelRelease authority is changed by this result.
