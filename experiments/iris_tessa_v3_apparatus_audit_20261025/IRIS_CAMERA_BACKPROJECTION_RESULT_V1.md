# IRIS Camera / Backprojection Court — Result V1

**Date:** 2026-10-26  
**Preregistration:** `IRIS_CAMERA_BACKPROJECTION_PREREG_V1.md`  
**Workflow:** `iris-camera-backprojection-v1-stage`  
**Run:** `40312959634`  
**Exact SHA:** `fb8946714b941ead5f686cf84625875066ad859f`  
**Artifact:** `iris-camera-backprojection-v1-report`, digest `sha256:61abd403d88973b983e78eaa17aadb088b21fbd10e91757e321e7ab10052eed7`  
**Status:** `CAMERA_GEOMETRY_INCONCLUSIVE_TINY_V1`

## Result

| arm | mean accuracy | min accuracy |
|---|---:|---:|
| CURRENT_FACTORIZED | 0.499829 | 0.497787 |
| DETERMINISTIC_BACKPROJECTED | 0.499835 | 0.497787 |
| RANDOM_LABEL_FIREWALL | 0.499332 | 0.496078 |
| RAW_CORRESPONDENCE_POSITIVE_CONTROL | 1.000000 | 1.000000 |

The deterministic arm used no more trainable parameters than the current-like arm. The firewall and raw construction control passed, but neither learned arm solved the task and their paired difference was effectively zero.

## Why this is not a camera-geometry verdict

The court jointly required:

1. recovering canonical correspondence from camera/UV geometry; and
2. computing four-view random-bit parity through the same decoder.

The positive control validated only the task construction; it did not establish that the learned decoder/optimizer can solve the parity aggregation once correspondence is explicit. Therefore the learned failure cannot be attributed specifically to factorized camera geometry or to deterministic backprojection.

## Carry-forward

No camera treatment is selected or rejected from V1.

V2 must isolate geometry from parity/aggregation capacity. Preferred next court: canonical-locus recovery under held-out yaw and orthographic half-extent, comparing current factorized camera+UV features against deterministic backprojected canonical coordinates with a learned positive control and matched budget. Cross-view query-mediated fusion remains governed by the already-passed cross-view fusion court.

## Authority

No ProductRevision, ModelRelease, Truth139, FIT8, LOFO or genericity authority changes.
