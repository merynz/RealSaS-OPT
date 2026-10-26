# Oracle-Z Representation Court — Result V2

**Date:** 2026-10-26  
**Preregistration:** `ORACLE_Z_REPRESENTATION_PREREG_V2.md`  
**Execution amendment:** `ORACLE_Z_V2_FAST_EXECUTION_AMENDMENT_V1.md`  
**Workflow:** `iris-oracle-z-representation-v2-fast`  
**Run:** `40312802934`  
**Exact SHA:** `526ff81242c765f34d7a3a15bb781f32726c04f8`  
**Artifact:** `oracle-z-representation-v2-report`, digest `sha256:8273c09ed58fa73de4cdf29dd7a1b6d9d288367bf13f4b7b8c1bdecee2c0b714`  
**Probe status:** `VALID_REPRESENTATION_CAPACITY_CURVES_V2`

## Apparatus validity

The V2 court is interpretable:

- independent optimizer calibration: PASS;
- selected LR: `1e-3`;
- calibration mean/worst nRMSE: `0.073784 / 0.076905`;
- random-target firewall: PASS, mean/worst nRMSE `1.175285 / 1.193230`;
- frozen optimizer and held-out continuous evaluation were used for the comparison arms.

The dense 3-D positive control solved the easy volumetric frequency-1 family at mean/worst nRMSE `0.072361 / 0.075248`, so unlike V1 the court can expose at least one genuine representation mismatch.

## Surface-capacity result

The preregistered triplane surface-support gates were not reached at any tested radius. However this is **not** a triplane failure verdict, because the matched dense-3D per-scene upper bound also failed the same surface-family gates.

Examples:

| family / radius | TRIPLANE per-scene mean / worst | DENSE3D per-scene mean / worst |
|---|---:|---:|
| surface 0.333333 | 0.3797 / 0.4209 | 0.2937 / 0.3065 |
| surface 0.166667 | 0.3507 / 0.4186 | 0.2662 / 0.3142 |
| thin 0.166667 | 0.3322 / 0.3733 | 0.2319 / 0.2428 |
| thin 0.041667 | 0.4052 / 0.4330 | 0.3813 / 0.3967 |

Accordingly every preregistered `credible_triplane_concern_curve` entry is false. The surface target / decoder / optimization combination remains a confound for carrier selection.

## Diagnostic treatments

- explicit xyz best preregistered worst-seed gain: `1.36%`; selection threshold `20%` -> **not selected**;
- multires triplane best thin worst-seed gain: `4.15%`; selection threshold `20%` -> **not selected**;
- neither treatment justifies changing the current baseline from this court.

## Interpretation

V2 successfully repaired the V1 volumetric-control failure but did not yet establish a surface carrier winner. The correct carry-forward is:

1. do **not** reject triplanes;
2. do **not** add explicit xyz or multires merely from these results;
3. calibrate the surface target itself with a learned/direct-coordinate positive control and a surface-relevant easier ladder before blaming a carrier;
4. only then compare triplane/dense/other carriers under the same validated surface task.

This is an apparatus/selection refinement, not a product or model-authority change.

## Authority

No ProductRevision, ModelRelease, Truth139, FIT8, LOFO or genericity authority changes.
