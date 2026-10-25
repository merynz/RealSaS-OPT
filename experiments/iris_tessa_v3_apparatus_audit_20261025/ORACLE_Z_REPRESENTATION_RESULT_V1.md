# Oracle-Z Representation Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `ORACLE_Z_REPRESENTATION_PREREG_V1.md`  
**Workflow:** `iris-oracle-z-representation-v1-stage`  
**Run:** `40227304772`  
**Status:** `APPARATUS_INCONCLUSIVE__ESCALATE`

## Outcome

The tiny oracle-Z court did **not** establish the preregistered surface-capacity gates for the current triplane+Q arm.

Current `TRIPLANE_Q` mean normalized RMSE was approximately:

- `SURFACE_STRUCTURED`: 0.318–0.319 across seeds;
- `THIN_SURFACE`: 0.315–0.318 across seeds;
- `VOLUMETRIC_3WAY_CONTROL`: ~1.001.

The matched-budget `DENSE3D_Q` control improved the ordinary surface family to roughly 0.231–0.239 mean nRMSE and was similar on the thin family, but it also remained near ~0.99–1.00 on the deliberately volumetric control.

`MULTIRES_TRIPLANE_Q` produced only a small thin-family improvement (reported worst-seed relative improvement ~4.77%), below the preregistered 20% selection threshold. Explicit xyz did not improve the base arm; the worst-seed relative change was negative (~-1.39%).

## Why this is not a triplane verdict

The court was intended to separate representation capacity from perception. It succeeded at bypassing DINO/camera/image inference, but it did **not** fully isolate representation from optimization and shared-query capacity.

The decisive diagnostic is that the supposedly favorable dense-3D control also failed the volumetric family. Therefore at least one of the following remains a confound:

- optimization horizon / learning-rate schedule;
- shared QueryMLP capacity;
- multi-scene shared decoder burden;
- target-frequency scale versus tiny carrier resolution;
- carrier initialization / conditioning;
- the synthetic target construction itself.

Accordingly the correct classification is `APPARATUS_INCONCLUSIVE`, not `TRIPLANE_FAIL`.

## Escalation contract

V2 must factor the question into separate courts:

1. **carrier-only oracle:** per-scene decoder or analytic/direct decoder upper bound, proving whether each carrier can encode the target at all;
2. **shared-Q court:** freeze or prefit carriers, then test whether one shared Q can decode across scenes;
3. **optimization court:** learning-rate/step schedule chosen on an independent calibration family before the actual representation comparison;
4. **frequency ladder:** sweep target spatial frequency / thickness against carrier cell size rather than testing one arbitrary difficulty;
5. **negative-control calibration:** require at least one arm to solve the volumetric control at an easier frequency before interpreting a failure at harder frequencies.

Only after those controls work may `TRIPLANE_Q` versus dense/multires/other carriers be treated as a representation result.

## Current architecture consequence

None. No representation choice is promoted or rejected from V1.

The result does, however, strengthen the audit requirement that **probe adequacy itself must be falsified** before a negative architecture result is accepted.

## Authority

No Product, Truth139, FIT8 or ModelRelease authority changed.
