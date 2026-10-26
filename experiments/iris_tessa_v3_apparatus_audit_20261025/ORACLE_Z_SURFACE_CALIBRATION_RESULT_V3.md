# Oracle-Z Surface Calibration Result V3

**Run:** `40315595228`  
**Head SHA:** `6ebd644613c9aaa3817857e684a1c7b14ca5ada9`  
**Artifact digest:** `sha256:37149f70c0962bf9cf9a8bd8a9fb00019cf6c06449831c5360cbf93bdfdc80fc`  
**Status:** `APPARATUS_INCONCLUSIVE__SURFACE_CALIBRATION_FAIL`

## Calibration

Direct-coordinate learned positive control, surface target:

| LR | mean nRMSE | worst nRMSE |
|---:|---:|---:|
| 3e-3 | 0.55451 | 0.61595 |
| 1e-3 | 0.70338 | 0.77431 |
| 3e-4 | 0.76776 | 0.82941 |

Best arm remains far above the preregistered `<= 0.10` calibration gate.

## Verdict

No carrier comparison is scientifically authorized from this target. A direct-coordinate positive control cannot solve the same surface-response task, so failure by TRIPLANE or DENSE3D would be uninterpretable.

Carry forward:

1. do not reject triplane or dense carrier from V3;
2. do not increase carrier complexity to compensate for an uncalibrated target;
3. redesign the surface target/easy ladder so an explicit direct-coordinate positive control first demonstrates learnability;
4. only then reopen matched carrier capacity and shared-Q courts.

This narrows the fault to task/target/optimization formulation before representation choice rather than providing evidence against the current carrier.
