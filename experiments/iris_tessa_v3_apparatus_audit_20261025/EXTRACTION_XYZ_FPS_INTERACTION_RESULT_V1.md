# Extraction x XYZ-FPS Interaction Result V1

**Run:** `40318067128`  
**Head SHA:** `0fb13be718207b87290ce8cb078a7ac49d090698`  
**Artifact digest:** `sha256:9aa16576471c0fd89062735ae2b71feb3e93abf6b4eef74ed579ef0a62181331`  
**Status:** `OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION_SELECTED_TINY_V1`

## Result

XYZ-FPS remained permutation invariant and achieved conditional support recall `1.0` at both 64 and 128 witness budgets for every tested extraction resolution.

Mean end-to-end qualified-support recall was therefore exactly extraction admission recall:

| extraction resolution | mean E2E recall | worst E2E recall | XYZ-FPS conditional recall |
|---:|---:|---:|---:|
| 24 | 0.438272 | 0.000000 | 1.0 |
| 32 | 0.709877 | 0.333333 | 1.0 |
| 48 | 0.938272 | 0.666667 | 1.0 |
| 64 | 1.000000 | 1.000000 | 1.0 |

Increasing the witness budget from 64 to 128 produced **0.0 end-to-end recall gain** at every resolution. Features absent from extraction cannot be recovered by sampler budget.

## Decision

Carry the architecture policy form:

`OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION -> deterministic XYZ_FPS`

Do not hard-wire res24 as a generic inference guarantee and do not repair extraction misses by increasing witness count.

This court does not select a final production resolution. The escalation thresholds/default resolution require a normalized real morphology thickness/gap census during Truth Builder/FIT8 preparation.
