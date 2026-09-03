# RealSaS — IRIS privileged-input firewall repair V1

**Date:** 2026-09-03  
**Status:** `IN_PROGRESS__REGRESSION_RUNNING__LEARNED_IRIS_RESULTS_REMAIN_QUARANTINED`  
**Branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Formal family selection:** `BLOCKED`

## 1. Epistemic continuity

This record does not rewrite the earlier leak finding. `canonical/IRIS_LEAK_SCOPE_20260903.md` remains the historical scope record.

The old IRIS V2 preregistration admitted a learned RGBA native path and padded visual-hull candidate domain. Under the stricter observation-only firewall adopted on 2026-09-03, those clauses are scientifically superseded. No learned fit is authorized until the repaired source is behaviorally closed and a new architecture fingerprint is explicitly sealed.

## 2. Causal confirmation before repair

Workflow run `33704040761` executed the original source with deterministic synthetic mutations and confirmed all three privileged-input paths:

- alpha-only mutation -> learned native feature max delta `2.012694835662842`;
- cyclic view re-enumeration with jointly permuted evidence/analytic projections -> pooled evidence max delta `0.18466269969940186`;
- foreground-mask mutation -> `8` candidate-valid flips and mask-selected anchor count `64 -> 32`.

The contemporaneous legacy source suite still reported `20 passed`, proving the old tests were blind to these causal paths.

These results remain historical evidence; they are not replaced by later PASS results.

## 3. Generic source repairs

### A. Renderer alpha removed from learned path

Commit `a259c845d2f56f8988bdff73b34b092f9ed0979f`:
- `NativeResolutionPyramidV2.INPUT_CHANNELS = 3`;
- learned native pyramid rejects 4-channel input;
- streamed native descriptor path is RGB-only.

Commit `c557637e686b0b54a4704ba8d038e38918fb4fe8`:
- direct `IrisReprojectionV2.forward()` also rejects non-RGB learner input.

Observation files may remain RGBA for renderer/provenance authority; alpha is removed before learner execution.

### B. Absolute view-slot identity removed

Commit `86407a0f686826c9f02a91c1d1df9757428cface`:
- removes learned absolute `vi` scalar;
- replaces it with analytic per-view `projected_depth` alongside projected grid and validity;
- intended consumer contract is view re-enumeration equivariance: jointly permuting observations and analytic camera relations may only permute view-local tokens/weights, not pooled evidence.

### C. Mask / externally selected Q-domain separated from production authority

Commit `a44aa4de03ea7e6de1d032b676f8b4d01deddb19`:
- introduces `RGB_CAMERA_FULL_FRAME_LATTICE_V1` as the only production Q-domain authority;
- retains mask-constrained domains as `MASK_CONSTRAINED_DIAGNOSTIC_V1` for deterministic Gate-0 measurements;
- caller-supplied anchors are explicitly unsealed;
- production lattice is deterministic, full-frame and camera-only;
- production validator reconstructs the expected full-frame lattice and rejects mask policies, pruned anchors, forged support/candidate validity and unsealed construction authority.

Commit `2856becf02df01a51eed33d22e72670f0a5ecfca` plus follow-up `5b8ef8642c2084e747056e345b2ca6002a138247`:
- production apparatus requires RGB learner interface and sealed camera-only Q-domain;
- Q-domain validation occurs before observation preprocessing, frozen-foundation extraction or learned execution.

## 4. Regression closure being executed

New frozen regression file:
`tests/iris/test_iris_v2_privileged_input_firewall_v1.py`

It asserts four generic causes, not any real-family fixture:

1. alpha mutations cannot change learned native features because learner input is RGB-only;
2. joint view re-enumeration preserves pooled evidence and only re-enumerates view-local outputs;
3. production Q-domain is full-frame camera-only and mask/pruned/forged domains fail closed;
4. apparatus rejects an unsealed domain before image/foundation execution.

Additional source fixture repair changes direct learner tests and native-capacity tests from 4-channel to 3-channel input without changing five-scale widths/capacity.

CI workflow `iris-v2-source-contract` now treats the firewall as a mandatory negative regression gate rather than re-running the old positive leak diagnostic.

## 5. Current authorization boundary

- deterministic Gate-0 mask/hull measurement lane: `ALLOWED_AS_DIAGNOSTIC_ONLY`;
- learned IRIS fitting: `NOT_AUTHORIZED`;
- affected historical learned IRIS evidence: `QUARANTINED`;
- Geppetto behavioral closure: unaffected and remains separately recorded;
- architecture refreeze: `BLOCKED` until IRIS firewall regressions and broader architecture-freeze prerequisites pass;
- formal Family-1 selection: `BLOCKED`.

## 6. Next closure actions

1. obtain clean `iris-v2-source-contract` PASS with firewall regressions;
2. run the same repaired source under architecture-freeze prerequisite suite;
3. update this record with run/job IDs, exact PASS counts and final source fingerprint candidate;
4. only then decide whether the repaired IRIS contract is ready for explicit refreeze; no learned fit or family selection occurs before that decision.
