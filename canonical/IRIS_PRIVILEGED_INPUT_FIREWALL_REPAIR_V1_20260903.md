# RealSaS — IRIS privileged-input firewall repair V1

**Date:** 2026-09-03  
**Status:** `PASS_SOURCE_FIREWALL_CLOSED__HISTORICAL_LEARNED_RESULTS_QUARANTINED__NO_FIT_AUTHORIZED`  
**Branch:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Formal family selection:** `BLOCKED`

## 1. Epistemic continuity

This record does not rewrite the earlier leak finding. `canonical/IRIS_LEAK_SCOPE_20260903.md` remains the historical scope record.

The 2026-08-31 IRIS V2 preregistration admitted a learned RGBA native path and padded visual-hull candidate domain. Under the stricter observation-only firewall adopted on 2026-09-03, those clauses are scientifically superseded. The original preregistration remains immutable historical provenance; the correction is recorded separately rather than silently editing history.

Source-firewall closure is not a learned-fit result. It does not retroactively clean any checkpoint or metric produced under the old privileged-input contract.

## 2. Causal confirmation before repair

Workflow run `33704040761` executed the original source with deterministic synthetic mutations and confirmed all three privileged-input paths:

- alpha-only mutation -> learned native feature max delta `2.012694835662842`;
- cyclic view re-enumeration with jointly permuted evidence/analytic projections -> pooled evidence max delta `0.18466269969940186`;
- foreground-mask mutation -> `8` candidate-valid flips and mask-selected anchor count `64 -> 32`.

The contemporaneous legacy source suite still reported `20 passed`, proving the old tests were blind to these causal paths.

These values remain historical causal evidence. Later PASS results do not erase or reinterpret them.

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
- jointly re-enumerating observations and their analytic camera relations may only re-enumerate view-local tokens/weights; pooled evidence must remain invariant.

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

## 4. Frozen negative regressions

Regression file:
`tests/iris/test_iris_v2_privileged_input_firewall_v1.py`

It asserts four generic causes, not any real-family fixture:

1. alpha mutations cannot change learned native features because learner input is RGB-only;
2. joint view re-enumeration preserves pooled evidence and only re-enumerates view-local outputs;
3. production Q-domain is full-frame camera-only and mask/pruned/forged domains fail closed;
4. apparatus rejects an unsealed domain before image/foundation execution.

Source fixtures were changed from 4-channel to 3-channel direct learner input without changing the five-scale native width/capacity profile.

`iris-v2-source-contract` now treats the firewall as a mandatory negative regression gate rather than re-running the old positive leak diagnostic.

## 5. Standalone source closure

### Run A

Workflow run `33751592814`, job `100636154068`, runner region `westcentralus`:
- dedicated privileged-input firewall: `4/4 PASS`;
- combined IRIS source + generic-strength + firewall suite: `24/24 PASS`.

### Run B — cross-region replay

Workflow run `33751730077`, job `100636581385`, runner region `westus3`:
- dedicated privileged-input firewall: `4/4 PASS`;
- combined IRIS source + generic-strength + firewall suite: `24/24 PASS`.

`IRIS_PRIVILEGED_INPUT_FIREWALL_CROSS_REGION_REPLAY = PASS`

This establishes deterministic source-contract replay at the test/status level. It is not a bit-level learned-output or fit-performance claim.

## 6. Architecture-freeze prerequisite integration

Commit `1c35357bd03b0589e04a70a59890f3ef0f7242a6` makes the IRIS privileged-input firewall a mandatory architecture-freeze prerequisite.

Architecture-freeze workflow run `33751730186`, job `100636583064`, runner region `northcentralus`:
- compile current learner source: `PASS`;
- family-independent / freeze-eligible source audit: `PASS`;
- expanded generic + behavioral prerequisite suite: `65/65 PASS`;
- generic source count: `38`;
- candidate fingerprint: `1c6878b2e1e8cbd30a055849a64c8fe68558924e0a874de2e8fffa2e24ad7575`;
- candidate status: `PASS_SOURCE_ELIGIBLE_FOR_FREEZE`;
- `family_selection_authorized = false`;
- final workflow failure is intentionally only `FAMILY_SELECTION_BLOCKED__SOURCE_CHANGED_AFTER_FREEZE`.

The candidate fingerprint is **not** a seal. No refreeze was performed.

## 7. Authorization boundary after source closure

- IRIS privileged-input source firewall: `PASS / CLOSED`;
- deterministic Gate-0 mask/hull measurement lane: `ALLOWED_AS_DIAGNOSTIC_ONLY`;
- learned IRIS fitting under repaired source: `NOT YET AUTHORIZED BY THIS RECORD`;
- affected historical learned IRIS checkpoints/metrics: `QUARANTINED`;
- Geppetto behavioral closure: unaffected and remains separately `PASS`;
- architecture candidate: `ELIGIBLE_FOR_FREEZE`, not frozen;
- formal Family-1 selection: `BLOCKED`.

## 8. Scientific interpretation

The repaired learner contract is now:

`RGB observations + exact camera/candidate analytic relations -> learned evidence`

and explicitly not:

`RGBA/renderer alpha + mask-selected hypothesis domain + absolute orbit slot -> learned evidence`.

Mask/hull measurements remain useful deterministic Gate-0 diagnostics. They no longer authorize or constrain the production learned hypothesis domain.

No real-family fit was used to define or validate these repairs.

## 9. Remaining program work

The privileged-input seam itself is source-closed. Broader architecture hardening continues before any global refreeze decision. Historical affected learner artifacts remain quarantined until replaced by evidence produced under a future explicitly sealed repaired architecture.
