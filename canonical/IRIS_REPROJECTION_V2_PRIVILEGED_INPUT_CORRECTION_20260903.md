# RealSaS — IRIS Reprojection V2 privileged-input correction

**Date:** 2026-09-03  
**Status:** `ACTIVE_CORRECTION__ORIGINAL_PREREG_PRESERVED__NO_FIT_AUTHORIZATION`  
**Corrects:** `canonical/IRIS_REPROJECTION_CENTERED_V2_PREREG_20260831.md`

## 1. Purpose

This document is a successor correction, not an edit of the 2026-08-31 preregistration. The original preregistration remains authoritative evidence of what was actually assumed at that time.

A later causal audit demonstrated that three assumptions allowed privileged renderer/domain metadata to influence the learned IRIS evidence path. The corresponding source seam was repaired and behaviorally locked by `canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`.

Only the clauses below are superseded. Unrelated IRIS V2 principles remain historical context until a future explicit architecture seal adjudicates them.

## 2. Observation-channel correction

### Superseded clause

The old architecture diagram and native-spatial-path description admitted `8 x native 1024 RGBA` directly into learned native descriptors.

### Corrected rule

Observation artifacts may still be stored and hash-validated as native RGBA for renderer/provenance integrity. Learned IRIS input is exactly native RGB.

Renderer alpha is not learner evidence and may not enter trainable native descriptors, learned evidence aggregation, support prediction or uncertainty prediction.

`LEARNER_OBSERVATION_CHANNELS = RGB_ONLY`

## 3. Candidate-domain correction

### Superseded clause

The old V2-A diagram and candidate-lattice section placed a padded visual-hull domain in the learned production path.

### Corrected rule

Mask-, alpha-, raster- or externally selected candidate domains are forbidden as learned production Q-domain authority.

The learned production Q-domain is a deterministic full-frame camera-only ray lattice generated from:
- exact observation camera authority;
- a globally declared anchor view and stride;
- declared depth hypotheses.

Its candidate validity is analytic in-frame validity only.

Foreground masks and visual-hull calculations remain permitted in the deterministic Gate-0 measurement lane for containment, computational accounting and diagnostic studies. They may not create/delete production learner rays or candidates.

`PRODUCTION_Q_DOMAIN_AUTHORITY = RGB_CAMERA_FULL_FRAME_LATTICE_V1`

`MASK_HULL_DOMAIN_ROLE = GATE0_DIAGNOSTIC_ONLY`

## 4. View-identity correction

### Superseded behavior

The inspected implementation supplied a learned absolute view-slot scalar `vi` to the evidence encoder.

### Corrected rule

Learned evidence aggregation receives no absolute view/orbit slot identity.

Camera/candidate relation may enter through analytic equivariant quantities derived from exact camera authority, including:
- projected image grid coordinates;
- camera-forward projected depth;
- analytic validity.

Jointly re-enumerating observations and these analytic camera relations must not change pooled evidence; view-local tokens/weights may only be re-enumerated correspondingly.

`ABSOLUTE_VIEW_SLOT_IDENTITY = FORBIDDEN`

## 5. Authority boundary

This correction does not relax the original Mode-G invariant:
- exact cameras remain analytic geometry authority;
- IRIS learns observation evidence, not camera geometry;
- no hidden-surface completion is authorized;
- common-frame P remains analytic from exact camera plus emitted forward depth;
- downstream Compiler authority remains unchanged.

The correction narrows learner inputs; it does not grant a new geometry authority.

## 6. Evidence status

The pre-repair causal audit remains valid historical evidence:
- alpha path was causally reachable;
- absolute view slot broke enumeration equivariance;
- mask domain causally controlled candidates/rays.

The repaired source firewall is closed by cross-region negative regressions and architecture-freeze prerequisites. This does not rehabilitate checkpoints or metrics produced under the superseded contract.

Affected historical learned IRIS checkpoints/metrics remain quarantined.

## 7. Authorization

This correction alone authorizes no optimizer step, no family selection and no architecture refreeze.

- repaired source-contract testing: `AUTHORIZED / COMPLETE`;
- historical learned result reuse under the corrected claim: `FORBIDDEN`;
- new learned fit: `REQUIRES FUTURE EXPLICIT SEALED ARCHITECTURE / RUN AUTHORIZATION`;
- formal Family-1 selection: `BLOCKED` until explicit global refreeze.

## 8. Provenance rule

Never edit the original 2026-08-31 preregistration to make it appear as if these corrected rules were known in advance.

The valid chain is:

`original prereg -> causal falsification -> repair ledger -> this correction -> future explicit architecture seal`.
