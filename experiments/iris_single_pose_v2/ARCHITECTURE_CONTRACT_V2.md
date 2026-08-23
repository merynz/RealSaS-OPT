# IRIS Single-Pose V2 — Architecture Contract

Status: `AUDIT_CANDIDATE__NO_TRAINING_AUTHORITY`
Date: 2026-08-24

## Problem

Input is one neutral character observed in eight ordered views under the controlled known-camera contract. Output is observation-grounded geometry/persistence evidence that a deterministic SurfaceBuilder can convert into a rigging-relevant surface substrate. Hidden authored rig identity is not an IRIS target.

## Evidence reconciled

V2 reconciles the canonical observable-substrate contract, native-1024 frontend contract, the 2026-08-21 correspondence paper transfer, and post-study experiments. The preserved experimental conclusion is explicit: D1-style observation-level correspondence is useful, reciprocal/cycle is a promoted deterministic primitive, and D2 fine evidence is supported locally but falsified as global rank authority.

## Neural architecture

For square input R divisible by 16:

```text
RGBA x 8
 -> shared encoder: f2=R/2, f4=R/4, f8=R/8, f16=R/16
 -> within-view axial reasoning on full f16
 -> adaptive pool f16 to fixed 16x16 context
 -> known-yaw row-wise cross-view Transformer
 -> upsample pooled context to full f16 and fuse with local f16
 -> skip decoder
      Z_coarse @ R/8
      P / N / U_geo / Z_fine @ R/2
```

Native 1024 therefore yields Z_coarse at 128x128 and fine/geometry evidence at 512x512. No learned fixed-width table or `max_w` limit is permitted.

## Head roles

- **P:** direct common/object-frame position evidence; no unproven hard output clip.
- **N:** unit local orientation/normal evidence.
- **U_geo:** geometry risk trained from detached P error. It is not match ambiguity.
- **Z_coarse:** only descriptor allowed to perform global/high-recall corridor search.
- **Z_fine:** local precision only; forbidden from global candidate admission or cross-basin ranking.
- **V/support:** direct alpha/raster visibility plus deterministic correspondence-derived support; no V neural head without causal evidence.
- **provenance:** deterministic.
- **H/ambiguity:** preserved first as top-k hypotheses; no neural H head assumed.

## Training objectives

Z_coarse is supervised before view pooling using observation-level multi-positive contrast, bidirectional view-pair matching, hard wrong-locus margin and a weak soft reciprocal term. Z_fine receives only a local offset/lattice objective around truth-containing local support. P receives direct geometry + cross-view consistency, N direct normal truth, and U_geo detached-error heteroscedastic supervision.

## D3-free matcher

```text
source query
 -> known-camera corridor on target Zc lattice
 -> Zc global top-Kc UNION P-nearest rescue Kp
 -> coarse basin set/order using Zc+P only
 -> local Zf search inside each admitted basin
 -> preserve basin order across basins
 -> top-k hypotheses + evidence/provenance
 -> reciprocal/cycle qualification
 -> later deterministic SurfaceBuilder
```

No singleton is authorized before a dedicated calibration gate.

## D3 boundary

A learned query-conditioned local cost-volume refiner is reserve-only. It may be opened only if correct coarse containment is proven and a reproducible local hard tail remains. D3 may improve the local stage; it may not silently replace high-recall coarse containment.

## Non-claims

This contract is not proof of downstream Geppetto sufficiency, SurfaceBuilder closure, U calibration, native-1024 GPU budget, exact hidden mesh recovery, or unique mechanical rig identifiability.
