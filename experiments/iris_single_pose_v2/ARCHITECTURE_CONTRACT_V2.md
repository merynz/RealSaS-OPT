# IRIS Single-Pose V2 — Architecture Contract

Status: `P_FORMULATION_V3_CANDIDATE__OPTIMIZER_ZERO_CLOSURE_REQUIRED__NO_TRAINING_AUTHORITY`
Date: 2026-08-24

## Problem

Input is one neutral character observed in eight ordered views under the controlled known-camera contract. Output is observation-grounded geometry/persistence evidence that a deterministic SurfaceBuilder can convert into a rigging-relevant surface substrate. Hidden authored rig identity is not an IRIS target.

The P ontology is unchanged by P Formulation V3: **P is the canonical/object-frame position of the observed physical surface locus.** V3 changes only how the image learner parameterizes that observable quantity.

## Evidence reconciled

V2 reconciles the canonical observable-substrate contract, native-1024 frontend contract, the 2026-08-21 correspondence paper transfer, post-study experiments, CI104 Representation Authority, and the CI202 learner diagnostic.

Preserved conclusions:

- exact legal P is a sufficient correspondence representation on the frozen CI104 panel (`P_GEOMETRY_SUFFICIENT`);
- D1-style observation-level correspondence is useful;
- reciprocal/cycle is a promoted deterministic primitive;
- D2 fine evidence is supported locally but falsified as global rank authority;
- CI202 learned substantial signal but the historical free-XYZ P head remained far above absolute P precision target;
- the CI202 failure does not reopen information-existence or justify an information-limit claim.

## Controlled camera geometry

Camera contract: `realsas.level_orthographic_z_orbit.v1`.

For yaw `theta`:

```text
right(theta)   = (cos theta, -sin theta, 0)
forward(theta) = (sin theta,  cos theta, 0)
up             = (0, 0, 1)
half_extent    = 0.54
```

Image coordinates use top-left origin, image Y down, NDC/grid Y up, and `align_corners=False`.

For normalized observation coordinate `g=(g_x,g_y)`:

```text
P · right   = 0.54 * g_x
P · up      = -0.54 * g_y
P · forward = d
```

Therefore P Formulation V3 reconstructs:

```text
P(g,theta) = 0.54*g_x*right(theta)
           - 0.54*g_y*up
           + d(g,theta)*forward(theta)
```

Only the camera-forward scalar `d` is learned. The two screen-plane coordinates are analytic geometry and are not rediscovered by a neural XYZ head.

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
      learned view-depth d @ R/2, conditioned on gx/gy + sin/cos(yaw)
      deterministic P reconstruction @ R/2
      N / U_geo / Z_fine @ R/2
```

Native 1024 therefore yields Z_coarse at 128x128 and geometry/fine fields at 512x512. No learned fixed-width table or `max_w` limit is permitted.

## Head roles

- **P:** public canonical/object-frame 3D surface position. P is reconstructed deterministically from exact normalized screen coordinate, known yaw/camera basis, and one learned camera-forward depth scalar. No free 3-channel XYZ head and no unproven hard depth clip.
- **N:** unit local observation-surface orientation evidence. It is auxiliary/diagnostic in the current learner line and is forbidden from correspondence admission/ranking and checkpoint-selection authority.
- **U_geo:** geometry risk trained from detached Euclidean P error. It is not match ambiguity.
- **Z_coarse:** only descriptor allowed to perform global/high-recall corridor search.
- **Z_fine:** local precision only; forbidden from global candidate admission or cross-basin ranking.
- **V/support:** direct alpha/raster visibility plus deterministic correspondence-derived support; no V neural head without causal evidence.
- **provenance:** deterministic.
- **H/ambiguity:** preserved first as top-k hypotheses; no neural H head assumed.

## Training objectives

P Formulation V3 does **not** average three XYZ regression channels. The learned P objective is SmoothL1 on camera-forward depth `d=P·forward`; the two screen-plane coordinates are analytic and therefore do not dilute the depth gradient. Evaluation and U_geo continue to use the reconstructed full 3D P and its Euclidean error.

Z_coarse is supervised before view pooling using observation-level multi-positive contrast, bidirectional view-pair matching, hard wrong-locus margin and a weak soft reciprocal term. Z_fine receives only a local offset/lattice objective around truth-containing local support. N receives direct local normal truth. U_geo receives detached full-P error heteroscedastic supervision.

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

## P Formulation V3 closure boundary

No optimizer is authorized merely because the code compiles.

Before any retraining, V3 must close:

1. synthetic camera/formulation invariants at R=256/512/1024;
2. finite full-loss backward with nonzero depth-head gradient;
3. real FIT-only corpus canonical gauge/envelope;
4. exact camera basis/half-extent/raster-1024 authority;
5. `triangle+barycentric -> P -> camera projection` agreement with raster observation;
6. exact-depth V3 reconstruction of canonical P.

Frozen authority:

- `P_FORMULATION_V3_CLOSURE_PREREG_20260824.md`
- `P_FORMULATION_V3_PANEL_V1.json`
- `p_formulation_corpus_audit_v1.py`

The closure is optimizer=0 and may not consume TUNE/CAL/DEV/EXTERNAL_HOLDOUT.

## D3 boundary

A learned query-conditioned local cost-volume refiner is reserve-only. It may be opened only if correct coarse containment is proven and a reproducible local hard tail remains. D3 may improve the local stage; it may not silently replace high-recall coarse containment.

## Non-claims

This contract is not proof of image extractability, generalization, SurfaceBuilder closure, Geppetto sufficiency, U calibration, native-1024 GPU budget, exact hidden mesh recovery, or unique mechanical rig identifiability. CI202 remains historical evidence for the former raw-XYZ parameterization and is not a checkpoint authority for P Formulation V3.
