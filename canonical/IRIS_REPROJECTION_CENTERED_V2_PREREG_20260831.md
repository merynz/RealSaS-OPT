# RealSaS — IRIS Reprojection-Centered V2 Preregistration Seal

**Date:** 2026-08-31  
**Status:** `SEALED_BEFORE_GATE0_OUTPUTS__NO_OPTIMIZER_AUTHORIZED`  
**Successor to:** `DINO_LADDER_S1_EARLY_TERMINATION` + `DINO_FROZEN_APPARATUS_CODE_AUDIT_V1`

## 1. Purpose

V2 does not ask a pointwise learner to infer depth from a low-resolution fused feature field. It restructures Mode G around the information that is analytically known: eight exact orthographic cameras and their reprojection geometry.

The central question becomes:

> for a canonical world candidate q=(X,Y,Z), do the observations obtained by exact reprojection into the eight views provide mutually compatible evidence for an observed surface?

The learned system evaluates evidence. It does not relearn camera geometry and does not complete unseen geometry.

## 2. Claim firewall

This seal does **not** claim:

- that the S1 failure was caused by the decoder;
- that DINO-S is sufficient under V2;
- that DINO is necessary;
- that visual hull is useful until its containment/reduction frontier is measured;
- that recurrence/iterative visibility refinement is required;
- that clean-C0 corpus changes explain S1.

Permitted starting claim:

> DINOv2-S is the compute-efficient default backbone for V2 apparatus development. A larger backbone is not authorized unless S first exhibits a representation ceiling under a qualified V2 apparatus.

## 3. External IRIS contract — unchanged

IRIS remains observation-grounded perception. The externally consumable geometric quantity remains camera-forward depth d plus validity/uncertainty/support evidence; common-frame P remains analytic from exact cameras.

V2 may use a canonical internal representation, but it must remain **partial**. It may not infer a closed occupancy volume, hidden back surface or learned completion.

Canonical candidate states are:

- `SURFACE_SUPPORTED`
- `SURFACE_AMBIGUOUS`
- `UNKNOWN_OCCLUDED`
- `UNKNOWN_UNOBSERVED`
- `OUTSIDE_ADMISSIBLE_DOMAIN`

Unknown states may not be rendered as invented depth.

## 4. Mode G / Mode E invariant

The Product Contract Mode scalability invariant remains binding: Mode G and future Mode E preserve the analytic-geometry / learned-evidence topology. Increased camera/observation uncertainty may increase learned correspondence/reliability burden, but may not silently create a second geometry authority or hidden completion.

## 5. V2-A minimal single-pass reference

V2-A is intentionally the simplest reprojection-centered reference. Architectural mechanisms may be added only after a measured failure demonstrates the need.

```text
8 x native 1024 RGBA
+ 8 exact orthographic camera.json authorities
        |
        +--> frozen DINOv2-S multi-level correspondence descriptors
        |
        +--> learned native high-resolution spatial pyramid
                         |
                         v
              padded visual-hull domain
               [constraint only]
                         |
                         v
              canonical q=(X,Y,Z) lattice
                         |
              exact q -> V0..V7 projection
                         |
              descriptor sampling per view
                         |
              robust view-evidence aggregation
                         |
              compact evidence field C(Z,X,Y)
                         |
              spatial regularization with
              approximately isotropic world
              receptive field
                         |
              supported/ambiguous surface modes
                         |
              local continuous refinement
                         |
           CanonicalObservedSurfaceEvidence
                         |
                exact camera rendering
                         |
        8 forward-depth maps + validity + uncertainty
                         |
                    P = O + dF
                         |
                  frozen DTB-ND1
```

V2-A explicitly excludes:

- learned hidden-surface completion;
- full occupancy/SDF authority outside the observed band;
- learned occlusion authority;
- recurrent visibility iteration;
- sparse-convolution dependency;
- S/B/L/g ladder;
- candidate-specific consumer tuning;
- Geppetto/Arachne labels in the learner.

## 6. DINO authority

Backbone: frozen official DINOv2-S/14 under the already sealed source/weight authority.

The S1 `384 -> zero-pad 1536 -> L2 -> 256` ladder adapter is **not** inherited.

For correspondence, V2 will expose multiple S transformer depths. Before optimizer step 1, exact tap indices, projection widths, preprocessing and cache hashes must be frozen. The provisional architecture target is four evenly distributed taps including the final block; no tap may be chosen using V2 learned-output quality.

A DINO-OFF / native-ON ablation is mandatory before any claim that the foundation prior is necessary.

## 7. Native spatial path

The 25-point RGBA stencil is retired for V2-A.

The native path must be a real shared 2D spatial pyramid and must preserve meaningful high-resolution capacity. Initial width profile to preflight:

- 1024: 32 channels
- 512: 48 channels
- 256: 64 channels
- 128: 96 channels
- 64: 128 channels

High-resolution blocks should be shallow and memory-aware; width may be reduced only by a pre-optimizer VRAM preflight rule recorded before learned outputs.

## 8. Exact camera geometry

`camera.json` is sole camera authority. Image-derived scale proxies are forbidden.

Required analytic operations:

- world q -> per-view normalized image coordinate;
- image coordinate + camera-forward depth -> world P;
- q -> per-view camera-forward depth;
- exact cross-view reprojection.

Gate 0 requires projection/backprojection roundtrip and Z-row invariance tests before any learned execution.

## 9. Candidate lattice and visual hull

V2-A uses **dense XY slices + hull masking + chunked/streamed candidate evaluation**. Raw eight-view descriptors are never stored as a 3D volume; they are sampled from 2D feature maps and immediately reduced to compact candidate evidence.

Sparse convolutions are not part of V2-A.

Coarse spacing is **not frozen to .008**. Gate 0 performs a training-free sweep over:

`{0.016, 0.008, 0.004}` world units.

For each spacing report at minimum:

- lattice dimensions and total candidate count;
- active hull candidate count;
- projected descriptor sample count;
- memory/runtime estimate;
- authoritative thin-structure thickness expressed in coarse cells;
- count/fraction of structures in `<1`, `[1,2)`, `[2,4)`, `>=4` cell strata.

Continuous refinement does not excuse a missing coarse mode. Any product-relevant structure class that is sub-cell must either trigger finer/adaptive search or be explicitly typed `RESOLUTION_UNSUPPORTED -> UNKNOWN` with quantified product loss.

Visual hull is **only** a candidate-domain constraint. It is never an input semantic feature, predicted surface or fallback geometry.

Gate 0 sweeps padding `{0,1,2,4,8}` native pixels and jointly reports:

1. truth containment;
2. hull-volume / bounding-domain ratio.

The selected padding must preserve required truth containment while retaining material search reduction. If containment requires padding that makes the hull computationally useless, hull pruning is disabled rather than claimed as useful.

## 10. Evidence aggregation and occlusion

In V2-A, geometric visibility is not learned because resolved visibility is a function of the still-unknown surface.

The first pass therefore uses permissive validity plus robust multi-view consensus. A minority of incompatible/occluded observations must not kill an otherwise supported candidate.

Learned **reliability** may later estimate whether a geometrically admissible observation is discriminative. Learned **occlusion authority** remains forbidden.

If V2-A failure anatomy demonstrates an occlusion-specific ceiling, a later fixed-iteration geometry-derived visibility refinement may be opened as a new versioned experiment.

## 11. Spatial regularization

Canonical coordinates are `[Z,X,Y]`; no target-view `[z,x,d]` notation is permitted in V2-A.

Implementation may be factorized for compute, but the effective receptive field must be approximately isotropic in world units. Z is privileged by camera orbit for computation, not by character morphology.

Mandatory evaluation strata include orientation/thickness categories sufficient to expose horizontal, vertical and oblique thin structures.

## 12. Refinement / uncertainty

Global soft-argmin across a multimodal distribution is forbidden.

V2 must retain ambiguity. Continuous refinement is local to a selected supported mode. Multimodality, mode margin, support-view count and evidence dispersion remain available as uncertainty evidence; unsupported averages between modes may not be emitted as confident surface.

## 13. Iteration policy

V2-A is single-pass.

No recurrent surface/visibility refinement may be added without a demonstrated V2-A failure it addresses.

If iteration is later opened:

- visibility is geometry-derived and conservative/fail-closed;
- every iteration is evaluated separately;
- intermediate supervision is recorded;
- an equal-supervision control is required before claiming iteration itself causes monotonic improvement;
- greater iteration depth is not evidence of refinement.

Program rule:

> `NO_ARCHITECTURAL_MECHANISM_WITHOUT_A_DEMONSTRATED_FAILURE_IT_ADDRESSES`.

## 14. Gate sequence

### Gate 0 — deterministic geometry/apparatus

No optimizer.

Must close:

- exact camera contract validation;
- projection/backprojection numerical parity;
- common-world-Z row invariance across orbit views;
- hull containment x search-reduction frontier;
- coarse spacing x thin-structure resolvability;
- descriptor sampling coordinate parity;
- robust evidence permutation/outlier tests;
- candidate streaming/memory accounting;
- no completion path;
- deterministic synthetic regression suite.

### Gate 0.5 — one-family single-pass ceiling

Only after Gate 0 passes and the learned V2-A source/config/runtime hashes are separately sealed.

Purpose: feasibility/ceiling, **not causal attribution**.

If one real eligible family cannot approach/enter frozen DTB-ND1 tolerance, stop. Do not run a large corpus.

### Gate 1 — sealed heterogeneous 8-family ceiling

Only after one-family success. Report morphology/thickness/orientation strata.

### Gate 2 — small causal ablations

Only after the reference architecture demonstrates adequate ceiling:

- DINO ON / native ON;
- DINO OFF / native ON;
- DINO ON / native OFF;
- other mechanisms only when justified by observed failure.

### Full training

Not authorized by this seal.

## 15. Loss firewall

Gate 0 has no learned loss.

The V2-A learned loss is not frozen by this document because the actual differentiable canonical evidence representation is not yet implemented or preflighted. Its loss must be separately sealed **before optimizer step 1**.

Consumer labels/gradients remain forbidden. DTB-ND1 is evaluation authority, not training target.

## 16. Corpus firewall

The first V2-A causal/feasibility work must not silently combine the new apparatus with a new clean-C0 membership and then attribute gains to architecture.

The completed native image-integrity audit and later semantic clean-C0 work are separate H0 interventions. S1 may not be retroactively reinterpreted from them.

## 17. Current authorization

- Gate 0 deterministic implementation/testing: **AUTHORIZED**.
- DINO/native descriptor extraction preflight without optimizer: **AUTHORIZED** after source hashes are recorded.
- Any optimizer step for V2-A: **NOT AUTHORIZED** until Gate 0 closes and a learned apparatus/loss/runtime seal is committed.
- New S/B/L/g ladder: **NOT AUTHORIZED**.
