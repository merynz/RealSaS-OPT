# RealSaS N1D — Descriptor Repair Ladder

**Date:** 2026-08-21  
**Status:** `FROZEN_RESEARCH_LADDER__D1_NEXT`  
**Parent:** `canonical/MODEL1_2P5D_MECHANICAL_SURFACE_CONTRACT.md`

## Motivation

Current post-hoc diagnostics show that the frozen descriptor is useful as a proposal prior but is not a qualified pixel/carrier identity discriminator. The current implementation has three known structural mismatches:

1. **Objective mismatch:** training pools visible views into one carrier vector before the legacy cross-pose InfoNCE loss, while inference asks same-view pixel/candidate identity questions.
2. **Spatial mismatch:** the descriptor is emitted on the 32x32 `f4`-resolution dense decoder lattice, while downstream proposal refinement distinguishes native 256x256 sites at 2-pixel spacing.
3. **Matcher mismatch:** inference uses local descriptor similarity/interpolation without a learned query-conditioned coarse-to-fine local matcher or calibrated reciprocal uncertainty.

These problems are repaired one controlled treatment at a time. No later treatment may be silently folded into an earlier gate.

## Frozen ladder

```text
D0 — frozen baseline
     exact current metrics
          │
          ▼
D1 — OBJECTIVE FIX
     architecture same
     head same
     ├─ per-view A↔B dual-softmax
     ├─ multi-positive carrier supervision
     ├─ symmetry/lookalike hard negatives
     └─ reciprocal-aware training
          │
          ▼
D2 — SPATIAL FIX
     ├─ coarse semantic descriptor 32×32
     ├─ genuine fine descriptor 64×64
     └─ interpolation ≠ evidence
          │
          ▼
D3 — MATCHER FIX
     ├─ query-conditioned local correlation
     ├─ coarse→fine refinement
     ├─ bidirectional matching
     └─ calibrated uncertainty
          │
          ▼
D4 — BACKBONE
     only if still required
     DINO/foundation transfer
```

## D1 controlled-treatment contract

D1 changes **descriptor supervision only**.

Held invariant from the frozen N1D baseline:

- `IRISSEESN1` architecture and parameter schema;
- total parameter count: 6,195,085;
- descriptor head parameter count: 8,385;
- descriptor head structure: `Conv2d(128,64,1)` + uncertainty `Conv2d(128,1,1)`;
- 128px network input;
- calibrated canonical eight-view camera contract;
- N1D parameter-free correspondence transport;
- geometry, visibility, differential, R/F/D losses and their weights;
- optimizer family, LR groups, weight decay and gradient clipping;
- family-disjoint canonical N1D split;
- fit uses only `e00..e05`; `e06,e07` remain same-family intervention holdouts;
- sealed21 and external10 remain CLOSED.

D1 replaces only the descriptor term `Z_match`. Operationally:

```text
D1_total = legacy_observation_loss
           - 0.5 * legacy_Z_match
           + 0.5 * D1_Z_match
```

`D1_Z_match` is a preregistered equal-weight mean of four observation-level terms:

1. same-view symmetric/dual softmax identity loss;
2. multi-positive contrastive loss over visible observations without view pooling;
3. online hardest wrong-carrier margin loss in the same view;
4. same-view reciprocal soft-assignment cycle loss.

Aligned carrier correspondence from training sidecars is **training truth only** for this designated descriptor component. It is never a forward/inference input. This is explicitly permitted under the Model-1 contract's training/evaluation authority boundary and is a deliberate change from the older N1D mechanics preregistration, not a silent reuse of forbidden inference authority.

## D1 decision semantics

D1 is a development experiment, not sealed qualification. It must report at minimum:

- frozen D0 and trained D1 legacy pooled cross-pose top-1;
- same-view carrier top-1 at ground-truth projected sites;
- reciprocal same-view top-1;
- positive-vs-hardest-negative cosine margin;
- per-family distribution and worst-family values;
- mechanics non-regression diagnostics from the unchanged observation metrics.

No D2 spatial head, D3 matcher, D4 foundation backbone, G1/G2 retuning, sealed21, or external10 access is authorized by D1.

## Advancement rule

After D1 completes, inspect its evidence before authorizing D2. D2 is justified if the corrected objective materially improves identity discrimination but a meaningful residual remains consistent with the known 32x32-vs-native-pixel spatial bottleneck. If D1 fails to improve identity discrimination, diagnose supervision/optimization before changing spatial architecture.
