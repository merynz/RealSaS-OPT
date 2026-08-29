# RealSaS-OPT — Current State

**Date:** 2026-08-29  
**Canonical continuation branch:** `main`  
**Status:** `E0_PASS__N_B3_PASS_NO_EXPLICIT_N__COMPILER_IN_SYSTEM_LOOP__DTB_ND1_CLOSED__FIT_PROXY32_AUDIT_CLOSED__CONSUMER_INTERLOCK_6_OF_6_PASS__DINO_CONTROLLED_PREREG_SEALED__ZERO_STEP_PREFLIGHT_NEXT__TRAINING_NOT_AUTHORIZED__DEV32_CLOSED`

This file is continuation authority only when present on `main`. On a non-main branch it is a proposed next canonical state and does not override the current `main` authority until that branch is explicitly promoted/merged.

Plain-language roadmap:
`canonical/HUMAN_READABLE_ROADMAP_20260829.md`.

## One-line state

`depth tolerance measured -> old learner failure decomposed -> real downstream consumer route + coupling validated -> controlled DINOv2 S/B/L/g experiment preregistered -> zero-step apparatus/data/weight preflight next -> only then training`

## 1. Product/system contract — STABLE

North star:

`ONE NEUTRAL 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

Current typed route:

```text
8 ordered neutral views + known orthographic cameras
 -> IRIS learned forward depth d
 -> analytic P = O + dF
 -> deterministic SurfaceBuilder / RiggingSurfaceIR S
 -> Geppetto SkeletonProposalIR G*
 -> Compiler graph qualification / QualifiedSkeletonIR G
 -> Arachne SkinProposalIR W*
 -> Compiler skin qualification / QualifiedSkinIR W
 -> CanonicalPuppetGraph Y
 -> deformation/motion proof -> attribution -> bounded repair -> mandatory re-proof
 -> proven runtime projection
```

Authority invariants:

- IRIS/Geppetto/Arachne emit evidence/proposals, not product truth;
- Compiler owns canonical joint IDs and product state;
- there is one skeleton/topology authority: `QualifiedSkeletonIR` materialized through the existing canonical graph optimizer;
- optimizer candidate/work objects are not product graphs;
- `CanonicalPuppetGraph` references admitted surface/skeleton/skin lineages and is the single product-state authority;
- proof and runtime bind to exact `product_state_hash`;
- stale bindings fail closed.

Compiler/runtime restoration remains closed for current execution. Exact historical vendor raw SHA-256:

`3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`

## 2. Geometry requirement — CLOSED FOR THE TESTED CONSUMER PROFILE

The structured predicted-depth bridge and refinement are closed.

Historical frozen-D2 ALL8 boundary before robust normal derivation:

`0.00225 <= epsilon_critical < 0.00250 RMS`

After the preregistered robust local-plane deterministic normal derivation (`DTB-ND1`):

`0.00250 <= epsilon_critical < 0.00275 RMS`

For the matched ell=0 Gaussian-like bridge this corresponds approximately to depth absolute-P95:

`0.00490–0.00539`

Clean non-inferiority passed 6/6.

The plane operator does not move/smooth `P`; it only derives `N_d`. Its supported mechanism is changed normal-gated correspondence/admission and therefore changed downstream evidence topology.

Canonical closure:
`canonical/DTB_ND1_ROBUST_LOCAL_PLANE_CLOSURE_20260829.md`.

### Stop rule

Do not open another cheap deterministic geometry intervention before the representation test. In particular do not tune plane window, MAD threshold, minimum neighbors, hull dilation, persistence thresholds, or similar operators from candidate outcomes.

## 3. FIT_PROXY32 coordinate/residual audit — CLOSED

Canonical closure:
`canonical/FIT_PROXY32_COORDINATE_AND_RESIDUAL_AUDIT_20260829.md`.

Status:
`CLOSED__CONTROLLED_CAPACITY_LADDER_WARRANTED__PRODUCT_SELECTION_NOT_AUTHORIZED`

Key facts:

- actual real witness: `asset_ea593d044e14f20abe6d2818`;
- 8 views x 4096 sample loci;
- screen-plane mismatch P95: `6.386590393958613e-05`;
- screen-plane mismatch max: `0.00012190263805678114`;
- relevant depth-tolerance absolute-P95 lower scale: about `0.00490`;
- therefore the historical P metric is depth-dominated at the relevant scale and is useful as a residual-scale diagnostic.

Historical small-fit evidence:

- P-V5 R256 8x2 fit aggregate P-P95: `0.003706276847515254`;
- worst cell: `0.005740759451873588`.

Historical family ladder was confounded by exposure:

- 32 families: nominal family exposure 1792; train median `.00859`; FIT_PROXY32 median `.17066`;
- 128 families: exposure 448; train `.01819`; FIT `.09814`;
- 512 families: exposure 112; train `.03138`; FIT `.06593`.

At the current upper matched tolerance around `.00539`, rung512 had `0/32` train diagnostic families and `0/32` FIT_PROXY32 families at/below tolerance.

Supported interpretation:

1. global fitting/precision deficit already exists on training families;
2. an additional unseen-family gap exists, especially on Objaverse;
3. family/source tails matter strongly;
4. increasing family diversity improved transfer while fixed total optimizer steps reduced per-family exposure;
5. the old 32 -> 128 -> 512 experiment is not a controlled capacity ladder;
6. failure must not be described as pure unseen-family generalization collapse.

Consequence: a controlled pretrained representation-capacity/accessibility test is warranted.

## 4. Consumer-validity interlock — CLOSED 6/6 FOR SACRIFICIAL G0/A0

Original interlock:
`experiments/g0_g1_single_pose_geometry/depth_tolerance_bridge_20260829/CONSUMER_VALIDITY_INTERLOCK_V1.md`.

Canonical closure:
`canonical/CONSUMER_VALIDITY_INTERLOCK_CLOSURE_20260829.md`.

The six required steps are closed for the sacrificial profile:

```text
ClosedRiggingVolume V0
 -> InteriorRiggingSubstrate V0
 -> G0
 -> exact Compiler skeleton qualification
 -> A0
 -> exact Compiler skin qualification
 -> deformation/proof
 -> explicit G0/A0 coupling probe
```

Clean witness `asset_551ea351b43a1787d0f55536` passed exact skeleton qualification, 512/512 skin qualification, finite/nontrivial/bounded deformation, exact-state proof/runtime binding, and canonical-ID firewalls.

The explicit preregistered coupling probe collapsed all non-root G0 joints 75% toward root. Corruption RMS / bbox diagonal was `0.21196194321606632`. After A0 recomputed weights, posed-surface difference remained visible: normalized RMSE `0.015168392024714485`, P95 `0.02837774072761807`, diagnostic compensation ratio `1.3976998717340667`. Both frozen detectability gates passed.

Interpretation: the downstream route genuinely consumes skeleton geometry; this does not make sacrificial G0/A0 product-quality models.

Product-level depth tolerance remains consumer-profile specific. Replacing/retraining Geppetto or Arachne requires replay under the new profile before previous rankings transfer.

## 5. Controlled DINOv2 frozen-representation ladder — PREREGISTERED

Scientific preregistration:
`experiments/iris_dino_controlled_20260829/DINO_CONTROLLED_REPRESENTATION_LADDER_PREREG_V1.md`.

Binding ambiguity closure:
`experiments/iris_dino_controlled_20260829/DINO_CONTROLLED_REPRESENTATION_LADDER_PREREG_AMENDMENT_V1_1.md`.

Machine-readable contract:
`experiments/iris_dino_controlled_20260829/DINO_CONTROLLED_REPRESENTATION_LADDER_PREREG_V1.json`.

Plain-language question:

> With everything after the frozen visual representation held constant, does stronger DINOv2 representation make the required camera-forward depth field accessible at the precision RealSaS currently needs?

Candidates:

- DINOv2 S / 384-d;
- DINOv2 B / 768-d;
- DINOv2 L / 1024-d;
- DINOv2 g / 1536-d.

Key controls now frozen before candidate outputs:

- one exact historical 512-FIT-family population for all four candidates;
- both frozen styles for every family;
- exact same deterministic sample/order stream;
- Mode-G 8 x native-1024 exact-camera observations;
- full-canvas deterministic 518 resize -> 37 x 37 patch grid;
- identical native-1024 detail/support path;
- fixed isometric zero-pad lift `Q_d : R^d -> R^1536`;
- identical non-affine post-lift normalization;
- identical trainable fusion/head bytes/config/parameter count;
- IRIS learns camera-forward depth `d` only;
- no learned normal, risk or camera head in Phase 1A;
- no constant `kappa_common * D_hull` ranking path;
- same optimizer protocol and no candidate-specific stopping/tuning.

Historical 512 membership authority is frozen by:

- source metadata SHA `475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`;
- seed `PV5_R256_FIT_SCALE_TRAIN_V1`;
- quota `483 / 24 / 5` Objaverse/Quaternius/KayKit;
- complete canonical membership JSON SHA `8531360f1c61dc4cdb699c095790c35ab3af0da5a7a290b420e5777bb4e9169b`.

No missing family may be substituted.

### Fixed exposure budget

Mandatory checkpoints:

- `MATCHED_7168`: historical-budget diagnostic, 112 nominal family exposures;
- `PRIMARY_32768`: primary comparison, 512 nominal family exposures.

The 32768-step checkpoint is not claimed asymptotic. It is a fixed, substantially less-starved comparison chosen before candidate outputs; every candidate receives exactly the same budget.

### Frozen PASS operator

At `PRIMARY_32768`, `REPRESENTATION_ACCESSIBLE_V1` requires:

- FIT_PROXY32 actual-residual direct-replay PASS on at least `61/64` family-style cells;
- FIT_PROXY32 hard route failures `0`;
- TRAIN_DIAG32 actual-residual direct-replay PASS on at least `61/64` cells;
- TRAIN_DIAG32 hard route failures `0`.

Scalar depth statistics remain diagnostics and cannot override direct replay.

If all four candidates fail training-fit adequacy, the conclusion is shared learner/optimization adequacy not established at the fixed budget—not capacity falsification.

## 6. Zero-step apparatus/data/weight preflight — NEXT / CURRENT BLOCKER

Status object:
`experiments/iris_dino_controlled_20260829/DINO_LADDER_ZERO_STEP_PREFLIGHT_STATUS_V1.json`.

Scientific optimizer steps remain exactly `0`.

Already verified at planning level:

- native-1024 master corpus exists;
- historical 512 selection rule/quota/membership SHA are known;
- FIT_PROXY32 was frozen disjoint from the historical 512 train set;
- ObservationContract 518/patch14 framing exists;
- S/B/L/g token widths are fixed;
- Phase-1A risk calibration has been removed as a confound.

Hard blockers before training can open:

1. recover/reconstruct the exact historical membership object and verify canonical membership SHA;
2. verify all exact train512/FIT_PROXY32 members map to native-1024 authoritative assets with no substitution;
3. seal official DINO source revisions/model IDs and exact weight SHA-256 values;
4. seal the shared trainable architecture/config/code and prove equal parameter count;
5. seal one deterministic sample/order manifest;
6. pass cached-vs-online frozen-token parity;
7. prove the frozen DTB-ND1 evaluator can consume actual candidate residual fields without semantic reinterpretation;
8. verify closed sets remain unopened.

The old runner log establishes the historical membership filename was `P_V5_R256_FIT_SCALE_LADDER_MEMBERSHIP_V1.json`, but the exact object itself has not yet been recovered from persistent authority. Re-selecting a fresh 512-set with the same source quota is forbidden.

## 7. Training authorization

**FULL DINO S/B/L/g TRAINING: NOT AUTHORIZED.**

Next executable work is zero-step preflight/recovery only. Candidate training opens only after every preflight blocker above is sealed and the preregistration state is canonical on `main`.

DEV32 remains closed unless a separately preregistered and authorized gate explicitly opens it.
