# RealSaS-OPT — Current State

**Date:** 2026-08-29  
**Canonical continuation branch:** `main`  
**Status:** `E0_PASS__N_B3_PASS_NO_EXPLICIT_N__COMPILER_IN_SYSTEM_LOOP__DTB_ND1_CLOSED__FIT_PROXY32_AUDIT_CLOSED__CONSUMER_INTERLOCK_6_OF_6_PASS__DINO_CONTROLLED_PREREG_NEXT__TRAINING_NOT_AUTHORIZED__DEV32_CLOSED`

This file is continuation authority only when present on `main`. On a non-main branch it is a proposed next canonical state and does not override the current `main` authority until that branch is explicitly promoted/merged.

Plain-language roadmap:
`canonical/HUMAN_READABLE_ROADMAP_20260829.md`.

## One-line state

`depth tolerance measured -> old learner failure decomposed -> real downstream consumer route + coupling validated -> controlled DINOv2 S/B/L/g preregistration next -> only then training`

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

Consequence: a controlled pretrained representation-capacity test is warranted.

## 4. Consumer-validity interlock — CLOSED 6/6 FOR SACRIFICIAL G0/A0

Original interlock:
`experiments/g0_g1_single_pose_geometry/depth_tolerance_bridge_20260829/CONSUMER_VALIDITY_INTERLOCK_V1.md`.

Canonical closure:
`canonical/CONSUMER_VALIDITY_INTERLOCK_CLOSURE_20260829.md`.

The six required steps are now closed for the sacrificial profile:

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

### Clean exact-Compiler witness

Asset:
`asset_551ea351b43a1787d0f55536`

Results:

- surface nodes: `512`;
- G0 joints / edges: `5 / 4`;
- qualified joints: `5 / 5`;
- graph solver: `chu_liu_edmonds_maximum_spanning_arborescence`;
- optimizer status: `optimal_arborescence_super_root`;
- optimality proven: `true`;
- proposal IDs disjoint from product canonical IDs: `true`;
- all product IDs minted `J:*`: `true`;
- qualified skin rows: `512 / 512`;
- skin max simplex residual before bounded repair: `1.341104507446289e-07`;
- deformation finite / nontrivial / bounded: PASS;
- proof bound to exact product state: PASS;
- runtime bound to exact product state: PASS.

Clean result:
`experiments/consumer_interlock_20260829/CONSUMER_INTERLOCK_EXACT_COMPILER_CLEAN_RESULT_V1.json`.

### Explicit coupling probe

Preregistered before outcome:
`experiments/consumer_interlock_20260829/CONSUMER_COUPLING_PROBE_PREREG_V1.md`.

Corruption:

`p_bad = root + 0.25 * (p_clean - root)` for every non-root G0 joint.

Measured corruption RMS normalized by surface bbox diagonal:

`0.21196194321606632`

After A0 was allowed to recompute its weights on the corrupted skeleton:

- posed-surface RMSE / bbox diagonal: `0.015168392024714485`;
- posed-surface P95 point delta / bbox diagonal: `0.02837774072761807`;
- mean A0 row-L1 weight change: `0.2687926005329492`;
- diagnostic compensation ratio: `1.3976998717340667`.

Both preregistered response-detectability gates passed. The coupling proof is bound to exact corrupted product state.

Result:
`experiments/consumer_interlock_20260829/CONSUMER_INTERLOCK_COUPLING_PROBE_RESULT_V1.json`.

Interpretation:

The real downstream route can consume the current RealSaS 2.5D substrate, and A0 does not silently erase a severe systematic G0 skeleton error on this sacrificial witness. This validates the route, not product-quality Geppetto/Arachne.

Product-level depth tolerance remains consumer-profile specific. Replacing/retraining Geppetto or Arachne requires replay under the new profile before previous rankings transfer.

## 5. Next scientific gate — CONTROLLED DINOv2 REPRESENTATION LADDER PREREGISTRATION

Do **not** start candidate training yet.

Next executable action is to write and seal the controlled frozen-representation ladder:

- DINOv2 S / 384-d;
- DINOv2 B / 768-d;
- DINOv2 L / 1024-d;
- DINOv2 g / 1536-d.

The experiment must isolate representation accessibility/capacity rather than recreate the historical family-exposure confound.

Required controls:

- exact same scientific training-family population across S/B/L/g;
- exact same sample/order stream and augmentations;
- foundation raster `518 x 518`, patch-14 grid `37 x 37`;
- same native `1024 x 1024` detail path;
- no adaptive crop/zoom or rung-specific framing;
- fixed isometric lift `Q_k` into 1536-d;
- identical non-affine post-lift normalization;
- identical trainable fusion/head architecture;
- identical trainable parameter count after the frozen extractor;
- matched optimizer/exposure policy fixed before scientific candidate outcomes;
- no candidate-specific stopping/tuning from outcome inspection;
- candidate interpretation based on actual residual fields replayed through the frozen downstream consumer/proof route, not summary depth metrics alone.

Prospective V-next design remains under:
`prospective/iris_vnext_20260829/IRIS_VNEXT_EXPERIMENT_PLAN_V4.md`.

Important prospective amendment remains binding for future preregistration: do not blindly use the spatially constant `kappa_common * D_hull` common risk band. Replace it with a candidate-independent observable-conditioned band with non-degeneracy checks, calibrate cross-view inconsistency on consistent and contradictory observation sets, and retain camera perturbation as non-binding sensitivity unless separately promoted.

## 6. Training authorization

**FULL DINO S/B/L/g TRAINING: NOT AUTHORIZED YET.**

Authorization sequence:

1. consumer closure promoted to `main` through normal branch governance;
2. current-state/roadmap synchronization present on `main`;
3. controlled DINO ladder preregistration written and sealed;
4. only then execute candidate training/evaluation.

DEV32 remains closed unless a separately authorized gate explicitly opens it.
