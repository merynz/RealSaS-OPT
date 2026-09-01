# RealSaS — Full-Product Single-Specimen Fit Contract V1

**Date:** 2026-09-01  
**Status:** `FROZEN_FOR_INVESTOR_DEMO_IMPLEMENTATION`  
**Architecture authority:** `canonical/SYSTEM_ARCHITECTURE_V3_20260901.md` from architecture PR #19 / current product topology.  

## 0. Non-negotiable rule

The investor demo does **not** define a reduced RealSaS architecture.

The only intentional difference from the generic product program is the training distribution:

```text
GENERIC REALSAS ARCHITECTURE / INTERFACES / AUTHORITY / FUNCTIONAL OBLIGATIONS = UNCHANGED
TRAINING DISTRIBUTION = ONE SELECTED FULL-TRUTH SPECIMEN
GENERALIZATION CLAIM = FALSE
```

Single-specimen fitting may memorize the selected specimen in learned parameters. It may not move truth, specimen identity, handwritten geometry, handwritten skeleton, handwritten skin, or per-specimen control flow into product inference.

## 1. Final demo route

```text
8 native 1024 RGBA views + exact orthographic cameras
  -> IRIS Reprojection-Centered V2-A
  -> forward depth d + validity/support/uncertainty evidence
  -> ObservationEvidenceIR
  -> GeometricSubstrateAssembler
     -> analytic P = O + dF
     -> teacher-free persistence/support/provenance
     -> frozen operational N_d / local deterministic geometry
  -> RiggingSurfaceIR S

S
  -> GeppettoConditioningAdapter
  -> Geppetto R6 learned proposal model
     -> template-free variable control count
     -> continuous joint/control location evidence
     -> spatial uncertainty
     -> root evidence
     -> directed parent/edge evidence
     -> endogenous stop / unsupported behavior
  -> SkeletonProposalIR G*
  -> Compiler.rig_qualification
  -> QualifiedSkeletonIR G

S + G
  -> ArachneConditioningAdapter
  -> SkinFieldCodec / compact per-control skin-field representation
  -> Arachne R6 learned semantic influence-field proposal
     -> dense influence evidence
     -> uncertainty
     -> deformation-sensitive functional training path
  -> SkinProposalIR W*
  -> Compiler.skin_qualification
  -> QualifiedSkinIR W

S
  -> deterministic mesh/discretization candidate producer
  -> Compiler.geometry_qualification
  -> QualifiedEditableMeshIR M

M + W
  -> deterministic mesh-weight binder
  -> QualifiedMeshSkinIR B

S + G + W + M + B
  -> CanonicalPuppetGraph.v2 Y
  -> exact-state deformation/contact/motion probes
  -> ProofFrame
  -> PASS only -> RuntimePackageIR / animation render
```

## 2. IRIS obligations — unchanged from product V3

Final IRIS is **not** the earlier demo U-Net scaffold.

Required product-intent apparatus:

- 8 native 1024 RGBA observations and exact known orthographic cameras;
- frozen DINOv2-S/14 multilevel correspondence descriptor path;
- learned native high-resolution shared spatial pyramid;
- padded visual-hull candidate-domain constraint only;
- canonical world-space q lattice;
- exact q -> V0..V7 reprojection;
- per-view descriptor / learned-feature sampling;
- robust view-evidence aggregation preserving view validity;
- compact 3D evidence field;
- approximately isotropic world-space regularization;
- supported/ambiguous mode extraction;
- local continuous refinement;
- exact camera rendering to forward depth + validity/support + uncertainty;
- external learned geometry authority = forward depth d only;
- P remains analytic; N_d remains the frozen operational deterministic downstream geometry unless a separately versioned product contract changes it.

Forbidden even in the fitted demo:

- GT depth/surface as final-inference input;
- source mesh/triangle/barycentric identity as final-inference input;
- hidden back-surface completion as authority;
- skeleton/weights/part labels/product importance shaping IRIS inference;
- per-specimen lookup tables or branching.

## 3. Geppetto obligations — unchanged from product V3

Geppetto consumes only deterministic information legally derivable from admitted RiggingSurfaceIR S.

Required:

- template-free **variable** control count;
- no fixed product `max_joints` / `max_controls` class cap;
- continuous joint/control location evidence;
- spatial uncertainty;
- root evidence;
- directed parent/edge evidence;
- structural relation participates in the generated/contextual mechanical state, not only an evaluator-side auxiliary head;
- endogenous learned stop / unsupported behavior;
- anonymous proposal identity only;
- enough scored hypotheses for Compiler global graph qualification;
- token/input ordering must not become product identity;
- Compiler owns final root/tree/canonical `J:*` IDs.

A runtime safety budget may exist solely to prevent infinite generation. It must be derived from current input/evidence size or a general resource bound and may not define product cardinality.

The historical 48/64/72-query experiment controls are **not** product architecture authority.

## 4. Arachne obligations — unchanged from product V3

Arachne consumes `RiggingSurfaceIR S + QualifiedSkeletonIR G` through a deterministic conditioning adapter.

Required:

- dynamic surface count and dynamic qualified-joint count;
- point/surface contextual encoder;
- skeleton/tree contextual encoder;
- explicit point-control / bone-segment geometry;
- cross-entity attention or an equivalently expressive interaction mechanism;
- per-control compact skin-field representation (`SkinFieldCodec` family);
- codec reconstruction ceiling before relying on predicted latents;
- dense decoded influence proposal over current legal controls;
- uncertainty evidence;
- training objective includes dense skin reconstruction **and** deformation-sensitive functional error under generic probes;
- no source bone tail / source rig ID / teacher column identity as product inference input;
- no model-owned final simplex/legal-reference authority;
- `SkinProposalIR -> Compiler.qualify_skin` remains mandatory.

Compiler-side sparsification/repair policy is not Arachne model architecture and may not be used to hide a weak predictor.

## 5. Mesh / product / proof obligations — unchanged

- mesh is derived only from admitted S support;
- no hidden source mesh is copied into final product inference;
- QualifiedEditableMeshIR and QualifiedMeshSkinIR remain typed Compiler products;
- CanonicalPuppetGraph.v2 is the single canonical mesh-aware product truth;
- exact state hash binds S/G/W/M/B;
- deformation/contact/motion proof is bound to that exact state;
- stale proof is invalid;
- runtime/export is non-owning and requires a passing proof;
- historical ARAP/XPBD/runtime authority may be restored/promoted/bridged without changing canonical ownership.

## 6. Single-specimen fitting privileges

Allowed **only during training/evaluation**:

- exact rendered depth/support geometry truth;
- anonymous skeleton teacher projection from M4/R6 corpus authority;
- dense skin truth from M5/Master corpus authority;
- source mesh/rig metadata needed to construct teacher targets;
- teacher-forcing, assignment/matching and codec encoding;
- repeated optimization until the one specimen is memorized;
- deterministic image augmentations that preserve exact target transforms;
- CPU or GPU fitting.

All such information is barred by the final inference firewall.

## 7. Final inference firewall

Fresh-process final run may read only:

- canonical 8 RGBA observations;
- exact camera authorities;
- frozen IRIS checkpoint;
- frozen Geppetto checkpoint;
- frozen SkinFieldCodec/Arachne checkpoint(s);
- current Compiler/runtime code and declared numerical authorities;
- generic configuration.

It may not read:

- specimen ID as a prediction key;
- mechanical truth bundle;
- source skeleton/parents;
- source skin matrix;
- source mesh as hidden product geometry;
- manual patch files;
- per-specimen threshold overrides;
- hard-coded joint coordinates/parent edges/weights.

## 8. Specimen-selection gate

**SPECIMEN_SELECTION_FORBIDDEN** until all of the following are true:

1. IRIS full-product fitted architecture is source-complete and zero-step executable;
2. Geppetto full-product fitted architecture is variable-cardinality and zero-step executable;
3. SkinFieldCodec + Arachne full-product fitted architecture is source-complete and zero-step executable;
4. Compiler typed seams are wired;
5. historical/current proof-runtime route is wired;
6. synthetic capacity tests prove each learned architecture can drive its exact generic output contract;
7. source scan confirms no specimen IDs/constants/known joint count/product-truth inference path.

Only then may candidates be drawn from M4/M5/Master.

## 9. Specimen-selection requirements after architecture closure

Selection is not random. Candidate must be visually inspected and machine-audited for:

- exact 8-view / renderable observation authority;
- full IRIS geometry truth;
- valid anonymous deform-control projection / hierarchy truth;
- dense valid skin truth aligned to those controls;
- clean single subject and scene extraction;
- no helper/background/alternate-geometry leakage;
- no pathological zero rows or unrelated meshes;
- a visually presentable stylized/product-adjacent character when available;
- manageable but not architecture-defining mechanical complexity;
- clear licensing/provenance.

A selected specimen binds **data only**. Selection may not cause model source changes.

## 10. Demo claim

Permitted claim:

> `Single-specimen fitted end-to-end RealSaS product-architecture closure.`

Not permitted:

> `Generic/unseen-character generalization is solved.`

The demo must show the real chain from image-only observations to an editable, deformable, animated, proof-bound product artifact.