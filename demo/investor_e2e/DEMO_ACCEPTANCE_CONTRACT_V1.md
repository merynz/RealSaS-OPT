# RealSaS — Investor Demo Acceptance Contract V1

**Status:** `FROZEN_BEFORE_SPECIMEN_SELECTION__FROZEN_BEFORE_OPTIMIZER_STEP_1`  
**Branch scope only:** `demo/investor-single-specimen-e2e`

This contract defines what `fit`, `PASS`, and `end-to-end` mean for the single-specimen investor demo. It is intentionally a memorization/architecture proof, not a generalization claim.

## A. Global hard rules

A demo PASS requires all of the following:

1. IRIS, Geppetto, and Arachne are actual learned modules with saved/loaded parameter checkpoints.
2. Their source code is generic and frozen before specimen selection.
3. Fitting may consume teacher truth; final inference may not.
4. Final inference receives only the canonical eight RGBA observations plus the fixed canonical camera contract and model checkpoints.
5. No source-rig skeleton, source-rig weights, teacher projections, teacher mesh, specimen ID, manual correction, or lookup table is reachable by the final-inference process.
6. Geppetto must emit `SkeletonProposalIR`; final skeleton authority remains `CompilerFacade.qualify_skeleton`.
7. Arachne must emit `SkinProposalIR`; final skin authority remains `CompilerFacade.qualify_skin`.
8. No proposal is relabeled canonical without the existing Compiler qualification path.
9. All recorded metrics must be computed by versioned evaluator code, not manually transcribed.
10. A failed threshold is a FAIL. Thresholds are not relaxed after seeing the selected specimen.

## B. IRIS single-specimen fit PASS

IRIS preserves the current external product boundary: camera-forward depth + support/validity/uncertainty evidence; common-frame `P` is analytic from the fixed cameras.

Frozen demo-fit thresholds over authoritative visible/admitted target samples:

- finite output fraction: `1.000000`
- required target support recall: `1.000000`
- invented supported samples in teacher-UNKNOWN/unobserved regions: `0`
- normalized/world depth RMS: `<= 0.00250`
- absolute depth error P95: `<= 0.00539`
- all eight canonical views represented: `8 / 8`
- final emitted evidence must compile through the same `ObservationEvidenceIR -> RiggingSurfaceIR` deterministic geometry route.

The `.00250` RMS / `.00539` P95 limits intentionally bind the demo to the already established robust local-plane consumer tolerance neighborhood rather than inventing a looser investor-demo tolerance.

IRIS may overfit the specimen. It may not consume Geppetto/Arachne labels or source-rig identities as inference features.

## C. Geppetto single-specimen fit PASS

Teacher truth is the generic anonymous deform-control projection produced by the existing R6 teacher projection policy. The selected specimen envelope will require one projected deform root for this demo.

Evaluation first matches predicted proposal joints to teacher controls by minimum-distance bipartite matching in normalized object coordinates. Let `D` be the admitted subject 3D bounding-box diagonal.

Frozen thresholds:

- predicted active joint count == teacher deform-control count
- unmatched teacher controls: `0`
- unmatched active proposal joints: `0`
- matched joint `PCK@0.05D = 1.000000`
- matched joint RMS `<= 0.01000 D`
- matched root accuracy: `1.000000`
- matched directed parent-edge accuracy: `1.000000`
- duplicate active proposal positions within evaluator epsilon: `0`
- `CompilerFacade.qualify_skeleton(...)`: `PASS`
- qualified skeleton root count: exactly `1`
- manual joint/edge injection: `0`

The Compiler may mint different canonical IDs. Evaluation maps identities geometrically/structurally; source teacher IDs never become product IDs.

## D. Arachne single-specimen fit PASS

Teacher skin targets are constructed offline by a generic adapter from authoritative dense skin truth onto the admitted demo surface and the anonymous/qualified skeleton correspondence. Teacher weights are training/evaluation-only.

Frozen thresholds:

- admitted surface rows with a prediction: `100%`
- invalid/NaN/negative predicted weights before Compiler: `0`
- zero-sum predicted rows: `0`
- mean per-row L1 error to teacher: `<= 0.02000`
- per-row L1 error P95: `<= 0.05000`
- predicted simplex residual P95 before Compiler: `<= 0.02000`
- `CompilerFacade.qualify_skin(..., max_simplex_repair_l1=0.02)`: `PASS`
- missing qualified skin rows: `0`
- illegal joint references: `0`

A frozen generic deformation probe set is additionally required before architecture READY. On that probe set, after mesh binding:

- detached required surface/mesh components caused by skinning: `0`
- non-finite deformed positions: `0`
- normalized deformed-position RMS vs teacher deformation: `<= 0.01000 D`
- normalized deformed-position P95 vs teacher deformation: `<= 0.02500 D`

## E. Mesh / binding PASS

The demo cannot stop at skin rows; it must produce an editable deformation carrier.

Before specimen selection, a generic branch-local mesh candidate path must be implemented that:

- derives every rest vertex from admitted `RiggingSurfaceIR` support only;
- never introduces hidden/back-side geometry truth;
- records exact support bindings and lineage;
- passes existing mesh validation;
- transfers qualified Arachne skin through a typed `QualifiedMeshSkinIR` route;
- has no specimen-specific topology constants.

For the demo, a view-local triangulated deformation carrier is acceptable. It is not promoted to `main` MWB-2 authority by this branch.

## F. Final image-only E2E PASS

A fresh final-inference process must start after teacher/fitting objects are destroyed or made inaccessible.

Allowed inputs:

```text
- 8 canonical RGBA images
- fixed canonical camera contract
- frozen IRIS checkpoint
- frozen Geppetto checkpoint
- frozen Arachne checkpoint
- generic source/config code
- preset generic animation clip definition
```

Forbidden inputs:

```text
- specimen ID as a model feature or branch condition
- GT depth
- GT surface
- GT skeleton / bone arrays / parent arrays
- GT dense weights
- teacher projection
- source rig or source mesh
- manually supplied output coordinates/edges/weights
- cached fitted predictions masquerading as inference
```

Required outputs/closures:

- IRIS evidence PASS
- `RiggingSurfaceIR` constructed
- Geppetto `SkeletonProposalIR` emitted
- Compiler skeleton qualification PASS
- Arachne `SkinProposalIR` emitted
- Compiler skin qualification PASS
- typed mesh + mesh-skin qualification PASS
- `CanonicalPuppetGraph.v2` assembled
- generic deformation probes PASS
- at least one generic preset animation produces a finite frame sequence from the qualified product state
- all stage artifacts carry hashes and one run manifest binds checkpoints, inputs, outputs, code commit, metrics, and verdict.

## G. Claims permitted after PASS

Permitted:

> RealSaS has a genuine learned end-to-end single-specimen closure: images -> learned perception -> learned skeleton proposal -> learned skin proposal -> Compiler-qualified editable deformation state -> animation.

Not permitted:

- unseen-character generalization
- broad style/domain generalization
- production readiness
- anything-class coverage
- equivalence to a full production native runtime

Those remain separate research/product gates.
