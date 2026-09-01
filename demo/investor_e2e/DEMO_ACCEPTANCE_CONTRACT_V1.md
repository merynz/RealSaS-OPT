# RealSaS — Investor Demo Acceptance Contract V1

**Status:** `FROZEN_BEFORE_SPECIMEN_SELECTION__FROZEN_BEFORE_OPTIMIZER_STEP_1`  
**Branch scope only:** `demo/investor-single-specimen-e2e`

This contract defines what `fit`, `PASS`, and `end-to-end` mean for the single-specimen investor demo.

The demo uses the **full current RealSaS product architecture**. The only intentional change from the generic program is the optimization distribution: one subsequently selected full-truth specimen. Generalization is not claimed.

## A. Global hard rules

A demo PASS requires all of the following:

1. IRIS, Geppetto and Arachne are real learned modules with saved/loaded checkpoints.
2. Model/inference source is specimen-agnostic and frozen before specimen selection.
3. Fitting/evaluation may consume teacher truth; fresh final inference may not.
4. Final inference receives only the canonical eight RGBA observations, exact camera contract, frozen checkpoints and generic code/configuration.
5. No source-rig skeleton, source-rig weights, teacher projection, teacher mesh, specimen ID, manual correction, lookup table or cached fitted prediction is reachable by final inference.
6. Geppetto emits `SkeletonProposalIR`; Compiler owns final admissibility, root/tree choice and canonical `J:*` IDs.
7. Arachne emits `SkinProposalIR`; Compiler owns legal references, simplex policy, bounded repair and final qualified skin.
8. `CanonicalPuppetGraph.v2` is the final canonical product state.
9. Exact-state proof/runtime consumes only that qualified state.
10. Metrics are computed by versioned evaluator code. Thresholds may not be relaxed after specimen selection.

## B. IRIS single-specimen fit PASS

IRIS preserves the current external product boundary: camera-forward depth + support/validity/uncertainty evidence; common-frame `P` is analytic from the exact cameras.

Hard visible/admitted geometry gates:

- finite output fraction: `1.000000`
- required target support recall: `1.000000`
- invented supported samples in authoritative UNKNOWN/unobserved regions: `0`
- normalized/world depth RMS: `<= 0.00250`
- absolute depth error P95: `<= 0.00539`
- all eight canonical views represented: `8 / 8`
- emitted evidence compiles through the normal `ObservationEvidenceIR -> RiggingSurfaceIR` deterministic route
- no skeleton/skin/product-importance truth is an IRIS inference feature

The `.00250` RMS neighborhood binds the fitted demo to the already measured downstream clean/robust consumer regime rather than inventing a demo-only geometry tolerance.

IRIS is allowed to memorize this specimen in parameters. It is not allowed to receive the answer as an inference input.

## C. Geppetto single-specimen fit PASS

### C.1 Teacher role

The existing anonymous R6 deform-control projection is a **training/evaluation reference**, not the product skeleton definition.

Teacher-exact count, source topology, source bone identity and source helper layout are **not hard product targets**.

The fitted model may happen to converge very close to the clean projected teacher skeleton. That is acceptable and useful, but the architecture must not require it.

Teacher diagnostics to report, not use as sole PASS authority:

- matched-locus PCK/RMS under anonymous geometric matching;
- predicted-vs-projected control-count difference;
- root agreement when a meaningful anonymous mapping exists;
- directed parent agreement when the two structures are directly comparable.

### C.2 Hard product skeleton gates

Geppetto inference must determine cardinality itself; oracle K is forbidden.

`Compiler.qualify_skeleton(...)` must PASS and the resulting `QualifiedSkeletonIR` must satisfy:

- finite joint positions: `100%`
- exactly one Compiler-qualified root for the selected demo product
- connected, acyclic qualified hierarchy
- missing/illegal parent references: `0`
- duplicate canonical joint positions within evaluator epsilon: `0`
- unsupported accepted joints: `0`
- every accepted joint has admitted `RiggingSurfaceIR` support evidence
- all accepted bone segments are finite and non-degenerate under the frozen geometry epsilon
- no accepted joint/bone lies outside the frozen conservative rigging-volume/support envelope beyond the declared geometric tolerance
- learned stop/endogenous count is used at inference; no teacher count is supplied
- manual joint/edge injection: `0`

### C.3 Functional skeleton gate

The qualified skeleton proceeds to fitted Arachne skin and the frozen generic deformation/probe bank.

A skeleton is not accepted merely because its joints are geometrically plausible. The final coupled `G + W` system must satisfy section D's deformation gates and the exact downstream proof route.

This makes the product target:

> a clean, geometrically supported, mechanically usable skeleton,

not:

> byte-for-byte reconstruction of the authored teacher rig.

## D. SkinFieldCodec + Arachne single-specimen fit PASS

### D.1 Teacher role

Dense authoritative skin truth is training/evaluation authority. When predicted and teacher control spaces admit a direct anonymous correspondence, weight-space errors are reported. They are not the sole product criterion when the product skeleton is a cleaner/different valid structure.

### D.2 SkinFieldCodec gate

Before accepting the Arachne predictor:

- codec output finite: `100%`
- decoded nonnegative field evidence: `100%`
- all admitted surface rows reconstructed/present: `100%`
- selected-specimen codec reconstruction must meet the frozen weight/deformation ceiling established by the generic pre-specimen codec tests
- codec encoder/teacher dense weights are absent from final inference

### D.3 Arachne / Compiler hard gates

For the predicted `SkinProposalIR`:

- admitted surface rows with prediction: `100%`
- NaN/Inf/negative predicted influences: `0`
- zero-sum required rows: `0`
- illegal qualified-joint references: `0`
- `Compiler.qualify_skin(...)`: `PASS`
- missing qualified skin rows: `0`
- Compiler repair remains within its frozen bounded correction policy

When direct teacher-control correspondence exists, report at minimum mean/P95 row L1 and dominant-control agreement as diagnostics.

### D.4 Coupled deformation hard gate

The **primary Arachne/product criterion is functional deformation** of the complete qualified product state.

On the frozen generic multi-probe bank:

- detached required surface/mesh components caused by skinning: `0`
- non-finite deformed positions: `0`
- exploded/collapsed required components: `0`
- illegal transform propagation: `0`
- skin simplex/reference violations after qualification: `0`
- all required probe families execute
- exact downstream proof has no unsafe accepted state

Where a teacher deformation response can be transported into the product control space without importing teacher identity into inference, additionally require the frozen deformation-response RMS/P95 ceiling. If direct transport is not mathematically meaningful because the valid product skeleton differs, the exact-state Compiler/proof gates remain authoritative and the non-comparability is reported rather than forcing teacher topology back into the product.

## E. Mesh / binding PASS

The demo cannot stop at skin rows; it must produce an editable deformation carrier.

The mesh path must:

- derive every rest vertex from admitted `RiggingSurfaceIR` support only;
- introduce no hidden/back-side source geometry authority;
- record support bindings and lineage;
- pass existing mesh validation;
- transfer qualified Arachne skin through typed `QualifiedMeshSkinIR`;
- contain no specimen-specific topology constants.

A bounded view-local/support-bound triangulated deformation carrier is acceptable for this branch if it satisfies the current typed product contract and exact downstream proof. It is not automatically promoted as generic MWB-2 authority on main.

## F. Final image-only E2E PASS

A fresh final-inference process starts with training/evaluator truth absent or unreachable.

Allowed inputs:

```text
- 8 canonical 1024 RGBA observations
- exact orthographic camera authorities
- frozen IRIS checkpoint
- frozen Geppetto checkpoint
- frozen SkinFieldCodec/Arachne checkpoints
- generic source/configuration
- generic preset motion/probe definition
```

Forbidden inputs:

```text
- specimen ID as prediction key
- GT depth/surface
- GT skeleton / parent arrays
- GT dense skin
- teacher projection
- source rig
- source mesh as hidden final geometry
- manually supplied joints/edges/weights
- cached fitted predictions masquerading as inference
- per-specimen inference thresholds
```

Required closure:

```text
8 images
 -> IRIS
 -> ObservationEvidenceIR
 -> GeometricSubstrateAssembler / RiggingSurfaceIR
 -> Geppetto
 -> SkeletonProposalIR
 -> Compiler skeleton qualification
 -> QualifiedSkeletonIR
 -> SkinFieldCodec + Arachne
 -> SkinProposalIR
 -> Compiler skin qualification
 -> QualifiedSkinIR
 -> typed mesh + mesh-skin
 -> CanonicalPuppetGraph.v2
 -> exact deformation/contact/motion probes
 -> PASSING exact-state proof
 -> RuntimePackageIR / finite animation frame sequence
```

One run manifest binds input hashes, checkpoint hashes, code commit/tree, every typed stage hash, proof hash, metrics and final verdict.

## G. Claims permitted after PASS

Permitted:

> RealSaS has a genuine learned end-to-end single-specimen fitted closure: images -> learned observation-grounded geometry -> learned skeleton proposal -> learned skin proposal -> Compiler-qualified editable/deformable product -> proof-bound animation.

Also permitted, if visually demonstrated:

> The same product architecture intended for generic RealSaS has been fitted end-to-end on one specimen.

Not permitted:

- unseen-character generalization
- broad style/domain generalization
- anything-class coverage
- full production readiness

Those remain later research/product gates.