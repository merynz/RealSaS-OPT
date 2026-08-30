# RealSaS Geppetto G0.1 / Arachne A0.1 Architecture

**Date:** 2026-08-30  
**Status:** `FREEZE_CANDIDATE__NOT_SEALED__NOT_PRODUCT_QUALIFIED`

V0 is closed as a discussion draft. V0.1 is the implementation freeze-candidate. Learned consumers propose; Compiler qualifies and owns canonical product state.

## Canonical route

```text
IRIS d
 -> analytic P = O + dF
 -> deterministic SurfaceBuilder / persistence
 -> RiggingSurfaceIR S
 -> Geppetto G0.1 -> SkeletonProposalIR G*
 -> Compiler -> QualifiedSkeletonIR G
 -> Arachne A0.1 -> SkinProposalIR W*
 -> Compiler -> QualifiedSkinIR W
 -> CanonicalPuppetGraph
 -> deformation / proof / bounded repair / mandatory re-proof
```

Invariants:

1. proposal IDs never become product-canonical IDs;
2. Compiler owns root/tree qualification and canonical ID minting;
3. Arachne consumes `QualifiedSkeletonIR`, never raw Geppetto topology as authority;
4. source bone indices/names/tails/rest matrices/skin columns are teacher/evaluator provenance only;
5. current IRIS learned authority is forward depth `d`; learned risk/normal/camera fields are not mandatory consumer inputs;
6. C0 is trained on clean exact-observable substrate; C1 may later use a prospectively frozen task-agnostic geometry corruption generator;
7. projected deformation/proof is product authority; teacher parameter closeness is auxiliary.

## Parallel development before IRIS completion

The master corpus already supplies exact geometry and downstream teacher fields. A clean observation-equivalent substrate can therefore be constructed without learned IRIS predictions:

```text
exact geometry/renders
 -> exact d/P-equivalent observable substrate
 -> same deterministic SurfaceBuilder / volume operators
 -> Geppetto/Arachne training and qualification
```

Real IRIS residuals are replayed later through the frozen consumer profile. Candidate-specific downstream retuning after opening IRIS outcomes is forbidden.

## SkeletonTeacherProjectionV1

Teacher arrays are `bone_heads[J,3]`, `bone_tails[J,3]`, `parents[J]`, `deform_mask[J]`.

Projection:
- one anonymous G0 control per deform source bone;
- control position = source bone head;
- helper/non-deform bones are skipped to the nearest deform ancestor;
- multiple deform roots are preserved; no synthetic super-root is invented;
- source bone index/tail remain teacher-only provenance;
- source skin-column transport remains explicit and auditable;
- deterministic BFS serialization is authority; sibling randomization is training-only and may not change depth/ancestry.

## Typed substrate tokens

Surface and interior samples are distinct token types with explicit validity bits. V0.1 token width is 27D:

```text
position xyz                         3
token type one-hot                   2
normal xyz                           3
normal_valid                         1
8-view support mask                  8
geometry uncertainty                 1
geometry_uncertainty_valid           1
medialness                           1
local thickness                      1
branch likelihood                    1
volume-state one-hot                 5
                                    --
                                    27
```

Geometry uncertainty is optional/versioned. Missing IRIS risk is represented by validity=0, never authoritative zero error.

Typed volume states are exactly:
`CERTAIN_OUTSIDE`, `SUPPORTED_SURFACE_BAND`, `POSSIBLE_INTERIOR`, `HIGH_CONFIDENCE_INTERIOR`, `UNKNOWN_CONCAVITY`.
Interior sampling may never admit outside or supported-surface voxels.

## Geppetto G0.1

Input is a typed set of production-available surface + interior tokens. Forbidden inference inputs include source bone/joint IDs/names, source parent labels, source tails/rest matrices, skin/weights, and product canonical joint IDs.

The encoder may use deterministic common-frame geometry: absolute position, relative deltas/distances and local KNN neighborhoods. The earlier V0 prohibition against positional evidence beyond raw P is withdrawn; spatial relation is required, while hidden teacher identity remains forbidden.

The first decoder treatment is autoregressive anonymous-control generation. Each proposal control emits:
- position xyz;
- position uncertainty;
- exist/continue evidence;
- root evidence;
- proposal confidence.

After control generation, G0.1 emits full directed parent evidence for every legal parent/child pair. It does not select the final root/tree. The proposal adapter emits the existing `SkeletonProposalJoint`, `SkeletonProposalEdge`, `SkeletonProposalIR`; Compiler performs global graph qualification and independently mints `J:*` IDs.

BFS serialization is an optimization/representation treatment, not product authority. Historical M4 failure is not claimed to prove that Hungarian/set decoding must fail.

## Arachne A0.1

A0.1 input authority is `RiggingSurfaceIR / RefinedRiggingVolume evidence + QualifiedSkeletonIR`.

Production-available point/control features may include point-to-qualified-segment distance, normalized coordinate along the qualified parent->child segment, normal-vs-segment orientation with an explicit validity bit, and deterministic point->control volume path evidence. Root controls have no parent segment; segment-dependent terms must be invalid rather than inventing an axis from teacher tails.

The point->control segment carries fractions and lengths in all five typed volume states plus total path length. This supersedes the V0 single outside-crossing boolean.

V0's SkinTokens description is withdrawn. The V0.1 treatment is:

```text
point encoder + qualified-bone encoder
 -> relational cross-attention / pair evidence
 -> per-bone dense influence-field latent
 -> compact discrete skin-field codec (SkinTokens/FSQ-CVAE-inspired treatment)
 -> decoded influence proposal
 -> SkinProposalIR
 -> Compiler qualify_skin
```

A SkinToken represents a bone influence field, not bone identity. Exact codec capacity/FSQ levels are not frozen yet and require a separate preflight.

Top-4 is not early A0 authority. A0 may produce wider evidence; product max-influence/sparsification policy remains Compiler qualification policy. Differentiable nonnegative/simplex normalization inside training is allowed but does not transfer authority from Compiler.

## Training / qualification

Geppetto teacher losses may supervise anonymous locations, sequence count/continue, root evidence, directed parent evidence and uncertainty calibration. Teacher IDs are construction/loss provenance only.

Arachne may use teacher skin error as auxiliary supervision, but primary functional evaluation must use a frozen motion-probe bank and eight known orthographic projections. Functional terms should include projected carrier displacement, silhouette/contour agreement and local deformation/rigidity behavior. Silhouette-only authority is rejected.

Sequence:
1. teacher-projection corpus audit;
2. clean exact-substrate G0.1 training;
3. Compiler qualification replay;
4. A0 skin-column/qualified-control transport audit;
5. SkinFieldCodec ceiling;
6. A0 learned proposal model;
7. multi-probe 8-view deformation replay + `qualify_skin` + proof;
8. freeze C0;
9. train/promote C1 only after prospective corruption definition and clean non-inferiority.

Any learned-consumer checkpoint/architecture change creates a new consumer profile and invalidates transfer of an old IRIS admissible region until replayed.

## Open gates before seal

V0.1 is not sealed until TRAIN-authority audits quantify helper bones, multi-root rigs, zero-length deform bones, skin mass on non-deform columns, maximum deform-control count, and the teacher control-position convention. SkinFieldCodec capacity and the frozen functional motion-probe bank also require preregistration before outcome-based selection.

## Implementation boundary

Current first slice:

```text
SkeletonTeacherProjectionV1          implemented
SubstrateConsumerAdapterV1           implemented
Typed volume path evidence           implemented
Geppetto G0.1 neural scaffold        implemented
Arachne neural model                 not yet implemented
Architecture seal                    not authorized
Product qualification                not claimed
```
