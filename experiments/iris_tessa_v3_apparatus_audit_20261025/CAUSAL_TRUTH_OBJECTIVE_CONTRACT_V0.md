# RealSaS — Causal Truth / Objective Contract V0

**Date:** 2026-10-25  
**Status:** `DESIGN_CONTRACT__NO_BUILDER_OR_TRAINING_AUTHORIZED`  
**Purpose:** define what must be true mathematically before a Truth139 builder is allowed to emit student supervision for IRIS/TESSA V3.

This document is deliberately upstream of implementation. It does not select final loss weights, thresholds, sampling counts, or a final V3 architecture. Its job is to prevent the teacher from producing labels that are convenient for the current heads but do not correspond to an identifiable causal quantity.

## 1. Central rule

The teacher's canonical output is **qualified response evidence**, not a pre-decided hidden identity or binary part label.

For an intervention family `c` and a typed source locus `l`, retain a privileged response object

`R_c(l)`

with authority, provenance, intervention parameters, validity and ambiguity state.

A student target is a **derived view** of this response evidence. It may be changed in a later experiment without rebuilding or redefining the privileged authored truth.

The builder therefore owns:

`authored source -> intervention -> measured response -> qualification`

and a versioned objective adapter owns:

`qualified response -> student target / loss geometry`.

The two are not the same authority.

## 2. Endpoint domains

No evidence row is meaningful without an endpoint domain.

### 2.1 Pre-M domains

`SourceObservationRef(view, pixel/ray/source-footprint, provenance)`

Use where the distinction is authored/view-observable and may not be single-valued at one 3D position.

`SceneLocusRef(xyz, local support, provenance)`

Use only for a quantity that is well-defined as a function of a scene locus under the admitted representation. If two causally distinct authored states collapse to the same locus/observation available to the student, the target is not identifiable on this domain.

`LocusPairRef(a, b)`

Typed ordered or unordered pair of valid endpoint refs. Pair algebra must be declared explicitly.

`RayHitRef(view, ray, ordered hit, locus/source binding)`

Use for complete-stack visibility/compositing evidence. A pair relation must not pretend to own a quantity whose definition requires the complete ray.

### 2.2 Post-M domain

`CanonicalMeshAddress(mesh_hash, face, barycentric)`

Minted only after Compiler accepts canonical M. Pre-M truth must not require this address to exist.

### 2.3 Identifiability gate

Before a truth family is projected onto IRIS H(x), ask:

> Is the target single-valued on the observation/scene quotient visible to the student?

If `x1` and `x2` receive identical admitted student observation but teacher truth differs, the target cannot be supervised as a deterministic function of that representation. Options are:

1. preserve a richer endpoint/observation representation;
2. move the quantity to deterministic external evidence;
3. model a justified conditional distribution;
4. abstain.

Never punish the learner for an unidentifiable private authored distinction.

The RGBA collision probe is a concrete example of this gate.

## 3. Relation algebra is part of the target contract

Different evidence families obey different algebra. They must not all be routed through one unconstrained pair scalar merely because implementation is convenient.

### 3.1 Symmetric relations

Examples:

- appearance-response similarity/equivalence;
- rigid-transform coherence;
- deformation-response coherence;
- separation compatibility/distance where defined symmetrically.

Required property:

`r(a,b) = r(b,a)`

A student parameterization should enforce this by construction or the training/evaluation contract must explicitly verify symmetry. Merely presenting both orders is weaker and can waste capacity.

Potential symmetric pair features include commutative terms such as

`H_a + H_b`, `|H_a - H_b|`, `H_a * H_b`

plus symmetric relative-geometry features where justified.

### 3.2 Directed relations

Example:

`attachment_dependency(a -> b)`

Required property: order is meaningful. In general

`r(a -> b) != r(b -> a)`.

The student must retain ordered source/destination features or an explicitly directional operator.

### 3.3 Antisymmetric / complementary view relations

For tie-qualified front order on a comparable pair:

`P(a front b) = 1 - P(b front a)`

except an explicit tie/abstain state.

Training/evaluation should test this relation algebra rather than score two unrelated logits as if they were independent facts.

### 3.4 Ray-global quantities

Global visibility/compositing is neither symmetric nor a pairwise quantity. It is defined on the ordered complete ray stack. Pairwise front/overlap evidence may be inputs to Compiler reasoning, but it must not be substituted for complete-stack visibility truth.

## 4. Privileged response families

The following are candidate teacher families. Names do not imply that every family requires a learned IRIS head.

### 4.1 Appearance response

For a bounded reversible appearance intervention `c`:

`R_app_c(l) = source-space appearance response`

and separately, for view `v`:

`S_app_c^v(x) = visible raster response`.

Do not infer causal sameness from material ID alone. Material/node/component IDs are privileged intervention handles and provenance, not student ontology.

Candidate derived relation:

`K_app(i,j) = similarity({R_app_c(i)}_c, {R_app_c(j)}_c)`

with explicit intervention coverage.

### 4.2 Rigid-transform response

Under a bounded transform intervention, retain displacement/transform response. A useful pair residual is

`D_rigid(i,j) = E_c || (x_i^c - x_j^c) - (x_i^0 - x_j^0) ||`.

Low residual is evidence of common rigid response under the tested intervention family. It is not by itself a final drawable identity.

### 4.3 Deformation response

Retain trajectory/kernel information before thresholding:

`R_def(i,t,c) = x_i(t,c) - x_i(0)`

or an equivalent canonicalized response after removing explicitly defined global rigid motion.

Derived pair quantities may include correlation, normalized trajectory distance, response-kernel similarity or ranking. Do not reduce to binary `same_deform` until a court demonstrates a stable threshold and sufficient coverage.

### 4.4 Directed attachment/dependency response

Intervene on candidate parent/support A and measure response of B, with a reverse intervention control where legal.

A candidate influence statistic may take the form

`I(A -> B) = normalized_response(B | do(A))`

with a corresponding `I(B -> A)` and authored hierarchy/skin/contact evidence retained only as teacher qualification/provenance.

Correlation from native animation alone is not sufficient to claim causal direction.

Compiler owns final legal attachment after combining learned dependency evidence with canonical anchors and mechanical/contact proof.

### 4.5 Separation evidence

Positive evidence means at least one qualified intervention distinguishes the response of nearby/adjacent loci.

It is evidence of independent controllability, **not** a direct topology cut command.

A negative target requires sufficient intervention coverage to establish non-separability within the admitted family; absence of a distinguishing intervention is otherwise UNKNOWN/N/A.

### 4.6 Front / depth order

Use deterministic complete tie-safe depth authority where available. Equal-depth or numerically ambiguous cases abstain.

This is naturally view-conditioned and should preserve camera provenance.

### 4.7 Occlusion / compositing contribution

Positive support is insufficient for a calibrated objective. The teacher must define:

- qualified positive blocking/compositing support;
- qualified negative support for comparable non-occluding cases;
- N/A for non-comparable pairs;
- alpha/compositing ambiguity handling.

A stronger causal quantity may compare visible contribution with and without a qualified hit/intervention, e.g. a counterfactual visibility contribution change.

### 4.8 Global visibility

Use complete all-hit ray/compositing truth only. Store per-hit contribution rather than silently reducing every stack to opaque first-hit visibility if alpha requires a richer model.

Incomplete stack or ambiguous compositing abstains.

## 5. Objective geometry

No universal loss is authorized for all evidence families.

### 5.1 Calibrated probability objectives

Suitable candidates: front probability, qualified binary dependency events, properly balanced/qualified occlusion events.

Candidate loss: BCE or a calibrated alternative.

Required evaluation: Brier score / ECE plus precision/recall or order accuracy as appropriate. AUC alone is insufficient.

### 5.2 Continuous response objectives

Suitable candidates: rigid residual, deformation distance, response magnitude, compositing contribution.

Candidate losses: SmoothL1/Huber, normalized robust regression, cosine/directional loss, or distributional regression depending on response geometry.

Loss scale/normalization must be derived from the teacher distribution, not chosen merely to make gradients numerically convenient.

### 5.3 Relative / ranking objectives

When authored ambiguity makes an absolute metric unjustified but response ordering is identifiable, use relative constraints such as

`D(i,j) < D(i,k)`

with a margin/ranking/contrastive objective.

This is especially relevant to role-free latent organization where a universal absolute part-distance metric may not exist.

### 5.4 Trajectory/kernel objectives

For deformation, preserve temporal/condition response structure. Candidate objectives may combine normalized trajectory correlation, spectral/kernel distance, directional response and magnitude terms. A single BCE on a thresholded coherence label is not presumed sufficient.

### 5.5 Structural/topology objective

TESSA token CE/coordinate metric remains a proposal-generation objective. Free-running decoded topology quality is an independent court. Improvement in teacher-forced loss cannot substitute for structural validity or downstream Compiler behavior.

## 6. Shared H does not imply one metric space

The shared-scene hypothesis requires common substrate, not identical relation geometry for every task.

It is admissible to decode family-specific projections from one H:

`z_app = g_app(H)`

`z_rigid = g_rigid(H)`

`z_def = g_def(H)`

`z_attach_src, z_attach_dst = g_attach(H)`.

The scientific question is whether the same shared scene preserves enough information for these qualified observables, not whether every observable can be represented by one Euclidean metric.

## 7. Geometry as deterministic side information

A relation head should not be forced to reconstruct deterministic relative geometry from H if the scientific target legitimately depends on geometry and the coordinates are already admitted observations.

For each causal family, preregister whether explicit features such as

- `x_a`, `x_b`, `x_a - x_b`, distance;
- local normals / tangent frames where admitted;
- camera-relative depth/projected distance for view-conditioned quantities

are part of the input contract.

This is an inductive-bias question and must be ablated at matched capacity where material. It must not be decided after seeing FIT8 outcomes.

## 8. Truth qualification and negatives

Every target must report at least:

- positive count;
- negative count where meaningful;
- N/A count;
- abstain/ambiguous count;
- authority distribution;
- intervention-family coverage;
- morphology/source-provider distribution.

A negative label requires positive qualification evidence for the negative proposition; UNKNOWN is not zero.

The current V3 positive-only occlusion adapter is therefore not sufficient for calibrated occlusion learning.

## 9. Sampling contract

Training sampling must preserve/measure rare causal strata. Candidate strata include:

- morphology;
- connected component / low-area support;
- thinness and curvature;
- boundary cause;
- attachment/contact;
- adjacent/non-adjacent pair;
- overlap/non-overlap;
- grazing angle;
- alpha/compositing complexity;
- positive/negative/N/A;
- intervention response magnitude.

Sampling policy is part of the scientific apparatus. If a target exists in truth but is systematically absent from sampled training rows, representation failure cannot be inferred.

## 10. Shared-trunk optimization court

When multiple families train the shared scene, record per-family gradient norm and pairwise cosine statistics on shared-trunk parameters.

Persistent destructive interference is an optimization/apparatus finding, not automatically a representation falsification.

No adaptive loss weighting, gradient surgery or task sampling repair may be introduced after candidate outcomes are visible without a new preregistration.

## 11. TESSA presentation evidence requirement

Before the matched TESSA A/B:

1. every presentation token must carry a stable typed endpoint/address;
2. the mapping from IRIS evidence to presentation token must be deterministic/versioned;
3. presentation evidence must be student-produced in the B arm, not teacher truth leaked at inference;
4. A arm uses explicit null presentation with identical overall model capacity/budget;
5. B-arm evidence quality is measured independently before topology outcome is opened.

If independent scene/surface/presentation compression destroys endpoint association, fusion placement is an apparatus treatment and must be versioned before the reference A/B.

## 12. Compiler boundary

Truth and learned evidence do not mint final graphics identity, attachment, order or canonical M.

Compiler may:

- bind accepted pre-M endpoints to canonical addresses;
- combine typed evidence families;
- enforce graph legality / acyclicity / canonical anchors;
- derive quotients where multiple intervention families agree and coverage is sufficient;
- abstain/fail closed.

Compiler must not become a second perception model or fabricate missing negatives.

## 13. Readiness conditions for Truth Builder implementation

Truth139 builder implementation may begin only when the apparatus audit has at least versioned answers for:

1. which target families are functions of current admitted observation/H and which require richer/external evidence;
2. endpoint domain for every family;
3. relation algebra (symmetric/directed/complementary/ray-global);
4. privileged response representation;
5. qualification/negative/N-A semantics;
6. candidate objective geometry;
7. sampling/coverage court;
8. representation/fusion treatment arms still requiring preregistration.

The builder may still be implemented before final loss hyperparameters are known, because privileged response truth is intentionally richer than a particular student objective.

## 14. Explicitly forbidden shortcuts

- material ID == drawable truth;
- mesh component == drawable truth;
- bone ID == attachment truth;
- absent positive == negative;
- candidate index == identity;
- current head name == teacher ontology;
- thresholded response == canonical truth without a court;
- pairwise visibility == global visibility;
- teacher topology or authoring names as student inference inputs;
- changing representation/loss after FIT8 outcomes without a new controlled experiment.
