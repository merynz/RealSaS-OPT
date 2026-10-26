# Truth Foundry V1: geometric and mechanical response truth

Design date: 2026-10-26. Source baseline: `311ade52c6c118d3d1d88076507fc6b7fc0d3b1f`.
Status: **design/measurement contract locked; teacher adapters, real FIT8 qualification,
real-model integration and training execution remain open**. This is the next work
after the pre-Truth IRIS/TESSA V3.1 architecture freeze. No ModelRelease,
ProductRevision, corpus admission or genericity PASS is created by this document.

## Decision

Primary truth describes **where source geometry and radiance go under explicit
interventions, how local geometry changes, which relative freedoms remain, and
what each ray contributes**. Attachment, grip, support, drawable membership and
draw-order policy are downstream interpretations. They are not obligatory primary
class labels and are not interchangeable with source bone/material/component IDs.

The measurable object for subject s is

\[
\mathcal T_s=(\mathcal L_s,\mathcal C_s,\mu_s,
R_s(\ell;c,t),\Pi_s,\mathcal V_s,\mathcal P_s).
\]

Here L is the exact source locus domain; C and mu describe interventions and their
measure; R contains geometric, mechanical and appearance responses; Pi contains
camera/raster/ray operators; V contains measurement validity and student
observability; P contains immutable provenance. An author's kinematic animation
law is not automatically a physical constitutive/contact law.

Rich teacher truth does not make every mechanism identifiable from static images.
Two visually identical handles may be welded, hinged or merely touching. The
teacher can distinguish them after intervention while the student cannot. Retain
that ambiguity; train calibrated conditional predictions/abstention rather than
requiring an impossible deterministic answer. FIT8 fit alone cannot prove unseen
bird/tank/snake/zombie genericity.

## Existing data: extend the same eight

Snake, Bat, T-Rex, Horse, Spider, GreenBlob, Shark and George remain the selected
eight. Knight is a separate regression witness. The 139 GLTF audit is not a new
selection: four selected inputs are FBX outside that GLTF subset. Preserve raw
corpus seal `2f15bf84694417cee3862c39a89e04bfa131f51331d6da5880739d40b7eab5ab`.

The read-only audit tool verifies the exact three historical manifest byte hashes,
schemas and eight-subject order. See `EXISTING_FIT8_AUDIT_20261026.json`. Historical
execution source is `eb11004e4d8c7fba6f6ba95fa8eacaa0c24fcb3d`.

| Existing component | Reuse | Boundary |
|---|---|---|
| source RGBA/cameras/normalization and geometry/rig/skin lineage | exact qualified parents | each subject needs source/import/rest/camera admission |
| all-hit CSR relation evidence | generic ray sampler and complete depth/tie stack | same-ray positive support is not authority for arbitrary negative pairs |
| raster and boundary builders | footprint, alpha and appearance response machinery | source-space edit must be separated from visible pixel delta |
| presentation queries/memory and TESSA corpus | existing addresses, manifests and training plumbing | model-surface-derived evidence is not an independent causal teacher |
| existing unavailable channels | preserve `ABSTAIN` | occlusion, deformation_coherence and attachment_persistence have no new measurement yet |

The three manifests were read and hash checked; this task has **not** verified the
`.pt` payload bytes, rerun all eight imports or admitted a training corpus.
Historical generic producers must be promoted/reviewed on main before a new
Platform attempt; a historical branch is not execution authority. Derived packages
are additive children, never mutation of raw files or silent mask replacement.

## Wire contract and address domain

The normative envelope is `canonical/schemas/truth_package_manifest_v1.schema.json`;
measurement layouts and policies are in `canonical/TRUTH_FOUNDRY_V1_CONTRACT.json`.
These six logical records form one package lineage:

| Record | Required content | Current implementation |
|---|---|---|
| TeacherSceneIRV1 | source+topology+normalization+appearance hashes; native deformation evaluator; supported controls/clips/constraints; unit/frame convention and adapter version | design; source adapters pending |
| InterventionSpecV1 | law and weights/covariance, baseline/sham, common-root controls, action units/amplitude/sign, times, initial solver state and teacher capability | design; real probe scheduler pending |
| CausalResponseBatchV1 | exact source loci; family arrays, masks, observability witness and measurement provenance | numeric kernels implemented; real producer pending |
| RayCompositingTruthV1 | full hit CSR, depth/ties, coverage/opacity, blend and colour law, T/weights/counterfactual response | complete source-over kernel implemented; renderer adapter pending |
| TruthQualificationReportV1 | source/teacher equivalence, masks/coverage/strata, interventions, leakage, replay, consumer behaviour and explicit failed/unsupported requirements | courts specified; real qualifier pending |
| TruthPackageManifestV1 | typed payload descriptors and byte hashes; exact parent identities and producer commit; all families explicitly covered or unavailable | local writer/reader implemented; Registry import separate |

A source locus is `(source_hash, topology_hash, primitive, face, barycentric)`.
It is **pre-M**, never a CanonicalMesh address. Use source skin weights only on
their exact source topology. Native deformations must be reproduced before sampling
truth; no nearest-face repair may silently invent equivalence. Compiler binds
accepted evidence to `(CanonicalMesh hash, face, barycentric)` only after M exists.
Keep teachers' bone/control/node IDs in privileged adapter provenance; the student
does not receive those IDs, teacher topology or source ownership labels as input.

Every family has independent measurement status and observability. Observation
conditioning requires a content-addressed witness and specified student input
domain. `MECHANISM_CONDITIONAL`, `PRIVILEGED_ONLY`, `AMBIGUOUS` and `UNKNOWN` remain
distinct; native availability is not observation-conditioned supervision authority.
The local writer cannot mint qualification. Platform/Compiler must validate the
witness and eligibility at import; a nonempty witness hash alone is insufficient.

## Required primitive measurements

1. **Surface trajectory:** absolute and root-normalized position response at fixed
   source loci, with action amplitudes, timestamps, displacements and actual
   interval velocities. Retain magnitude and units, including zero responses.
2. **Local surface differential:** metric strain, area change, anisotropy and
   differential orientation. A triangle has a two-dimensional tangent domain;
   do not fabricate a full-rank volumetric F from it. Degenerate triangles abstain.
   Curvature/bend needs qualified neighboring patches and a separate mask.
3. **Relative affine transport:** `A(c,t)^-1 B(c,t)`, variation from rest, reflection,
   shear and scale. Explicit local-frame construction and confidence are required.
   Preserve full affine information before deriving a rigid SE(3) approximation.
   Near-singular frames abstain. Shared root motion cancels, but common-parent
   co-motion alone still does not establish a directed causal connection.
4. **Oriented pair geometry:** gap along an oriented normal, tangent slip, relative
   velocity and approach/separation events, each with orientation/contact-domain
   validity. A point-pair normal gap is not a universal signed surface distance.
   Proximity or zero gap cannot establish force, support, grip or nonpenetration.
5. **Admissible motion subspace:** anchor/pivot residuals, allowed relative degrees
   of freedom, constraint Jacobian rank/nullspace over a verified excitation set.
   Record singular values and excitation coverage, not an arbitrary hard joint
   name. Observed trajectory rank is an empirical lower bound on explored motion,
   not proof of all allowed freedoms. Rank thresholds require units/metric policy.
6. **Physical impulse/temporal response:** only when an authored solver and known
   parameters genuinely supply it. Record initial state, integration step, force,
   mass/stiffness/damping provenance, lag/decay and reset/replay. Otherwise abstain.
   Native pose clips do not reveal physical mass, force or load-bearing support.
7. **Raster and ray response:** complete stack alpha, incoming transmittance,
   premultiplied contribution and render response under removal/appearance edits;
   preserve geometric hidden support separately from observable pixel differences.
8. **Boundary response:** discontinuities in local differential/appearance/ray
   response under matched interventions. Do not collapse occlusion, alpha edge,
   material change, deformation seam and authored visibility switch into one cut.

All ten V1 wire families must explicitly report `MEASURED`, `UNAVAILABLE` or
`UNSUPPORTED`. Missing prerequisites block any consumer requirement that needs
them. Absence is not automatically a negative example or a passing N/A stratum.

## From Spine: study observable laws, preserve the authority boundary

The official TypeScript 4.2 runtime was inspected at commit
`e7dc1435fa4a0083ab431f1b28e083c14a1f5c68`. Exact files and findings are in
`RESEARCH_AND_RUNTIME_REFERENCES.md`. This is a behavioral reference; no runtime
source is copied or vendored. The independent NumPy kernels implement general
affine geometry, surface metrics and source-over arithmetic.

Spine separates attachment instances, slot state, deform/weights, transform
constraints, dynamic state, clipping and draw-order timelines. That separation is
useful here. A weapon can follow selected coordinates of another frame without
being its direct hierarchy child. A soft patch may blend several frames and carry
local offsets. Visibility can change without any mechanical detachment.

For our 3D teacher, sample geometric response under isolated positive/negative
control perturbations, root-only perturbations, sham/no-op and native motion.
Measure anchor positional residual, relative orientation/affine residual, local
strain, constraint residual and temporal transport. Include partial freedoms and
switching intervals. A constant relative transform over that envelope supports a
binding hypothesis, but it does not identify weld versus fixed shared motion
outside the probed envelope. Preserve alternatives and the intervention scope.

Spine's authored slot/triangle layering is not 3D depth truth. Capture geometric
ray order and any source-authored artistic presentation order as separate channels.
Our ray-local order may interleave across a surface; a global drawable DAG may
not exist. Never force a cycle into a global order by deleting weak measurements.

## Intervention design and response metric

Record `R(l;c,t)` before lossy clustering. Include native clips for teacher
equivalence and isolated probes for causal separation. Probe enough signs,
amplitudes, axes, time horizons and states to detect sliding, hinge-like freedom,
strain and state switches. A disabled/unsupported control is an explicit outcome.
Constraint on/off is permitted only where the adapter exposes an authored law;
do not invent ablation semantics. Keep camera changes out of mechanical response.

For a discrete action law, use
`K(i,j)=sum_c mu[c] <R_i[c],R_j[c]>` with preregistered channel units/metric.
An orthogonal change of isotropic control basis preserves the integral; arbitrary
author control reparameterization does not unless the intervention law is
transported. Duplicating an action with split probability mass must not change K.
Never normalize every subject's flow by its largest motion: that discards response
strength. Missing positive-mass probes invalidate the metric rather than silently
renormalizing each pair to a different action distribution.

Sampling must include thin surfaces, near overlap, grazing views, alpha boundaries,
crossing limbs, separable rigid/soft responses and available interaction domains.
Source IDs may stratify privileged coverage reports, never supply target ownership.
If the selected eight do not exhibit a required mechanism, report the coverage gap;
add a declared diagnostic source or later heldout set rather than claiming coverage.

## Compositing and visibility

For a complete source-over stack in front-to-back order:
`T_i=product_(j<i)(1-alpha_j)` and `w_i=T_i*alpha_i`.
RGB is evaluated in linear light with declared premultiplication, sampling filter,
background and clipping law. Distinguish material opacity from footprint coverage
where the renderer exposes both; do not multiply already resolved coverage twice.
Other blend modes, refraction or an incomplete stack abstain in the V1 kernel.

Pair front/back means comparable projected loci, not global visibility. Qualified
negative domains require complete comparison evidence, including hard nearby
non-overlapping examples. An unrelated distant pair is not a universal occlusion
negative. A tie group retains group membership; face ID must not resolve truth.
V1 conservatively invalidates per-hit contribution/RGB on depth ties while keeping
order-independent total alpha when known. Later equivalence proofs may recover
special tied cases without changing the primary records.

Removal must recompose surviving fragments. Equal-colour opaque layers can yield
zero RGB removal delta while the rear layer has zero transmittance. Pixel delta,
geometric occlusion and perceptual importance therefore remain separate quantities.
Unsupported hidden artwork is not regenerated as qualified source appearance.

## Construction and qualification order

1. Read the Platform campaign/Attempt and exact deployed main; bind admitted raw
   source/observation parents and acknowledged handoff. Never resubmit a timed-out
   attempt without inspecting durable state.
2. Import/reuse teacher adapters; prove source rest/topology/skin/animation/camera
   equivalence on exact parents. Emit TeacherSceneIR and capability report.
3. Seal the intervention/sampling/evaluation policy before examining fit results.
4. Evaluate native and isolated responses in source space; render source rays and
   counterfactuals independently of the student. Retain all masks and reasons.
5. Qualify identity, geometry, replay, source raster parity, differential invariance,
   action-law invariance, ambiguous ties, observability and no teacher-input leakage.
6. Write a fresh derived package; import through Platform Registry/CAS. The local
   staging writer's content hashes are not Registry admission.
7. Run the **teacher-oracle consumer court first** on fixed admitted M/G/W/CAA.
   Then substitute predicted evidence under the same consumers. Follow each family
   through `DEFINED → PRODUCED → QUALIFIED → SERIALIZED → CONSUMED →
   BEHAVIORALLY_EFFECTIVE`. A schema field alone does not close a demo defect.
8. Seal exact qualified truth hashes, real model code, numeric optimizer/sampling/
   loss budgets, stopping rules, eval thresholds and seeds; then launch the new
   common-weight FIT8 through the Platform campaign. See the preregistration.

The new CPU workflow covers source kernels/contracts and architecture impact only.
It does not run a scientific teacher build, train a model or promote an artifact.

## Compiler consumers and production continuation

| Consumer | Evidence to bind after accepted M | Required consumer proof |
|---|---|---|
| Mechanics / anchor / grip proposal | local frames, residuals, excitation-qualified relative freedoms, gap/slip response | orientation/depth/anchor stability under heldout motion; distinguish common root and free sliding |
| Graphics partition/boundary | differential and appearance discontinuities plus local ray response | no bone/material-ID shortcut; source appearance preserved; alpha/occlusion boundaries separated |
| Graphics ordering/visibility | comparable pair order, complete ray T/weights, artistic order if authored | ray-local interleaving, cycles and ties retained; transparent layers and removal response correct |
| Motion | bound relative constraints and temporal response, validity and alternatives | transitions and constraint activation do not silently teleport geometry; secondary response only where measured |
| Proof / product | exact model/evidence/CanonicalMesh/Rig/Skin/CAA identities and consumer courts | cannot activate from research fit or a Git merge |

The current source responsibility sequence is confirmed:
`Input → Shape → Mesh → Rig → Skin → Mechanics → Graphics → Motion → Proof → ProductRevision`.
After IRIS/TESSA we continue this sequence, preserving unrelated qualified parents.
**IRIS/TESSA is not proven to be the only remaining blocker.** Source inspection
found these explicit closure requirements:

- The frozen V3.1 observation-to-Dense3D real model is not yet integrated/promoted.
- The product Shape worker calls the IRIS/TESSA input bridge without presentation;
  optional TESSA presentation support alone does not mean the production path uses it.
- Current Graphics still uses scalar confidence, pair visibility and semantic
  same-drawable/cut/attachment thresholds. Its global DAG weak-edge removal cannot
  certify richer ray-local compositing truth. Upgrade the actual consumer contract.
- Current Motion presentation uses physical z-buffer with slot ties. That is not
  a generic artistic-overlap implementation. The existing surface anchor primitive
  is useful but its XY correction and caller-supplied depth do not prove causal grip.
- Fresh Skin requires exact typed Mesh/Rig/rigging-surface/material-support parents
  and an admitted matching MIRA release/channel. Source DRAFT recovery is not live
  promotion. Required CUDA/BF16 execution cannot silently become CPU inference.
- Responsibility descriptors for Mechanics/Graphics/Motion/Proof are not evidence
  that every provider is activated or qualified.

The actual latest Knight parent execution inspected at this source snapshot was
run `40352421537`, job `121675884503`, artifact `12663594482`. It reached the
self-hosted runner and Platform API, then failed with HTTP 409:
`MODEL_SELECTION_CHANNEL_MISSING:IRIS:RESEARCH`. That run demonstrates a missing
selected model release/channel, not a Go-language or GitHub execution ceiling.
Readiness workflow `40352246444` was green but its inspector job was skipped and
produced no receipt. See `PRODUCTION_AUDIT_SNAPSHOT_20261026.json`; this snapshot
must not replace a fresh live read after deployment or authorize bypassing admission.

Reuse the existing **read-only Knight Skin readiness inspector** after exact-main
deployment to read current Platform parents/model channels. Its request revision
is advanced with this source work; it must retain its exact-deployed-SHA guard.
If runner/deployment is unavailable, live verification remains pending. It must
not import, evaluate, qualify, promote models or activate product Skin.

## Next execution boundary

Implemented now: independent numeric measurement kernels, conservative ray
compositing/removal, strict local envelope validation, reproducible numeric array
packaging/hash verification, exact existing FIT8 manifest audit and CPU tests.
No real teacher adapters, renderer hookup, real eight-subject truth qualification,
Platform truth stage or IRIS/TESSA retraining is claimed complete. Those are the
next owned implementation/execution tasks; the design below prevents them from
silently turning privileged source labels into observable truth.
