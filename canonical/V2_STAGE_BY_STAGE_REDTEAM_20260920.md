# RealSaS V2 — Stage-by-Stage Red-Team Audit — 2026-09-20

**Status:** CURRENT RED-TEAM AUTHORITY  
**Scope:** exact 46-stage V2 DAG on `main`  
**Witness used:** **NO**  
**Thresholds selected from Knight:** **NO**  
**Trigger:** the exact functional head `ca9d2d2e59bc517a139c743eb6a91df2e48ac761` passed the full self-hosted current-mainline gate and the subject-free Stage01–08 orchestration/artifact dry-run before this audit was opened.

## Why this audit exists

Module-family PASS is not enough. A system can have individually green components while still making an over-broad product claim at a boundary between them. This audit attacks each stage separately and asks:

- what truth does the stage actually consume?
- what can it silently fail to prove?
- can a downstream PASS hide that omission?
- does the stage prove geometry, mechanics, appearance, presentation, runtime integrity, or only byte/lineage integrity?
- is the claim appropriate for the first controlled eight-view Knight witness?
- does it justify the phrase **Spine-class 2D art under motion**, or merely prepare us to test that phrase?

Verdicts:
- **CLOSED** — no new implementation blocker found for the frozen first-witness contract.
- **CLOSED_WITH_SCOPE** — correct for the first controlled witness, but intentionally narrower than future real-world input/generalization.
- **DECLARATIVE_ONLY** — records provenance/intent but does not independently verify the external fact.
- **REOPENED** — a real product-quality measurement/authority gap remains and readiness may not be resealed until it is resolved or the claim is explicitly narrowed.

## Stage audit

| # | Stage | Red-team attack | Verdict |
|---:|---|---|---|
| 01 | `SOURCE_BYTES_SEALED` | Moving aliases, duplicate roles, byte drift and size drift are rejected. An expected SHA is not mandatory; when absent the stage mints authority from the bytes it actually reads rather than proving a prior external commitment. That is acceptable for run-local byte identity but must not be described as prior preregistration. | **CLOSED** |
| 02 | `SOURCE_LICENSE_PROVENANCE` | The adapter proves required provenance fields are present and sealed. It does **not** contact a registry or legally validate the asserted license. | **DECLARATIVE_ONLY** |
| 03 | `SOURCE_MECHANICAL_AUDIT` | Exact source raster hashes are Stage01-bound; exact masks are hash-bound through the manifest/input fingerprint and Stage07 authority. The audit proves foreground non-emptiness/border safety, not cross-view semantic consistency or arbitrary-pose validity. | **CLOSED_WITH_SCOPE** |
| 04 | `FULL_SUBJECT_ADMISSION` | Admission requires the controlled eight-view mode and Stage03 PASS. It is deliberately not a general real-world View Contract Resolver. | **CLOSED_WITH_SCOPE** |
| 05 | `CAMERA_CONTRACT_SOLVED` | The exact hashed camera bundle is qualified; cameras are not inferred from the images. A wrong-but-self-consistent supplied camera bundle remains an upstream acquisition error. | **CLOSED_WITH_SCOPE** |
| 06 | `OBSERVATION_RENDER_8VIEW` | The implementation does not synthetically rerender the subject; it materializes exact source observation bytes under the qualified cameras. The old word “Render” can mislead reviewers into assuming a new image-generation authority. | **CLOSED; TITLE SHOULD SAY MATERIALIZE** |
| 07 | `OBSERVATION_CONTRACT_QUALIFIED` | Raster, foreground, camera and admission identities are joined exactly and out-of-frame remains UNKNOWN. No aggregate score can replace those bindings. | **CLOSED** |
| 08 | `NORMALIZATION_DOMAIN_QUALIFIED` | Center/half-extent are exact manifest authority, finite and positive. They are not inferred from arbitrary user imagery. | **CLOSED_WITH_SCOPE** |
| 09 | `IRIS_FIT_PREREGISTERED` | Model source, architecture, executor, seed, upstream bindings, output contract and dense source-coverage policy are frozen before execution; teacher inference inputs are forbidden. | **CLOSED** |
| 10 | `IRIS_FIT` | External GPU execution is admitted through a hash-bound receipt. This proves execution identity and declared policy compliance, not bit-for-bit deterministic replay of GPU training. | **CLOSED_WITH_SCOPE** |
| 11 | `IRIS_CHECKPOINT_SEALED` | Checkpoint/result bytes bind to the exact execution receipt. No qualified-output binding exists yet because geometry qualification is downstream; this is an execution seal, not product authority. | **CLOSED** |
| 12 | `ZERO_SURFACE_DECODED` | Zero-surface artifact, arrays, checkpoint, observation and normalization hashes are exact; nonfinite/index-invalid arrays and declared teacher truth are rejected. The orchestrator validates the external decode artifact rather than recomputing decoding from the checkpoint itself. | **CLOSED_WITH_SCOPE** |
| 13 | `GEOMETRY_SUBSTRATE_QUALIFIED` | Eight-view silhouette recall/precision, coherent holes, interior coverage, connected-component recall and edge distance are hard gates. They prove renderable geometric support, not hidden-depth semantic correctness. Hidden geometry is owned by later mesh/dynamic/exposure gates. | **CLOSED_WITH_SCOPE** |
| 14 | `GSA_BUILD` | Dense-to-compact construction is policy-bound; relation quality cannot borrow a downstream mesh PASS. Parent-relation feasibility is independently attacked at Stage18. | **CLOSED** |
| 15 | `RIGGING_SURFACE_QUALIFIED` | The surface/relation substrate is qualified as mechanics/evidence substrate only. It has no RGB or categorical authority. | **CLOSED** |
| 16 | `OUTPUT_PRESENTATION_DIRECTIONS_SEALED` | Output directions are a distinct authority object, but the first-witness implementation derives the eight output directions from the same controlled camera set. Different source/output cardinalities remain deliberately deferred. | **CLOSED_WITH_SCOPE** |
| 17 | `MECHANICAL_PARTITION_QUALIFIED` | Manual component/carrier injection is forbidden; structural relations own partitioning and consequential UNKNOWN cannot silently become truth. The result is intentionally **not** presentation segmentation. | **CLOSED** |
| 18 | `CANONICAL_MESH_ADDRESSING_BUILD` | Candidate mesh, stable surface addressing and appearance domain are born together. Relation-parent corners below the frozen conditioning floor fail before an incapable CDT can pretend to repair them. | **CLOSED** |
| 19 | `STATIC_CANONICAL_MESH_QUALIFIED` | Static topology/addressability/conditioning are checked without allowing appearance or later mechanics to mint a replacement topology. This stage does not prove dynamic deformation. | **CLOSED** |
| 20 | `CAA_BACKEND_PREREGISTERED` | Backend identity and numerical appearance policy are explicit; an unbound backend cannot execute. Knight-derived threshold selection is forbidden. | **CLOSED** |
| 21 | `CAA_COMPILE` | Every renderable sample receives explicit provenance. Deterministic completion can still be visually poor on truly unobserved surface; totality is not confused with fidelity and Stage24/45 must detect unacceptable quality/exposure. | **CLOSED_WITH_RISK** |
| 22 | `CAA_COMPILE_SEALED` | Compile outputs and execution identity are sealed before bake; runtime cannot silently regenerate missing art. | **CLOSED** |
| 23 | `COMPLETE_APPEARANCE_ASSET_BAKED` | Premultiplied internal alpha, unique face barycentric atlas, bleed and provenance arrays are materialized into a complete asset. This is an asset identity, not a quality PASS by itself. | **CLOSED** |
| 24 | `COMPLETE_APPEARANCE_QUALIFIED` | Source-lock exactness, totality, structured holdout, seam color/gradient and sampling policy are first-class static appearance gates. These are subject-free guardrails, not a perceptual-optimality claim. | **CLOSED** |
| 25 | `CAA_REFERENCE_REST_RENDER_PROOF` | Exact rest/source directions prove source-foreground RGBA/alpha preservation and geometry-visible hole limits. This is strong rest art proof but does not prove appearance quality after large animation deformation. | **CLOSED_WITH_SCOPE** |
| 26 | `GEPPETTO_FIT_PREREGISTERED` | Source, upstream surface binding and execution contract are frozen before training; teacher inference inputs are forbidden. | **CLOSED** |
| 27 | `GEPPETTO_FIT` | External fit is hash/receipt bound; execution PASS does not itself mint a skeleton. | **CLOSED** |
| 28 | `SKELETON_QUALIFIED` | Compiler qualification owns root/tree/identity/evidence validity; learned proposal identity cannot bypass qualification. Semantic names are not required product truth. | **CLOSED** |
| 29 | `GEPPETTO_CHECKPOINT_SEALED` | Exact fit execution/checkpoint is sealed after qualified skeleton lineage exists. | **CLOSED** |
| 30 | `ARACHNE_FIT_PREREGISTERED` | Surface+skeleton authority and skin output contract are frozen before execution. | **CLOSED** |
| 31 | `ARACHNE_FIT` | External fit is receipt/hash bound and cannot itself mint qualified skin. | **CLOSED** |
| 32 | `SKIN_QUALIFIED` | Missing rows, negative/nonfinite/non-simplex weights and repair accounting are fail-closed. Compiler owns the qualified skin. | **CLOSED** |
| 33 | `ARACHNE_CHECKPOINT_SEALED` | Exact fit/checkpoint bytes are sealed against the qualified skin lineage. | **CLOSED** |
| 34 | `DEFORMATION_CAPABILITY_ENVELOPE` | This is a bounded numerical conditioning envelope and joint-frame witness, **not** professional motion capability. Actual artist motion is proven later. | **CLOSED** |
| 35 | `DYNAMIC_MECHANICAL_MESH_QUALIFIED` | The exact frozen mesh is stressed under the exact qualified rig/skin/envelope; repair must create new lineage rather than mutate a passing mesh in place. This proves mechanics/conditioning, not 2D art quality. | **CLOSED** |
| 36 | `QUALIFIED_MESH_SKIN_TRANSFER` | Skin is deterministically transferred onto the already-qualified product mesh; no second runtime skin truth is allowed. | **CLOSED** |
| 37 | `QUALIFIED_PRESENTATION_STRUCTURE` | Current automatic segmentation creates separate slots for disconnected face islands inside a mechanical component. It **cannot automatically split two independently addressable visual regions that are mechanically equivalent and topologically connected** (the head/hat class of counterexample). This is exactly where structural partition and presentation segmentation can diverge. | **REOPENED — HIGH** |
| 38 | `CANONICAL_PUPPET_SEALED` | Exact mechanics, CAA and presentation bindings are sealed correctly, but sealing cannot repair Stage37’s missing automatic presentation partition. | **CLOSED AS SEAL; INHERITS STAGE37 GAP** |
| 39 | `MOTION_SOURCE_OR_PRESET_SEAL` | Current V2 implementation accepts professional external MotionSourceClip.v2 assets and deliberately rejects unsupported source kinds. The stage name still says “or preset,” while inline preset execution is not current authority. | **CLOSED; TITLE/CLAIM SHOULD BE PRECISE** |
| 40 | `MOTION_COMPILE_RUN` | Full-3D local quaternion/root-translation compilation is bound to exact skeleton/envelope/presentation/mechanical state. No legacy single-axis contract remains in current closure. | **CLOSED** |
| 41 | `MOTION_DYNAMIC_PROOF` | Exact FK/LBS/contact/conditioning is proven over compiled clips. Rest-unseen exposure is correctly diagnostic here because appearance exposure is owned by Stage45. | **CLOSED** |
| 42 | `RUNTIME_PROJECTION_AND_CAA_BINDING` | Runtime binds exact posed XYZ/topology/cameras/sealed CAA and forbids donor search, skin solve and appearance generation. | **CLOSED** |
| 43 | `RSS_MATERIALIZE_COMPACT` | Package entries are replay-hashed and self-contained; materialization cannot mint a new authority. | **CLOSED** |
| 44 | `NATIVE_PACKAGE_OPEN_PLAYBACK` | Native reader opens the actual package and is byte-compared with Python reference. This stage samples a representative frame; exhaustive frame/view checking is deliberately Stage45’s job. | **CLOSED** |
| 45 | `DYNAMIC_VISUAL_INTEGRITY_PROOF` | The implementation exhaustively gates undefined visible provenance, compiled-unobserved exposure and native/reference byte parity over all frame/view pairs. It records alpha-hole diagnostics. **It does not yet directly measure dynamic 2D-art deformation quality** such as screen-space texture stretch/shear, line-art distortion, or temporal edge/flicker integrity. Mechanical triangle conditioning is an important proxy but is not the same claim. | **REOPENED — HIGH** |
| 46 | `PRODUCT_CLOSURE_SEAL` | Binding closure and editable archive integrity are strong. However `appearance_authority_passed=true` / `product_pass=true` currently inherit Stage45’s narrower runtime-integrity proof and Stage37’s limited automatic segmentation. Therefore “PASS_PRODUCT_V2” must not be read as proof of universal Spine-class authored appearance/editability until those gaps are closed or the claim is narrowed. | **REOPENED BY INHERITANCE** |

## Red-team conclusions

### RT-37 — automatic presentation segmentation gap — HIGH

The frozen architecture correctly states:

> structural/mechanical partition and presentation segmentation are different questions.

The current V2 implementation does not fully realize that distinction. It creates presentation groups from **connected mesh-face islands within each mechanical component**. That is useful and non-semantic, but insufficient when two visually distinct regions share mechanics **and** are topologically connected.

This does not corrupt geometry, mechanics or CAA. It limits automatic authoring/editability. A correct repair must remain role-free and may not smuggle categorical labels such as “hat” or “sword” into product authority.

**Required before broad automatic Spine-class authoring claim:** add a qualified presentation-partition authority that may use generic evidence (mechanical boundaries, disconnected topology, source-visible appearance boundaries, occlusion/composition evidence) without categorical recognition, with exact merge/split provenance and author-editable override.

### RT-45 — dynamic appearance-quality measurement gap — HIGH

Stage45 is an excellent **runtime integrity** proof:

- every dynamic frame/view is executed by the native package reader;
- native bytes must equal the deterministic reference;
- visible pixels require defined provenance;
- compiled-unobserved visible exposure is budgeted.

But runtime correctness is not identical to **artist-quality appearance under deformation**. A reference renderer can be perfectly deterministic while both renderers agree on ugly texture stretch or line distortion.

**Required before claiming dynamic Spine-class art quality as scientifically proven:** either

1. add a subject-free preregistered dynamic appearance-conditioning gate (for example bounded screen-space surface/UV conditioning plus temporal boundary/edge integrity, with thresholds calibrated without Knight), or
2. explicitly narrow Stage45/46 claims to runtime visual integrity and leave professional dynamic appearance quality as a witness/product-performance observation rather than a proven implementation invariant.

The preferred engineering direction is (1), because geometry and mechanics already receive quantitative conditioning proofs and appearance deserves the same standard.

### RT-46 — closure wording currently over-broad — HIGH by inheritance

Stage46 is allowed to say the exact V2 contract closed. It is not yet allowed to use that closure as evidence that all aspects of automatic Spine-class authorability and dynamic art quality have been proven. The product closure report must either consume the repaired RT-37/RT-45 authorities or make the narrower claim explicit.

## What remains genuinely green

The red-team findings do **not** invalidate the successful infrastructure/implementation evidence:

- self-hosted current-mainline CI on `ca9d2d2e59bc517a139c743eb6a91df2e48ac761`: PASS;
- repository/governance tests: 48 PASS;
- model/learned-source tests: 51 PASS;
- compiler regressions: 281 PASS;
- subject-free Stage01–08 orchestration/artifact dry-run: PASS;
- exact dry-run implementation closure SHA-256 before this red-team documentation update: `a46361fc2d9b989a2a4490829b73b835202521c5efed6916e44aab253cabf882`;
- Knight data used during this audit: **none**.

Those green results prove that the current system executes and enforces its existing contract. The red-team audit asks whether that contract is strong enough for the product claim, and found two high-value places where it is not yet strong enough.

## Promotion rule

**Knight remains forbidden.** Do not reseal `READY_FOR_WITNESS_EXECUTION` merely because CI and orchestration are green. RT-37 and RT-45/46 must first be resolved or have their claims deliberately narrowed and re-approved, followed by a fresh exact implementation-closure hash, full self-hosted CI, subject-free orchestration dry-run, and readiness reseal.

No Knight-derived threshold or design choice may be used to resolve either finding.
