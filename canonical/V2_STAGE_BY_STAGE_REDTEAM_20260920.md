# RealSaS V2 — Stage-by-Stage Red-Team Audit — 2026-09-20

**Status:** PASS — 46-STAGE SECOND-PASS HARDENED SUBJECT-FREE  
**Scope:** exact 46-stage V2 DAG on `main`  
**Witness used:** **NO**  
**Thresholds selected from Knight:** **NO**  
**Closure evidence:** documentation-inclusive head `4419509ee0f62a7e5ca088b87feb682b1a664105`; mainline run `35539075794`; subject-free orchestration run `35539075769`; exact closure `c2ba1569285811ecbac448c54a4fbe07d57111f76ff4f7003c7482c7ce26306e`; no Knight data used.\n
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
| 06 | `OBSERVATION_RENDER_8VIEW` | Exact source observation bytes are materialized under qualified cameras; no new subject render is synthesized. The plan title now states materialization explicitly. | **CLOSED** |
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
| 37 | `QUALIFIED_PRESENTATION_STRUCTURE` | The second-pass attack showed disconnected topology alone could not split mechanically equivalent connected visual regions. Repair adds hash-bound `PresentationPartitionEvidenceIR.v2`: shared-edge adjacency may be cut only by frozen source-backed CAA appearance-boundary evidence. Categorical identity remains forbidden. | **CLOSED_WITH_SCOPE** |
| 38 | `CANONICAL_PUPPET_SEALED` | Stage38 re-verifies presentation-partition evidence against exact mesh, CAA asset, CAA qualification and structure bindings before sealing. | **CLOSED** |
| 39 | `MOTION_SOURCE_OR_PRESET_SEAL` | Current authority accepts professional external `MotionSourceClip.v2`; unsupported kinds fail closed. Plan title no longer implies an inline preset authority. | **CLOSED** |
| 40 | `MOTION_COMPILE_RUN` | Full-3D local quaternion/root-translation compilation is bound to exact skeleton/envelope/presentation/mechanical state. No legacy single-axis contract remains in current closure. | **CLOSED** |
| 41 | `MOTION_DYNAMIC_PROOF` | Exact FK/LBS/contact/conditioning is proven over compiled clips. Rest-unseen exposure is correctly diagnostic here because appearance exposure is owned by Stage45. | **CLOSED** |
| 42 | `RUNTIME_PROJECTION_AND_CAA_BINDING` | Runtime binds exact posed XYZ/topology/cameras/sealed CAA and forbids donor search, skin solve and appearance generation. | **CLOSED** |
| 43 | `RSS_MATERIALIZE_COMPACT` | Package entries are replay-hashed and self-contained; materialization cannot mint a new authority. | **CLOSED** |
| 44 | `NATIVE_PACKAGE_OPEN_PLAYBACK` | Native reader opens the actual package and is byte-compared with Python reference. This stage samples a representative frame; exhaustive frame/view checking is deliberately Stage45’s job. | **CLOSED** |
| 45 | `DYNAMIC_VISUAL_INTEGRITY_PROOF` | Stage45 now gates native/reference parity, visible provenance and compiled-unobserved exposure plus rigid-motion-invariant intrinsic UV→posed-surface conditioning, rest-relative condition/principal stretch and adjacent-frame intrinsic stretch. A raw screen-space shipping gate was explicitly rejected because legitimate 3D foreshortening is not art deformation. Thresholds are frozen subject-free. | **CLOSED_WITH_SCOPE** |
| 46 | `PRODUCT_CLOSURE_SEAL` | Stage46 now refuses product closure unless Stage37 partition evidence is bound and Stage45 dynamic appearance conditioning passed with non-empty evidence. Editable authoring export carries the partition evidence. Claim is scoped to the exact controlled V2 contract. | **CLOSED_WITH_SCOPE** |

## Red-team conclusions

### RT-37 — CLOSED_WITH_SCOPE

The original finding was valid: mechanical partition is not presentation segmentation. Stage37 now emits a separate hash-bound role-free presentation-partition authority. Shared-edge adjacency inside one mechanical component may be cut only when qualified source-backed CAA evidence across that edge exceeds the frozen subject-free policy. Insufficient evidence means continuity, not semantic invention.

This closes the controlled-witness implementation blocker. It does not claim recovery of an unobservable semantic object boundary.

### RT-45 — CLOSED_WITH_SCOPE

Native/reference parity alone was insufficient because two deterministic renderers can agree on ugly art deformation. The first repair idea, raw screen-space conditioning, was itself rejected by red-team because rigid 3D rotation can legitimately create foreshortening.

Shipping gates therefore use intrinsic textured-surface metrics: UV→posed-3D conditioning, rest→posed condition/principal stretch and previous→posed temporal stretch. Screen projection conditioning is diagnostic only. Relative and temporal thresholds are frozen by `DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json` without Knight data.

This proves bounded local texture/line-art deformation under the frozen contract; it does not claim semantic correctness, animation acting quality or human aesthetic optimality.

### RT-46 — CLOSED_WITH_SCOPE

Stage46 consumes both repaired authorities. It will not emit `PASS_PRODUCT_V2` if presentation-partition evidence drifts or if dynamic appearance conditioning failed/was empty. The editable authoring archive includes partition evidence.

`PASS_PRODUCT_V2` therefore means the exact controlled V2 contract closed, not unseen generalization or universal semantic/perceptual optimality.

## Exact closure evidence

- Documentation-inclusive implementation head: `4419509ee0f62a7e5ca088b87feb682b1a664105`.
- Self-hosted mainline run `35539075794`, job `106153297550`: **PASS**.
- Repository/governance: **49 PASS**.
- Learned/model ownership: **51 PASS**.
- Compiler regressions: **288 PASS**.
- Plan SHA-256: `b703b139d30987de0df8c7753eb0ecb8a69134d4c87fd0247ff5e478d1d6163f`.
- Implementation closure SHA-256: `c2ba1569285811ecbac448c54a4fbe07d57111f76ff4f7003c7482c7ce26306e`.
- Native V2 CAA player SHA-256: `b1440a6c421123620ec7052d83100bd0fbea1116843f987ac312d419de1422e1`.
- Subject-free Stage01–08 orchestration run `35539075769`, job `106153297245`: **PASS**.
- Orchestration evidence ZIP SHA-256: `e5d994baccae4baee42edb9b00381504f972e619d97521694f10efdb0398c1f1`.
- Knight data used for repair/threshold selection: **none**.

## Promotion rule

The second-pass implementation blockers are closed and readiness may be sealed against the exact closure above. **READY does not itself authorize Knight.** Subject-2 Knight remains held until explicit user approval; only then may a fresh run-local witness ledger be created and Stage01 begin.
