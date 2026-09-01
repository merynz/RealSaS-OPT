# RealSaS — Human-Readable Continuation Roadmap — 2026-09-01

**Purpose:** explain the current program without requiring the reader to reconstruct state from experiment codenames.

**Architecture authority:** `SYSTEM_ARCHITECTURE_V3_20260901.md`  
**Detailed visual:** `REALSAS_END_TO_END_CANONICAL_ARCHITECTURE_20260901.svg`  
**Operational checklist:** `MASTER_EXECUTION_BACKLOG_20260901.md`

## 1. What we are building

RealSaS converts one controlled neutral eight-view character sheet into one editable, rigged, animatable puppet.

The product pipeline is deliberately split into four authority layers:

1. **observe geometry** — IRIS predicts observation-grounded forward depth/evidence;
2. **propose mechanics** — Geppetto proposes skeleton evidence and Arachne proposes skin evidence;
3. **compile canonical truth** — Compiler qualifies hierarchy, skin, editable mesh, mesh weights and exact product lineage;
4. **prove and project** — motion/deformation proof gates repair/export/runtime.

Models propose. Compiler owns canonical IDs and product state. Runtime consumes one proven product; it never becomes a second truth.

## 2. Geometry: what is already known

The downstream geometry tolerance study is closed for the tested consumer profile.

Current robust DTB-ND1 transition:

`0.00250 <= depth RMS critical boundary < 0.00275`

with the matched high-frequency depth absolute-P95 around `0.00490–0.00539`.

This is a measured consumer boundary, not a universal RealSaS constant.

The old DINO SharedLearner S/B ladder was also closed early. S→B scaling did not materially improve accessibility under that particular shared apparatus; that does **not** prove DINO-S itself is globally sufficient or insufficient.

## 3. IRIS: current direction

The old “learn depth independently at each pixel and hope a stronger backbone fixes it” treatment has been replaced by **Reprojection-Centered V2-A**.

The central idea is simple:

> exact cameras generate common-world candidate points; learned image features judge which candidate is supported across the eight observations.

Current sealed V2-A structure:

- frozen DINOv2-S/14 multi-level descriptors;
- native high-resolution spatial features;
- canonical world candidate lattice;
- exact eight-view reprojection;
- robust evidence aggregation;
- approximately isotropic world-space regularization;
- multimodal supported/ambiguous surface modes;
- local continuous refinement;
- exact projection back to forward depth, validity and uncertainty.

IRIS may not invent hidden back-side geometry. Unobserved regions remain UNKNOWN.

### Current IRIS gate

Synthetic Gate0 is PASS.

Real TRAIN512 Gate0 is running externally. Until its result is ingested:

- hull padding is not frozen;
- spacing/resolution is not frozen;
- no learned V2 optimizer step is authorized.

After Gate0, the next order is:

1. freeze padding/spacing policy;
2. freeze exact DINO taps/native pyramid/evidence/loss/runtime hashes;
3. Gate 0.5 — one real family ceiling;
4. Gate 1 — heterogeneous eight-family ceiling;
5. Gate 2 — DINO/native causal ablations;
6. only then decide whether full IRIS training is justified.

## 4. GeometricSubstrateAssembler: deterministic geometry boundary

IRIS does not directly own world-space product geometry.

`ObservationEvidenceIR` is compiled deterministically through:

`P = O + dF`

plus admitted persistence/support/provenance into `RiggingSurfaceIR S`.

The current implementation name is `GeometricSubstrateAssembler`. Historical `SurfaceBuilder` identities remain preserved only in old artifacts.

This layer may derive observation-grounded geometry. It may not choose a skeleton tree, mint canonical joint IDs, synthesize semantic skin, or fill hidden surfaces.

## 5. Geppetto: skeleton proposal, not skeleton authority

The clean-room RigAnything audit established that a sufficiently rich 3D surface substrate admits a demonstrated skeleton-solution class. The unresolved question for RealSaS is whether our **observation-limited** substrate exposes enough information.

That is now tested by R6 rather than argued abstractly.

Geppetto receives a deterministic consumer substrate:

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`.

Its job is to propose:

- variable control count/existence;
- continuous joint/control positions;
- root evidence;
- directed parent evidence;
- uncertainty/unsupported behavior.

It emits `SkeletonProposalIR G*`.

The existing Compiler graph optimizer then selects/admis root and parent structure and mints new canonical `J:*` IDs into `QualifiedSkeletonIR G`.

### Current Geppetto status

R0-R5 external-reference work is closed; R6 causal protocol is frozen.

A legacy G0.1 neural scaffold was **not** promoted. Only its still-valid teacher/evaluator skeleton projection was selectively ported into the R6 experiment namespace.

Next:

1. freeze an independent `GeppettoConditioningAdapter`;
2. freeze an independent candidate architecture;
3. run one-family U0 rich-surface ceiling;
4. run one-family then heterogeneous U1 observation-oracle ceilings;
5. route every output through the real Compiler qualifier;
6. open U2 predicted-IRIS coupling only after U1.

No Geppetto optimizer step is currently authorized.

## 6. Arachne: skin-field proposal, codec first

SkinTokens provides the cleaner external solution-class reference: geometry-conditioned influence fields that can be decoded at requested surface points.

RealSaS separates representation from product authority:

`S + QualifiedSkeletonIR G -> Arachne -> SkinProposalIR W* -> Compiler.qualify_skin -> QualifiedSkinIR W`.

Compiler owns legal references, nonnegativity, bounded sparsification/simplex correction and failure behavior.

### Why SkinFieldCodec comes before Arachne predictor

Before asking a neural predictor to infer the field, we must prove the representation itself can reconstruct authoritative skin truth well enough for deformation.

R6-A0 therefore tests:

- encode/decode reconstruction;
- Compiler-qualified decoded rows;
- static weight error;
- deformation-sensitive proof;
- one-family then heterogeneous small-family ceilings.

If the codec fails, predictor training is forbidden.

If it passes, only then freeze `ArachneConditioningAdapter`, skeleton serialization and the predictor architecture.

No Arachne optimizer step is currently authorized.

## 7. Editable mesh and mesh-weight seam: no longer missing

The old architecture documents correctly identified a missing mesh/mesh-weight typed seam. That gap is now closed at the type/product-lineage level.

MWB-0 added:

- `SurfaceSupportBinding`;
- `MeshDiscretizationCandidateIR M*`;
- `QualifiedEditableMeshIR M`;
- `QualifiedMeshSkinIR B`;
- `CanonicalPuppetGraph.v2`.

Every admitted mesh vertex must derive its rest position from admitted surface support. Mesh topology is a product component, not a second hidden geometry truth.

MWB-1 then passed on a real sealed witness:

- 512 admitted surface nodes;
- 5 Compiler-qualified joints;
- 512 qualified skin rows;
- deterministic identity-bound three-node mesh;
- exact skin-row copy to mesh vertices;
- topology and weight mutations changed product hashes;
- stale proof was rejected.

### What MWB-1 does not prove

The current executable baseline does not yet perform production triangulation or local-convex inserted-vertex construction.

Next is MWB-2:

- local convex support and/or CDT candidate;
- unsupported/UNKNOWN crossing checks;
- topology validity;
- deformation non-inferiority.

Historical CDT is a reference/promotion candidate, not current authority.

## 8. Canonical product: one exact state

The current target product is `CanonicalPuppetGraph.v2`.

Its state hash explicitly binds:

- S — admitted surface;
- G — qualified skeleton;
- W — qualified surface skin;
- M — qualified editable mesh;
- B — qualified mesh skin;
- qualification ledger and current editable/deformation/contact/motion state.

There is no second product graph for proof or runtime.

Any admitted rest-state mutation creates a new product hash and invalidates old proof.

Legacy `CanonicalPuppetGraph.v1` remains only for compatibility/sacrificial historical tests.

## 9. Proof, repair and runtime

The current executable envelope already enforces exact-state proof binding:

- `ProofFrame` carries the product-state hash;
- stale proof is rejected;
- export requires PASS proof for the exact current product;
- `RuntimePackageIR` records source product hash and source proof hash.

The historical v0.5 compiler contains richer heavy machinery for:

- motion/deformation probes;
- playback validation;
- failure signatures;
- owner attribution;
- bounded causal repair;
- retry/feedback/qualification loops.

Those semantics remain architectural obligations. They will be rebound around V2 product state only when the current product gates require them; they are not silently assumed to be current executable code.

Intended future loop:

`Y -> probes -> proof -> PASS export`

or

`Y -> failure signatures -> owner attribution -> RepairDirective -> Y'/ABSTAIN -> mandatory re-proof`.

## 10. Historical numerical compiler: preserved but not hidden

The 25-Aug historical audit established that the strongest compiler is a composite restoration target, not “whatever file is newest.”

Important external authorities remain:

- exact-predicate/constrained CDT + cotangent/topology work;
- BBW + active-set QP + ADMM + coupled/sparse KKT/Schur family;
- ARAP/corrective deformation;
- XPBD/contact/SDF contact;
- heavy motion-proof/attribution/repair;
- full native C++17 runtime SDK.

The 28-Aug GitHub restoration intentionally did **not** mirror the whole historical tree. It embedded the exact narrow historical closure required by the current execution path.

The migration completeness audit is now CLOSED PASS:

- 13/13 intended vendored records reconciled;
- branch-stranded restoration files: 0;
- current entrypoint byte-verifies the vendored closure before import;
- full historical source mirror was not the intended migration contract.

`solver_registry.py` is now fail-closed provenance metadata: there is no hidden current CDT/BBW/ARAP/XPBD execution path.

Every future solver promotion requires provenance, typed compatibility, residual/invariant tests, downstream parity and proof that it does not duplicate semantic ownership.

## 11. Clean-C0: remaining human-data gate

Native image/raster integrity is already closed exact on 2874 assets / 22992 views.

The remaining Clean-C0 work is semantic:

1. run the sealed visual-anomaly triage to prioritize review;
2. perform blind image-only single-riggable-character labels;
3. independently label deterministic random controls;
4. freeze final semantic membership and set hash.

Diagnostic anomaly scores cannot auto-exclude an asset.

## 12. What was deliberately left behind

The architecture audit explicitly excludes from the current product path:

- old image-decomposition/front-brain ontology;
- authored-owner/teacher-exact product ownership;
- parallel product truths;
- Unity-specific canonical ownership assumptions;
- old unsealed GeppettoG01 architecture;
- old 27D surface+interior token contract unless separately requalified.

Older architecture/type documents remain historical evidence, but their “mesh seam is open” / “explicit mesh types do not exist” topology is superseded by V3 + MWB-0/MWB-1.

## 13. Current order of work

While the real IRIS Gate0 runs externally, autonomous work can proceed in parallel:

1. **architecture reconstruction / SVG** — closed by V3 after migration audit;
2. **Geppetto R6 adapter + independent candidate preflight**;
3. **SkinFieldCodec design / preregistration**;
4. **MWB-2 local-convex/CDT policy and provenance preflight**;
5. **Clean-C0 diagnostic execution preparation**.

When Gate0 output arrives, IRIS result ingestion/padding/spacing becomes the highest-priority dependent path.

## One-line state

`geometry tolerance known -> old shared DINO ladder closed -> reprojection-centered IRIS V2 awaiting real Gate0 -> deterministic S boundary stable -> Geppetto/Arachne R6 causal gates frozen -> mesh/mesh-weight V2 product lineage closed through MWB-1 -> historical compiler/runtime migration reconciled -> one exact product/proof/runtime authority -> next: R6 apparatus + SkinFieldCodec + MWB-2 while Gate0 finishes`

**Learned optimizer authority remains CLOSED.**
