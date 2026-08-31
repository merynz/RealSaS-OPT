# RealSaS — Canonical End-to-End System Architecture V3 — 2026-09-01

**Status:** `CANONICAL_END_TO_END_ARCHITECTURE__MWB_V2_BOUND__MIGRATION_RECONCILED__HISTORICAL_EXTERNAL_AUTHORITIES_EXPLICIT`

**Supersedes for current topology:** `SYSTEM_ARCHITECTURE_V2.md` and the current-topology portions of `IR_TYPE_SYSTEM_V1.md` / `PRODUCT_CONTRACT_V1.md` where they still describe the mesh/mesh-weight seam as open or `SurfaceBuilder` as a pending rename. Those files remain historical contract evidence; they are not rewritten.

**Visual authority:** `canonical/REALSAS_END_TO_END_CANONICAL_ARCHITECTURE_20260901.svg`.

## 1. North star and authority rule

`ONE NEUTRAL 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

The system is an authority pipeline, not a sequence of equally authoritative models.

`EVIDENCE -> PROPOSAL -> QUALIFIED -> CANONICAL -> DERIVED -> RUNTIME`

IRIS, Geppetto and Arachne may emit learned evidence/proposals. Compiler qualification owns admissibility and canonical IDs/state. Runtime is a non-owning projection of one exact proven product state.

## 2. Current end-to-end target

```text
8 ordered neutral RGBA views + exact orthographic cameras
  -> IRIS Reprojection-Centered V2-A
     -> forward-depth d + validity/uncertainty/support evidence
  -> ObservationEvidenceIR
  -> GeometricSubstrateAssembler
     -> analytic P = O + dF
     -> admitted teacher-free persistence/support/provenance
  -> RiggingSurfaceIR S

S -> GeppettoConditioningAdapter B_G [deterministic, future frozen adapter]
  -> Geppetto [learned proposal model; R6 apparatus not yet sealed/trained]
  -> SkeletonProposalIR G*
  -> Compiler.rig_qualification
     -> restored canonical graph optimizer
     -> Compiler-minted J:* IDs
  -> QualifiedSkeletonIR G

S + G -> ArachneConditioningAdapter B_A [deterministic, after codec gate]
      -> Arachne [learned proposal model; predictor not yet authorized]
      -> SkinProposalIR W*
      -> Compiler.skin_qualification
      -> QualifiedSkinIR W

S -> deterministic mesh/discretization candidate producer
  -> MeshDiscretizationCandidateIR M*
  -> Compiler.geometry_qualification
  -> QualifiedEditableMeshIR M

M + W -> deterministic mesh-weight binder
      -> QualifiedMeshSkinIR B

S + G + W + M + B
  -> Compiler.product_assembly
  -> CanonicalPuppetGraph.v2 Y
  -> exact-state deformation/contact/probe execution
  -> ProofFrame / failure evidence
  -> PASS: RuntimePackageIR
  -> FAIL: owner attribution -> RepairDirective -> new Y' or abstain -> mandatory re-proof
  -> historical C++17 runtime ABI when native runtime is materialized/promoted
```

The present executable MWB path is deliberately narrower than the target mesh path: MWB-1 implements identity-bound admitted surface nodes and exact weight copy. Local convex interpolation is typed; a production CDT/interpolation producer is MWB-2 work, not current execution.

## 3. IRIS Reprojection-Centered V2-A

### Current sealed design

Inputs:

- 8 native 1024 RGBA views;
- 8 exact orthographic `camera.json` authorities.

Parallel 2D evidence paths:

- frozen DINOv2-S/14 multi-level correspondence descriptors;
- learned native high-resolution shared spatial pyramid.

Canonical candidate apparatus:

- padded visual hull is a candidate-domain constraint only;
- canonical `q=(X,Y,Z)` lattice;
- exact `q -> V0..V7` reprojection;
- per-view descriptor sampling;
- robust view-evidence aggregation;
- compact evidence field `C(Z,X,Y)`;
- approximately isotropic world-space regularization;
- supported/ambiguous mode extraction;
- local continuous refinement;
- exact camera rendering to forward-depth, validity and uncertainty.

External learned geometric authority remains forward depth `d`; `P` is analytic.

### Forbidden IRIS authority

- hidden back-surface completion;
- full occupancy/SDF truth outside observed support;
- learned camera geometry in known-camera Mode G;
- learned occlusion as canonical visibility truth;
- product skeleton hierarchy, skin weights or canonical IDs.

### Current gate status

Synthetic Gate0 is PASS. Real TRAIN512 Gate0 executes externally; no result is ingested, so no real padding/spacing choice is yet canonical. No IRIS optimizer step is authorized until the deterministic gate and a separate learned-apparatus/loss seal close.

## 4. GeometricSubstrateAssembler

Current executable owner of deterministic observation-grounded geometry:

- `ObservationEvidenceIR -> P=O+dF`;
- admitted persistence/grouping;
- support-view/raster/provenance bookkeeping;
- `geometry_lineage_hash`;
- separately qualified local operators only when causally justified.

Output: `RiggingSurfaceIR S`.

It may not own hierarchy, canonical joint IDs, semantic skin, or hidden-surface completion.

`SurfaceBuilder` is a historical implementation identity only; current default producer identity is `RealSaS.GeometricSubstrateAssembler.current`.

## 5. Geppetto R6 boundary

Scientific consumer boundary:

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`.

A future frozen adapter may deterministically derive/query only information available from exact admitted `S`: resampling, normalized coordinates, qualified normals/local differentials, neighborhoods, masks and packaging. It may not import full source mesh, source rig IDs, teacher identity, hidden completion or other geometry authority.

Geppetto functional obligations:

- template-free variable control count;
- continuous joint/control location evidence;
- root evidence;
- directed parent/edge evidence;
- endogenous count/stop/unsupported behavior;
- enough evidence for downstream Compiler qualification.

External output is `SkeletonProposalIR G*`, never product truth.

Current training/evaluator-only support under `experiments/geppetto_arachne_r6_20260901/` includes the selectively ported anonymous skeleton teacher projection. It preserves deform-only controls, nearest-deform-ancestor helper skipping and multi-root teacher truth without minting product IDs. This apparatus is not product inference code.

R6 causal arms remain:

- `U0_REFERENCE_FULL_SURFACE` — diagnostic rich-surface upper bound;
- `U1_OBSERVATION_ORACLE_SUBSTRATE` — shipping-observable oracle substrate;
- `U2_PREDICTED_IRIS_SUBSTRATE` — only after U1 and IRIS qualification.

## 6. Compiler skeleton qualification

Current executable route:

```text
SkeletonProposalIR G*
 -> CanonicalGraphNodeCandidate / CanonicalGraphEdgeCandidate
 -> CanonicalGraphOptimizationRequest
 -> optimize_canonical_graph_v18_98
 -> CanonicalGraphOptimizationResult
 -> newly minted Compiler J:* IDs
 -> QualifiedSkeletonIR G
```

The optimizer is in the byte-verified historical vendor closure. Geppetto proposal IDs and internal graph-candidate IDs do not survive as canonical product IDs.

`ShapeSkeletonGraph`, surface-local relations and teacher projections are not product hierarchy authority.

## 7. Arachne R6 boundary

Scientific consumer boundary:

`B_A = ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`.

Arachne owns learned semantic influence-field proposal evidence. It does not own legal simplex/reference policy or final mesh weights.

Before predictor training, a separate `SkinFieldCodec` candidate must pass one-family and heterogeneous small-family reconstruction plus deformation-sensitive ceilings using authoritative eligible skin truth.

After codec PASS, Arachne may emit `SkinProposalIR W*`; all product candidates pass through the existing Compiler skin qualifier.

No Arachne full training is currently authorized.

## 8. Compiler skin qualification

Current executable `qualify_skin` owns:

- exact S/G lineage checks;
- legal surface/joint references;
- finite/nonnegative checks;
- deterministic optional influence sparsification within the same correction budget;
- bounded simplex repair;
- fail-closed rejection outside budget;
- `QualifiedSkinIR.skin_lineage_hash`.

It may not use BBW/QP/KKT as a hidden second semantic skin generator.

A full deterministic BBW solution, if ever opened, is a separate typed proposal/fallback arm and must traverse normal qualification and deformation proof.

## 9. MWB — editable mesh and mesh-skin binding

### Current frozen types

- `SurfaceSupportBinding`;
- `MeshDiscretizationCandidateIR M*`;
- `QualifiedEditableMeshIR M`;
- `QualifiedMeshSkinIR B`;
- `CanonicalPuppetGraph.v2`.

Every mesh vertex rest position derives from admitted S support:

`P_v = Σ_s a(v,s) P_s`, with `a>=0` and `Σa=1`.

Admissible binding modes:

- `IDENTITY_SURFACE_NODE`;
- `LOCAL_CONVEX_INTERPOLATION`.

### Current executable implementation

MWB-1 provides:

- identity-subset candidate validation;
- Compiler-minted `MV:*` mesh IDs;
- exact identity weight-row copy;
- mesh/mesh-skin/product hashing;
- stale-proof rejection.

It does **not** triangulate, insert local-convex vertices, run CDT or synthesize skin.

### Next typed mesh gate

MWB-2 may add local convex inserted vertices and/or a promoted CDT candidate only under explicit topology/support/UNKNOWN/provenance/deformation rules.

## 10. Canonical product and lineage

`CanonicalPuppetGraph.v2` is the single mesh-aware canonical product truth.

Its exact state binds:

- admitted surface hash S;
- admitted skeleton hash G;
- admitted qualified skin hash W;
- admitted editable mesh hash M;
- admitted mesh-skin hash B;
- qualification ledger;
- optional parent-state hash;
- deformation/contact/motion/editable state included in the product-state payload.

Any accepted rest-state topology/geometry/skeleton/skin/mesh-weight mutation creates a different `product_state_hash`. Proof from another state is stale and invalid.

Legacy `CanonicalPuppetGraph.v1` remains executable only for compatibility/sacrificial historical consumers; it is not the final mesh-aware architecture target.

## 11. Proof, attribution, repair and export

### Current executable exact-state envelope

- `bind_proof(Y, plan, measurements)` -> `ProofFrame(product_state_hash=Y.hash)`;
- `require_current_proof` rejects hash mismatch;
- runtime/export requires `proof.passed == true` for the exact current product;
- `RuntimePackageIR` records both source product hash and source proof hash.

### Historical heavy authority awaiting typed promotion

The historical v0.5 maximum contains richer:

- motion/deformation probes;
- playback/deformation validation;
- failure signatures;
- owner attribution;
- retry/feedback/qualification loops;
- bounded causal repair machinery.

These are architectural obligations but not all are present as current-main executable modules. The intended future loop is:

`Y -> probe -> proof -> failure signatures -> owner attribution -> RepairDirective -> Y' or abstain -> re-proof`.

Repair is owner-routed and bounded. It is not a hidden monolithic auto-rigger.

## 12. Runtime boundary

Current executable Python boundary emits `RuntimePackageIR` only from a passing proof bound to the exact product.

Historical native authority is the full v0.5 C++17 runtime SDK (`runtime/realsas_cpp` in the historical source archive), SHA-256:

`1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`.

Restoration evidence records clean CMake build, ABI smoke `1/1 PASS` and Python package -> native open/render PASS.

Current GitHub intentionally stores the runtime boundary/authority record, not a full native source mirror. Native runtime is a consumer/projection, not a canonical product owner.

Host adapters may target Unity, Unreal, Godot, custom engines or WASM without becoming product truth.

## 13. Historical external numerical authorities

The 25-Aug historical maximum and 1-Sep migration audit require these components to remain visible in architecture even though they are not current-main executable:

### Mesh / geometry

- production/exact-predicate CDT;
- constrained triangulation;
- cotangent/intrinsic repair and topology quality.

Potential future route: `S -> M*`, never hidden geometry completion.

### Weights

- BBW;
- active-set QP;
- simplex/ADMM;
- coupled/sparse KKT;
- Schur/direct sparse solves.

Potential future roles are either bounded mathematical projection preserving Arachne semantics or an explicit separately typed deterministic proposal/fallback arm.

### Deformation

- ARAP / posed ARAP / corrective deformation.

Consumes exact admitted Y; may not silently rewrite rest state.

### Physics/contact

- implicit/graph XPBD;
- garment/body and SDF contact;
- shared contact binding.

Consumes exact admitted Y; accepted rest-state mutation requires new Y' and re-proof.

### Registry rule

Current `solver_registry.py` is provenance metadata only. `EXECUTABLE_SOLVERS={}` and heavy solver resolution fails closed until a record is explicitly promoted through provenance + typed compatibility + residual/invariant + downstream parity gates.

## 14. Byte-verified historical closure inside current execution

Current main embeds a deliberately narrow vendor closure:

`compiler/vendor/realsas_v05_current_execution_closure.b64`

- 9 byte-exact v0.5 leaf modules;
- 4 controlled namespace rebinds;
- 13 declared records total;
- decoded raw SHA `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`.

The package entrypoint verifies transport part size/SHA, decoded archive size/SHA, safe extraction and every restored record byte-size/SHA before import.

This closure provides current graph/contracts/artifact dependencies. It is not a full historical source mirror.

## 15. Training / evaluator-only lane

Not product inference authority:

- clean-C0 native image integrity and visual triage apparatus;
- semantic single-riggable-character review;
- Geppetto R6 teacher projection;
- U0/U1/U2 oracle substrate experiments;
- future SkinFieldCodec ceiling apparatus;
- external RigAnything and SkinTokens clean-room reference audits.

Source-rig IDs, tails, helper identities, dense truth and full meshes may exist here strictly as teacher/evaluator evidence under leakage firewalls; they cannot cross into shipping product inputs.

## 16. Dead / superseded / reference-only lineage

### `DEAD_SUPERSEDED_CONFIRMED`

- old image-decomposition/front-brain ontology;
- authored-owner/teacher-exact product ownership assumptions;
- parallel production truth paths;
- Unity-specific product ownership as canonical truth.

### `REFERENCE_ONLY / SUPERSEDED CURRENT TOPOLOGY`

- old unsealed `GeppettoG01` neural scaffold;
- old 27D surface+interior `ConsumerTokenV1` contract;
- `SYSTEM_ARCHITECTURE_V2.md` open-mesh-seam topology;
- `IR_TYPE_SYSTEM_V1.md` claim that explicit mesh types do not exist;
- `PRODUCT_CONTRACT_V1.md` wording that treats the GeometricSubstrateAssembler rename as pending.

Historical reports remain scientific/provenance evidence even when their topology is superseded.

## 17. Architecture status classes used by the SVG

The SVG uses these semantic classes:

1. **CANONICAL MAINLINE EXECUTABLE** — exists in clean current main and is on/available to the canonical product route.
2. **SEALED / FUTURE LEARNED APPARATUS** — frozen responsibility/gate, but final model/optimizer product not yet authorized.
3. **TYPED CURRENT, IMPLEMENTATION PARTIAL** — type/contract exists but only bounded baseline implementation is executable.
4. **TRAINING / EVALUATOR ONLY** — may use teacher truth but cannot become shipping input authority.
5. **HISTORICAL EXTERNAL BYTE / NUMERICAL AUTHORITY** — preserved/re-materializable authority, not current-main execution.
6. **PROMOTION GATE / OPTIONAL FALLBACK** — potential path requiring explicit evidence before execution/product admission.
7. **DEAD / SUPERSEDED** — explicitly excluded from current canonical path.

## 18. Current open gates

- IRIS real Gate0 result ingestion and padding/spacing freeze;
- Geppetto R6 independent conditioning/candidate seal and U0/U1 ceilings;
- SkinFieldCodec design and R6-A0 reconstruction/deformation ceiling;
- Arachne predictor only after codec PASS;
- MWB-2 local-convex/CDT candidate;
- MWB-3 Arachne field-to-mesh deformation ceiling;
- ARAP/XPBD/native runtime promotion only when exact product gates require them;
- final learned training authorization remains separately gated.

No learned optimizer step is authorized by this architecture document.
