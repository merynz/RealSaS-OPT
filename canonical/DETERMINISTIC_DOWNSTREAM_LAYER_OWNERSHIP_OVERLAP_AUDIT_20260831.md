# RealSaS — Deterministic / Compiler Downstream Layer Ownership Overlap Audit — 2026-08-31

**Status:** `CLOSED_WITH_BINDING_BOUNDARY_AMENDMENTS__MESH_WEIGHT_TYPED_SEAM_OPEN_BEFORE_FINAL_ARACHNE_PRODUCT_SEAL`

## Question

After reintroducing learned Geppetto and Arachne into a system whose Compiler historically attempted much more generic model-free rigging, do the current/new deterministic layers and the restored old Compiler layers duplicate or compete for the same semantic job?

This audit covers the full current downstream stack, not only the immediate model boundaries:

1. `ObservationEvidenceIR`;
2. deterministic geometric substrate assembly (`SurfaceBuilder`, planned rename `GeometricSubstrateAssembler`);
3. morphology/local geometric relations and the historical `ShapeSkeletonGraph` concept;
4. Geppetto / `SkeletonProposalIR`;
5. Compiler graph optimizer / `QualifiedSkeletonIR`;
6. Arachne / `SkinProposalIR`;
7. Compiler skin qualification and historical BBW/QP/KKT weight solvers;
8. view-local mesh / CDT;
9. `CanonicalPuppetGraph` assembly;
10. ARAP deformation;
11. XPBD/contact;
12. motion proof / attribution / bounded repair;
13. export/runtime projection;
14. corpus-only evaluated-geometry/skin repair staging, to ensure it is not confused with runtime product repair.

## Authorities inspected

Current typed code:

- `compiler/realsas_compiler_core/types.py`
- `surface.py`
- `rig.py`
- `skin.py`
- `product.py`
- `bundle_routes.py`
- `api.py`
- `solver_registry.py`
- compiler entrypoint / vendor manifest

Canonical contracts:

- `IR_TYPE_SYSTEM_V1.md`
- `SYSTEM_ARCHITECTURE_V2.md`
- `SOLVER_AUTHORITY_MATRIX_V1.md`
- `PRODUCT_CONTRACT_V1.md`
- `OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`
- `COMPILER_ARTIFACT_ROUTING_V1.md`
- `COMPILER_RUNTIME_RESTORATION_CLOSURE_20260828.md`
- `CONSUMER_VALIDITY_INTERLOCK_CLOSURE_20260829.md`
- Geppetto/Arachne external-reference clean-room closure and R5 equivalence matrix.

Historical/restoration evidence:

- current vendor closure manifest;
- `COMPILER_RUNTIME_RESTORATION_RESULT_V1.json`;
- `HISTORICAL_AUTHORITY_POINTERS_V1.json`;
- v97.39 single-product-truth report;
- post-corpus B4/B5/B6 evaluated-geometry/skin staging code and interpretation.

Late-May v97.43 heavy numerical source remains SHA-bound external byte authority rather than source text in the current GitHub tree. Therefore this audit may bind its role from the canonical restoration contract, but may not invent uninspected implementation details.

---

# 1. Executive verdict

The overall decomposition is sound, but three seams need explicit treatment.

## PASS — no duplicate authority

- IRIS vs deterministic analytic geometry;
- geometric substrate vs mechanical skeleton hierarchy;
- Geppetto proposal vs Compiler canonical graph optimizer;
- Compiler graph optimizer vs `CanonicalPuppetGraph` assembly;
- product state vs proof/runtime projection;
- Arachne semantic weight proposal vs ARAP/XPBD downstream deformation;
- corpus repair staging vs product repair.

## BINDING AMENDMENT — real overlap risk

### A. Arachne vs historical BBW/QP/KKT

If BBW/QP/KKT is allowed to independently regenerate semantically new weights from `S+G` after Arachne has predicted `W*`, then two separate weight authorities exist and Arachne becomes partly redundant.

Therefore the normal learned product path is now bound as:

`Arachne semantic influence proposal -> Compiler bounded legality/projection -> QualifiedSkinIR`.

A numerical skin solver inside qualification may enforce mathematical admissibility, smoothness or bounded projection **only while preserving the admitted semantic support/evidence supplied by Arachne within a preregistered correction budget**. It may not silently create a materially different skinning solution.

If a deterministic BBW system is intentionally used as a full alternative skin generator, it must be represented as a separately typed **proposal producer / fallback arm** that emits a `SkinProposalIR` (or a future explicitly equivalent proposal type) and is compared/qualified through the same downstream route. It cannot hide inside `Compiler.skin_qualification` as a second semantic owner.

### B. Mesh/discretization typed seam is missing

The current executable IR library contains `RiggingSurfaceIR`, skeleton proposal/qualified IRs, skin proposal/qualified IRs and `CanonicalPuppetGraph`, but **no explicit typed editable mesh/discretization IR and no explicit mesh-vertex weight projection/binding IR**.

Meanwhile the solver authority documents retain view-local CDT and historical mesh/weight/deformation solvers. This means the conceptual responsibility exists but the modern typed seam by which it enters product state is underspecified.

This is not evidence of current duplicate execution; it is an **integration gap that could create duplication later** if CDT, Arachne weights and historical BBW are wired independently.

Before final Arachne/product architecture seal, a separate pre-output contract must freeze:

1. what the editable/view-local mesh representation is;
2. whether it is canonical product data or a deterministic projection of `RiggingSurfaceIR`;
3. exact lineage from mesh vertices/elements back to observation-grounded surface/raster evidence;
4. how qualified skin fields are evaluated/transferred to mesh vertices;
5. whether transfer is interpolation/query/projection and what residual is allowed;
6. which component owns topology edits;
7. how mesh/weight changes create a new `CanonicalPuppetGraph` state and invalidate stale proof.

Until this contract is frozen, no historical CDT/BBW path may silently become the hidden product mesh/weight authority.

### C. Current-execution provenance wording needs precision

`solver_registry.py` names imports for `realsas_mesh`, `realsas_weight` and `realsas_deformation`, but those source packages are not present in the current GitHub tree and the canonical package entrypoint does not import `solver_registry.py`.

The narrow vendor closure contains the historical graph optimizer/contracts/artifact pieces, not mesh/BBW/ARAP/XPBD modules. The restoration documents separately preserve those heavier numerical implementations as historical/external authorities by SHA.

Therefore `EXECUTABLE_BASELINE` in the solver matrix must be read as **restored/historical executable baseline authority**, not as proof that a clean checkout of the current `main` facade presently contains all numerical solver modules. No final architecture may assume local availability without an explicit executable-restoration/parity step.

---

# 2. Layer-by-layer ownership matrix

| Layer / artifact | May own | Must not own | Overlap verdict |
|---|---|---|---|
| `ObservationEvidenceIR` / IRIS | learned observation evidence, depth/posterior/uncertainty under its contract | canonical P authority, skeleton, weights, product IDs | CLEAN |
| `GeometricSubstrateAssembler` | analytic P, admitted persistence, provenance/support/raster binding, qualified local geometric operators | mechanical parent/root graph, skin semantics, hidden completion | CLEAN if local relations remain geometric |
| `SurfaceRelation` / historical morphology graph | local geometric/morphological adjacency/evidence | canonical skeleton hierarchy | CLEAN but naming/typing firewall binding |
| Geppetto / `SkeletonProposalIR` | control existence/position, root and directed-edge evidence, uncertainty/support | final canonical tree/IDs | CLEAN |
| Compiler graph qualification | global admissible root/parent selection, graph invariants, canonical IDs, fail closed | discovering missing mechanical anatomy from raw geometry | CLEAN |
| Arachne / `SkinProposalIR` | semantic per-joint influence field/proposal, uncertainty/field representation | legal/canonical IDs, silent product truth | CLEAN |
| Compiler skin qualification | lineage/legal refs, nonnegative/finite checks, simplex/influence policy, bounded mathematical projection | independent semantic skin synthesis in normal Arachne path | AMENDED |
| BBW/QP/KKT | bounded projection/regularization under Arachne semantics, or explicit separate fallback proposal arm | hidden second normal-path skin authority | AMENDED |
| CDT/view mesh | deterministic editable discretization/projection of admitted geometry | hidden surface completion, changing observation geometry truth, rig hierarchy | CONTRACT GAP: explicit mesh IR needed |
| `CanonicalPuppetGraph` | single product-state lineage/hash and admitted component bindings | re-solving geometry/skeleton/skin | CLEAN |
| ARAP | pose/deformation solve on exact product state + residuals | rest-rig semantics or weight rewriting without new state | CLEAN |
| XPBD/contact | contact/constraint correction/report on exact state | canonical rest geometry/skin rewrite | CLEAN |
| Proof | measure/pass/fail exact Y | mutate Y | CLEAN |
| Attribution | owner diagnosis from failures | silently repair | CLEAN |
| Repair | issue bounded directive / apply admitted owner-scoped mutation creating Y' | become hidden generic auto-rigger | CLEAN if owner-routed and re-proofed |
| Runtime/export | projection of PASS-proven exact Y | second canonical truth | CLEAN |
| post-corpus B4/B5/B6 | teacher/corpus extraction, normalization and quarantine staging | shipping product repair/runtime semantics | CLEAN, separate scientific data authority |

---

# 3. Detailed findings

## 3.1 Geometric substrate vs skeleton graph

The current `surface.py` is narrower than the older substrate design notes. It analytically forms `P = O + dF`, validates admitted persistence groups, fuses observations and preserves support/provenance/raster lineage. It currently rejects requested normal derivation rather than inventing an implementation.

`RiggingSurfaceIR.local_relations` therefore remains safe **only if `relation_kind` is geometric/morphological** (neighborhood, sheet continuity, reprojection support, local surface relation, etc.). A relation that asserts `PARENT`, `ROOT`, `BONE`, authored control identity or other mechanical hierarchy semantics belongs in Geppetto proposal evidence or the Compiler graph work layer, not in the substrate.

The historical `ShapeSkeletonGraph` naming is especially dangerous because it sounds like product hierarchy. Its canonical status remains morphology/surface evidence only; it must never be consumed as a final tree.

## 3.2 Geppetto vs Compiler graph optimizer

No problematic duplication was found.

Geppetto supplies proposal-local joints/edges and scores. `qualify_skeleton` rebinds those as graph candidates, invokes the restored global optimizer, rejects invalid solutions, then mints new canonical joint IDs. This is a clean evidence-vs-global-authority split.

The existing optimizer is valuable historical Compiler work and should remain. Geppetto should not be expanded to duplicate exact arborescence legality merely because a reference model predicts parents directly.

However the Compiler must not use geometry-only heuristics to invent materially missing joints/edges when Geppetto evidence is absent, except through an explicitly separately typed bounded-completion/fallback contract. Otherwise the old model-free generic-rig problem reappears inside qualification.

## 3.3 Arachne vs skin qualifier vs BBW

This is the main overlap seam.

The current Python qualifier is clean: it checks lineage/references, finite/nonnegative values, optional bounded top-k and bounded simplex correction. It does not infer semantic influence ownership.

The historical BBW/QP/KKT authority is different: a full BBW solve can itself generate a weight field from geometry and handles. If inserted after Arachne without a preservation contract, it becomes a second Arachne.

Binding normal-path rule:

`W* semantic evidence is owned by Arachne; Compiler numerical work may only make a bounded admissible projection of W* unless a separately typed fallback/proposal route is explicitly selected before qualification.`

A solver correction that changes the material support pattern, moves major influence mass to different joints, or repairs a semantically bad Arachne field is **not qualification**. It is a new proposal and must be treated/attributed/tested as such.

This rule is compatible with the historical stronger QP/KKT work: strong numerical machinery remains useful, but its role is mathematical feasibility/regularity or an explicit alternative proposal arm, not hidden semantic authorship.

## 3.4 Mesh/CDT vs geometric substrate

CDT should discretize an admitted observation-grounded surface/view for editing, deformation and rendering. It must not define a second geometric truth.

A triangle or mesh edge is therefore a **discretization relation**, not evidence that an unobserved 3D surface exists. Any fill across unknown/occluded regions requires an explicit product policy and typed uncertainty/coverage contract; it cannot enter because triangulation needs a closed mesh.

The current type library does not make this relationship explicit. This is the main IR-layer gap found by the audit.

## 3.5 Skin field vs mesh-vertex weights

The current `QualifiedSkinIR` indexes `surface_id`, while the historical/runtime mesh stack naturally operates on mesh vertices/elements. A deterministic mapping is therefore required somewhere.

That mapping is not allowed to become a second weight predictor. It must either:

- evaluate an already-qualified continuous Arachne field at mesh vertices; or
- interpolate/project admitted qualified surface-node weights under a frozen local geometric rule;

and report coverage/residual/failure. If the mesh asks for weights in a region unsupported by the qualified field/substrate, fail/UNKNOWN/owner-routed repair is preferable to unconstrained BBW invention in the normal learned path.

Exact representation choice remains intentionally unsealed; the obligation is binding.

## 3.6 CanonicalPuppetGraph

No duplication found. The current object is a product-state/lineage envelope carrying admitted S/G/W hashes plus deformation/contact/motion/editable state. `assemble_product` validates lineage and hashes the exact admitted state; it does not re-solve any proposal.

This is the correct location for single product truth, not another geometry/skeleton/weight algorithm.

## 3.7 ARAP / XPBD / proof

These are genuinely downstream numerical operators and do not overlap Geppetto/Arachne **provided they consume exact admitted Y and emit posed/derived state + residuals rather than silently rewriting rest semantics**.

A deformation/contact solver may expose failure evidence that causes a new owner-scoped attempt, but any accepted rest skeleton/skin/mesh mutation must create a new product state and mandatory re-proof.

## 3.8 Repair

Repair must remain orchestration/owner routing, not a second generic rigging engine.

Allowed pattern:

`proof failure -> attribution -> RepairDirective(owner=GEPPETTO/ARACHNE/MESH/...) -> bounded owner-specific new candidate/state -> Y' -> re-proof`.

Forbidden pattern:

`proof failure -> global heuristic silently edits arbitrary joints + weights + mesh until proof passes`.

The latter recreates the historical monolithic compiler and destroys causal attribution.

## 3.9 Corpus repair is not product repair

Post-corpus B4/B5/B6 operates on authoritative training assets: evaluated REST geometry, preserved deform layers, canonical normalization, capability/quarantine checks and read-only/frozen staging. It explicitly avoids inventing missing skin and keeps corpus mutation separate.

These scripts are data-authority preparation and have no runtime product-repair ownership. Their use of the word `repair` must not be interpreted as an executable Compiler repair layer.

---

# 4. Current execution / historical numerical authority distinction

The current `realsas_compiler_core` entrypoint reconstructs a narrow SHA-verified historical closure containing graph/contracts/artifact dependencies and then exposes current `types/surface/rig/skin/product` code.

The vendor manifest does **not** include the full mesh/BBW/ARAP/XPBD source modules. `solver_registry.py` references those package names, but it is not imported by the canonical package entrypoint and the referenced packages are not in the current GitHub tree.

Therefore:

- graph qualification is locally/restored executable through the current canonical loop;
- the current Python skin qualifier is locally executable;
- historical mesh/BBW/ARAP/XPBD capability has regression/restoration evidence and SHA-bound source authority;
- but the final modern typed integration of those numerical layers is **not yet a self-contained current-main implementation contract**.

This distinction must remain explicit to prevent accidental architecture claims.

---

# 5. Binding owner rules after this audit

1. **One semantic owner per question.** Geometric observation truth, skeleton semantics, skin semantics, mesh discretization, numerical deformation and product admission are separate domains.
2. **Qualification may constrain; it may not secretly re-infer the proposal semantics.**
3. **Alternative deterministic solvers are proposal arms when they create semantics.** A full BBW-generated skin is a proposal, not invisible qualification.
4. **Derived topology is not hidden geometry.** CDT cannot convert UNKNOWN into observed surface truth.
5. **Mesh-weight transfer must be typed and measurable.** No implicit nearest-neighbor/barycentric/BBW handoff becomes product authority by convenience.
6. **Every rest-state mutation produces new Y.** ARAP/XPBD/proof artifacts cannot mutate canonical rest state in place.
7. **Repair routes to the owner.** It does not become a parallel monolithic auto-rigger.
8. **Historical stronger numerical code stays reference/external authority until exact typed executable restoration/parity.**

---

# 6. Required pre-seal follow-up

## Does not block

- deterministic IRIS real Gate0;
- Geppetto external-reference closure already completed;
- Geppetto R6 U0/U1 candidate work once its own prerequisites close.

## Blocks final Arachne/product apparatus seal

Before final Arachne learned architecture/loss/training seal, freeze a `MESH_WEIGHT_BINDING_CONTRACT` (name may change before seal) that states:

- editable mesh/discretization type and owner;
- CDT role;
- surface-to-mesh lineage;
- Arachne field query/projection semantics;
- qualified surface-weight to mesh-weight mapping;
- BBW/QP/KKT normal-path vs fallback role;
- residual/failure thresholds;
- exact relationship to `CanonicalPuppetGraph` and proof invalidation.

No optimizer step is authorized by this audit.

---

# 7. Final verdict

`GEOMETRIC_SUBSTRATE_VS_GEPPETTO_OVERLAP = NO`

`GEPPETTO_VS_COMPILER_GRAPH_OVERLAP = NO__COMPLEMENTARY`

`ARACHNE_VS_CURRENT_PYTHON_SKIN_QUALIFIER_OVERLAP = NO__COMPLEMENTARY`

`ARACHNE_VS_UNCONSTRAINED_FULL_BBW_OVERLAP = YES__MUST_BE_ROLE_SEPARATED`

`CDT_VS_GEOMETRIC_TRUTH_OVERLAP = NO_IF_DERIVED_DISCRETIZATION_ONLY`

`TYPED_MESH_DISCRETIZATION_IR = MISSING_IN_CURRENT_V1__MUST_CLOSE`

`TYPED_MESH_WEIGHT_BINDING_IR = MISSING_IN_CURRENT_V1__MUST_CLOSE`

`CANONICAL_PUPPET_GRAPH_DUPLICATES_S_G_W = NO`

`ARAP_XPBD_VS_LEARNED_RIG_SEMANTICS_OVERLAP = NO_IF_DERIVED_EXACT_STATE_ONLY`

`PRODUCT_REPAIR_PARALLEL_AUTORIGGER = FORBIDDEN`

`CORPUS_REPAIR_VS_PRODUCT_REPAIR = DISTINCT`

Overall result:

> The architecture is not suffering from a general duplicate-Compiler problem. The old Compiler work is mostly sitting at the correct lower numerical/qualification layers. The material architecture risk is specifically the weight-solver boundary plus an under-typed mesh/weight handoff. Closing that seam preserves the value of the historical Compiler without allowing it to compete with Geppetto/Arachne.
