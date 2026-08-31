# RealSaS — Solver Authority Matrix V1

**Updated:** 2026-08-31  
**Status:** `CANONICAL_SOLVER_RESPONSIBILITY_MAP__SEMANTIC_NONOVERLAP_BOUND__EXECUTION_PROVENANCE_SPLIT_EXPLICIT`

Solvers are typed compiler transformations/validators. Every solver has an authority owner, exact input/output class, invariants, residual/failure behavior and provenance. Newer timestamp is never authority.

**Execution-provenance distinction:** the current GitHub `realsas_compiler_core` facade contains the typed surface/rig/skin/product code and a narrow restored graph/contracts vendor closure. Historical CDT/BBW/ARAP/XPBD implementations have restoration/regression evidence and SHA-bound source authority, but their source packages are not contained in the current GitHub tree/facade closure. Therefore the table distinguishes **current-main self-contained execution** from **historical/external executable authority**.

| Domain | Typed input | Typed output | Canonical owner | Current-main status | Historical / stronger numerical authority |
|---|---|---|---|---|---|
| Surface/persistence | ray/depth observation evidence | `RiggingSurfaceIR S` | `GeometricSubstrateAssembler` (historical implementation name `SurfaceBuilder`) | self-contained analytic `P=O+dF` + admitted persistence | deterministic-first; preserve ambiguity/fail closed |
| View-local/editable mesh | `S + view` | **typed mesh/discretization IR TO BE FROZEN** | Compiler geometry | typed seam not yet explicit in IR V1 | v0.5 production CDT regression authority; v97.43 exact-predicate CDT/cotangent reference until parity |
| Rig graph | `S + G*` | `QualifiedSkeletonIR G` | Compiler rig | self-contained adapter + restored `optimize_canonical_graph_v18_98` | root/parent arborescence + deterministic tie break; bounded MILP shadow/escalation where configured |
| Skin qualification | `S + G + W*` | `QualifiedSkinIR W` | Compiler skin | self-contained bounded fail-closed Python qualifier | v97.43 QP/KKT family may be promoted only under the semantic-preservation rule below |
| Full deterministic skin synthesis, if deliberately used | `S + G` (+ explicit solver policy) | **`SkinProposalIR` / future equivalent proposal type, not silently `QualifiedSkinIR`** | separate deterministic proposal arm | not current normal learned path | historical BBW/QP/KKT is candidate/reference; must be explicitly restored, typed and qualified |
| Mesh-weight binding | qualified skin + typed mesh | **typed mesh-weight binding/projection IR TO BE FROZEN** | Compiler projection | missing explicit IR seam | interpolation/query/projection or bounded numerical solve only under frozen semantics |
| Deformation | exact `Y + pose/probe` | posed geometry + residuals | Compiler deformation | not self-contained in current GitHub facade | v0.5 ARAP historical executable baseline; v97.43 sparse ARAP reference until residual/deformation parity |
| Contact/physics | exact `Y + contact` | corrected pose + report | Compiler contact | not self-contained in current GitHub facade | v0.5 XPBD/SDF/contact historical baseline; v97.43 implicit/graph XPBD + contact binding until parity |
| Motion proof | exact `Y + probe plan` | proof/failure artifacts | Compiler proof | typed proof binding self-contained; heavy historical proof/playback remains restoration authority where required | exact-state binding required |
| Attribution | proof failure + exact `Y` | attribution report | Compiler proof/repair | historical/restored evidence; exact heavy route restored only when required | causal regression required |
| Repair | attribution/directive + exact `Y` | `Y'` or abstain | Compiler repair/orchestrator | typed `RepairDirective` self-contained; heavy mutation implementations are owner-specific/restoration work | mandatory new state + re-proof |
| Export | PASS-proven exact `Y` | runtime projection | Compiler export | current typed proof/runtime binding self-contained; native runtime has separate restored source authority | binds product hash + proof hash; native ABI fixture |

## Rig hierarchy solver

Geppetto proposals are converted to `CanonicalGraphNodeCandidate/EdgeCandidate`, then to `CanonicalGraphOptimizationRequest`. `optimize_canonical_graph_v18_98` owns admitted root/parent graph selection. `CanonicalGraphOptimizationResult` is qualification evidence, not product truth. Compiler then mints new canonical joint IDs and materializes `QualifiedSkeletonIR`.

`ShapeSkeletonGraph` is not this solver and is not product hierarchy. `RiggingSurfaceIR.local_relations` are geometric/morphological relations only and may not encode canonical parent/root semantics.

## Skin semantic non-duplication rule

Arachne owns the normal learned-path **semantic influence proposal**. Compiler qualification owns legality and bounded mathematical projection.

A numerical QP/KKT/BBW operation used inside normal qualification must preserve the admitted Arachne semantic support/evidence within a preregistered correction budget. It may enforce finite/nonnegative/simplex/influence constraints and justified bounded regularity, but it may not silently synthesize a materially different skin field.

If BBW/QP/KKT is intentionally asked to create the semantic skin field from `S+G`, that operation is a **separate deterministic proposal producer/fallback arm**. Its output must be typed/provenanced as proposal evidence and pass the same qualification/deformation-proof route. It does not get canonical authority merely because it is deterministic.

Simple clipping/renormalization is not a substitute for a stronger qualified numerical projection when it materially changes semantics. Conversely, a strong solver is not license to hide semantic re-inference inside qualification.

## Mesh / CDT rule

View-local/editable triangulation is a discretization/projection of admitted observation-grounded geometry. It may not convert UNKNOWN/unobserved surface into geometric truth by convenience.

The current IR V1 lacks an explicit typed mesh/discretization and mesh-weight-binding seam. Final Arachne/product seal is blocked on a separate mesh-weight binding contract defining lineage, topology ownership, qualified skin transfer/query semantics, residuals and proof invalidation.

## ARAP / XPBD rule

Deformation/contact solvers consume an exact admitted product state and emit posed/derived geometry, corrections and residuals. They may not silently rewrite canonical rest geometry, skeleton or skin. Any accepted rest-state mutation creates a new `CanonicalPuppetGraph Y'` and requires re-proof.

## Current-main execution note

`compiler/realsas_compiler_core/solver_registry.py` records historical solver module names, but it is not imported by the canonical package entrypoint and the referenced `realsas_mesh`, `realsas_weight` and `realsas_deformation` packages are not present in the current GitHub tree. The narrow vendor manifest likewise does not contain those modules.

Accordingly, terms such as `v0.5 executable baseline` mean **historically restored/executable authority with regression evidence**, not that every numerical solver is presently bundled into a clean checkout of `main`. Exact modern product integration requires explicit typed restoration/promotion.

## Solver report envelope

Where mathematically meaningful, solver-backed promotion records implementation/provenance, input IR hashes, config hash, convergence status, objective, primal/dual residuals, active/violated constraints, warnings, deterministic tie-break and output IR hash.

## Promotion rule

A replacement/promoted solver requires:

1. exact source/implementation provenance;
2. typed IR compatibility;
3. invariant/residual regressions;
4. downstream non-inferiority or improvement;
5. exact product/proof lineage compatibility;
6. fail-closed behavior outside admitted policy;
7. proof that it does not acquire a second semantic authority already owned by IRIS/Geppetto/Arachne or another deterministic layer.

Performance is considered only after correctness and authority-boundary parity.

Authority for the overlap findings: `DETERMINISTIC_DOWNSTREAM_LAYER_OWNERSHIP_OVERLAP_AUDIT_20260831.md`.
