# RealSaS — Solver Authority Matrix V1

**Date:** 2026-08-28  
**Status:** `CANONICAL_SOLVER_RESPONSIBILITY_MAP__EXECUTABLE_BASELINES_RESTORED__NUMERICAL_PROMOTION_EVIDENCE_CONTROLLED`

Solvers are typed compiler transformations/validators. Every solver has an authority owner, exact input/output class, invariants, residual/failure behavior and provenance. Newer timestamp is never authority.

| Domain | Typed input | Typed output | Canonical owner | Current executable | Stronger historical reference / rule |
|---|---|---|---|---|---|
| Surface/persistence | ray/depth observation evidence | `RiggingSurfaceIR S` | SurfaceBuilder | analytic `P=O+dF` + admitted persistence | deterministic-first; preserve ambiguity/fail closed |
| View-local mesh | `S + view` | editable triangulation | Compiler geometry | v0.5 production CDT baseline | v97.43 exact-predicate CDT/cotangent remains reference until parity |
| Rig graph | `S + G*` | `QualifiedSkeletonIR G` | Compiler rig | restored `optimize_canonical_graph_v18_98` | root/parent arborescence + deterministic tie break; bounded MILP shadow/escalation where configured |
| Skin qualification/solve | `S + G + W*` | `QualifiedSkinIR W` | Compiler skin | bounded fail-closed Python qualifier / v0.5 BBW baseline | v97.43 bound-active QP, coupled simplex/ADMM/KKT, sparse LDLT/Schur and dual-feasibility authority until parity |
| Deformation | exact `Y + pose/probe` | posed geometry + residuals | Compiler deformation | v0.5 ARAP baseline | v97.43 sparse ARAP reference until residual/deformation parity |
| Contact/physics | exact `Y + contact` | corrected pose + report | Compiler contact | v0.5 XPBD/SDF/contact baseline | v97.43 implicit/graph XPBD + contact binding until parity |
| Motion proof | exact `Y + probe plan` | proof/failure artifacts | Compiler proof | restored v0.5 proof/playback | exact-state binding required |
| Attribution | proof failure + exact `Y` | attribution report | Compiler proof/repair | restored v0.5 causal attribution | causal regression required |
| Repair | attribution/directive + exact `Y` | `Y'` or abstain | Compiler repair | restored bounded repair | mandatory re-proof |
| Export | PASS-proven exact `Y` | runtime projection | Compiler export | restored `.rss/.rsr` | binds product hash + proof hash; native ABI fixture |

## Rig hierarchy solver

Geppetto proposals are converted to `CanonicalGraphNodeCandidate/EdgeCandidate`, then to `CanonicalGraphOptimizationRequest`. `optimize_canonical_graph_v18_98` owns admitted root/parent graph selection. `CanonicalGraphOptimizationResult` is qualification evidence, not product truth. Compiler then mints new canonical joint IDs and materializes `QualifiedSkeletonIR`.

`ShapeSkeletonGraph` is not this solver and is not product hierarchy.

## Skin solver rule

Simple clipping/renormalization is not a substitute for a stronger qualified numerical solve when it materially changes semantics. The current typed qualifier permits only bounded, auditable simplex correction and fails closed outside budget. v97.43 BBW/QP/KKT remains the numerical ceiling/reference until an executable implementation demonstrates formulation + residual + deformation parity.

## Solver report envelope

Where mathematically meaningful, solver-backed promotion records implementation/provenance, input IR hashes, config hash, convergence status, objective, primal/dual residuals, active/violated constraints, warnings, deterministic tie-break and output IR hash.

## Promotion rule

A replacement/promoted solver requires:

1. exact source/implementation provenance;
2. typed IR compatibility;
3. invariant/residual regressions;
4. downstream non-inferiority or improvement;
5. exact product/proof lineage compatibility;
6. fail-closed behavior outside admitted policy.

Performance is considered only after correctness parity.
