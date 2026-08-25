# RealSaS — Solver Authority Matrix V1

**Date:** 2026-08-25  
**Status:** `CANONICAL_SOLVER_RESPONSIBILITY_MAP__IMPLEMENTATION_SELECTION_PENDING_SOURCE_DIFF`

Solvers are first-class compiler architecture. Each solver must have explicit typed inputs/outputs, invariants, residuals, failure behavior and authority ownership.

Historical implementation selection is **not** frozen here. The 2026-08-25 Compiler/Runtime Historical Maximum audit establishes source families to restore and reconcile; exact files remain pending forensic diff/dependency closure.

## 1. Matrix

| Domain | Typed input | Typed output | Canonical owner | Historical authority / candidate | Required evidence |
|---|---|---|---|---|---|
| Surface fusion / topology | `ObservationEvidenceIR E` | `RiggingSurfaceIR S` | SurfaceBuilder / Compiler geometry authority | deterministic fusion; constrained topology; late-May CDT/cotangent where applicable | reprojection/cycle, sheet validity, downstream non-inferiority |
| View-local mesh | `RiggingSurfaceIR S + view` | editable 2D mesh/triangulation | Compiler geometry authority | production CDT / constrained triangulation | contour fidelity, no invalid triangles, deterministic fixture |
| Rig graph qualification | `S + SkeletonProposalIR G*` | `QualifiedSkeletonIR G` | Compiler rig authority | historical rig assembly/graph optimization; constrained DAG solver | root/parent/DAG invariants, support sanity, determinism |
| Skin qualification / solve | `S + G + SkinProposalIR W*` | `QualifiedSkinIR W` | Compiler skin authority | late-May BBW / active-set QP / KKT family; v0.5 wrappers where stronger/cleaner | simplex/bounds, KKT/residual thresholds, deformation quality |
| Deformation | `CanonicalPuppetGraph Y + pose/probe` | posed geometry + residual report | Compiler deformation authority | ARAP / posed ARAP / corrective deformation | energy/residual, foldover/sanity, playback fixture |
| Contact / physics | `Y + contact state` | corrected posed state + contact report | Compiler contact authority | implicit/graph XPBD, SDF/contact | penetration/contact residual, stability |
| Motion proof | exact `Y` + probe plan | `MotionProofReport` | Compiler proof authority | Aug v0.5 proof/playback system | exact-state binding, deterministic failure signatures |
| Attribution | proof failure + exact `Y` | `OwnerAttributionReport` | Compiler proof/repair authority | Aug v0.5 causal attribution | attribution calibration/causal regression |
| Repair | attribution + `RepairDirective` + exact `Y` | candidate `Y'` or abstain | Compiler repair authority | Aug v0.5 bounded repair/orchestrator | boundedness, causal effect, mandatory re-proof |
| Export | proven final `Y` | runtime package | Compiler export authority | Aug `.rss/.realsas/.rsr` | source-state hash binding, schema/ABI fixture |

## 2. Surface geometry / topology solver

Current canonical starting point is deterministic-first:

```text
P/N/support/provenance/persistence/image-grid evidence
        ↓
candidate local relations
        ↓
reprojection + distance + normal + silhouette/discontinuity tests
        ↓
constrained component/sheet graph
        ↓
RiggingSurfaceIR S
```

A point cloud alone is not topology. Surface topology is distinct from skeleton topology and deformation influence topology.

Hard requirements:

- touching-but-disconnected sheets must not be merged merely by Euclidean proximity;
- cross-view duplicate observations may fuse only with support/cycle-consistent evidence;
- boundaries/seams remain representable;
- topology construction must be deterministic given admitted evidence/configuration;
- if global topology is underdetermined, preserve ambiguity or fail closed rather than fabricate hidden mesh truth.

Learned local relation evidence is admissible only after deterministic topology is shown downstream-inferior.

## 3. View-local triangulation / CDT

The shipping product is 2D/8-view. A canonical hidden watertight 3D face mesh is not required merely because historical 3D riggers had one.

Where a 2D editable deformation mesh is needed:

```text
RiggingSurfaceIR S
        ↓ project to view
qualified sheet/component boundaries
        ↓
constrained triangulation / CDT
        ↓
view-local editable mesh
```

Historical late-May CDT/cotangent implementations are restoration candidates and must be source-diffed against v0.5 ports before replacement.

## 4. Rig graph solver

Geppetto produces `G*`; Compiler owns `G`.

Potential deterministic/global tools include:

- constrained parent selection;
- maximum spanning arborescence where formulation fits;
- MILP/ILP for coupled discrete constraints where justified;
- duplicate/helper suppression;
- bounded structural completion;
- root uniqueness and DAG normalization.

The exact solver is not frozen. The invariants are.

Required hard checks:

```text
valid references
acyclic
root policy
surface support
bounded control count / completion policy
canonical IDs
reproducible objective + tie-break
```

## 5. Skin weight solver

Arachne may generate a strong learned `W*`; Compiler still owns admissible `W`.

Historical numerical families to preserve/reconcile:

```text
BBW
bound-active QP
simplex coupling
ADMM variants
coupled KKT
sparse KKT
Schur/direct sparse solves
hard residual validation
```

Typical invariant class:

```text
w_ij finite
w_ij >= 0 where policy requires
Σ_j w_ij = 1
invalid component influences forbidden
influence sparsity/count policy respected after controlled projection
```

A solver must emit residuals/active constraints. Silent clipping and renormalization cannot substitute for a qualified solve when they materially change deformation semantics.

## 6. ARAP / deformation solver

ARAP is both a production deformer/corrective candidate and a proof instrument.

Required outputs include enough information to distinguish:

- numerical non-convergence;
- foldover/inversion;
- excessive local distortion;
- rig/weight-caused deformation failure;
- contact-caused failure;
- unsupported geometry.

A visually plausible frame is insufficient without residual/sanity evidence.

## 7. XPBD / contact solver

Historical families include implicit XPBD, graph XPBD, garment/body contact and SDF contact.

Canonical rules:

- contact state is derived from the same product lineage;
- contact correction cannot silently mutate canonical rest state;
- penetration/stability residuals are reported;
- solver failure routes to typed proof failure/repair or abstention.

## 8. Motion proof solver

Motion proof is not a separate product owner.

```text
exact CanonicalPuppetGraph Y
        ↓
MotionProbeSpecification
        ↓
ResolvedMotionProbePlan
        ↓
measurement/deformation/contact execution
        ↓
MotionMeasurementReport
        ↓
MotionProofReport
```

The proof system must refuse a stale/mismatched `Y` hash.

## 9. Attribution and repair

Attribution answers **which responsibility domain of the current product caused the observed failure**, not which hidden teacher owner should have existed.

Repair may target:

```text
geometry/surface
rig graph
skin weights
deformer
contact
motion binding
export/runtime
```

Every repair is bounded and produces either:

```text
new candidate canonical state Y' → mandatory re-proof
```

or

```text
ABSTAIN / FAIL-CLOSED
```

## 10. Solver selection rule

For every restored numerical subsystem compare:

1. mathematical formulation;
2. constraints and boundary conditions;
3. convergence/residual checks;
4. deterministic tie-breaking;
5. downstream consumers;
6. product/proof lineage binding;
7. test evidence and historical regressions;
8. performance only after correctness parity.

Prefer a cleaner Aug wrapper only if it preserves or exceeds the stronger late-May mathematics. Never choose solely by timestamp or language.

## 11. Required solver report envelope

Every solver-backed canonical transition should produce a machine-readable envelope comparable to:

```text
SolverReport
  solver_id
  implementation_hash
  input_ir_hashes[]
  config_hash
  convergence_status
  objective_before / after where meaningful
  primal_residual
  dual_residual where meaningful
  constraint_violations[]
  numerical_warnings[]
  output_ir_hash
  deterministic_seed/tie_break_policy
```

Not every solver has primal/dual residuals, but every solver must expose the strongest meaningful correctness diagnostics for its formulation.

## 12. Promotion policy

A restored or new solver may become production authority only when:

- exact implementation/source provenance is recorded;
- typed IR contract is satisfied;
- invariant/residual regression fixtures pass;
- downstream quality is non-inferior or better;
- proof/repair sees the same canonical product lineage;
- failure is fail-closed rather than silently approximate beyond policy.
