# RealSaS — Skin Field / Topology Continuity Execution Plan — 2026-10-03

**Status:** ACTIVE — FAIL-CLOSED — AUDIT BRANCH ONLY  
**Branch:** `audit/canonical-caa-mechanics-geometry-lock-20261002`  
**Owner:** RealSaS compiler/mechanics audit  
**Witness:** Knight is a witness/adversarial case only; no Knight-specific thresholds, regions, semantics, vertex IDs or patches are permitted.  
**Main:** MUST NOT be modified until the generic contract is proven and separately admitted.

## 1. Problem statement

V9 static topology/geometry reached zero residual and static topology PASS, but fresh skin realization on the V9 carrier fails deformation qualification badly. The evidence currently localizes the failure to the contract between skin-field evidence and the final carrier discretization, not to Geppetto skeleton kinematics.

Observed facts:

- archived carrier + archived skin under the same motions: max edge stretch about 3.62x;
- fresh V9 CUDA/BF16 inference: G3 max condition about 518.27, 51/51 sampled motion frames fail, max edge stretch about 521.37x;
- archived-kinematics counterfactual on V9 did not improve the result (G3 about 752.11, max edge about 551.14x);
- V9 skin-field edge gradient p99 about 153.54 vs archived about 29.08; max about 3475.76 vs archived about 125.03;
- geometry-only carrier smoothing reduced the catastrophe substantially (max edge about 521.37x -> 13.48x at the strongest tested setting), proving that continuity is causal, but required an unacceptably large p95 L1 skin correction of about 1.06 and still failed 49/51 frames;
- topology-aware prolongation from the archived skin remained mechanically invalid;
- some Stage15 source components have no Stage18 coarse face, so the archived coarse carrier cannot define a globally complete FEM prolongation field.

## 2. Architectural conclusion under test

The product authority must not be "weights attached to one exact mesh vertex list."

Target abstraction:

```
Arachne qualified evidence
        ↓
canonical skin / deformation field evidence
        ↓
deterministic carrier-specific realization
        ↓
G3B skin-topology compatibility qualification
        ↓
G3 local deformation qualification
        ↓
Stage35 exact-motion qualification
        ↓
QualifiedSkinIR
```

A topology/refinement change is allowed to change the discretization. It is not allowed to create an unrelated deformation field.

## 2A. Existing canonical contract alignment

This work is **not** authorization for a new learned-owner architecture.

`canonical/PRODUCT_CONTRACT_V1.md` already states:

- Arachne owns learned **semantic influence-field / editable skin proposal** evidence, including proposed continuous field/weight evidence;
- Compiler skin qualification may perform only a **bounded** mathematical/admissibility projection of that evidence;
- a materially different BBW/QP/KKT semantic solution would be a separately typed proposal producer, not invisible qualification;
- an explicit editable-mesh / mesh-weight projection seam **must close before final Arachne/product seal**;
- that seam must define how Arachne's qualified field/weights are evaluated/interpolated/projected onto mesh vertices, with residual/failure/coverage policy.

Therefore the active program is best understood as **closing the already-preregistered mesh-weight binding seam and enforcing its continuity under carrier changes**, not inventing a Knight-driven architecture revision.

A new IR type such as `SkinFieldEvidenceIR` is optional implementation detail only. Prefer the least-disruptive typed extension compatible with the existing product contract.

## 2B. Existing G3B implementation boundary

`compiler/realsas_compiler_core/mesh/skin_topology_compatibility_v1.py` already provides a deterministic topology x skin compatibility proof. It can:

- stress the exact candidate under canonical joint-frame probes;
- detect unsafe faces/edges;
- propose seam-cut candidate repair or mechanical repartition evidence;
- remain fail-closed and preserve skin weights.

It explicitly **does not** repair the skin field, and its automatic repartition path is forbidden until trustworthy skin/reliability authority exists. Therefore it must not be used to hide the present V9 failure by cutting away enough topology. The current program must first establish whether the admitted Arachne field can be realized continuously with bounded correction. G3B remains the verifier / downstream owner-attribution court, not a license for topology-specific witness surgery.

## 3. Genericity invariants

A proposed repair or representation is admissible only if all of the following hold:

1. **Subject-free:** no Knight names, parts, regions, vertex IDs, clip-specific constants or hand-authored masks.
2. **Topology continuity:** the same qualified evidence can be realized on multiple qualified carrier discretizations without catastrophic mechanical divergence.
3. **Bounded correction:** compiler realization may reconcile evidence with carrier mechanics, but may not silently rewrite the learned proposal wholesale.
4. **Evidence preservation:** corrections must be measured against the admitted Arachne evidence and reported.
5. **Mechanical truth:** final acceptance is based on exact LBS deformation, G3/G3B and exact authored-motion courts, not on a visual heuristic.
6. **Resolution stability:** refinement/densification must not make edge stress diverge merely because edges became shorter.
7. **Fail-closed:** if no bounded realization exists, the compiler must reject and attribute ownership rather than invent a passing skin.
8. **Same operator / same policy:** topology variants use the same subject-free operator and frozen policy; thresholds are not tuned per witness.

## 4. SkinTokens-derived lesson

SkinTokens is not adopted as an implementation dependency. The transferable architectural lesson is:

- skin is represented as a field that can be queried on surface points rather than as authority tied to a specific vertex indexing;
- training/evaluation samples the surface continuously and can transfer to the final mesh;
- deformation quality is explicitly tested under LBS rather than inferred only from per-vertex weight error;
- optional geometry/geodesic priors can constrain learned skin proposals.

RealSaS equivalent:

- Arachne remains evidence producer;
- compiler owns deterministic field realization and qualification;
- G3/G3B provide mechanical authority;
- the final carrier is a consumer of the field, not the owner of its semantics.

## 5. Execution phases

### Phase A — Recover the current deformation-sensitive audit

Goal: make the current G3-sensitive solver execute correctly before judging the idea.

- read the exact failure from run `37113657436`;
- patch only the implementation defect;
- rerun on the frozen V9 evidence;
- do not change model checkpoints or training.

PASS means: all projection variants are produced and the frozen G3 + 51-frame courts execute.

### Phase B — Direct deformation-constrained projection

Replace proxy-only weight smoothness with direct LBS mechanical evidence.

For fixed subject-free stress poses, vertex motion is linear in skin weights:

```
x'_i(p) = Σ_j w_ij T_j(p) x_i
```

Use deterministic optimization to minimize evidence deviation while bounding mechanically destructive relative edge deformation.

Primary objective family:

```
min_W  evidence_deviation(W, W0)
     + λ * deformation_energy(W)

subject to:
    W >= 0
    row_sum(W) = 1
    support / component / coverage policy
    bounded correction policy
```

Optimization must not use Knight clip semantics. The G3 local-frame probe bank is permitted because it is subject-free and already part of qualification.

PASS means: G3 and exact-motion results improve materially with small, bounded evidence correction.

### Phase C — Exact qualification loop

For each candidate realization record at minimum:

- p50 / p95 / p99 / max L1 correction from admitted evidence;
- support changes and dominant-joint changes;
- G3 minimum/maximum area ratio;
- G3 maximum condition number;
- 51-frame failure count;
- worst edge stretch and p99 edge stretch;
- flipped/collapsed face counts where available;
- failure ownership.

No candidate is promoted because it "looks better."

### Phase D — Topology-invariance court

Run one frozen evidence field through multiple qualified carrier discretizations:

- archived/reference carrier where legally comparable;
- V9;
- V9 alternate refinement;
- denser refinement / perturbation generated by the same subject-free rules.

Acceptance question:

> Does the same field realization policy preserve the same mechanical class as the carrier discretization changes?

FAIL if one topology passes only after topology-specific policy tuning or if stress diverges under refinement.

### Phase E — Productize the contract

Only after Phases B-D pass:

- formalize `SkinFieldEvidenceIR` or the least-disruptive equivalent contract;
- make per-vertex Arachne outputs evidence samples rather than final product authority;
- implement deterministic carrier realization;
- bind policy hashes and provenance;
- add G3B continuity/refinement gates;
- add adversarial unit tests for tiny-edge / cross-influence discontinuity;
- rerun static + source fidelity + dynamic full courts.

Target seal:

```
Static PASS
+ Source Fidelity PASS
+ G3B PASS
+ G3 PASS
+ 51/51 Exact Motion PASS
= MECHANICS / GEOMETRY SEALED
```

Only then continue to final presentation/render attribution.

## 6. A100 escalation rule

Do **not** use A100 merely because a compiler experiment fails.

A100/readout refit is allowed only if all of the following are established:

1. the direct deformation-aware deterministic realization is implemented and correctly executed;
2. no bounded evidence-preserving realization can pass the generic mechanical courts;
3. failure is attributed to insufficient Arachne evidence rather than Geppetto or carrier topology;
4. V9-compatible training truth / teacher mapping is legal and complete enough to train against;
5. the refit plan freezes the backbone by default and trains only the minimal necessary readout unless evidence proves otherwise;
6. expected fit cost fits the remaining A100 budget.

Before any A100 run, report to the user with:
- exact reason compiler repair was rejected;
- exact trainable parameter scope;
- estimated run configuration and stop criteria;
- checkpoint/rollback plan.

## 7. Explicit non-goals

- no Knight-specific skin painting;
- no per-clip repair;
- no topology rollback merely to recover an old demo;
- no unbounded smoothing;
- no main-branch merge while the contract is under audit;
- no claim of genericity from a single passing carrier;
- no A100 use without the escalation rule above.

## 8. Current decision state

- V9 static mesh/topology: retained as current static witness; no rollback is justified by current evidence.
- Geppetto refit: not indicated by current counterfactual/replay evidence.
- Arachne full refit: not authorized.
- Arachne readout-only refit: contingency only, gated by Section 6.
- Active next step: recover and complete the deformation-sensitive solver, then move to direct LBS deformation-constrained projection.

## 8A. Decision after deformation-sensitive projection run 37116286772

The implementation issue from run `37113657436` was a linear-solver convergence limit, not a scientific result. Commit `a72c4c88fc2127ade696d92885f210c168c39ca4` added a fail-closed sparse-direct fallback and run `37116286772` completed all 8 variants plus frozen G3 and 51-frame courts.

Measured range:

- low-correction variants (~0.034-0.036 p95 L1) still had G3 max condition ~314-330 and max motion edge ~149-292x;
- medium variants (~0.168-0.449 p95 L1) still had G3 max condition ~123-199 and max motion edge ~60-130x;
- strongest variants (~0.765-0.792 p95 L1) still failed G3 and all 51 motion frames, with best max motion edge ~26.38x and best observed G3 max condition ~120.11.

Decision:

- reject the G3-joint-metric proxy projection as a sufficient product repair;
- do not increase lambda further because correction is already too large and all exact-motion frames still fail;
- do not use A100 yet;
- advance to **Phase B: direct LBS deformation-constrained projection**, which optimizes measured carrier deformation rather than proxy weight-space motion similarity.

## 9. Progress ledger

| Step | State | Evidence |
|---|---|---|
| V9 static topology | PASS | zero residual / topology clean audit |
| Fresh V9 rig replay | STRUCTURAL ONLY | Geppetto CUDA replay close to archived; not product seal |
| Fresh V9 skin + G3 | FAIL | G3 ~518.27 |
| Fresh V9 exact motion | FAIL | 51/51 failed; max edge ~521.37x |
| Archived-kinematics counterfactual | FAIL | G3 ~752.11; max edge ~551.14x |
| Skin-field discontinuity audit | FAIL / OWNER LOCALIZED | V9 gradient p99 ~153.54, max ~3475.76 |
| Geometry-only projection | DIAGNOSTIC IMPROVEMENT, NOT ADMISSIBLE | max edge ~13.48x at large correction; 49/51 still fail |
| Topology-aware old-skin prolongation | FAIL | all variants 51/51 fail |
| Coarse FEM prolongation | STRUCTURALLY INCOMPLETE | source components without coarse faces |
| G3-sensitive projection | FAIL / PROXY INSUFFICIENT | run 37116286772 PASS as workflow; all 8 variants mechanically fail. Best max motion edge ~26.38x, G3 max condition ~120.11, 51/51 frames fail, p95 correction ~0.765 L1 |
| Direct LBS deformation-constrained projection | TODO | next scientific experiment |
| Multi-topology invariance court | TODO | required before genericity claim |
| Product contract integration | BLOCKED | requires preceding PASS |
