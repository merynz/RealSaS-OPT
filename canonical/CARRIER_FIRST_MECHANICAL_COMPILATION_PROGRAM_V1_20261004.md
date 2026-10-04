# RealSaS — Carrier-First Mechanical Compilation Program V1 — 2026-10-04

**Status:** `IMPLEMENTATION_AUTHORIZED_ON_ISOLATED_BRANCH`  
**Branch:** `impl/carrier-first-mechanical-compilation-20261004`  
**Parent research head:** `545418a8664b33de4133558483b91477dc1b1bee`  
**Main policy:** `DO_NOT_TOUCH_MAIN`

## 0. Executive decision

RealSaS mechanics becomes **carrier-first**:

```
IRIS / GSA perception substrate
        ↓
Mechanical partition
        ↓
Canonical mechanical carrier candidate M^k
        ↓
Static qualification + attempt seal
        ↓
derive carrier-bound model evidence
        ↓
ATLAS -> R^k
        ↓
MIRA  -> W^k
        ↓
hard dynamic mechanical proof
        ↓
PASS -> seal mechanical state
FAIL -> owner attribution -> bounded repair -> new immutable attempt k+1
```

The key invariant is:

> Rig and skin may not silently survive a mechanical carrier change.

A topology change is a new mechanical world unless a typed equivalence proof explicitly says otherwise.

## 1. Terminology

### Perception substrate

`RiggingSurfaceIR` / GSA is observation-grounded evidence. It is **not** the final mechanical carrier.

### Mechanical carrier

For attempt `k`:

`M^k = (V, E, F, N, addressing, provenance, lineage)`

The carrier owns the interpolation/deformation basis used by downstream mechanics.

### Qualified mechanical state

`Q^k = (M^k, R^k, W^k, P^k)`

where `P^k` is the hard proof result over the admitted probe/motion family.

### Reference evidence

Coherent rigged assets provide useful `(M_ref, R_ref, W_ref, A_ref)` examples. They are **reference evidence**, not product truth.

## 2. Non-negotiable invariants

1. `Stage15/GSA != mechanical carrier authority`.
2. A carrier is built and statically qualified before ATLAS/MIRA mechanics are product-valid.
3. ATLAS/MIRA outputs bind an exact carrier hash.
4. `mesh_hash` change invalidates carrier-bound rig/skin/proof by default.
5. No hidden Stage35 -> Stage18 filesystem back-edge.
6. Repair creates a new immutable attempt. No in-place carrier mutation.
7. Differentiable losses are optimization surrogates; hard Compiler proof remains authority.
8. Teacher/reference cardinality or weight values are not product authority.
9. Stage numbers and total stage count are implementation details, not product contracts.
10. Final product closure may contain fewer or more than 46 stages.

## 3. Current-stage semantic migration

We preserve existing IDs during the first implementation pass to avoid meaningless repository-wide churn.

### Stage15 — RiggingSurface qualification

Role:
- preserve GSA/perception evidence;
- source visibility/support/provenance;
- not mechanical topology authority.

### Stage17 — Mechanical partition

Role:
- derive conservative structural partition constraints before carrier construction.

### Stage18 — Canonical mechanical carrier build

Target role:
- build immutable candidate `M^k`;
- consume only upstream evidence plus an **explicit parent-attempt repair input**;
- never read Stage35 artifacts from the same run;
- mint a new carrier lineage for every repaired attempt.

### Stage19 — Static carrier qualification and attempt seal

Role:
- manifold/connectivity legality;
- no degenerates/slivers beyond policy;
- static conditioning/min-angle/aspect policy;
- boundary/component correctness;
- source/silhouette fidelity;
- addressing/provenance completeness;
- emit the exact carrier binding used by mechanics;
- emit carrier-derived `XYZ + normal` evidence.

Stage19 does **not** claim dynamic articulation correctness.

### Stage26/27 — ATLAS preregistration/inference

Target dependency:
- Stage19 carrier seal/evidence;
- Stage15 only as auxiliary observation evidence where explicitly mapped.

Output must bind:
- carrier hash;
- evidence contract hash.

ATLAS does not need raw triangle-face IDs as neural input merely because the carrier owns topology. Initial contract remains `XYZ + normals + admitted observation evidence`.

### Stage28 — Skeleton qualification

Compiler owns:
- root/tree legality;
- canonical IDs;
- structural constraints.

Skeleton output becomes carrier-bound for product mechanics.

### Stage30/31 — MIRA preregistration/inference

Target dependency:
- same Stage19 carrier;
- qualified carrier-bound skeleton;
- optional semantic memory/reference features.

The existing point-query decoder is tested first on exact carrier vertices before any new topology-conditioned MIRA head is authorized.

### Stage32 — Skin qualification

Target output is a carrier-bound skin proposal/state.

No claim that a semantic GSA weight field can be transferred to an unrelated product topology without proof.

### Stage34 — Deformation capability / probe envelope

Keep as subject-free probe-plan authority. Bind it to the skeleton and carrier attempt.

### Stage35 — Joint dynamic mechanical proof

Stage35 remains after carrier + rig + skin.

It evaluates the tuple:

`(M^k, R^k, W^k, Q)`

Required output classes:
- hard G3/G3B-style consequence report;
- failure signatures;
- owner attribution;
- repair directive when bounded repair is justified;
- no direct mutation of Stage18 artifacts.

Stage35 is therefore **proof + diagnosis**, not hidden mesh builder.

### Stage36 — Mechanical closure seam

The historical meaning “deterministic surface-skin -> mesh-skin transfer” is transitional and should disappear once MIRA is carrier-native at query/output time.

Target role:
- mint/seal `QualifiedMechanicalStateIR` on PASS;
- or expose explicit repair handoff metadata for the next attempt.

Whether this remains Stage36 is not architecturally important.

## 4. Outer compiler loop: immutable attempt DAGs

A single DAG attempt must stay acyclic.

The fixed point lives **outside** one DAG execution:

```
Attempt k:
  Stage18 -> Stage19 -> ATLAS -> MIRA -> Stage35

Stage35 PASS:
  seal attempt k

Stage35 FAIL + mesh owner:
  MechanicalRepairDirectiveIR
        ↓
  create attempt k+1 with explicit parent attempt reference
        ↓
  Stage18 builds M^(k+1)
        ↓
  all carrier-bound mechanics revalidated/reinferred as required
```

This replaces the current hidden Stage35 -> Stage18 filesystem side-channel.

## 5. Evidence contract: models must continue from the same carrier

Stage19 must emit a typed carrier-evidence package.

Minimum V1:
- carrier binding hash;
- ordered carrier vertex IDs;
- carrier vertex positions;
- carrier-derived vertex normals;
- normal-valid flags;
- exact face/connectivity digest;
- Stage18/19 provenance;
- mapping to source/GSA addressing.

Auxiliary observation evidence may be mapped from GSA/source:
- view support;
- raster coordinates;
- observed/completed state;
- uncertainty/provenance.

But the positions/normals and any product-mechanical query points are derived from **the exact sealed carrier**.

## 6. Model strategy and fit policy

### ATLAS

Immediate:
- no full retrain;
- make current ATLAS output explicitly evaluated on the sealed carrier;
- build carrier-evidence compatibility court.

Target:
- ATLAS consumes carrier-derived `XYZ + normals` plus admitted mapped observation evidence;
- teacher joint count is evidence, not cardinality authority;
- mechanically sufficient control basis is the objective.

Retrain policy:
- only if zero-train/adapter compatibility fails or carrier-conditioned mechanical utility exposes a real residual;
- prefer freezing reusable encoder/backbone and fitting the smallest locus/diffusion/control head first;
- full fit is last resort.

### MIRA

Immediate:
- use existing GSA/skeleton backbone if useful;
- query the existing point-query decoder at exact Stage19 carrier vertices;
- no MIRA-M topology head yet.

First fit candidate, only if required:
- freeze backbone;
- fit decoder/tail or minimal mechanical residual head;
- add joint mechanical consequence loss on exact carrier faces;
- preserve semantic/reference supervision as a regularizer.

Full MIRA retrain is not authorized unless the minimal path is falsified.

## 7. Loss program

### Reference terms

Reference assets may supervise:
- joint locus/role;
- influence semantics;
- deformation behavior;
- motion coverage.

They do not define product topology authority.

### Joint mechanical loss

Extend current `JointMechanicalLossV1`:

`L_mech = w_cond L_cond + w_area L_area + w_edge L_edge + w_fold L_fold + ...`

Required principles:
- exact carrier faces are used;
- exact probe family is bound;
- zero/low penalty inside admitted region;
- finite gradients;
- hard G3 remains separate.

Future terms may include:
- self-intersection/penetration;
- locality/editability;
- control complexity;
- source silhouette/appearance consequence.

### Training/qualification split

Training:
- differentiable mechanical surrogate;
- reference supervision;
- held-out perturbation families.

Qualification:
- hard G3/G3B;
- held-out probes not identical to the training surrogate set;
- fail closed.

## 8. Redesigned execution gates

Legacy E1/E2 evidence is retained as foundation, but the old E3 plan is retired.

### F0 — causal foundation — CLOSED

Evidence already sealed:
- topology-basis equivalence;
- AC/BD diagonal connectivity causality;
- teacher-exact weights do not rescue an incompatible carrier.

### F1 — differentiable mechanical kernel — PARTIAL PASS

Existing:
- condition;
- area;
- edge;
- finite gradient court.

Remaining:
- fold/orientation term;
- ordering/parity regression against hard G3 cases.

### C0 — architecture contract migration — NOW

Tasks:
- freeze this plan;
- Stage26/30 product dependencies bind Stage19;
- remove hidden Stage35 -> Stage18 read;
- explicit attempt-parent repair input;
- carrier hash required in learned-mechanics preregistration.

Exit:
- architecture contract tests PASS.

### C1 — Stage19 carrier-evidence V1 — NOW

Tasks:
- derive exact carrier vertex positions/normals;
- seal connectivity/face digest;
- bind source/GSA addressing;
- emit typed evidence package.

Exit:
- deterministic hash;
- same Stage18 candidate -> same evidence;
- AC/BD connectivity -> different carrier/evidence hash.

### C2 — zero-training model compatibility

ATLAS:
- replay current checkpoint with carrier-first evidence/adapter;
- measure proposal drift and downstream utility.

MIRA:
- preserve current backbone;
- query existing decoder on exact carrier vertices.

Exit:
- no A100;
- determine exactly which model components, if any, require fit.

### C3 — Knight carrier-first one-shot mechanical court

Freeze one Stage19 carrier attempt.
Run:
- current/best ATLAS;
- current/best MIRA carrier query;
- hard Stage35.

This is the new one-shot baseline.

Metrics:
- G3/G3B closure;
- catastrophic faces;
- condition/edge/area/fold;
- source fidelity;
- carrier/rig/skin lineage coherence.

### C4 — minimal MIRA mechanical fit, conditional

Only if C3 identifies skin-owned residual.

Preferred order:
1. decoder/tail-only fit;
2. bounded residual head;
3. larger MIRA fit only if required.

Loss:
- reference semantic term;
- carrier-native joint mechanical loss;
- deformation consequence term when coherent reference motion exists.

A100 only here and only with a demonstrated residual.

### C5 — ATLAS mechanically sufficient control basis, conditional

Only if C3/C4 identifies rig-owned residual.

Tasks:
- carrier-bound downstream utility;
- RigAnything-faithful absolute-joint diffusion challenger;
- grow-to-need / prune-to-proof;
- Compiler retains discrete tree legality.

Prefer head/locus fit before full model fit.

### C6 — compiler repair loop

Tasks:
- Stage35 owner attribution contract;
- explicit `MechanicalRepairDirectiveIR`;
- parent-attempt input;
- new immutable carrier attempt;
- bounded iteration/budget;
- repair-effect report;
- no hidden back-edge.

Acceptance:
- static PASS;
- source-fidelity non-regression;
- mechanical consequence improvement;
- no new catastrophic class.

### C7 — Knight closure-to-render

Goal:
- return quickly to the product witness.

Path:
- selected sealed carrier;
- qualified ATLAS/MIRA state;
- Stage35 PASS;
- mechanical-state seal;
- existing appearance/runtime line;
- idle/run/slash render.

Do not block this milestone on FITK/LOFO or shared-encoder efficiency research.

### C8 — generalization and moat benchmark

After Knight:
- coherent FITK;
- LOFO;
- unseen;
- one-shot vs compiler-closed comparison;
- Mechanical Closure Rate;
- repair count;
- source fidelity;
- editability;
- control complexity;
- runtime stability.

This court measures the actual Compiler advantage.

## 9. Compute policy

- No A100 for C0/C1/C2.
- C3 uses current checkpoints.
- A100 only after owner attribution proves a model fit is necessary.
- Prefer partial/head fit over full retrain.
- GitHub Actions is final reproducibility evidence; exploratory loops belong in notebooks/local runner.

## 10. Promotion policy

This implementation branch is not merged wholesale into `main`.

Promotion unit:
- proven contract;
- typed artifact/IR;
- passing gate;
- explicit ProductRevision / qualified artifact graph.

Research chronology is evidence, not shipping authority.

## 11. Immediate implementation order

1. Add contract tests for carrier-first dependencies and no hidden Stage35 -> Stage18 read.
2. Make Stage18 repair input explicit and attempt-bound.
3. Emit Stage19 carrier-evidence V1.
4. Rebind ATLAS preregistration to Stage19 carrier/evidence.
5. Rebind MIRA preregistration/output to the same carrier.
6. Extend JointMechanicalLoss with fold/parity.
7. Run C2 no-training compatibility.
8. Run Knight C3 one-shot court.
9. Fit only the owner proven by C3.
10. Implement C6 fixed-point attempt orchestration.
11. Return to Knight render.
