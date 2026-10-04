# RealSaS — ATLAS / MIRA / Compiler Mechanical Coherence Execution Plan V1

**Date:** 2026-10-04  
**Status:** `ACTIVE_SINGLE_AUTHORITY__CARRIER_FIRST_REVISION`  
**Implementation branch:** `impl/carrier-first-mechanical-compilation-20261004`  
**Main policy:** `DO_NOT_TOUCH_MAIN_UNTIL_GATES_CLOSE`

> This document is the single execution authority for the ATLAS / MIRA / Compiler mechanical-coherence program.
> The original E1-E7 plan is preserved below as historical scientific lineage. Its old E3/MIRA-M authorization path is retired by the carrier-first revision; evidence is kept, authority is not duplicated.

---

# 0. Executive decision

RealSaS mechanics becomes **carrier-first**:

```text
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
FAIL -> owner attribution -> bounded repair -> immutable attempt k+1
```

Core invariant:

> Rig and skin may not silently survive a mechanical-carrier change.

A topology/connectivity change is a new mechanical world unless a typed equivalence proof explicitly authorizes reuse.

---

# 1. Why the plan changed

The original plan correctly identified a composition gap between:
- product topology;
- ATLAS control basis;
- MIRA skin field;
- Compiler dynamic proof.

But later causal evidence sharpened the failure:

1. The same rest geometry and vertex values can produce different deformation under different connectivity.
2. Teacher-exact weights do not rescue a mechanically incompatible carrier.
3. GSA relation topology, source-asset topology, and Stage18 product mechanical topology are different authorities and must not be conflated.
4. ATLAS/MIRA do not need triangle faces as mandatory neural input merely because the mechanical carrier owns connectivity.
5. The product must not treat source/teacher topology or projected teacher weights as product truth.

Therefore the correct first fix is not “add a topology-aware MIRA head.” The first fix is:

> **Create the product carrier before learned mechanics, derive model evidence from that exact carrier, bind all mechanics to it, and let Compiler proof/repair close the system.**

---

# 2. Terminology

## 2.1 Perception substrate

`RiggingSurfaceIR` / GSA is observation-grounded geometry/evidence.

It may provide:
- source support;
- visibility;
- raster correspondence;
- completion state;
- uncertainty/provenance;
- local relation evidence.

It is **not** the final product mechanical carrier.

## 2.2 Mechanical carrier

For attempt `k`:

[
M^k=(V,E,F,N,addressing,provenance,lineage)
]

The carrier owns:
- the interpolation basis;
- the deforming surface;
- the downstream mechanical coordinate system.

## 2.3 Qualified mechanical state

[
Q^k=(M^k,R^k,W^k,P^k)
]

where:
- `R^k` = carrier-bound rig/control basis;
- `W^k` = carrier-bound skin/influence state;
- `P^k` = hard proof/qualification evidence.

## 2.4 Reference evidence

Coherent rigged assets may expose:

[
(M_{ref},R_{ref},W_{ref},A_{ref})
]

These are coherent reference examples, not product truth.

---

# 3. Non-negotiable invariants

1. `Stage15/GSA != mechanical carrier authority`.
2. A carrier is built and statically qualified before ATLAS/MIRA mechanics are product-valid.
3. ATLAS/MIRA learned outputs bind an exact carrier evidence hash.
4. A carrier/topology hash change invalidates carrier-bound rig/skin/proof by default.
5. No hidden Stage35 -> Stage18 filesystem back-edge.
6. Repair creates a new immutable attempt; no in-place carrier mutation.
7. Differentiable losses are optimization surrogates; hard Compiler proof remains authority.
8. Teacher/reference cardinality and projected weight values are not product authority.
9. Stage numbers and total stage count are implementation details, not architecture truth.
10. Product closure may end at Stage40, Stage46, Stage50, etc.; DAG semantics matter, not the number.
11. GSA relation topology and mechanical carrier topology must have distinct names and hashes.
12. Any product-mechanical query point, normal, rig, skin, and proof record must resolve to one carrier attempt.

---

# 4. Stage/DAG semantic migration

Existing IDs are retained during the first migration to avoid repository-wide churn. Their semantics may change.

## Stage15 — RiggingSurface qualification

Role:
- perception substrate;
- source support/visibility/provenance;
- auxiliary observation evidence.

Forbidden claim:
- final mechanical topology authority.

## Stage17 — Mechanical partition

Role:
- derive conservative structural/component constraints before carrier construction.

## Stage18 — Canonical mechanical carrier build

Target role:
- build immutable candidate `M^k`;
- consume upstream evidence plus an explicit parent-attempt repair input if this is a repair attempt;
- mint a new carrier lineage for every repaired attempt.

Forbidden:
- reading Stage35 artifacts from the same run;
- silently mutating a previously sealed carrier.

## Stage19 — Static carrier qualification and attempt seal

Role:
- connectivity/manifold legality;
- no degenerates;
- sliver/min-angle/aspect/conditioning policy;
- boundary/component correctness;
- source/silhouette fidelity;
- addressability and provenance completeness;
- emit exact carrier-bound model evidence.

Stage19 emits at minimum:
- carrier binding hash;
- ordered carrier vertex IDs;
- carrier XYZ;
- carrier-derived normals;
- normal-valid flags;
- exact face/connectivity digest;
- source/GSA mapping/provenance.

Stage19 means:

> “This is a statically admissible mechanical carrier for this attempt.”

It does **not** claim dynamic articulation correctness.

## Stage26/27 — ATLAS preregistration / inference

Target dependencies:
- Stage19 carrier seal/evidence;
- Stage15 auxiliary source evidence only where explicitly mapped.

ATLAS V1 input principle remains:

[
XYZ+N+	ext{admitted source evidence}
]

Raw triangle-face IDs are not made mandatory neural input without evidence.

ATLAS output/evaluation must bind:
- carrier evidence hash;
- carrier topology hash.

## Stage28 — Skeleton qualification

Compiler retains:
- legal tree/root/IDs;
- structural hard constraints;
- discrete solver authority.

Qualified rig/control basis is product-valid only for its bound carrier attempt.

## Stage30/31 — MIRA preregistration / inference

Target dependencies:
- same Stage19 carrier;
- qualified carrier-bound skeleton;
- optional semantic/reference memory.

First implementation:
- preserve current useful MIRA backbone;
- query the existing point-query decoder at exact carrier vertices;
- do not authorize a new topology-conditioned MIRA-M head yet.

## Stage32 — Skin qualification

Target:
- a carrier-bound skin proposal/state.

Forbidden assumption:
- a semantic GSA weight field can be transferred to an unrelated product topology and remain mechanically equivalent by default.

## Stage34 — Deformation capability / probe envelope

Role:
- subject-free motion/probe family;
- bind to carrier attempt and skeleton;
- define the evaluation envelope consumed by dynamic proof.

## Stage35 — Joint dynamic mechanical proof

Stage35 evaluates:

[
(M^k,R^k,W^k,Q)
]

Required outputs:
- hard G3/G3B-style consequence report;
- failure signatures;
- owner attribution;
- bounded repair directive where justified;
- no direct Stage18 mutation.

Stage35 is:

> **proof + diagnosis**

not a hidden mesh builder.

## Stage36 — Mechanical closure seam

Historical role:
- deterministic surface-skin -> mesh-skin transfer.

Target role after carrier-native skinning:
- seal `QualifiedMechanicalStateIR` on PASS;
- or emit explicit repair handoff metadata.

Whether this remains Stage36 is not architecturally important.

---

# 5. Outer Compiler loop: immutable attempt DAGs

A single attempt DAG remains acyclic.

The fixed-point loop lives outside a single attempt:

```text
Attempt k:
  Stage18 -> Stage19 -> ATLAS -> MIRA -> Stage35

Stage35 PASS:
  seal (M^k,R^k,W^k)

Stage35 FAIL:
  owner attribution
        ↓
  bounded repair directive
        ↓
  create attempt k+1 with explicit parent-attempt reference
        ↓
  build M^(k+1) / R^(k+1) / W^(k+1) as required
        ↓
  re-run affected proof
```

Carrier mutation rule:

[
M^k 
eq M^{k+1}
]

means downstream mechanics are stale unless explicitly revalidated.

---

# 6. Evidence contract: all models continue from the same carrier

Stage19 emits typed carrier evidence.

Minimum V1:
- `candidate_mesh_binding_hash`;
- `static_mesh_qualification_binding_hash`;
- `surface_addressing_binding_hash`;
- ordered vertex IDs;
- positions;
- normals;
- normal-valid mask;
- exact face indices/connectivity digest;
- topology hash;
- geometry hash;
- carrier evidence hash.

Auxiliary GSA/source features can be mapped onto the carrier:
- view support;
- raster coordinates;
- observed/completed state;
- uncertainty;
- provenance.

Critical rule:

> Product-mechanical XYZ/normals/query points come from the exact sealed carrier, not from a different substrate.

---

# 7. ATLAS architecture program

## 7.1 Target role

ATLAS is not merely a teacher-skeleton reproducer.

Target:

> infer a **mechanically sufficient editable control basis** for the current canonical carrier.

## 7.2 Preserve

Preserve unless falsified:
- carrier-derived XYZ + normals as primary geometry evidence;
- full-surface attention, which already showed controlled value;
- Compiler ownership of final legal tree/root/IDs;
- uncertainty/confidence output;
- support-to-surface evidence.

## 7.3 Do not promote the current residual diffusion as the final answer

Current frozen ATLAS/Geppetto residual diffusion:
- refines a deterministic coarse joint;
- does not feed sampled/refined XYZ into the next causal step;
- is not RigAnything joint-diffusion parity.

Therefore the old negative residual-diffusion ablation is not evidence against reference-style absolute-joint diffusion.

## 7.4 ATLAS challenger direction

The next serious ATLAS challenger, if C2/C3 show rig-owned residual, should test:

- absolute next-joint XYZ diffusion;
- sampled joint XYZ feeds parent prediction;
- sampled joint/parent state feeds subsequent generation;
- full carrier-surface access each step;
- order augmentation only as an isolated later rung;
- Compiler retains discrete legality/final canonical identity.

## 7.5 Cardinality/objective

Teacher joint count is reference evidence, not authority.

ATLAS target becomes:

[
	ext{minimum sufficient control basis under mechanical proof}
]

Evaluate:
- mechanical closure;
- marginal articulation gain;
- editability/locality;
- control count after adequacy;
- stability under free-running generation.

Possible strategy:
- grow-to-need;
- prune-to-proof;
- keep a control only if removal causes measurable articulation loss.

## 7.6 ATLAS fit policy

No immediate full retrain.

Order:
1. zero-train carrier compatibility;
2. current best checkpoint one-shot court;
3. head/locus challenger if rig-owned residual is proven;
4. absolute-joint diffusion challenger;
5. larger/full fit only if smaller interventions are falsified.

---

# 8. MIRA architecture program

## 8.1 Immediate contract

No MIRA-M topology head is authorized yet.

First:
- keep useful semantic/skeleton backbone;
- query the existing point-query decoder at exact Stage19 carrier vertices;
- bind output to carrier + skeleton.

## 8.2 If skin-owned residual is proven

Preferred intervention order:
1. decoder/tail-only fit;
2. bounded residual/correction head;
3. larger MIRA fit;
4. topology-conditioned head only if exact carrier-query freedom is insufficient.

## 8.3 Product objective

Do not optimize only:

[
hat W approx W_{teacher}
]

Prefer:

[
D(M,R,hat W,q)
]

to satisfy RealSaS mechanical criteria while reference supervision regularizes semantics.

---

# 9. Loss program

## 9.1 Joint mechanical surrogate

Extend `JointMechanicalLossV1`:

[
L_{mech}=
w_{cond}L_{cond}+
w_{area}L_{area}+
w_{edge}L_{edge}+
w_{fold}L_{fold}+ldots
]

Requirements:
- exact carrier faces;
- bound probe family;
- finite gradients;
- low/zero penalty in admitted region;
- ordering correlation with hard G3;
- hard G3 remains independent authority.

Future terms may include:
- self-intersection/penetration;
- locality/editability;
- control complexity;
- source silhouette consequence.

## 9.2 Reference supervision

Coherent assets can supervise:
- joint loci/roles;
- influence semantics;
- deformation behavior;
- motion coverage.

Reference weights/skeletons are evidence, not product authority.

## 9.3 Training vs qualification split

Training:
- differentiable surrogate;
- reference losses;
- held-out perturbation families.

Qualification:
- hard G3/G3B;
- held-out probes;
- fail closed.

---

# 10. Redesigned execution gates

The old E1/E2 evidence is retained. The old E3 “optimize mesh-native MIRA-M first” path is retired.

## F0 — causal foundation — CLOSED

Already established:
- connectivity alone can change mechanical consequence;
- AC/BD diagonal microcourt;
- teacher-exact weights do not rescue an incompatible carrier.

## F1 — differentiable mechanical kernel — PARTIAL PASS

Existing:
- condition;
- area;
- edge;
- finite-gradient court.

Remaining:
- fold/orientation;
- parity/order regression against hard G3 cases.

## C0 — architecture contract migration — IN PROGRESS

Tasks:
- Stage26/30 depend on Stage19 carrier;
- remove hidden Stage35 -> Stage18 read;
- explicit parent-attempt repair input;
- carrier hash in learned-mechanics preregistration;
- stage count not treated as product architecture.

Exit:
- contract tests PASS.

## C1 — Stage19 carrier evidence — IN PROGRESS

Tasks:
- derive exact carrier positions/normals;
- seal face/connectivity digest;
- bind source/GSA addressing;
- emit typed evidence.

Exit:
- deterministic hash;
- same candidate -> same evidence;
- AC/BD -> different topology/carrier hash.

## C2 — zero-training model compatibility

ATLAS:
- replay current checkpoint with carrier-first evidence;
- measure representation/proposal drift;
- verify no illegal teacher/cardinality authority.

MIRA:
- query existing decoder at exact carrier vertices.

Exit:
- determine exactly which components need fit;
- no A100.

## C3 — Knight carrier-first one-shot court

Freeze one Stage19 carrier attempt.

Run:
- current/best ATLAS;
- current/best MIRA carrier query;
- Stage35 hard proof.

Metrics:
- G3/G3B closure;
- catastrophic faces;
- condition/edge/area/fold;
- source fidelity;
- carrier/rig/skin lineage coherence.

This is the new one-shot baseline.

## C4 — minimal MIRA mechanical fit — CONDITIONAL

Only if C3 identifies skin-owned residual.

Order:
1. decoder/tail-only;
2. bounded residual;
3. larger MIRA fit.

A100 only after residual is demonstrated.

## C5 — ATLAS mechanically sufficient control basis — CONDITIONAL

Only if C3/C4 identifies rig-owned residual.

Subcourts:
- current checkpoint vs carrier-first evidence;
- mechanical-capacity/marginal-control court;
- absolute-joint sampled-feedback diffusion challenger;
- grow-to-need/prune-to-proof;
- free-running stability.

Prefer minimal head/locus fit before full ATLAS fit.

## C6 — Compiler repair loop

Tasks:
- Stage35 owner attribution;
- typed repair directive;
- parent-attempt input;
- immutable new carrier attempt;
- bounded repair budget;
- repair-effect report;
- no hidden DAG back-edge.

Acceptance:
- static PASS;
- source-fidelity non-regression;
- mechanical improvement;
- no new catastrophic class.

## C7 — Knight closure-to-render

Goal:
- recover the product witness quickly.

Path:
- sealed carrier;
- qualified ATLAS/MIRA state;
- Stage35 PASS;
- mechanical-state seal;
- existing appearance/runtime;
- idle/run/slash render.

Do not block on FITK/LOFO/shared-encoder efficiency research.

## C8 — generalization and Compiler moat benchmark

After Knight:
- FITK;
- LOFO;
- unseen;
- one-shot vs compiler-closed;
- Mechanical Closure Rate;
- repair count;
- source fidelity;
- editability;
- control complexity;
- runtime stability.

This measures the actual Compiler advantage.

---

# 11. Compute policy

- No A100 for C0/C1/C2.
- C3 uses current checkpoints.
- A100 only after owner attribution proves a fit is necessary.
- Prefer partial/head fit over full retrain.
- Notebooks/local runner for exploratory iterations.
- GitHub Actions for reproducibility/final gates.

---

# 12. Promotion policy

Do not merge the implementation branch wholesale.

Promotion unit:
- proven contract;
- typed artifact/IR;
- passing gate;
- explicit product revision / qualified artifact graph.

Research chronology is evidence, not shipping authority.

---

# 13. Immediate work queue

1. Close C0 contract tests.
2. Close C1 Stage19 carrier-evidence tests.
3. Fix runner/tooling failures independently of scientific verdicts.
4. Add fold/parity to JointMechanicalLoss.
5. Run C2 ATLAS zero-train carrier compatibility.
6. Run C2 MIRA exact-carrier query compatibility.
7. Run C3 Knight one-shot carrier-first court.
8. Use owner attribution to decide whether ATLAS, MIRA, or Compiler needs the next change.
9. If rig-owned, run C5 ATLAS challenger ladder.
10. If skin-owned, run C4 minimal MIRA fit ladder.
11. Close C6 explicit fixed-point orchestration.
12. Return to Knight idle/run/slash render.

---

# 14. Preserved historical E1-E7 lineage

This section preserves the original execution logic so earlier evidence is not lost.

## Historical E1 — diagonal-flip causal microcourt

Purpose:
- isolate connectivity as a causal mechanical variable;
- verify finite gradients w.r.t. weights;
- verify identity/non-regressive arm.

**Disposition:** completed and promoted into F0 evidence.

## Historical E2 — differentiable G3 surrogate unit court

Purpose:
- condition;
- area;
- edge;
- fold;
- compare with hard G3 ordering.

**Disposition:** partially completed and continued as F1.

## Historical E3 — frozen Knight mesh-native optimization oracle

Original question:
- can mesh-vertex weight freedom on fixed product topology materially reduce G3B?

**Disposition:** retired as the next architectural decision because it assumed product mechanics after a carrier-transfer seam. The useful scientific question survives inside C3/C4, but only after carrier-first evidence/output binding is enforced.

## Historical E4 — topology move + reoptimized weights

**Disposition:** absorbed into C6 Compiler repair/effect courts.

## Historical E5 — short MIRA-M fit

**Disposition:** not authorized by default; replaced by C4 minimal carrier-native MIRA fit if skin-owned residual is proven.

## Historical E6 — ATLAS mechanical utility/control-basis court

**Disposition:** preserved and expanded into C5.

## Historical E7 — shared encoder

**Disposition:** deferred. Parameter sharing is an efficiency optimization, not a mechanical-coherence authority.

---

# 15. Core product principle

[
oxed{	ext{Mesh/Carrier first} ightarrow 	ext{Rig} ightarrow 	ext{Skin} ightarrow 	ext{Prove} ightarrow 	ext{Repair if needed}}
]

But:

> “Mesh first” does not mean “mesh is dynamically final before mechanics.”

Stage19 seals a statically admissible carrier attempt. Stage35 decides whether the full `(M,R,W)` mechanical state closes. Compiler repair then creates a new attempt rather than silently mutating the current one.

The quality moat is therefore:

[
oxed{	ext{Model proposes. Compiler proves. Compiler can repair and re-prove.}}
]
