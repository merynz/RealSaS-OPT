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

## 7.7 ATLAS pre-fit test ladder

The ATLAS redesign is evidence-gated. Run these courts before spending A100.

### A0 — carrier/GSA evidence decomposition

Question:
- which current ATLAS inputs are true perception evidence and which are accidental substrate coordinates?

Freeze:
- current checkpoint;
- one Knight carrier;
- current GSA.

Measure separate ablations for:
- GSA XYZ/N;
- mapped Stage19 carrier XYZ/N;
- view/raster support;
- GSA relation graph;
- carrier topology digest used only as lineage/evaluation, not neural input.

Exit:
- exact list of features that must remain source/GSA auxiliary evidence;
- exact list that must be carrier-derived.

### A1 — support-binding transport court

Every Stage18/19 carrier vertex already has a `SurfaceSupportBinding`.

Build deterministic source-evidence transport:
- carrier XYZ/N remain exact carrier values;
- support/raster evidence is transported only through admitted support coefficients;
- no teacher fields;
- no nearest-neighbor remapping;
- no hidden topology substitution.

Conservative view-valid rule:
- a transported raster/view field is valid only when the admitted support mass for that view is complete within tolerance;
- otherwise validity is false rather than hallucinated.

Exit:
- deterministic transport hash;
- identity-support vertices reproduce GSA evidence exactly;
- seam/generated vertices remain provenance-complete and fail closed on incomplete support.

### A2 — frozen ATLAS compatibility

Run current checkpoint without fitting on:
- legacy GSA contract;
- carrier-first transported contract.

Measure:
- joint proposal PCK/locus drift for diagnostics only;
- stop/existence drift;
- parent/root proposal drift;
- confidence/uncertainty drift;
- hard downstream Stage35 consequence on the same carrier.

Decision:
- if carrier-first evidence is compatible and mechanical utility is non-regressive, keep frozen backbone;
- if representation mismatch is large, do not immediately full-fit: open A3.

### A3 — carrier-awareness necessity court

Question:
- does ATLAS actually need explicit carrier conditioning beyond GSA semantics?

Construct paired carriers from the same upstream perception evidence:
- same GSA/source;
- different mechanically admissible carrier variants or bounded local connectivity changes;
- same MIRA ceiling/optimization policy.

Hold the ATLAS rig fixed and measure best achievable skin/mechanical closure on each carrier.

Then allow rig loci/control basis to optimize/search independently on each carrier.

Carrier-aware ATLAS is justified only if:

[
min_W L_{mech}(M_1,R_{fixed},W)
]

is materially worse than

[
min_{R,W} L_{mech}(M_1,R,W)
]

for a carrier change while source semantics are held fixed.

This prevents adding mesh topology to ATLAS merely because it is available.

### A4 — cardinality/control-capacity court

Use K0/K1/K2/K3 only as evaluation/training-policy challengers:
- K0 current core;
- K1 all deform + structural bridges;
- K2 independently proven mechanically necessary controls;
- K3 full legal source skeleton diagnostic ceiling.

Do not target a requested joint count.

Measure each additional control by marginal articulation consequence after common-mode rigid motion removal.

Exit:
- minimum evidence-backed control family;
- no teacher-cardinality authority.

### A5 — absolute-joint sampled-feedback challenger

Only if A2/A3 shows a rig-owned residual that locus generation can plausibly fix.

Compare controlled rungs:
- current coarse/residual locus;
- absolute joint diffusion;
- absolute joint diffusion + sampled XYZ -> parent;
- + sampled joint/parent -> next autoregressive state;
- + BFS-equivalent order augmentation as a separate final rung.

Required metrics:
- free-running structural closure;
- carrier-bound mechanical utility;
- locus error;
- parent/root correctness;
- terminal stability;
- no teacher feedback.

The historical C3/C4 source is the starting point; do not reimplement from scratch.

### A6 — mechanically sufficient basis loss

Only after a differentiable carrier-native `L_mech` is parity-checked against hard G3.

Potential ATLAS objective:

[
L_{ATLAS}
=
L_{reference_locus}
+
L_{structure}
+
lambda_m L_{mech}(M,R,hat W)
+
lambda_c L_{control_complexity}
]

where (hat W) comes from a frozen/admitted MIRA or a bounded inner optimization used only for training research.

Hard Compiler qualification remains separate.

### A7 — ATLAS promotion court

No promotion from teacher metrics alone.

Require:
- hard Stage35 improvement or non-regression;
- lower/equal catastrophic-face count;
- free-running stability;
- control-basis adequacy;
- editability/locality;
- held-out coherent assets;
- no carrier-lineage violation.

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
- rigid-transform-invariant fold/self-intersection surrogate;
- parity/order regression against hard G3 cases.

Safety correction:
- do **not** use `dot(rest_normal, posed_normal) < 0` as fold guilt;
- do **not** use fixed-camera projected winding sign as intrinsic fold guilt;
- `canonical/DEFORMATION_WITNESS_VALIDITY_AUDIT_V1_20260928.json` proved both can flip under valid proper rigid rotation;
- until a differentiable neighborhood/intersection surrogate is validated, hard self-intersection/geometry-integrity evidence remains the fold authority.

## C0 — architecture contract migration — PASS

Tasks:
- Stage26/30 depend on Stage19 carrier;
- remove hidden Stage35 -> Stage18 read;
- explicit parent-attempt repair input;
- carrier hash in learned-mechanics preregistration;
- stage count not treated as product architecture.

Exit:
- contract tests PASS.

## C1 — Stage19 carrier evidence — PASS

Tasks:
- derive exact carrier positions/normals;
- seal face/connectivity digest;
- bind source/GSA addressing;
- emit typed evidence.

Exit:
- deterministic hash;
- same candidate -> same evidence;
- AC/BD -> different topology/carrier hash.

## C2 — zero-training model compatibility — ATLAS PASS / MIRA IN PROGRESS

ATLAS:
- **PASS** — workflow `37200030790`, canonical result `canonical/ATLAS_CARRIER_FIRST_C2_RESULT_20261004.json`;
- 28/28 controls reproduced; changed parent count = 0;
- normalized archived-proposal RMSE = `3.50346e-6`;
- carrier-provenance-outside support count = 0;
- immediate ATLAS retrain = **NOT AUTHORIZED**;
- explicit carrier conditioning remains blocked on A3 necessity evidence.

MIRA:
- query existing decoder at exact carrier vertices;
- use GSA backbone/field tokens plus deterministic carrier support-bound memory transport;
- exact carrier XYZ/N + exact carrier↔joint pair geometry;
- produce direct carrier skin, not transferred surface weights.

Exit:
- determine exactly which components need fit;
- no A100.

## C3 — Knight carrier-first one-shot court — SCIENTIFIC FAIL / OWNER ATTRIBUTION ACTIVE

Measured run:
- workflow `37200683451` completed successfully as an execution;
- frozen/A2-verified Stage28 ATLAS rig;
- frozen MIRA V6 checkpoint;
- exact same Stage18/19 carrier;
- no training.

A/B:
- legacy GSA skin -> deterministic carrier transfer: **G3 PASS / G3B PASS / unsafe faces = 0**;
- zero-train direct carrier query using current Stage19 face-cross normals: **G3 FAIL / G3B FAIL / unsafe faces = 2348**;
- identity carrier vertices showed direct-vs-legacy weight-row L1 p95 ≈ `1.7789`.

This is a scientific FAIL of the first direct-carrier query construction, not a workflow failure, and it does **not** authorize MIRA refit by itself.

Source-level forensic:
- compact candidate faces preserve connectivity but not oriented winding because triangle IDs are canonicalized/sorted;
- Stage19 V1 then cross-producted those unordered tuples and treated the result as a signed carrier normal field;
- signed query normals were therefore not a valid authority.

### C3.1 — carrier normal authority causal court — COMPLETE / CAUSAL DRIVER CONFIRMED

Workflow:
- `37201156219`;
- verdict: `NORMAL_AUTHORITY_CAUSAL_DRIVER_SUPPORTED`.

Frozen:
- carrier XYZ/connectivity;
- ATLAS rig;
- MIRA checkpoint/backbone/readout;
- query support;
- hard G3/G3B family.

Changed only:
- current Stage19 face-cross normal field;
- versus signed GSA normal field transported through exact geometry `SurfaceSupportBinding`.

Normal evidence:
- all carrier vertices valid in the counterfactual;
- signed cosine mean ≈ `0.00278`;
- negative-dot fraction ≈ **0.4974**;
- strongly opposed fraction ≈ **0.4126**;
- identity subset negative-dot fraction ≈ **0.4978**.

Therefore the current Stage19 signed normal field is effectively unoriented with respect to the admitted GSA signed normal field.

Mechanical consequence:
- current normal arm: unsafe faces **2348**;
- GSA-oriented geometry-support normal counterfactual: unsafe faces **1327**;
- identity weight parity improves from p95 `1.7789` to exact `0.0`.

Conclusion:
- normal authority is a **proven causal bug**;
- it explains a large fraction of the direct-query failure;
- it is not the sole residual owner because 1327 unsafe faces remain.

Product direction:
- Stage19 requires a deterministic oriented carrier normal/basis authority;
- do not derive signed normals from canonical-sorted face tuples;
- transported GSA normals are a diagnostic counterfactual, not yet the final product normal authority.

### C3.1b — normal-path factorization — COMPLETE

Workflow:
- `37201079379`.

Additional factorization with exact carrier XYZ:
- current carrier normals in both paths: unsafe **2348**;
- transported GSA normals in geometry7 + pair geometry using current carrier-query support semantics: unsafe **903**, identity parity exact;
- GSA normals only in geometry7: unsafe **1033**, identity p95 L1 ≈ `0.1218`;
- GSA normals only in pair geometry: unsafe **2277**, identity p95 L1 ≈ `1.7502`.

Interpretation:
- geometry7 normal semantics dominate the identity-domain shift;
- pair-normal semantics are secondary but nonzero;
- support choice changes the residual, so carrier normal authority and semantic-memory support authority must remain separate contracts.

### C3.2 — MIRA support-mixture order court — COMPLETE / NOT THE OWNER

Workflow:
- `37201533011`;
- canonical result: `canonical/MIRA_SUPPORT_MIXTURE_C32_RESULT_20261004.json`;
- verdict: `SUPPORT_MIXTURE_ORDER_NOT_SUFFICIENT`.

All zero-train arms held fixed:
- exact carrier XYZ/connectivity;
- A2-verified ATLAS rig;
- frozen MIRA backbone/readout;
- signed-normal C3.1 counterfactual;
- support coefficients;
- no teacher inference input;
- no training.

Results:
- `LATENT_PREBLEND`: **1327 unsafe**;
- `LOGIT_POSTBLEND`: **1340 unsafe**;
- `PROB_POSTBLEND`: **1332 unsafe**;
- diagnostic `LEGACY_WEIGHT_TRANSFER_CEILING`: **0 unsafe / G3 PASS / G3B PASS**.

Therefore nonlinear support-mixture order is not the residual owner.

Residual localization is decisive:
- all-identity faces: **22,676 faces / 0 unsafe**;
- faces touching exactly one non-identity carrier vertex: **1,649 / 225 unsafe**;
- faces with two or more non-identity carrier vertices: **4,006 / 1,102 unsafe**.

There are 3,982 non-identity carrier vertices. On unsafe faces, direct-query rows differ from the diagnostic semantic ceiling by roughly L1 `1.55-1.60` on average and approximately `1.99` at p95.

Interpretation:
- the frozen MIRA semantic field is coherent on its trained GSA-node domain;
- after the signed-normal bug is removed, every pure identity face is mechanically safe;
- the remaining gap is specifically the **compiled-carrier/interpolated query domain**;
- merely moving convex support composition across the nonlinear readout does not solve it;
- because the same carrier + same rig closes under the diagnostic frozen semantic-field ceiling, a mechanically admissible field exists without carrier or ATLAS changes.

### C3.3 — carrier-bound frozen semantic projection vs exact motion — COMPLETE / MICROSTRESS PASS IS NOT MOTION CLOSURE

Workflow:
- `37202726161`;
- scientific result was emitted before the workflow surfaced a failure status;
- no training;
- no teacher weights;
- frozen MIRA semantic field;
- deterministic exact carrier-bound mechanical-support projection.

Micro-stress:
- G3 PASS;
- G3B PASS;
- unsafe faces = **0**;
- max condition ≈ `9.49`;
- max edge ratio ≈ `1.69`.

Exact idle/run/slash motion:
- **51 / 51 sampled frames FAIL**;
- maximum condition ≈ `381.66`;
- maximum edge ratio ≈ `9.33`.

Interpretation:
- the Stage34 micro-stress witness is functioning according to contract: it is a small numerical compatibility witness, not motion-capability authority;
- product motion capability remains owned by Stage35 exact quaternion clip execution;
- therefore “G3/G3B microstress PASS” must never be promoted as final mechanical closure.

### C3.4 — exact-motion residual localization — COMPLETE / RESIDUAL IS NOT NON-IDENTITY-ONLY

Workflow:
- `37203108738`;
- diagnosis: `MIXED_IDENTITY_AND_COMPILED_DOMAIN_FAILURE`.

Exact 51-frame idle/run/slash replay:
- unique bad faces = **90**;
- all-identity bad faces = **47**;
- non-identity-touching bad faces = **43**;
- identity fraction of bad faces ≈ `0.5222`;
- max condition ≈ `381.66`;
- max edge ratio ≈ `9.328`;
- min area ratio ≈ `0.00262`.

Important:
- several high-severity identity-domain offenders have acceptable static triangle shape;
- therefore the residual cannot be attributed only to Stage18 seam/generated vertices;
- the earlier “only adapt 3,982 non-identity rows” C4 authorization was too narrow.

### C4.0 — non-identity microstress direct-weight oracle — COMPLETE / FAIL

Workflow:
- `37202314087`.

Frozen:
- exact carrier;
- A2-verified ATLAS rig;
- 12,049 identity weight rows;
- frozen MIRA base field.

Optimized:
- only 3,982 non-identity carrier rows;
- differentiable condition/area/edge microstress surrogate;
- no teacher weight objective.

Result after 300 steps:
- hard microstress G3 FAIL;
- G3B FAIL;
- unsafe faces = **787**;
- max condition ≈ `1313.05`;
- max edge ratio ≈ `18.32`.

Verdict:
- `FAIL_DIRECT_WEIGHT_ORACLE__DO_NOT_FIT_HEAD_YET`;
- do not fit the previously proposed non-identity correction head from this objective;
- either the surrogate/probe family is insufficient for the true product motion objective, or ownership is broader than the non-identity skin domain.

### C4 — exact-motion owner attribution before any MIRA fit — ACTIVE

The previous non-identity-only MIRA adaptation authorization is **withdrawn**.

Not authorized yet:
- C4 correction-head training;
- decoder/tail fit;
- full MIRA retrain;
- ATLAS retrain;
- teacher-weight authority;
- A100 spend.

The next question is now:

> On the exact Stage19 carrier and frozen A2-verified ATLAS rig, can **skin weights alone** close the exact Stage35 idle/run/slash deformation court when optimization is driven by those real motion transforms rather than the Stage34 microstress surrogate?

#### C4.1 — actual-motion skin-only direct-weight ceiling — NEXT

This is an optimization oracle, not a model fit.

Freeze:
- Stage19 carrier topology/XYZ;
- A2-verified Stage28 ATLAS rig;
- exact current idle/run/slash retarget tracks;
- carrier/query lineage;
- Stage35 mechanical thresholds;
- no teacher skin;
- no topology repair.

Base:
- frozen MIRA semantic field deterministically bound/projected to the exact carrier;
- this base is diagnostic initialization, not teacher truth.

Active region:
1. run the exact 51-frame motion court;
2. collect union offender faces;
3. activate every weight row on those faces plus one carrier 1-ring closure;
4. keep every row outside the closure exact-frozen;
5. identity and non-identity rows are both admissible **only inside the proven offender closure**.

Optimize:
- exact-motion differentiable mechanical consequence over the real 51 frame transforms;
- condition / area / edge terms aligned with the hard court;
- trust/locality regularization to the frozen MIRA base;
- simplex-preserving weights;
- no teacher weight target.

Hard exit:
- replay the full exact 51-frame motion court on the full carrier;
- rerun Stage34 microstress G3/G3B as non-regression;
- do not claim PASS from surrogate loss alone.

Decision tree:

```text
C4.1 exact-motion skin-only ceiling PASS
    -> skin/model objective is a sufficient owner
    -> authorize minimal MIRA carrier-motion adapter/fit
    -> train against compiled-carrier actual-motion consequence

C4.1 FAIL
    -> do NOT fit MIRA
    -> open C5 actual-motion rig ceiling with skin reoptimized
       and/or C6 carrier repair attribution
```

This is the key owner-separation court before any expensive fit.

#### C4.2 — minimal carrier-motion MIRA adaptation — BLOCKED ON C4.1 PASS

If C4.1 proves skin-only sufficiency, preferred fit order:

1. frozen backbone/readout + small bounded carrier-motion correction head;
2. decoder/tail-only fit;
3. larger MIRA fit only if the smaller adapter is falsified.

Training requirements:
- compiled-carrier examples;
- exact carrier binding;
- coherent reference motion may supply behavioral reference evidence;
- product objective is mechanical consequence, not teacher weight equality;
- hard Stage35 remains qualification authority.

Identity rows are no longer assumed universally safe under product motion; any bypass/freeze mask must be justified by the exact-motion court, not by microstress identity alone.

#### C4.3 — larger MIRA fit — LAST RESORT

Requires evidence that the frozen semantic representation itself, rather than the carrier-domain adapter/objective, is the blocker.

Compute:
- C4.1 remains local/self-hosted if feasible;
- no A100 until C4.1 proves that a learnable skin-only solution exists and the selected training bundle exceeds local practical limits.

## C5 — motion adapter and ATLAS control-basis attribution — ACTIVE

C4.1 is COMPLETE / FAIL.

Workflow:
- `37204866091`.

Frozen:
- exact Stage19 carrier;
- A2-verified ATLAS rig;
- exact 51-frame idle/run/slash motion;
- no teacher weights;
- no topology repair.

Skin-only optimization result:
- bad faces: `90 -> 20`;
- failed frames: `51 -> 40`;
- max condition: `381.66 -> 94.66`;
- max edge ratio: `9.33 -> 4.05`;
- Stage34 microstress remained PASS;
- exact-motion closure still FAIL.

Verdict:
- skin is a material contributor but **not a sufficient sole owner**;
- MIRA fit remains blocked;
- next attribution must separate retarget/motion-adapter error from ATLAS rig/control-basis insufficiency and carrier insufficiency.

### C5.0 — tree-consistent retarget ceiling — COMPLETE / MATERIAL CONTRIBUTOR, NOT SOLE OWNER

The current demo retarget mapping is independently nearest-cost and historical forensic evidence found target-parent/source-child hierarchy reversal or cross-branch assignments.

Historical evidence:
- 4 target edges contain source hierarchy reversal/cross-branch mappings;
- ablating only those bad children improves catastrophic stretch but does not close the problem;
- therefore retarget is a real contributor but not proven sole owner.

Court:
- freeze carrier, ATLAS rig, and an admitted skin state;
- compare current mapping against a target-tree-constrained source mapping where:
  - target parent and child may map to the same source chain;
  - otherwise target parent must map to a source ancestor of the target child mapping;
  - side consistency and geometry cost remain secondary objectives;
  - no source-name hard coding in the generic solver;
- execute the exact 51-frame motion court.

Exit:
- quantify how much residual belongs to the motion-adapter layer;
- if tree-consistent retarget closes or materially changes owner ordering, repair retarget before changing ATLAS;
- otherwise proceed to C5.1.

Measured workflow:
- \`37205629997\`.

Result with the C4.1 optimized skin:
- hierarchy violations: \`4 -> 0\`;
- failed-frame sum across idle/run/slash: \`40 -> 35\`;
- unique-bad-face sum across clips: \`29 -> 16\`;
- run unique bad faces: \`15 -> 10\`;
- slash unique bad faces: \`12 -> 4\`;
- no clip fully closed.

Verdict:
- tree-consistent retarget is a real quality improvement and should replace the current independent nearest mapping;
- it is not sufficient to close the mechanical system;
- ATLAS/control-basis and/or carrier ownership remains open.


### C5.1 — subject-agnostic optimum control-basis court

**Do not optimize for a requested joint count.**

The goal is to infer the smallest control basis that is mechanically sufficient for the current carrier and admitted motion/editing envelope.

For a candidate control basis \(R\), define its best achievable skin state under the same carrier:

\[
W^*(R)=\arg\min_W L_{\text{mech}}(M,R,W,Q)
\]

subject to skin legality/locality constraints.

Define hard subject-specific admissibility with explicit robustness thresholds:

\[
A_\tau(R)=
[G_{\text{mech}}(R)\ge\tau_m]
\land
[G_{\text{source}}(R)\ge\tau_s]
\land
[G_{\text{edit}}(R)\ge\tau_e]
\land
[G_{\text{free}}(R)\ge\tau_f]
\]

where the \(G\) terms are measured on the current carrier and admitted probe/motion/editing envelope after skin is reoptimized/adapted for that rig.

Then select the minimum-sufficient basis:

\[
R^*=\arg\min_R C(R)
\quad\text{s.t.}\quad
A_\tau(R)=\text{true}
\]

Only after minimum complexity is established may excess proof margin be used as a tie-breaker between equally simple admissible bases.

This ordering is deliberate: maximizing proof margin before complexity can reward unnecessary controls and is therefore not the RealSaS optimum-rig objective.

\(C(R)\) may include:
- control count;
- unnecessary chain depth;
- redundant near-collinear controls;
- edit burden;
- instability/uncertainty penalty.

This formulation is subject-agnostic:
- Knight may close at one count;
- a creature, cloth-like appendage, quadruped, or unusual topology may require another;
- the number is an output of the proof, not a target label.

### Generic minimum-sufficient selector — IMPLEMENTED / CONTRACT PASS

Code:
- \`compiler/realsas_compiler_core/control_basis_selection_v1.py\`;
- \`tests/compiler/test_control_basis_selection_v1.py\`.

Contract:
- every candidate basis reports signed normalized margins for mechanics, source fidelity, editability and free-running stability;
- only bases with all hard margins \(\ge 0\) are admissible;
- selection minimizes complexity first;
- control count is the next deterministic simplicity term;
- excess robustness margin is only a tie-breaker after equal simplicity.

Synthetic regression explicitly proves that two subjects can select different optimum control counts from the same candidate-count family.

This selector does not decide how candidates are generated; C5.1a/C5.1b and later grow-to-need courts generate the measured candidate set.

### Marginal control utility

For candidate control (j), define mechanical marginal gain only after reoptimizing/adapting skin:

[
Delta_j(R)=
Phi(M,Rcup{j},W^*(Rcup{j}),Q)
-
Phi(M,R,W^*(R),Q)
]

where (Phi) is a proof-margin vector/score derived from:
- failed frame count;
- catastrophic face count;
- condition margin;
- area margin;
- edge-stretch margin;
- motion coverage;
- edit locality.

A control is not justified merely because a teacher/source skeleton contains it.

### Search policy

Use two complementary directions:

**Grow-to-need**
- start from the minimal legal/current core;
- add the control with highest positive marginal proof gain;
- stop when hard closure and robustness margin are reached.

**Prune-to-proof**
- start from an admitted richer basis;
- remove the lowest-utility control;
- reoptimize/adapt skin locally;
- keep the removal only if hard closure and editability remain non-regressive.

The intersection/stable fixed point is the candidate optimum basis.

### Historical Knight capacity evidence — diagnostic only

Existing teacher-side court:
- K0 current core = 20;
- K1/K3 = 41;
- K1-K0 adds 21 zero-direct-skin-mass deform controls;
- generic common-mode-removed marginal articulation court measured **0/21** additional controls with non-zero internal skin-deformation effect.

Interpretation:
- more joints do not automatically mean more useful mechanical capacity;
- this evidence is a prior/diagnostic only;
- it does **not** authorize 20 as the Knight product count and does not define counts for unseen subjects.

### C5.1a — effective-control prune screen — RUNNING

Generic operator:
- for each non-root control, remove its independent motion DOF;
- collapse that control's skin mass exactly to its parent;
- preserve the carrier and all remaining controls;
- use the tree-consistent retarget mapping;
- replay the exact 51-frame motion court;
- no teacher count or teacher weight is used.

This is a cheap screen only. A passing candidate advances to C5.1b; product pruning requires skin reoptimization, hard proof and editability/non-regression.

### C5.1b — full prune-to-proof — BLOCKED ON C5.1a

For screened candidates:
- construct the reduced/effective basis;
- reoptimize/adapt skin for that basis under exact-motion consequence;
- rerun Stage35 hard motion proof;
- rerun source fidelity and editability/locality;
- feed the resulting signed margins + complexity to the generic minimum-sufficient selector.

### C5.2 — actual-motion rig ceiling

After C5.0, use the C5.1 search formulation on the exact carrier.

For each rig challenger:
- reoptimize/adapt skin under the same exact-motion objective before comparing rigs;
- never compare one rig with stale skin from another rig;
- run hard Stage35 exact-motion proof;
- measure complexity and editability.

Rig challenger ladder:
1. current A2-verified ATLAS rig;
2. current rig with bounded locus perturbation in owner regions;
3. mechanically justified grow-to-need controls;
4. mechanically justified prune-to-proof simplification;
5. absolute-joint sampled-feedback ATLAS challenger only if locus/control generation remains the proven bottleneck.

### C5.3 — ATLAS architecture change — BLOCKED ON C5.2

Only if C5.2 proves that current ATLAS representation/generation cannot produce an admissible sufficient basis.

Then test:
- absolute next-joint XYZ diffusion;
- sampled joint XYZ -> parent prediction;
- sampled joint/parent state -> later generation;
- full carrier/source perception access;
- control-existence/stop head optimized against mechanical sufficiency, not teacher cardinality;
- Compiler retains tree/root legality and final proof authority.

No full ATLAS retrain before this point.

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

# 10.1 Training corpus use: reference domain vs compiled-carrier domain

Coherent rigged assets are used in two distinct modes.

## Reference-domain fit

Input/evidence comes from the coherent asset carrier itself:

[
(M_{ref}, R_{ref}, W_{ref}, A_{ref})
]

Use this to teach:
- articulation priors;
- joint locus/role priors;
- skin semantics;
- motion/deformation priors.

This is ordinary supervised/reference learning. It does **not** make reference topology product authority.

## Compiled-carrier fit

For the same legal reference asset:
1. render the admitted 2D observation contract;
2. run the RealSaS geometry path to Stage19;
3. obtain `M_compiled`;
4. run ATLAS/MIRA on `M_compiled`;
5. evaluate mechanical/deformation consequence against the coherent reference behavior through admitted correspondence/evaluation.

The product-side target is therefore not:

[
W_{compiled}=W_{ref}
]

on a different topology.

Instead train toward:

[
D(M_{compiled},R,W,Q) approx D(M_{ref},R_{ref},W_{ref},Q)
]

where correspondence/evaluation is explicit and topology is allowed to differ.

This compiled-carrier corpus is the preferred domain-bridge before FITK/LOFO/unseen.

## Fit ordering

- ATLAS: freeze reusable perception encoder first; fit the smallest locus/diffusion/control head proven necessary by C3/A3.
- MIRA: freeze GSA/skeleton backbone first; fit carrier query decoder/tail or bounded residual first.
- Full-model refit requires evidence that the frozen shared/perception representation itself is the blocker.
- No component is retrained merely because the architecture changed around it.

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
