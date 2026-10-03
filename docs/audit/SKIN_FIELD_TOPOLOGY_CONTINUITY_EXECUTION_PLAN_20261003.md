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

## 5A. Render-entry gate — Mechanics / Geometry Absolute Seal

**Render is forbidden as an attribution test until this seal is green.**

Purpose: when render is finally executed, any remaining visual failure must no longer have a plausible unclosed owner in rig, skin, weights, carrier topology, static mesh quality, deformation conditioning, or authored-motion mechanics. Only then may the program narrow the remaining owner to appearance / presentation / runtime transport.

The seal is conjunctive. Every item below must PASS on the exact same admitted lineage:

### A. Static geometry / topology
- canonical face provenance is complete; no invented faces or unsupported topology;
- manifold / incidence / component invariants PASS;
- no unresolved holes, self-intersections, duplicated faces, invalid winding, or topology-owner ambiguity;
- triangle quality PASS under frozen policy: minimum angle, aspect ratio, area/sliver limits, edge-length limits, and CDT/constrained-boundary rules where applicable;
- component-aware normals / local geometry evidence PASS;
- refinement/remeshing preserves all mechanical no-cross / continuity constraints;
- static quality repair may not erase qualified mechanical partition boundaries;
- zero unresolved static-policy residuals.

### B. Rig authority
- skeleton lineage and parent graph PASS structural qualification;
- rest positions / local frames / root / attachments satisfy current canonical contracts;
- reproducibility under the admitted inference backend is proven;
- no unresolved rig-owner counterfactual remains;
- motion retarget / joint-map coverage is complete for the court clips.

### C. Skin / weight authority
- simplex, non-negativity, influence sparsity / support policy, and coverage PASS;
- admitted Arachne evidence is preserved within the product correction budget;
- no unqualified / unmeasured population is shipped;
- topology x weight discontinuity court PASS;
- refinement / remeshing does not create unbounded skin gradients or tiny-edge opposite-influence failures;
- mesh-weight binding seam is fully typed, provenance-bound, and residual-accounted.

### D. Mechanical topology / deformation domain
- G3B skin-topology compatibility PASS with zero unsafe faces / consequential UNKNOWNs;
- no visual/mechanical triangle spans an unqualified deformation-domain boundary;
- mechanical partition constraints are preserved through all static remesh / quality operations;
- seam-owner / multi-support generated-vertex ownership is total and deterministic;
- repeated bounded topology feedback reaches a fixed point or fails closed.

### E. Local deformation conditioning
- production G3 local-frame micro-stress PASS;
- area-ratio lower/upper bounds PASS;
- condition-number bound PASS;
- edge-ratio lower/upper bounds PASS;
- zero flipped / collapsed faces under the complete G3 probe bank;
- no catastrophic response hidden by aggregate quantiles.

### F. Actual authored motion mechanics
- the full preregistered Idle / Run / Slash court passes on every sampled frame;
- zero failed frames;
- zero catastrophic edge / area / condition events;
- no face flips / collapses;
- no clip-specific repair or threshold;
- worst-case and tail metrics are reported, not only averages.

### G. Stability / adversarial courts
- alternate qualified triangulation / refinement replay does not change mechanical class;
- denser refinement does not cause stress divergence;
- tiny-edge / near-contact / cross-component / seam adversarial tests PASS;
- same subject-free operator and same frozen thresholds are used across all carrier variants;
- no success depends on reverting to a historically convenient topology.

### H. Source / physical geometry fidelity
- source-surface correspondence and addressing lineage remain valid after all repairs;
- no geometry repair changes the intended physical/source surface without explicit typed authority;
- camera / view / depth geometric contracts required by mechanics are intact;
- no hidden geometry warp is introduced by a downstream consumer.

Only when **A–H all PASS on one immutable lineage** may the state be called:

```
MECHANICS_GEOMETRY_ABSOLUTE_SEAL = PASS
```

At that point, and only at that point, run the render / appearance / presentation chain.

### Attribution rule after the seal

If the exact sealed mechanical/geometric lineage renders incorrectly while:

- no geometry or mechanics are mutated after the seal;
- source appearance bytes and provenance are known;
- the renderer consumes the sealed lineage exactly;

then the mechanical/geometric side is closed by construction and the remaining defect owner is restricted to the downstream **appearance / presentation / runtime-transport** domain.

This does **not** mean G3 alone proves that attribution. G3 is only one sub-gate inside the absolute seal.

## 6. A100 escalation rule

Do **not** use A100 merely because a compiler experiment fails.

A100/readout refit is allowed only if all of the following are established:

1. the direct deformation-aware deterministic realization is implemented and correctly executed;
2. no bounded evidence-preserving realization can pass the generic mechanical courts;
3. failure is attributed to insufficient Arachne evidence rather than Geppetto or carrier topology;
4. V9-compatible training truth / teacher mapping is legal and complete enough to train against;
5. the refit plan freezes the backbone by default and trains only the minimal necessary readout unless evidence proves otherwise;
6. A100 capacity is available; compute budget is no longer the limiting gate. The remaining gate is scientific legality/completeness of V9-compatible truth and minimal trainable scope.

Before any A100 run, report to the user with:
- exact reason compiler repair was rejected;
- exact trainable parameter scope;
- run configuration and stop criteria;
- checkpoint/rollback plan.

## 6A. FITK=2 escalation before any genericity claim — DEFERRED UNTIL KNIGHT E2E CLOSURE

Single-witness iteration is not sufficient evidence for genericity. However, FITK=2 is **not an active execution target before Knight reaches the Mechanics/Geometry Absolute Seal and a real end-to-end render**. Knight remains the current integration witness. After Knight E2E closure, freeze architecture/policy and cold-replay the second subject before deciding whether joint fitting is required.

Terminology note: do not reuse the historical repo label `FIT2`, which referred to a same-Mage full-subject reclosure. This program means **K=2 distinct subjects trained/evaluated under one frozen architecture and one frozen compiler policy**.

Required subjects:

- Knight: current V9 mechanics witness.
- Mage: preferred second subject because historical exact rig/skin authorities and Arachne/Geppetto evidence exist, but all Mage truth must be regenerated/rebound under the current product contracts; stale historical FIT1/FIT2 product claims are not reusable as current authority.

Rules:

1. Freeze subject-free representation, compiler qualification policy, G3/G3B thresholds, topology operators, and training objective before exposing the second subject to policy development.
2. No per-subject thresholds, masks, topology heuristics, lambdas, loss weights, or repair paths.
3. Arachne readout is the first trainable scope. Geppetto remains frozen unless independent cross-subject evidence reopens skeleton ownership.
4. Use balanced subject sampling and report all metrics separately per subject plus the aggregate; one subject may not hide the other's regression.
5. Every compiler/topology repair must run unchanged on both subjects.
6. A fitted subject may be used for capacity/continuity closure, but **FITK=2 does not prove generalization**.
7. After both fitted subjects close, require at least one truly unseen third subject (or a preregistered LOFO/unseen court) before any genericity/generalization claim.
8. If a policy change is motivated by failure on either fitted subject, both subjects must be rerun from frozen inputs before admission.

Decision rationale:

- Knight has demonstrated that a static-quality improvement can regress dynamic mechanical compatibility.
- Historical Mage evidence shows a second independent subject already exists in the project lineage.
- Joint fitting reduces the risk that learned evidence or compiler seam policies silently encode Knight-specific coincidences.
- A third unseen subject remains mandatory because two fitted subjects can still be jointly overfit.

A100 authorization under this program therefore requires not only a legal Knight target but a current-contract Mage target and a frozen two-subject training/evaluation manifest.

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

## 8B. Decision after direct-LBS mesh-weight projection

Phase B directly optimized carrier deformation under the generic G3 local-frame probe bank while:

- keeping the qualified Arachne surface skin fixed;
- hard-preserving original joint support;
- hard-parameterizing per-vertex simplex conservation;
- using no authored clip data in optimization;
- evaluating results separately under frozen G3 and the 51-frame exact-motion court.

The self-hosted workflow was stabilized against GitHub artifact transport failures by reusing a lineage-verified frozen cache. Verified bindings:

- surface lineage: `25b8ff0acd2154d8a3fc333d65af70b65f2658309135cf630225ecdfc35ab08d`;
- V9 candidate lineage: `3ab6b0ae02e362079fed11ceb627acf9be795004e790ee897c98a1eadc14edab`;
- fresh CUDA skeleton/skin hashes were independently SHA-verified before reuse.

Measured direct-LBS variants from run `37120181421`:

| lambda | numerical KKT | p95 row L1 | p99 row L1 | max row L1 | G3 max condition | failed frames | max motion edge |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | ~2.98e-9 | ~1.25e-4 | ~0.572 | ~1.010 | ~356.97 | 51/51 | ~369.39x |
| 1.0 | ~4.91e-8 | ~9.59e-5 | ~0.618 | ~1.372 | ~2650.42 | 51/51 | ~337.08x |
| 10.0 | ~9.89e-8 | ~4.77e-5 | ~0.810 | ~2.000 | ~883.90 | 51/51 | ~298.84x |

The correction is highly localized, so p95 alone is misleading: the catastrophic tail requires near-complete semantic reassignment on some rows while all exact-motion frames still fail.

Current product code is much stricter than these diagnostics:

- `product_mesh_skin_v1.py::_TRANSFER_EPS = 1e-9`;
- `mwb2_skin.py::_MESH_SKIN_TRANSFER_REPAIR_L1 = 1e-9`;
- the shipping Stage36 binder uses those default bounds.

Decision:

- **REJECT** direct-LBS carrier repair as the normal-path solution;
- do not pursue lambda=100 merely to force the Knight witness, because lambda=10 already requires materially different semantic weights and still fails catastrophically;
- preserve the result as owner-attribution evidence: the carrier binder cannot legally repair the admitted Arachne evidence within the existing semantic-ownership contract;
- advance to Arachne evidence/training-domain diagnosis;
- A100 remains unauthorized until V9-compatible truth/evidence mapping is proven legal and sufficiently complete.

## 8B.1. Static/mechanical composition replay — run 37124108389

The first explicit composition replay tested whether the existing generic static-quality operator family can close the one-pass teacher-oracle repartition child **without losing the mechanical class that motivated the repartition**. Qualified teacher skin remained frozen and no product authority was minted.

Measured result:

- input static-policy violations: **3,217**;
- after four admitted static cycles: **66** violations remain;
- residual classification: **1 interior/unprotected**, **65 boundary-or-protected**;
- topology manifold report remained PASS;
- final vertices / faces: **16,811 / 31,328**;
- final G3B unsafe faces: **277**;
- final G3: **FAIL**, max condition **~10,362.18**;
- exact authored motion: **51/51 FAIL**;
- maximum motion edge ratio: **~3,200.14x**.

This is a decisive composition failure, not a request for a larger repair budget. Static quality improved by ~97.95%, but the static operator sequence destroyed the mechanical improvement of the one-pass repartition child (previous max motion edge ~56.15x) and left the candidate mechanically much worse.

Decision:

- **REJECT** independent terminal sequencing of "dynamic repartition, then generic static cleanup";
- do not increase static cycles or weaken the 7.5-degree/aspect policy merely to close this witness;
- do not mutate QualifiedSkinIR;
- do not use A100: Section 6 remains unsatisfied because carrier topology / basis compatibility is still an open owner;
- next evidence court must test **deformation-field equivalence across topology/basis**, rather than treating exact projected teacher vertex weights as proof of equivalent teacher mechanics.

The next court must distinguish:

1. teacher behavior on its own source topology/basis;
2. the same teacher-induced field sampled/realized on the current product carrier;
3. alternate qualified carrier triangulations/refinements under the same subject-free realization policy.

Required measurements are deformation-space, not only vertex-weight-space: sampled displacement error under the frozen G3 probe bank, local edge/Jacobian response error where measurable, G3/G3B class, and exact-motion class. This court is diagnostic/attribution only and may not mint product weights or topology.

## 8C. V9 exact-teacher reprojection oracle — A100 escalation paused

Run `37122879585` rebuilt the teacher bank for the **current 15,490-node V9 RiggingSurface** from the exact artist/source authority using the frozen historical projector:

`EXACT_SOURCE_TRIANGLE_CLOSEST_POINT_BARYCENTRIC_WITH_COMPONENT_MECHANICAL_ATTACHMENT`.

Authority checks:

- V9 surface lineage: `25b8ff0acd2154d8a3fc333d65af70b65f2658309135cf630225ecdfc35ab08d`;
- V9 candidate lineage: `3ab6b0ae02e362079fed11ceb627acf9be795004e790ee897c98a1eadc14edab`;
- exact teacher source SHA-256: `96435646a0084a0a038040ee748cb33d3bfa0b1ed550661befd3b18539e28e6f`;
- exact Geppetto target SHA-256: `26d4d5411b14b23b894f0351ab08fc2da2b18d8fb3ac4776b25b8b2434b4bbc1`;
- generated V9 teacher bank SHA-256: `5f23adc084fdc15c95a6807e5f86fd365d547e7f558a0d341f9ec3be9bfa2987`;
- clean rows: **13,840 / 15,490 = 0.8934796643**;
- invalid / distance-warning rows: **1,650**;
- projection simplex residual: ~`2.22e-16`.

Critical oracle result:

- exact all-projected teacher skin on the current V9 carrier: **G3 FAIL**;
- G3 max condition: **~177.6753**;
- exact authored-motion court: **51/51 FAIL**;
- max motion edge ratio: **~188.6986x**.

Fresh Arachne versus this reprojected teacher:

- teacher-valid p95 row-L1: **~0.18872**;
- teacher-invalid p95 row-L1: **~0.03598**.

Decision:

- **do not start A100 readout refit yet**;
- the exact teacher itself is not mechanically admissible on current V9 connectivity, so readout-only fitting to this field cannot by itself establish closure;
- the previous compiler-weight-repair rejection remains valid, but ownership must now be split between **field prediction** and **carrier seam/connectivity compatibility**;
- next court: attribute exact-teacher failures by source-triangle topology class and test the existing subject-free G3B seam-cut / repartition mechanism as a teacher-oracle topology ceiling while keeping teacher weights frozen;
- A100 escalation resumes only if a mechanically admissible target topology/field realization is demonstrated.

## 8D. One-pass production repartition feedback on V9 teacher oracle

Run `37123289786` applied the existing production dynamic topology feedback path to the exact V9 teacher field, without changing any skin weight:

`SOURCE_EDGE_PROBE_RATIO_V1 -> mechanical_repartition_v2 -> Stage18 holeless dense -> COMPONENT_HARMONIC_DIRICHLET_V1`.

Measured one-pass change:

- parent partition components: **50**;
- parent G3B unsafe faces: **418**;
- direct source-edge separation seeds: **274**;
- partition-closed final SEPARATE constraints: **1,125**;
- child components: **88**;
- child vertices / faces: **17,733 / 33,172**;
- child G3B unsafe faces: **166**;
- teacher max authored-motion edge: **188.70x -> 56.15x**;
- child G3: still FAIL, max condition **~366.92**;
- exact authored motion: still **51/51 FAIL**;
- child static policy violations: **3,217**.

Interpretation:

- the existing generic repartition loop materially attacks the correct owner; this is not evidence for a Knight-specific weight patch;
- one pass is insufficient on the refined V9 carrier;
- Stage18 rebuild also reopens static mesh-quality violations, so dynamic partition repair and static quality qualification must eventually be composed rather than treated as independent terminal passes;
- do not start A100 while the exact teacher target itself is still mechanically inadmissible.

Next court:

- execute the already-bounded generic topology feedback loop for at most `DEFAULT_MAX_REPAIR_ITERATIONS=4`;
- use `SOURCE_EDGE_PROBE_RATIO_V1` where new source-edge cannot-links remain;
- use the existing `UNSAFE_FACE_LOCAL_L1_V1` residual proposal path if source-edge probing has no new admissible pair;
- never mutate QualifiedSkinIR weights;
- after topology feedback closes or exhausts, evaluate static quality + G3B + G3 + exact 51-frame motion;
- if mechanics closes but static quality reopens, compose the V9 source-surface static quality repair onto that child and re-run all mechanical courts.

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
| Direct LBS deformation-constrained projection | FAIL / COMPILER OWNER REJECTED | run 37120181421: lambda 10 still 51/51 fail, max motion edge ~298.84x, p99 correction ~0.81, max correction ~2.0; shipping binder budget is 1e-9 |
| V9 exact teacher reprojection oracle | FAIL / TOPOLOGY-FIELD OWNER EXPOSED | run 37122879585: teacher G3 ~177.68, 51/51 motion fail, max edge ~188.70x; A100 paused |\n| Teacher-oracle production repartition one-pass | IMPROVES / NOT CLOSED | run 37123289786: G3B unsafe 418→166, motion max edge 188.70x→56.15x, 51/51 still fail; child static violations 3217 |\n| Iterative teacher-oracle production repartition | SUPERSEDED BY COMPOSITION COURT | one-pass improved mechanics but reopened static quality; independent static cleanup then regressed mechanics |\n| Static↔mechanical composition court | FAIL / OWNER LOCALIZED | run 37124108389: 3217→66 static violations, but G3 max condition ~10362.18, 51/51 motion fail, max edge ~3200.14x; independent static cleanup destroys mechanical class |\n| Deformation-field / topology-basis equivalence court | NEXT | compare teacher-own-basis behavior vs teacher-induced field on V9/alternate carriers in deformation space; no product mutation |\n| Multi-topology invariance court | TODO | required before genericity claim |
| Product contract integration | BLOCKED | requires preceding PASS |
