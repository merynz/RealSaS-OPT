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
- Arachne minimal residual mechanical adapter: **AUTHORIZED FOR CPU PREFLIGHT ONLY**; A100 fit waits for the sealed multi-carrier bundle and Run-All handoff.
- Active compiler step: proposal-guarded static ↔ mechanical fixed-point court.
- Active model step: seal topology-robust multi-carrier Arachne adapter bundle; no GPU fit before that seal.

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

## 8E. Dynamic-child static composition court — composition failure localized

Run `37124108389` completed successfully as an audit workflow, but **did not close either product domain**. This is a scientific FAIL, not a workflow failure.

Starting from the one-pass dynamically repartitioned child:

- static policy violations: **3,217**
- G3B unsafe faces: **166**
- max authored-motion edge ratio: **~56.15x**
- mechanics already remained FAIL.

After four cycles of existing generic static-quality repair operators:

- static policy violations: **3,217 -> 66**
- topology audit: **PASS** (`nonmanifold_edge_count=0`, `illegal_vertex_link_count=0`)
- final minimum angle: **~0.213°** (policy floor 7.5°)
- final maximum aspect: **~537.77** (policy ceiling 16)
- 65 / 66 residual violations are boundary/protected; only 1 is interior/unprotected.

However the same static repair sequence materially regressed the mechanical domain:

- G3B unsafe faces: **166 -> 277**
- G3 max condition: **~366.92 -> ~10,362.18**
- G3 max edge ratio: **~315.98**
- G3 area ratio: **~0.0114 to ~7,811.46**
- authored motion: **51/51 FAIL**
- max authored-motion edge ratio: **~56.15x -> ~3,200.14x**
- max authored-motion condition number: **~145,625.55**.

No skin weight was mutated. Product authority was not minted.

Decision:

- the active blocker is now explicitly **STATIC-QUALITY × MECHANICAL-PARTITION COMPOSITION**, not a standalone G3 threshold issue;
- existing static repair operators are individually effective but are not currently constrained to preserve the deformation-domain / mechanical-partition invariants that made the dynamic child better;
- a static PASS obtained after unconstrained remeshing cannot retain prior mechanical credit;
- the next repair must make mechanical no-cross / partition / deformation-field constraints first-class hard constraints inside static quality operations, then rerun static + G3B + G3 + full motion on the exact same candidate;
- do not render and do not start A100 on this state.

## 8F. Teacher topology/basis equivalence court — hypothesis confirmed

Run `37139120074` directly tested the hypothesis that near-exact teacher weight transfer does **not** imply equivalent deformation mechanics when the carrier topology/basis changes.

The court compared the same teacher-induced field under the same frozen stress probe on:

1. the teacher/source topology/basis;
2. the current V9 product topology/basis.

Weight-transfer fidelity was effectively exact:

- teacher-bank barycentric rebind row-L1 max: **~7.45e-8**.

Teacher/source basis:

- vertices / faces: **3,665 / 6,952**;
- minimum area ratio: **~0.72536**;
- maximum area ratio: **~1.27937**;
- maximum condition number: **~1.32708**;
- maximum edge ratio: **~1.22730**;
- qualification class: **PASS**.

Current V9 basis:

- vertices / faces: **14,399 / 28,810**;
- minimum area ratio: **~0.00731**;
- maximum area ratio: **~56.6407**;
- maximum condition number: **~829.9104**;
- maximum edge ratio: **~62.6450**;
- qualification class: **FAIL**.

Sampled field-equivalence displacement error was small in absolute source scale (p99 ~`1.876e-4`, max ~`0.00611`, max/bbox diagonal ~`0.001368`) while the local deformation/Jacobian class changed catastrophically. This is precisely why vertex-weight similarity or even small point-displacement error is not sufficient authority for a discretized deformation system: local derivatives and triangle basis can amplify a small field mismatch.

Decision:

- **HYPOTHESIS CONFIRMED**;
- retire the assumption `teacher-like vertex weights => teacher-like deformation mechanics`;
- teacher remains a semantic/source-space supervisory oracle, not direct product-space mechanical truth;
- product truth must be evaluated on the actual compiler carrier;
- Arachne evaluation/training must include product-space deformation consequence, not only teacher weight error;
- carrier topology qualification must include field/deformation compatibility.

## 8G. Iterative repartition after harmonic multi-support owner closure

The harmonic seam multi-support owner contract is now generic and gated:

- contract gate run `37136685525`: **PASS**;
- multi-support is preserved as a complete convex support rather than collapsed to an arbitrary single owner;
- residual repartition evidence deterministically ranks admissible support-pair Cartesian products.

With that blocker removed, iterative teacher-oracle repartition run `37137357080` completed the bounded four-repair court:

| evaluation | components | SEPARATE | G3B unsafe | static violations |
|---:|---:|---:|---:|---:|
| 0 | 50 | 0 | 418 | 0 |
| 1 | 88 | 1,125 | 166 | 3,217 |
| 2 | 143 | 2,166 | 66 | 3,336 |
| 3 | 180 | 2,452 | 24 | 3,371 |
| 4 | 194 | 2,541 | 14 | 3,388 |

Final bounded-court result:

- G3B: **14 unsafe remain**;
- G3: **FAIL**, max condition **~489.11**;
- exact motion: **51/51 FAIL**;
- max motion edge: **~32.98x**;
- static violations: **3,388**.

Interpretation:

- the generic mechanical repartition operator is strongly convergent on the correct mechanical owner;
- repartition alone is insufficient and progressively reopens static quality;
- static quality and mechanical partition therefore require a true closed-loop composition, not terminal sequencing.

## 8H. Proposal-level static × mechanical composition

The static operator family now exposes fail-closed proposal-level mechanical admissibility for:

- fixed-vertex edge flip;
- endpoint collapse;
- vertex-cavity retriangulation;
- edge-cavity retriangulation;
- synchronized interior-edge midpoint split.

The local mechanical guard precomputes the frozen G3B pose bank and evaluates only the old/new proposal patch. Acceptance is hard/lexicographic:

```
new unsafe-face count <= old unsafe-face count
AND
new worst normalized mechanical severity <= old severity
```

Static quality cannot purchase a mechanical regression.

For split proposals, the temporary midpoint is evaluated with **exact LBS** using:

- midpoint rest position;
- the exact convex transferred endpoint skin field;
- the frozen joint probe matrices.

It is not approximated as the midpoint of posed endpoints.

Evidence:

- proposal-level static mechanical contract: **PASS**;
- exact-LBS temporary split guard: **PASS**;
- split mechanical-support preservation contract: **PASS**;
- topology-only synthetic adversary is rejected when a diagonal change raises normalized mechanical severity despite unchanged vertex positions/weights.

The closed-loop target is now:

```
G3B
 -> mechanical repartition / Stage18 rebuild
 -> mechanically guarded static proposals
 -> global production + all-face G3B non-regression
 -> repeat to fixed point
 -> STATIC + G3B + G3 + 51-frame exact motion
```

No skin weight is mutated by this compiler loop.

## 8I. Arachne mechanical residual adapter — minimal trainable scope

A full Arachne refit is not the active plan.

Implemented trainable scope:

- frozen existing Arachne field/backbone;
- zero-initialized `MechanicalResidualAdapterV1`;
- **1,582,849 trainable parameters**;
- output remains a non-negative simplex over the admitted joint mask;
- step-0 output reproduces the frozen base field up to floating-point tolerance.

Contract gate run `37138462111`: **PASS** (identity/simplex/gradient/mechanical-loss tests).

Training objective is product-space:

```
L =
    mechanical_consequence_on_compiler_carrier
  + lambda_teacher * teacher_valid_semantic_tether
  + lambda_trust   * frozen_base_field_trust
```

The mechanical term uses differentiable LBS consequences aligned with G3 classes:

- edge stretch;
- area lower/upper violation;
- local deformation condition number.

Teacher-invalid rows are not promoted to exact truth. The compiler exact G3B/G3/actual-motion courts remain final authority.

Because Section 8F proves topology/basis sensitivity, the A100 fit must not optimize against only one carrier. The sealed training bundle should contain a **preregistered carrier ensemble** drawn from subject-free compiler lineages (at minimum current V9 plus mechanically repartitioned child variants). The same Arachne prediction is evaluated across those carriers. This makes the intended target:

> a topology-robust mechanically conditioned influence field,

not a Knight/V9-topology-specific correction.

A100 becomes scientifically justified only after the CPU preflight seals:

- exact frozen base field;
- exact conditioning tensors;
- typed sparse carrier support transfer;
- full frozen probe bank;
- carrier-ensemble manifest and hashes;
- semantic/trust correction budgets;
- deterministic training schedule and rollback checkpoint.

At that point provide a single **Run All Colab notebook**. The notebook trains only the residual adapter and produces a sealed checkpoint + receipt for compiler requalification.



## 8J. Representation-conditioned truth — teacher labels are conditional, not universal

The topology/basis equivalence court in Section 8F changes how both rig and skin supervision must be interpreted.

Binding principle:

> A teacher rig/skin label is truth on the representation that generated it. Once RealSaS owns the carrier, product-space mechanical truth must be measured on the RealSaS representation.

For skinning, the product target is not an abstract vertex label W. It is a conditional object:

W* = f(M, G, D)

where:

- M is the admitted carrier geometry/topology/basis;
- G is the qualified skeleton;
- D is the admitted deformation/motion objective.

Therefore a teacher field W_T that is correct on M_T need not equal the mechanically optimal field W_R on a different RealSaS carrier M_R, even when vertex positions and transferred weights are numerically almost identical.

This is now measured, not hypothetical: Section 8F observed teacher-basis PASS and V9-basis catastrophic FAIL with barycentric weight rebind row-L1 max only about 7.45e-8.

Consequences:

1. teacher weight similarity is retained as semantic/source-space supervision, not final mechanical authority;
2. product-space deformation consequence is mandatory in Arachne training/evaluation;
3. carrier topology is an active representation owner;
4. final skin truth is minted only after exact Compiler qualification on the exact product carrier;
5. a high teacher-match score can mean the model learned the wrong representation-conditioned target very accurately.

The same caution applies to skeleton targets. A teacher skeleton is a strong articulation prior, but the mechanically optimal RealSaS skeleton may contain more or fewer controls if the RealSaS surface, carrier, motion envelope or editability contract differs.

## 8K. Geppetto articulation-capacity reopening — remove the "clean rig" aesthetic prior

The current Knight Geppetto training target is not the full source rig. The target builder at experiments/geppetto_reference_strength_fullstack_v1/mechanical_core_target_v1.py uses:

SKIN_SUPPORTED_PLUS_SUPPORTED_BRIDGES__ASSEMBLY_ONLY_ROOT_EXCLUDED

and explicitly:

- seeds skin-supported deform controls;
- retains unsupported controls only when structurally required as bridges;
- excludes unsupported assembly-only root chains;
- excludes helper/IK-only leaves;
- projects parents to the nearest retained ancestor.

This is a defensible mechanical-core projection, but it is not evidence that the projected rig has sufficient articulation capacity for the final product.

Current Geppetto V2 also does not have a binding 28-joint product architecture. GeppettoCandidateV2 is resource-bounded by admitted surface support; the historical Knight witness happened to qualify 28 joints. The research question is therefore not "raise a hard 28 cap". It is:

> Did target-construction / regularization teach Geppetto a rig that is mechanically too sparse?

### Capacity hypothesis

H_RIG_CAPACITY:

A denser but still justified skeleton can lower downstream deformation residual and reduce cardboard-like articulation without requiring a template-specific rig or weakening Compiler authority.

### Target-policy arms

Do not synthesize arbitrary extra bones just to hit a requested count.

Use teacher-only training/evaluation construction to define increasingly rich anonymous target policies:

- K0 — CURRENT_CORE: current skin-supported deform controls plus required bridges.
- K1 — ALL_DEFORM: all legal teacher deform controls, including low-direct-mass controls.
- K2 — DEFORMATION_NECESSARY: K1 plus non-deform/helper controls whose removal causes a preregistered measurable increase in deformation or retarget residual under the source motion/probe court.
- K3 — FULL_LEGAL_DIAGNOSTIC: full legal source skeleton after removing only controls that cannot participate in the product deformation contract; diagnostic ceiling only, not automatic shipping target.

K2 is the intended scientific target:

> no unnecessary bone, not minimum bone count.

### Marginal articulation gain

Define a subject-free quantity MARGINAL_ARTICULATION_GAIN(control) as the downstream improvement caused by admitting a control, measured with the same skin fitter and motion/deformation court.

A retained control must improve at least one preregistered deformation-capacity quantity without violating rig legality:

- local deformation condition;
- edge/area stress;
- authored-motion residual;
- retarget residual;
- mechanically defined endpoint/segment trajectory;
- downstream skin semantic correction required.

Joint count itself is never a quality metric. Complexity is a tie-breaker only after mechanical quality.

### Articulation-capacity court

Hold constant:

- exact RiggingSurfaceIR lineage;
- Geppetto architecture arm;
- optimizer budget;
- initialization/seeds;
- Compiler graph qualifier;
- skin training/fitting budget;
- carrier and G3/G3B/motion courts.

Compare K0/K1/K2/K3 using:

1. proposal matched-position error and calibrated uncertainty;
2. root/parent evidence and Compiler graph PASS;
3. qualified joint count and unsupported-control rate;
4. downstream skin correction budget;
5. G3B / G3;
6. exact authored motion;
7. marginal articulation gain;
8. model/runtime cost.

A richer rig is promoted only if product-space mechanics improves. "Looks cleaner" is not an admissible selection criterion.

### Code changes

Planned:

- keep mechanical_core_target_v1.py frozen as K0 baseline;
- add experiments/geppetto_reference_strength_fullstack_v1/mechanical_capacity_target_v2.py;
- add typed target-policy receipts and source-provenance hashes;
- add tools/audit_geppetto_articulation_capacity_court_v1.py;
- add unit tests proving target policies are anonymous, parent-before-child, family-count agnostic and source-name free.

No current Geppetto checkpoint or product authority changes merely by defining these challengers.

## 8L. RigAnything-derived joint-locus diffusion challenger — exact insertion point


### 8L.0 Correction after live-checkpoint audit — current product Geppetto already has diffusion

A live-checkpoint audit on 2026-10-03 corrected the assumed baseline.

The exact Knight checkpoint used by backend replay run 37104998946 is:

- class: GeppettoReferenceStrengthNoLearnedSlotV1;
- architecture: RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion.DeterministicViewDirection.v1;
- checkpoint SHA256: 1c124f876858f6f5b5fd0d4639fc7f2b596a25e962d19de8799ecd64ae59b08f.

The live model already contains ConditionalResidualDiffusionV1.

Its actual locus factorization is:

state -> learned coarse XYZ
state + noisy residual -> conditional residual diffusion
final XYZ = coarse XYZ + sampled residual

The recurrence deliberately feeds back coarse XYZ, not the sampled residual-refined XYZ. Final all-pair parent evidence is recomputed from the final refined XYZ.

Therefore the earlier GeppettoCandidateV2 D0/D1 implementation on this research branch is an interface/mechanism prototype only. It is not the current product baseline and is not eligible for promotion as a new diffusion feature.

The corrected live research arms are:

- L0 — LIVE_DIFFUSION: frozen current ReferenceStrength checkpoint, coarse + conditional residual diffusion, coarse-only recurrence.
- L-1 — COARSE_ONLY_ABLATION: exact same checkpoint/control states/STOP/root/support/cardinality, but emit coarse XYZ and recompute final parent evidence from coarse XYZ.
- L1 — DIFFUSION_VARIANT only if L0 contribution is insufficient: change residual diffusion mechanism/schedule while retaining coarse-only recurrence.
- L2 — REFINED_XYZ_FEEDBACK: only after L0/L-1 attribution; feed refined/generated XYZ into later autoregressive state, closer to the stronger RigAnything causal formulation.

The immediate court is L0 vs L-1. No training is needed. This measures whether current diffusion is already materially helping Knight and prevents duplicating an architecture feature that exists.

Research tool:
tools/audit_geppetto_live_diffusion_ablation_v1.py

The L2 feedback arm remains high risk because it introduces sample-path dependence and exposure/cascade sensitivity. It requires a separate court and cannot inherit L0 credit.


RigAnything's useful mechanism for RealSaS is conditional diffusion for the next continuous 3D joint locus, not replacement of the Compiler and not automatic adoption of its whole autoregressive rig authority.

The reference factorizes skeleton generation as:

shape context + previous skeleton -> next-joint context -> diffusion XYZ -> parent evidence -> next step.

Current independent Geppetto V2 already satisfies the same functional obligation with a different mechanism:

- latent autoregressive control state;
- three continuous locus modes;
- heteroscedastic per-mode sigma;
- stable MAP representative;
- parent/root/support evidence;
- hard-MAP XYZ deliberately not fed back into recurrent state.

Therefore diffusion is a challenger to the current locus distribution head, not a missing mandatory component.

### D0 — current Geppetto baseline

Keep the current mapping:

h_t -> three locus modes + sigma + mode logits

with latent-state-only recurrence.

### D1 — conditional diffusion locus head — recommended first challenger

Keep unchanged:

- current SurfaceSetEncoderV2;
- current recurrent control state;
- current STOP/existence/root/support heads;
- current parent evidence;
- current Compiler graph ownership.

Replace only the locus head with:

h_t + pooled/surface context -> conditional diffusion -> XYZ sample distribution.

Requirements:

- predict normalized 3D joint locus;
- expose multiple deterministic-seed samples as typed position hypotheses;
- expose calibrated uncertainty / sample dispersion;
- never mint canonical joint identity;
- do not feed sampled XYZ back into recurrence in D1;
- emit the same SkeletonProposalIR authority class.

Initial fitting strategy:

1. freeze surface encoder plus recurrent state plus non-locus heads;
2. train only the diffusion locus head against the same matched anonymous targets;
3. evaluate locus quality and downstream product consequence;
4. unfreeze shared latent blocks only if the D1 head-only ceiling is insufficient.

This makes the causal question clean:

> Does diffusion improve continuous locus evidence at fixed Geppetto representation?

### D2 — generated-XYZ causal-feedback challenger

Only if D1 indicates that locus quality helps but latent recurrence remains limiting, test the stronger RigAnything-like mechanism:

sampled/generated XYZ -> joint position token -> next autoregressive state.

Potential upside:

- later controls explicitly observe earlier realized geometry;
- local skeleton geometry may become more coherent;
- richer rigs may become easier to sequence.

Risks:

- sample-path dependence;
- error cascade;
- seed sensitivity;
- serialization/order sensitivity;
- conflict with the current deliberate safety property that hard-MAP locus does not rewrite later anonymous control state.

D2 therefore requires a separate adversarial court and may not silently replace D0/D1.

### Diffusion court

Run D0 vs D1 first. D2 only after D1 result.

Measure:

- matched PCK / normalized MAE / p95;
- calibrated coverage / NLL where comparable;
- position-hypothesis diversity without unsupported spread;
- Compiler root/tree qualification;
- downstream Arachne/skin correction;
- G3B/G3;
- authored motion;
- seed-to-seed skeleton stability;
- inference wall time and memory.

Primary decision metric is downstream mechanically qualified product behavior, not lower XYZ loss alone.

### Code changes

Refactor without deleting the current head:

- models/geppetto/v2/geppetto_candidate_v2.py
  - extract a JointLocusDistributionHead interface;
  - retain current multimodal head as MULTIMODAL_GAUSSIAN_V1.
- new models/geppetto/v2/joint_locus_diffusion_v1.py
  - independent conditional denoiser/sampler;
  - clean-room implementation; no RigAnything source import.
- models/geppetto/v2/geppetto_loss_v2.py
  - add diffusion-loss arm while leaving matching/root/parent/support losses unchanged.
- models/geppetto/v2/geppetto_train_v2.py
  - add frozen-base/head-only challenger mode and deterministic sampling seeds.
- tests/models/
  - conditionality, finite samples, deterministic-seed replay, no canonical-ID leakage, no parent-authority duplication.
- new audit:
  - tools/audit_geppetto_joint_locus_challenger_v1.py.

Reference-specific 300-step sampling is not adopted by default. Sampling schedule is an implementation hyperparameter and must earn its cost.


### D1 implementation evidence — 2026-10-03

Implemented on research/rig-skin-architecture-challengers-20261003:

- checkpoint-compatible D0 locus interface; historical parameter keys preserved;
- D0 exact timestep-shape parity gate: PASS, run 37151040534;
- clean-room conditional diffusion locus head with deterministic DDIM-style eta=0 sampling;
- D1 diffusion head contract: PASS, run 37151271350;
- actual-sample medoid representative with deterministic relative hypothesis scoring;
- isolated Geppetto D1 subclass:
  - same recurrence/control states;
  - same STOP/root/support evidence;
  - D1 locus only;
  - parent logits recomputed from D1 XYZ;
  - historical base checkpoint load permits only diffusion-head missing keys;
- isolated D1 semantics gate: PASS, run 37151486678;
- frozen-D0 anonymous matching -> fixed control-state/target XYZ training pairs;
- head-only fit scope; all non-diffusion Geppetto parameters frozen;
- head-only fit contract: PASS, run 37151641152.

No Knight D1 fit or product claim has been made yet. The next scientific step is a D0-vs-D1 locus court on the same frozen Geppetto checkpoint/targets before any wider unfreeze.




## 8M. Canonical Rig Tokens — shared mechanical language between Geppetto and Arachne

SkinTokens/TokenRig demonstrates a useful structural dependency: skeleton state and skin state can be represented in one causal rig sequence.

RealSaS should adopt the dependency structure, not unified model authority.

Compiler remains between the learned systems:

Geppetto proposal -> Compiler -> QualifiedSkeletonIR -> CanonicalRigTokens -> Arachne.

This gives Arachne a compact, already-qualified description of the rig rather than forcing it to rediscover the skeleton graph from loosely packaged joint features.

### Canonical token contract

Tokens are a deterministic serialization of QualifiedSkeletonIR, not new truth.

Minimum token content:

- sequence/start/end grammar;
- normalized joint position;
- parent reference in deterministic serialization;
- root marker;
- bounded local-frame / segment geometry derived from qualified rig;
- optional capability-envelope attributes only when already qualified.

Forbidden:

- source bone names;
- teacher row IDs;
- source semantic labels unavailable at product inference;
- token order as canonical joint identity;
- learned rewrite of Compiler parent/root decisions.

The token-stream hash binds:

- skeleton lineage;
- normalization;
- serializer version;
- capability-envelope lineage where used.

### T0 — current Arachne conditioning baseline

Current explicit joint and point-joint / parent-segment geometry only.

### T1 — additive canonical rig-token conditioning

Add:

CanonicalRigTokenAdapter(QualifiedSkeletonIR) -> token embeddings

alongside current explicit pair geometry.

Question:

> Do canonical rig tokens reduce the relational structure Arachne must relearn while preserving or improving product-space skin mechanics?

No existing conditioning is removed in T1.

### T2 — token-conditioned compact skin state

Only if T1 passes:

rig token -> compact per-joint skin-field state.

Dense influence remains geometry-conditioned and is queried on admitted surface nodes.

### T3 — architecture reduction

Only after T1/T2 quality parity or improvement:

- ablate redundant Arachne graph/relational blocks;
- reduce width/layers;
- measure quality-per-FLOP and memory;
- never assume tokenization automatically reduces compute.

### Required courts

- deterministic serialization replay;
- permutation / canonical-ID invariance;
- source-name leakage;
- 28-control and richer-rig scaling;
- current conditioning vs additive RigTokens vs reduced token architecture;
- exact Compiler requalification after every arm;
- G3B/G3/motion;
- compute/memory/latency.

### Code changes

- new Compiler-owned serializer:
  - compiler/realsas_compiler_core/canonical_rig_tokens_v1.py;
- new Arachne adapter:
  - models/arachne/v4/rig_token_conditioning_v1.py;
- extend conditioning batch with optional token stream/hash;
- do not change QualifiedSkeletonIR schema for the experiment;
- add token-specific tests before any training.

## 8N. SkinTokens-derived Arachne evolution — keep Arachne, adopt the stronger field formulation

SkinTokens remains a reference mechanism, not a model replacement.

Current Arachne strengths to preserve:

- exact QualifiedSkeletonIR conditioning;
- explicit point-joint / parent-segment geometry;
- RealSaS lineage/provenance;
- Compiler-owned legal/simplex qualification;
- product-space mechanical residual adapter from Section 8I.

Mechanisms worth testing:

1. per-bone continuous influence field rather than treating final N x J rows as the primary learned object;
2. geometry-conditioned field decode at requested admitted surface points;
3. compact per-joint field state;
4. optional quantization/token state only after the continuous codec ceiling passes;
5. active-region / sparse-positive supervision;
6. canonical rig-token conditioning from Section 8M.

### A0 — current baseline

Frozen current Arachne + 1.58M mechanical residual adapter + multi-carrier product-space consequence.

### A1 — continuous field-codec challenger

Introduce an independently trained geometry-conditioned per-joint field codec.

The first court is a representation ceiling, not predictor training.

Required:

- decode on all admitted surface nodes;
- reconstruct semantic support;
- preserve local influence transitions;
- product-space multi-carrier G3-like consequence;
- exact Compiler requalification.

Teacher row-L1 reconstruction alone cannot PASS the codec.

### A2 — compact field-state predictor

Only after A1 ceiling PASS:

Arachne conditioning -> per-joint compact field state -> frozen/qualified field decoder.

Compare against A0 with identical skeleton/carrier courts.

### A3 — optional quantized field state

Only if A2 works and quantization provides measurable operational value:

- deterministic bounded codebook/FSQ-like state;
- exact decode reproducibility;
- no material mechanical regression;
- lower storage/sequence complexity sufficient to justify added representation machinery.

Quantization is not a goal by itself.

### Active-region supervision

Teacher skin may be used offline to construct sparse-positive / active-region sampling for training, but:

- teacher topology is not product authority;
- teacher-invalid regions are not exact labels;
- final loss includes product-space multi-carrier deformation consequence;
- no source mesh is required at product inference.

### Code changes

Potential independent modules:

- models/arachne/v4/skin_field_codec_v1.py;
- models/arachne/v4/skin_field_codec_loss_v1.py;
- models/arachne/v4/skin_field_state_predictor_v1.py;
- models/arachne/v4/active_region_sampler_v1.py;
- optional later models/arachne/v4/skin_field_quantizer_v1.py.

Existing Arachne remains the baseline and is not deleted.

## 8O. Shared Surface Encoder — reduce duplicate geometry work only after functional equivalence

Geppetto and Arachne both consume the same admitted surface family. There is potential duplicated compute in independently re-encoding surface geometry.

Do not force a shared encoder before proving that both tasks can use the same representation without loss.

### E0 — current baseline

Independent Geppetto and Arachne geometry encoders.

### E1 — shared frozen surface-token contract

Create a typed internal learned tensor:

SharedSurfaceTokenSet

bound to:

- exact RiggingSurfaceIR lineage;
- normalization;
- encoder architecture/checkpoint;
- token mask/order;
- local P+N / qualified feature contract.

One encoder produces surface tokens once. Geppetto and Arachne consume task-specific projections from the same frozen token set.

E1 must preserve task-specific heads and task-specific downstream losses.

### E2 — jointly trained shared encoder

Only after E1 demonstrates no material information loss:

- unfreeze the shared encoder;
- optimize a multi-task objective;
- maintain separate task heads;
- require no-regression on rig and skin courts.

Avoid allowing a single scalar weighted objective to become authority. Both domains keep hard PASS gates.

### Shared-encoder courts

Compare E0/E1/E2:

- Geppetto skeleton proposal metrics;
- Compiler graph qualification;
- Arachne semantic/support metrics;
- product-space mechanical metrics;
- memory;
- FLOPs;
- latency;
- representation size;
- gradient conflict diagnostics for E2.

Promotion requires:

- no hard-gate regression;
- meaningful compute/memory gain or quality gain;
- deterministic lineage binding.

### Code map

Potential:

- new models/shared/surface_encoder_v1.py;
- new models/shared/surface_token_contract_v1.py;
- adapters in Geppetto/Arachne rather than schema changes to RiggingSurfaceIR;
- no Compiler dependence on learned hidden tokens.



## 8P. Experiment matrix — prevent architecture confounding

Do not run all new ideas in one model. Promotion is staged and orthogonal.

### Rig axis

| ID | Target policy | Locus head | Recurrence feedback | Purpose |
|---|---|---|---|---|
| R0 | K0 current core | D0 current 3-mode | latent-only | frozen baseline |
| R1 | K1/K2/K3 sweep | D0 | latent-only | isolate articulation-capacity / target-policy effect |
| R2 | winning target policy | D1 diffusion | latent-only | isolate diffusion-locus effect |
| R3 | winning target policy | D2 diffusion | generated-XYZ feedback | only if R2 justifies |

Do not compare R3 directly against R0 and attribute the entire delta to diffusion; the intermediate courts are mandatory.

### Skin axis

| ID | Rig conditioning | Skin representation | Mechanical adaptation | Purpose |
|---|---|---|---|---|
| S0 | current explicit | current Arachne | residual adapter | baseline |
| S1 | + CanonicalRigTokens | current Arachne | residual adapter | isolate token-conditioning effect |
| S2 | winning conditioning | continuous field codec | product-space loss | representation ceiling |
| S3 | winning conditioning | predicted compact field state | product-space loss | learned field challenger |
| S4 | optional quantized state | field decode | product-space loss | operational compression only |

### Shared encoder axis

Evaluate only after independent rig and skin winners are known:

- E0 independent encoders;
- E1 shared frozen surface tokens;
- E2 shared jointly trained encoder.

### Final combination rule

The integrated candidate contains only independently supported mechanisms.

A mechanism that wins only when paired with another unproven mechanism is marked interaction-dependent, not independently promoted.

## 8Q. Court definitions and winner criteria

### Q1 — Rig articulation-capacity court

Primary:

- Compiler-qualified skeleton legality;
- downstream G3B/G3/motion;
- marginal articulation gain.

Secondary:

- joint count;
- inference cost;
- retarget complexity.

No winner may be selected by raw joint count or visual rig cleanliness.

### Q2 — Joint-locus distribution court

Primary:

- downstream mechanically qualified product behavior.

Secondary:

- matched locus PCK/MAE/p95;
- calibrated uncertainty;
- sample stability;
- inference cost.

Diffusion must beat D0 by enough to justify sampling cost.

### Q3 — Canonical RigToken court

Primary:

- downstream skin mechanics under the exact same rig/carrier;
- semantic support fidelity.

Secondary:

- Arachne parameter/FLOP reduction opportunity;
- scaling with richer rigs.

### Q4 — Skin-field representation ceiling

The decoder receives authoritative training field state / truth state where legally available.

This isolates representation from predictor difficulty.

Required PASS:

- semantic support;
- multi-carrier deformation consequence;
- exact Compiler requalification;
- bounded reconstruction error;
- no hidden topology-specific smoothing.

### Q5 — Skin predictor court

Only after Q4.

Compare current Arachne state prediction against compact field-state prediction.

### Q6 — Shared encoder court

Only after rig and skin independent winners.

Requires no hard-gate regression for either subsystem.

### Q7 — Integrated Knight court

Exact same immutable lineage must pass:

- rig;
- skin;
- static mesh;
- topology;
- G3B;
- G3;
- 51/51 motion;
- source geometry;
- adversarial/refinement invariance.

Then and only then architecture becomes eligible for Mage cold replay / later FITK.

## 8R. Training and refit schedule

### Stage 0 — no A100: contracts and ceilings

CPU/1660Ti where practical:

- richer Geppetto target-policy builders and receipts;
- CanonicalRigToken serializer;
- diffusion-head unit/shape/seed tests;
- field-codec synthetic/teacher ceiling setup;
- active-region sampler;
- shared-token contract;
- exact court plumbing.

No model refit is authorized until each challenger can be evaluated without training ambiguity.

### Stage 1 — Geppetto target-policy sweep

Prefer reuse of existing architecture/checkpoint.

For K1/K2/K3:

1. build new anonymous teacher targets;
2. first test current checkpoint inference against richer truth for attribution;
3. fine-tune only necessary Geppetto blocks;
4. keep D0 locus head fixed during target-policy causality court.

This determines whether the current cardboard-risk is caused primarily by impoverished truth rather than architecture.

### Stage 2 — Geppetto D1 diffusion head

Start head-only:

- freeze shared encoder;
- freeze recurrent state;
- freeze parent/root/support/STOP heads;
- train diffusion locus head;
- deterministic validation seeds;
- compare against D0.

Only unfreeze common latent blocks if head-only ceiling is insufficient.

### Stage 3 — Arachne A0 mechanical adapter

If still useful after the architecture courts, fit the existing 1.58M adapter on the preregistered multi-carrier bundle.

This remains the cheapest product-space correction baseline.

### Stage 4 — A1 continuous field-codec ceiling

Train representation only; no compact predictor yet.

A100 is justified if CPU preflight seals at least two distinct admitted carriers and exact training manifests.

### Stage 5 — A2 compact field-state predictor + RigTokens

Train predictor against the qualified field representation.

Start with existing Arachne backbone frozen where possible; add token adapter / new state head first.

### Stage 6 — shared encoder

E1 frozen shared tokens first.

E2 joint training only after independent Geppetto and Arachne quality gates are stable.

### Retrain principle

Never do a full end-to-end refit merely because a local head changed.

Unfreeze from the outside in:

1. new head/adapter only;
2. nearest latent block;
3. shared task encoder;
4. full subsystem only when smaller scopes fail a preregistered ceiling.

Every escalation needs a causal receipt.

## 8S. Branch isolation and promotion governance

Architecture research is separated from the active mechanics seal.

### Active mechanics branch

audit/canonical-caa-mechanics-geometry-lock-20261002

Purpose:

- finish current Knight mechanical/static fixed-point;
- close existing contracts;
- no new rig/skin architecture experiments;
- keep current audit lineage interpretable.

### Architecture challenger branch

research/rig-skin-architecture-challengers-20261003

Fork point:

f271f87d8085e2df08ae8774ab67a19790eefe42

Purpose:

- richer Geppetto target policies;
- diffusion locus challenger;
- CanonicalRigTokens;
- SkinTokens-derived Arachne field architecture;
- shared surface encoder;
- training/court infrastructure.

### One-way synchronization rule

Validated generic compiler fixes from the active audit branch may be selectively brought into the challenger branch.

Experimental architecture commits never flow back into the audit branch.

### Main promotion rule

Never merge the challenger branch wholesale.

Promotion unit is one independently evidenced mechanism:

implementation commit(s) + tests + court artifact/receipt + no-regression evidence.

Examples:

- K2 target policy may promote while D1 diffusion remains rejected;
- CanonicalRigTokens may promote while field codec remains experimental;
- shared encoder may remain rejected even if rig/skin challengers pass.

Use isolated cherry-picks or narrowly scoped PRs after evidence review.

## 8T. Naming / branding — explicitly deferred from architecture identity

The names Geppetto/Arachne and the product name RealSaS are not architectural contracts.

A later naming pass may replace model/product display names with concise acronym-style names without rewriting schemas or scientific lineage.

Rules:

- architecture IDs and historical receipts remain immutable;
- aliases/display names may change;
- no code rename during active causal courts;
- product-name decision is separate from model architecture.

Desired future model naming properties:

- short pronounceable acronym;
- distinct visual identity next to IRIS;
- name expands to a real functional description rather than a forced backronym;
- no collision with major existing graphics/ML systems.

Naming is deliberately postponed until the current architecture candidates stabilize.


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
| V9 exact teacher reprojection oracle | FAIL / TOPOLOGY-FIELD OWNER EXPOSED | run 37122879585: teacher G3 ~177.68, 51/51 motion fail, max edge ~188.70x; A100 paused |\n| Teacher-oracle production repartition one-pass | IMPROVES / NOT CLOSED | run 37123289786: G3B unsafe 418→166, motion max edge 188.70x→56.15x, 51/51 still fail; child static violations 3217 |\n| Iterative teacher-oracle production repartition | CONVERGENT / BUDGET-EXHAUSTED | run 37137357080 after multi-support fix: G3B unsafe 418→166→66→24→14; static violations 0→3388; G3/motion still fail |\n| Static↔mechanical composition court | FAIL / OWNER LOCALIZED | run 37124108389: 3217→66 static violations, but G3 max condition ~10362.18, 51/51 motion fail, max edge ~3200.14x; independent static cleanup destroys mechanical class |\n| Deformation-field / topology-basis equivalence court | PASS AS ATTRIBUTION / HYPOTHESIS CONFIRMED | run 37139120074: teacher/source basis PASS (cond ~1.33, edge ~1.23), V9 basis FAIL (cond ~829.91, edge ~62.65) despite weight rebind L1 max ~7.45e-8 |\n| Dynamic-child static composition | FAIL / COMPOSITION OWNER EXPOSED | run 37124108389: static violations 3217→66, but G3B unsafe 166→277, G3 condition ~10362, 51/51 motion fail, max edge ~3200x |
| Proposal-level mechanical static guard | CONTRACT PASS | flip/collapse/vertex-cavity/edge-cavity/split; exact-LBS split temporary vertex guarded; hosted gates PASS |
| Arachne mechanical residual adapter | CONTRACT PASS / CPU PREFLIGHT PENDING | 1,582,849 trainable params, zero-init identity, product-space mechanical loss; full Arachne frozen |
| Multi-topology invariance court | TODO | required before genericity claim |
| Product contract integration | BLOCKED | requires preceding PASS |
