# RealSaS — Topology / Mechanical Coherence Forensic Lineage — 2026-10-04

**Status:** `FORENSIC_CLOSED__HYPOTHESIS_CONFIRMED__ARCHITECTURE_REVISION_REQUIRED`

## Question

Was the bad Knight deformation primarily compatible with the hypothesis that individually-correct rig/skin evidence was being composed on a mechanically different product topology, or is that a retrospective story unsupported by the recorded experiments?

## Verdict

**Confirmed, with important scope limits.**

The repo history contains repeated causal evidence that:

1. static geometric quality does not imply dynamic mechanical compatibility;
2. teacher/exact skin evidence does not imply teacher deformation after a topology/basis change;
3. topology/connectivity is a causal variable independent of MIRA/Arachne prediction error;
4. a coherent topology + skin intervention can remove catastrophic deformation where teacher weights alone do not;
5. naive post-hoc skin localization does not solve the problem.

This does **not** prove that every remaining failure is topology-owned or that exact source triangulation must always be preserved. It proves that any topology change requires a mechanical-equivalence or downstream joint-compatibility proof.

---

## A. 2026-09-28: teacher-weight causality isolates topology

### Teacher projection / target integrity

- run `36469103384` — `knight-teacher-projection-integrity-v1` — PASS
  - sealed evidence commit emitted by job: `e43af209`;
  - strict-local artist-source and bank discontinuity counts were both zero.
- run `36469579542` — `knight-teacher-target-alignment-proof-v1` — PASS
  - graph exact `28/28`;
  - evidence commit `ada4c3f5`.

These gates reduce the plausibility that the later result is merely a teacher-bank corruption artifact.

### Teacher weights fix source-local deformation but not nonincident candidate topology

- run `36469747325` — `knight-teacher-weight-oracle-causality-v1` — PASS
  - evidence commit `17cdc05e`.
  - On `STRICT_SOURCE_LOCAL` faces, teacher oracle:
    - `edge_gt_10 = 0`;
    - `edge_gt_4 = 0`;
    - run max edge ~`3.21`.
  - On `NONINCIDENT` candidate faces, teacher oracle still produced:
    - `edge_gt_10 = 592`;
    - `edge_gt_4 = 926`;
    - max edge ~`118.62`.

**Interpretation:** correcting the skin values removes a large error class where topology remains source-local, while catastrophic deformation survives on candidate faces whose connectivity/basis is not source-incident.

### Topology + teacher weights closes the catastrophic edge class

- run `36470745655` — `knight-combined-topology-teacher-weight-oracle-v1` — PASS
  - evidence commit `4d6aebfc`;
  - retained `edge_gt_10 = 0`;
  - worst retained edge ~`3.576`.

This establishes a topology/skin interaction, not a skin-only explanation.

### Explicit topology-only court

- run `36479489630` — `knight-teacher-weight-topology-only-court-v1` — PASS
  - evidence commit `49442881`;
  - exact finding recorded by the court:
    - `correct_weights_still_leave_topology_unsafe_faces = true`;
    - `unsafe_is_dominated_by_source_nonincident = true`;
    - `face_deletion_creates_new_open_boundaries = true`.
  - `unsafe_total = 1553`;
  - `unsafe_source_nonincident = 1510`;
  - only `43` unsafe faces were source-local/share-vertex.

- run `36480554559` — `knight-teacher-weight-seam-morphology-court-v1` — PASS
  - evidence commit `c2c9858c`;
  - largest unsafe component: `1194` faces;
  - ~`99.66%` source-nonincident.

### First teacher-free topology closure was insufficient

- run `36492191858` — execution PASS, scientific result `FAIL_TOPOLOGY_REMAINS_OPEN`
  - unsafe stress `1553 -> 1269`;
  - motion still `gt10=429`, `gt4=667`, worst edge ~`238.39`.

This is important negative evidence: “some topology repair” is not enough; the topology operation itself must be mechanically correct.

---

## B. 2026-09-29: product topology closure succeeds when mechanically coupled

Production repair lineage included:

- `158fcbd9...` — close Stage35 seeds into valid Stage17 partition cuts — contract PASS;
- `3c407c8d...` / `4009497c...` — bind seam stress to mechanical skin ownership;
- `6e45bfa4...` → `8c2ec84c...` → `f27716c9...` — component-harmonic seam support implementation/gates;
- `37f6c4c2...` / `0c3de8c0...` — source-edge probe repartition seed strategy.

### Production replay

- run `36601730812` — `knight-production-topology-replay-court-v1` — PASS
  - evidence commit `a1cd5ffb`;
  - `component_count = 73`;
  - `direct_seed_count = 1417`;
  - `final_separate_count = 1818`;
  - `motion_gt_10 = 0`;
  - `motion_gt_4 = 0`;
  - p99 ~`1.7301`, worst ~`3.6155`.

- run `36602572907` — `knight-production-g3-motion-court-v1` — PASS
  - evidence commit `d6c79cf9`;
  - `G3 PASS`;
  - motion `gt10=0`, `gt4=0`.

**Interpretation:** the repo already demonstrated that mechanically-informed topology/partition plus coherent seam support can close the catastrophic deformation class.

---

## C. 2026-10-03: V9 repeats and strengthens the causal result

### “Better transfer” alone fails

- run `37109019558` — topology-aware skin prolongation — diagnostic execution PASS.
  - parent-constant through graph-IDW variants all fail `51/51` motion frames.
  - best reported graph-IDW-P4 still:
    - G3 max condition ~`609`;
    - max motion edge ~`288`.

- run `37109348560` — dynamic source-topology attribution — PASS.
  - failures dominated by `HOP1_LOCAL_RELATION` edges.

FEM and deformation-sensitive projection experiments were not scientifically closed:
- `37109629340`: `FEM_SOURCE_COMPONENT_HAS_NO_COARSE_FACE`;
- `37113657436`: `DEFORM_REPAIR_CG_FAIL:1500`.

They are **inconclusive**, not evidence against the topology hypothesis.

### Exact teacher projection still fails V9 topology

- runs `37122879585` and `37124893895` — exact teacher projection oracle — execution PASS.
  - teacher projection itself: PASS with coverage warning;
  - coverage ~`0.89348`, `13840` clean rows / `15490`;
  - simplex residual ~`2.22e-16`;
  - exact teacher oracle on V9:
    - `teacher_g3_passed = false`;
    - `teacher_failed_motion_frames = 51`;
    - max condition ~`177.68`;
    - max motion edge ~`188.70`.

### Static-perfect does not mean mechanically-valid

- run `37137357080` — iterative repartition — execution PASS.
  - evaluation 0:
    - `static_violations = 0`;
    - `G3B unsafe = 418`.
  - iterative repartition reduces G3B:
    - `418 -> 166 -> 66 -> 24 -> 14`;
  - while static violations rise to `3388`.

This directly falsifies the implication:

```text
static mesh quality PASS  =>  mechanical deformation PASS
```

### Topology-basis equivalence court: direct causal proof

- run `37139120074` — `knight-teacher-topology-basis-equivalence-v1` — PASS
  - head around `ace2ca75...`;
  - court status: `HYPOTHESIS_CONFIRMED`;
  - same source teacher skin, same target skeleton, same ±10° probes;
  - teacher-bank barycentric rebind L1 max ~`7.45e-08`;
  - weight-prediction error explicitly absent.

Source topology:
- vertices `3665`, faces `6952`;
- min area ratio ~`0.725`;
- max area ratio ~`1.279`;
- max condition ~`1.327`;
- max edge ratio ~`1.227`;
- **PASS**.

V9 topology:
- vertices `14399`, faces `28810`;
- min area ratio ~`0.00731`;
- max area ratio ~`56.64`;
- max condition ~`829.91`;
- max edge ratio ~`62.65`;
- **FAIL**.

The court’s own claim:

`EXACT_TEACHER_VERTEX_WEIGHTS_DO_NOT_IMPLY_EXACT_TEACHER_DEFORMATION_FIELD_ACROSS_TOPOLOGY_BASIS_CHANGE`.

---

## D. 2026-10-04: residual seam and locality courts

Completed V9 static-mechanics notebook:
- static violations `3281 -> 1225`;
- production G3B `55 -> 48`;
- final candidate `a49170c3...`;
- final partition `b2503c49...`.

Residual classifier:
- 48 production unsafe / 58 all-face unsafe;
- 46/48 production unsafe are seam-bearing;
- 44/48 are dynamic-only;
- all 48 are component-pure.

Naive locality counterfactual:
- baseline: `48 / 58`;
- top-1: `141 / 143`;
- top-2: `72 / 83`;
- top-4: `50 / 64`.

Therefore simple “make seam weights more local/sparse” is falsified.

---

## E. Current typed architecture explains the gap

Current V2 DAG already builds product topology **before** learned rig/skin stages:

- Stage15: `RiggingSurfaceIR`;
- Stage17: mechanical partition;
- Stage18: canonical mechanical mesh addressing;
- Stage19: static canonical mesh qualification;
- Stage26–29: ATLAS/legacy Geppetto fit + skeleton qualification;
- Stage30–32: MIRA/legacy Arachne fit + skin qualification;
- Stage35: first joint dynamic mechanical qualification on exact Stage18 mesh;
- Stage36: deterministic surface-skin -> mesh-skin transfer.

Yet the learned proposal contracts are only:

- `SkeletonProposalIR.surface_binding_hash`;
- `SkinProposalIR.surface_binding_hash + skeleton_binding_hash`.

They are **not bound to the already-existing Stage18 mesh topology**.

By contrast `QualifiedMeshSkinIR` already binds:
- surface;
- skeleton;
- qualified semantic skin;
- exact mesh hash.

Thus the missing concept is not a generic hash system. The missing concept is **learned/joint mechanical optimization on the exact product topology before Stage35 discovers the mismatch**.

---

## F. Historical contract that must be revised

`canonical/MESH_WEIGHT_BINDING_CONTRACT_V1.md` froze the normal path:

```text
S + G -> MIRA semantic skin W
M + W -> deterministic mesh-weight transfer B
```

and explicitly required MWB-3 to prove that field-to-mesh transfer preserves deformation quality.

The later courts above demonstrate that deterministic transfer of semantically-correct weights is not sufficient across arbitrary topology-basis changes.

Therefore V1 remains valid as lineage/semantic decomposition but is **scientifically incomplete as the final normal mechanical path**.

---

## Final forensic conclusion

The user’s AC-vs-BD diagonal example is the correct minimal intuition.

Equal vertex positions, a clean manifold, good angles and even exact teacher vertex weights do not define the same continuous deformation field when face connectivity/interpolation basis differs.

The product truth must therefore be treated as a coherent tuple:

```text
MechanicalTruth = (Topology T, Control/Rig R, Influence W, Probe/Motion family Q)
```

and topology mutation is admissible only if:
1. exact topology is preserved; or
2. downstream mechanical equivalence/compatibility is proven.

