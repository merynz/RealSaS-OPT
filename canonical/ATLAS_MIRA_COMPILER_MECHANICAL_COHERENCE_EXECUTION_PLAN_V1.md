# RealSaS — ATLAS / MIRA / Compiler Mechanical Coherence Execution Plan V1

**Date:** 2026-10-04  
**Status:** `SUPERSEDED_BY_TRUTH_CARRIER_PLAN_V2__DO_NOT_EXECUTE_MIRA_M_OR_E3_FROM_THIS_DOCUMENT`

> **2026-10-04 correction:** Source-level re-audit of RigAnything, SkinTokens, current ATLAS/MIRA, and historical MIRA FIT1 shows that an explicit topology-conditioned MIRA-M head is not yet justified. Existing MIRA already supports point-query decoding, while the demonstrated gap is truth-carrier coherence. See `canonical/MECHANICAL_TRUTH_CARRIER_ARCHITECTURE_PLAN_V2_20261004.md`. E3 from this document is blocked.

## Goal

Eliminate the composition gap between:
- product topology;
- ATLAS control basis / skeleton;
- MIRA skin field;
- Compiler dynamic proof.

The exact Stage18/19 mechanical mesh already exists before ATLAS/MIRA execute. Future learned mechanics must exploit and be evaluated on that exact topology instead of discovering incompatibility only at Stage35.

---

# 1. Preserve what already works

Do **not** discard:

- `RiggingSurfaceIR` as observation-grounded geometry evidence;
- ATLAS semantic/articulation reasoning;
- MIRA semantic skin field;
- existing surface/skeleton/skin lineage hashes;
- `QualifiedMeshSkinIR` exact mesh binding;
- G3/G3B as fail-closed Compiler authority;
- deterministic Compiler graph/root/tree qualification;
- static mesh quality gates.

The architecture change is additive and binding-focused, not a rewrite.

---

# 2. MIRA: semantic field + topology-conditioned product head

## MIRA-S — semantic head

Keep the current semantic field objective on `RiggingSurfaceIR + QualifiedSkeletonIR`:

```text
MIRA-S(S, G) -> semantic surface influence field W_s
```

Teacher weight supervision remains useful here.

It is a prior / semantic representation, not final product weight authority.

## MIRA-M — mesh-native mechanical refinement head

Add a topology-conditioned product-space head:

```text
MIRA-M(
  Stage18/19 exact mesh M,
  surface memory from S,
  QualifiedSkeletonIR G,
  semantic field W_s
) -> MeshSkinProposalIR W_m*
```

Properties:
- rows indexed by exact candidate/canonical mesh vertex IDs;
- explicitly bound to exact mesh lineage + skeleton lineage + semantic-skin lineage;
- graph attention/message passing uses exact mesh edges/faces;
- output is simplex-valued mesh-vertex influences;
- may start as bounded residual/correction around deterministic surface-support transfer;
- no unrestricted semantic rewrite in V1.

Compiler qualifies this proposal into existing `QualifiedMeshSkinIR`.

### Why a correction head instead of deleting semantic MIRA

It preserves:
- reusable semantic skin evidence;
- coherent teacher supervision;
- current lineage and fallback path;
- a deterministic baseline.

It adds only the missing topology-conditioned degree of freedom demonstrated by the courts.

---

# 3. Joint mechanical loss: train on consequence, not only weights

For frozen exact product topology `M`, qualified/proposed skeleton `G`, and mesh weights `W_m`, define differentiable probe deformation.

First version:

```text
L_joint =
  w_cond * L_condition
+ w_area * L_area
+ w_edge * L_edge
+ w_flip * L_fold
```

using the same frozen probe family semantics as G3/G3B where practical.

Hard G3 remains authority. The differentiable loss is a training surrogate only.

### Deformation-truth supervision

When a coherent source asset provides source mesh/rig/skin/motion:

```text
D_product(M, G, W_m, q)
  ~= correspondence(
       D_teacher(T_teacher, R_teacher, W_teacher, q)
     )
```

for frozen probe/motion family `q`.

This is stronger than copying projected teacher weights.

Teacher weights remain one loss term; teacher deformation consequence becomes the cross-topology truth.

---

# 4. ATLAS: mechanically optimal control basis

ATLAS must not optimize teacher cardinality.

Keep semantic proposal losses, but add downstream utility:

```text
ATLAS(S [, M mechanical descriptors])
 -> candidate controls / loci / topology evidence

Compiler -> legal tree / root / IDs

utility(control set)
 := best achievable joint mechanical quality with MIRA-M on exact M
```

V1:
- continuous joint loci can receive gradient through FK/LBS joint loss;
- parent/tree legality remains Compiler/structured-solver owned;
- cardinality is grow-to-need / prune-to-proof;
- teacher skeleton is evidence, not cardinality authority.

Do not make raw arbitrary triangulation a mandatory ATLAS semantic input immediately. First prove value through joint mechanical loss; then open a topology-conditioned ATLAS sidecar only if the same geometry with different connectivity requires materially different control proposals.

---

# 5. Compiler: topology becomes mechanically guarded, not only statically qualified

Stage18/19 may produce the initial candidate, but a topology operation is not mechanically final merely because it improves:
- manifoldness;
- slivers;
- minimum angle;
- static quality.

Add a typed `MechanicalTopologyEquivalenceReportV1` / compatibility report for every topology mutation used after a rig/skin state exists.

For a proposed local topology change:

```text
(T_old, G, W_old)
 -> propose T_new
 -> re-evaluate / infer W_new with MIRA-M
 -> joint mechanical court
 -> accept only if lexicographically non-regressive
```

Static gates remain mandatory.

## Repair-loop semantics

Because Stage18 precedes ATLAS/MIRA, use the existing DAG invalidation model:

```text
Stage18/19 candidate
 -> ATLAS
 -> MIRA-S/MIRA-M
 -> joint mechanical qualification
 -> topology repair if needed
 -> new mesh lineage
 -> invalidate mesh-bound learned outputs
 -> rerun only affected downstream inference
 -> fixed point or bounded fail
```

No model retraining occurs during product compilation; only inference/qualification reruns.

---

# 6. Binding changes

Do not invent a second geometry authority.

Introduce only exact proposal bindings:

### `MeshSkinProposalIR.v1`
Must bind:
- `surface_binding_hash`;
- `skeleton_binding_hash`;
- `semantic_skin_binding_hash`;
- `mesh_binding_hash`.

### ATLAS mechanical evaluation binding
ATLAS proposal may remain surface-bound in V1, but every mechanical training/evaluation record must additionally bind the exact mesh lineage used for its downstream joint-loss score.

If a future topology-conditioned ATLAS head is promoted, create an explicit V2 proposal with `mesh_binding_hash`.

---

# 7. Coherent corpus contract

Each supervised mechanical item should expose one coherent source bundle:

```text
MechanicalTruthBundle = {
  source_topology,
  source_control_hierarchy,
  source_skin,
  source_animation_or_probe_family,
  source_to_product_correspondence
}
```

No individual teacher cardinality or projected weight field is final authority outside its topology.

KayKit / Quaternius-style rigged assets are useful specifically because mesh + rig + skin + animation are mutually coherent, not merely because they provide more labels.

---

# 8. Experiment order — no A100 before E3

## E1 — diagonal-flip causal microcourt — CPU
Same geometry, same weights, same rig, one connectivity flip.
Required:
- demonstrate different deformation Jacobians/mechanical loss;
- finite gradients w.r.t. weights;
- exact identity arm zero/non-regressive.

## E2 — differentiable G3 surrogate unit court — CPU
Implement:
- condition;
- area;
- edge;
- fold penalties.
Compare ordering against hard G3 on synthetic cases.

## E3 — frozen Knight mesh-native optimization oracle — CPU/GPU, no learned fit
Freeze:
- Stage18/19 mesh;
- current qualified skeleton;
- semantic teacher/current skin.

Optimize only mesh-vertex logits against:
- semantic weight-distance budget;
- joint mechanical loss.

Question:
**Does product-space weight freedom materially reduce G3B on the exact topology without topology mutation?**

If NO:
- topology must carry more responsibility; do not train MIRA-M yet.

If YES:
- MIRA-M is scientifically justified.

## E4 — topology move + reoptimized weights court — CPU
For bounded flip/split/cut candidates:
- baseline weights;
- reoptimized mesh weights;
- hard G3/G3B.
Estimate how much failure is weight-correctable vs topology-irreducible.

## E5 — short MIRA-M fit — A100 only if E3/E4 justify
Freeze ATLAS.
Train topology-conditioned correction head on coherent bundles.
Compare:
- semantic-only transfer;
- MIRA-M without joint loss;
- MIRA-M + joint loss.

## E6 — ATLAS mechanical utility / control-basis court
Only after MIRA-M ceiling is known.
Then evaluate control grow/prune with joint mechanical utility.

## E7 — shared encoder
Continue shared-encoder E1/E2 only after joint objective contract is frozen.
Parameter sharing is an efficiency optimization, not compatibility authority.

---

# 9. Promotion criteria

No architecture promotion from local loss alone.

MIRA-M must improve:
- hard G3/G3B;
- actual motion;
- no new catastrophic faces;
- teacher deformation-field error;
- generalization on coherent held-out assets.

ATLAS changes must improve:
- mechanical adequacy under best admitted MIRA-M;
- control-basis efficiency after adequacy;
- not merely teacher PCK/parent accuracy.

Compiler topology changes must:
- pass static gates;
- pass mechanical-equivalence/joint compatibility;
- preserve exact lineage/invalidation semantics.

---

# 10. Immediate work

1. implement E1 diagonal-flip court;
2. implement differentiable G3 surrogate;
3. run E2 synthetic regression;
4. build E3 frozen Knight mesh-weight optimization oracle;
5. **then** decide whether MIRA-M deserves training;
6. keep ATLAS RigAnything-parity and mechanically-optimal-basis research active, but do not spend A100 until E3 resolves the product-space ceiling.

