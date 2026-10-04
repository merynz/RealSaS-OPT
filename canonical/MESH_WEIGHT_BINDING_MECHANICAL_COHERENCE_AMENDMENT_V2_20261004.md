# RealSaS — Mesh / Weight Binding Mechanical-Coherence Amendment V2 — 2026-10-04

**Status:** `RESEARCH_AMENDMENT_PREREGISTERED__V1_LINEAGE_PRESERVED__PRODUCT_MIGRATION_BLOCKED_ON_E3`

This document amends, but does not erase, `MESH_WEIGHT_BINDING_CONTRACT_V1.md`.

## Why V1 is insufficient

V1 correctly separated:
- observation-grounded surface semantics;
- editable product mesh topology;
- semantic skin;
- deterministic mesh-weight binding.

Its open MWB-3 question was whether semantic surface skin could be transferred to the product mesh without losing deformation quality.

The September 28–October 4 courts answer:

**not in general.**

Exact/teacher skin can be mechanically correct on the source topology and mechanically invalid after a topology-basis change. Therefore deterministic semantic transfer remains a valid baseline and provenance path, but it cannot be the only normal product mechanism.

## Existing DAG opportunity

The current V2 DAG already creates:

- Stage18 canonical mechanical mesh;
- Stage19 static canonical mesh qualification;

**before** ATLAS (legacy Geppetto) and MIRA (legacy Arachne) execute.

The architecture therefore does not need a late topology retrofit or a second geometry authority.

The exact product topology is available in time for learned mechanical compatibility.

---

# 1. MIRA becomes dual-domain

## MIRA-S — semantic surface field

Keep:

```text
RiggingSurfaceIR S + QualifiedSkeletonIR G
 -> MIRA-S
 -> SkinProposalIR W_s*
 -> Compiler
 -> QualifiedSkinIR W_s
```

Responsibilities:
- semantic influence prior;
- coherent teacher-weight supervision;
- reusable surface-domain representation;
- topology-independent fallback/baseline.

`QualifiedSkinIR` remains surface-bound and is not itself final product deformation truth.

## MIRA-M — topology-conditioned mesh head

Add:

```text
Stage18/19 exact mechanical mesh M
+ S
+ G
+ MIRA-S field/features
 -> MIRA-M
 -> MeshSkinProposalIR W_m*
```

V1 head should be a **bounded correction** around the deterministic V1 transfer, not an unrestricted second skin generator.

It may use:
- exact mesh vertices;
- exact mesh edges/faces;
- surface-support coefficients;
- shared surface memory;
- qualified skeleton/control features;
- semantic MIRA-S weights/features.

It must not use:
- hidden teacher payload at inference;
- future motion truth;
- product proof result as an input shortcut.

---

# 2. New candidate type — MeshSkinProposalIR.v1

Required bindings:

- `surface_binding_hash`;
- `skeleton_binding_hash`;
- `semantic_skin_binding_hash`;
- `mesh_binding_hash`;
- model/checkpoint provenance;
- per-canonical/candidate-mesh-vertex simplex influence proposal;
- correction-from-baseline statistics;
- optional joint-mechanical training metadata.

A topology mutation changes `mesh_binding_hash` and makes the proposal stale.

Compiler must reject stale mesh-skin proposals before G3.

---

# 3. Minimal DAG migration — no wholesale renumbering

Do not insert a new geometry authority stage.

Research target semantics:

### Stage31 MIRA inference/fit
Emit both:
- semantic `SkinProposalIR`;
- optional experimental `MeshSkinProposalIR` bound to Stage18/19 mesh.

### Stage32 skin qualification
Continue to mint `QualifiedSkinIR` for semantic authority.

Additionally:
- validate `MeshSkinProposalIR` lineage/simplex/legal-joint/bounded-correction contract;
- keep it candidate authority until dynamic qualification.

### Stage33 checkpoint seal
Seal the exact dual-head checkpoint/config when MIRA-M is enabled.

### Stage35 dynamic mechanical mesh qualification
Baseline arm:
- existing deterministic `QualifiedSkinIR -> mesh` transfer.

Experimental/product-candidate arm:
- consume validated `MeshSkinProposalIR`;
- run exact hard G3/G3B on Stage18 topology.

### Stage36 qualified mesh skin
Mint `QualifiedMeshSkinIR` **only after Stage35 PASS** from the exact mesh-skin candidate that passed the court.

The existing deterministic transfer remains a fallback/control arm, not hidden product correction.

---

# 4. Topology-repair invalidation

If Stage35 produces a topology/partition repair that changes the Stage18 mesh lineage:

Keep reusable:
- IRIS;
- RiggingSurfaceIR;
- ATLAS semantic skeleton proposal / qualified skeleton in V1;
- MIRA-S semantic surface field, if its surface/skeleton bindings remain unchanged.

Invalidate and rerun:
- MIRA-M mesh proposal;
- Stage35 joint dynamic qualification;
- Stage36 QualifiedMeshSkinIR;
- all downstream product/proof artifacts.

This is intentionally cheaper than retraining or rerunning every learned stage.

If later evidence shows ATLAS control-basis choice materially depends on exact triangulation, a future ATLAS V2 mesh-bound sidecar may be added; do not assume that now.

---

# 5. Training objective

MIRA-M training objective:

```text
L =
  lambda_sem * distance(W_mesh, transfer(W_surface))
+ lambda_teacher * teacher_semantic_term
+ lambda_joint * JointMechanicalLoss(M, G, W_mesh, Q)
+ lambda_field * DeformationFieldCorrespondenceLoss
```

The semantic-distance term is a bounded prior, not a requirement to reproduce transferred weights exactly.

The deformation-field term compares the product mesh consequence against the coherent source rigged asset through correspondence.

Hard G3/G3B remains the product authority.

---

# 6. ATLAS interaction

ATLAS remains surface-semantic in the first coherence revision.

Changes now:
- every ATLAS training/evaluation court records the exact Stage18 mesh lineage used for downstream mechanical utility;
- mechanically-optimal control-basis evaluation uses the best admissible MIRA-M result on that exact mesh;
- joint loci may later receive differentiable mechanical gradients through FK/LBS.

No raw triangulation-conditioning is promoted into ATLAS until a controlled same-geometry/different-connectivity court proves it changes the optimal control proposal itself.

---

# 7. Compiler topology policy

Static topology quality is a prerequisite, not an acceptance objective.

When a rig/skin state exists, any topology mutation must be compared by:

1. static validity;
2. exact lineage;
3. best admissible mesh-skin state after bounded MIRA-M re-inference/optimization;
4. hard G3/G3B;
5. actual-motion nonregression where available.

A topology change that improves slivers/min-angle but worsens joint mechanics is rejected.

A future `MechanicalTopologyEquivalenceReportV1` records this comparison.

---

# 8. Promotion blocker

No production DAG migration until E3:

`FROZEN_KNIGHT_MESH_NATIVE_WEIGHT_OPTIMIZATION_ORACLE`

answers whether mesh-native weight freedom can materially reduce hard G3B on the exact Knight topology.

- If E3 produces strong closure: implement/train MIRA-M.
- If E3 cannot materially close G3B even with direct optimization: prioritize Compiler topology optimization; do not spend A100 training MIRA-M.

