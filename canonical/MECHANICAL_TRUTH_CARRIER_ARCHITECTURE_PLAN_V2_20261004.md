# RealSaS — Mechanical Truth Carrier Architecture Plan V2 — 2026-10-04

**Status:** `CANONICAL_RESEARCH_DIRECTION__E3_BLOCKED_PENDING_CARRIER_COURT`

## Correction

This plan supersedes the premature assumption that ATLAS or MIRA must consume the final product triangle connectivity as neural input, and supersedes the provisional MIRA-M topology-conditioned head proposal.

The demonstrated problem is **truth-carrier / mechanical-basis coherence**, not an established lack of topology features in the neural networks.

## External reference re-audit

### RigAnything

The released RigAnything rig model consumes sampled point coordinates and normals. Surface samples and normals are obtained from the input mesh; original/full mesh vertices remain available for skinning/output processing. Explicit triangle faces are not the core autoregressive rig transformer's learned input.

### SkinTokens / TokenRig

TokenRig conditions generation on sampled vertices and normals. The data path samples geometry from an asset's exact mesh faces and barycentrically samples vertex-group/skin values at those same face samples. Predicted rig/skin is materialized back on the original asset vertices/faces.

Therefore the relevant reference property is:

```text
sample geometry P,N
and supervision W/R
are derived from one coherent mesh/mechanical asset
```

not:

```text
triangle face indices must be neural input
```

## Current RealSaS re-audit

### GSA / RiggingSurfaceIR

Current GSA is not raw IRIS point output.

IRIS zero-surface has vertices/faces. GSA compaction derives deterministic local-relation edges from those source faces. Current face-provenance repair also preserves compact faces with dense-face witnesses and forbids three-clique face minting.

Thus GSA supplies a topology-derived surface representation.

### ATLAS

Current ATLAS is richer than the shorthand “XYZ+N”:

- normalized position;
- normal + validity;
- observation/view support;
- raster evidence;
- observed/completed state;
- deterministic GSA local-relation graph/edge attributes.

Its graph is the **GSA relation graph**, not Stage18 product mesh connectivity.

No current evidence requires adding Stage18 faces to ATLAS input.

### MIRA

Current MIRA backbone consumes:

- the same GSA surface evidence/relation graph;
- Compiler-qualified skeleton/tree;
- point-joint/parent-segment geometry;
- sparse mechanical support anchors.

Again, its surface graph is **GSA relation topology**, not Stage18 product triangles.

### MIRA decoder

The current V5 direct-simplex decoder is already a point-query field decoder.

After the GSA/skeleton backbone produces joint field tokens, the decoder consumes:
- query geometry `XYZ + normal + normal-valid`;
- point-joint pair geometry;
- joint field tokens.

The pair-geometry builder itself requires only query point position/normal and skeleton geometry, not triangle connectivity.

Therefore the current MIRA can, in principle, query its learned field at Stage18 mesh vertices without a new topology-conditioned graph head.

## Historical MIRA FIT1 truth

The sealed V4/V5 FIT1 trains/evaluates against a frozen supervision bank containing:

- GSA950 geometry + teacher weights;
- Dense8K query geometry + teacher weights;
- disjoint holdout geometry + teacher weights.

The predictor input explicitly excludes hidden source mesh geometry.

The loss includes row-weight terms and an articulated LBS consequence term evaluated on those query points, but no face-connectivity mechanical loss.

Historical A0 experiments also tried source-face/barycentric dense supervision; that treatment did not close the then-current GSA-target FIT1 objective and was rejected for that experiment. This is consistent with, but does not alone prove, a mismatch between source-topology supervision and the chosen GSA truth basis.

## The actual coherence gap

RealSaS currently has at least three distinct topology/basis notions:

1. **source coherent asset topology** — source mesh/rig/skin truth;
2. **IRIS/GSA topology-derived substrate** — compacted zero-surface nodes, relations, and now preserved compact-face provenance;
3. **Stage18 product mechanical topology** — partitioned/refined mesh used by final deformation.

The forensic courts prove that mechanically meaningful truth cannot be moved between these bases merely by copying/projecting vertex weights and assuming equivalence.

The architecture bug is therefore:

```text
separate truth/basis transitions were allowed without a closed mechanical-equivalence contract
```

not:

```text
ATLAS/MIRA definitely need final mesh adjacency as input
```

## Invariant to adopt

Every rig/skin/deformation truth record must bind to an explicit `mechanical_truth_carrier_hash`.

A carrier defines at minimum:

- rest positions;
- face/connectivity basis used for deformation interpolation;
- vertex/sample correspondence;
- normal convention;
- control/skeleton binding;
- skin field binding;
- probe/motion family used to validate consequence.

Two assets with identical XYZ but different triangulation are different mechanical truth carriers unless equivalence is proven.

## Model policy

### ATLAS

Do not add Stage18 topology input now.

Keep current GSA-derived representation.

Change evaluation/training authority:
- teacher skeleton/cardinality is evidence, not final target cardinality;
- mechanically-optimal control-basis utility is always scored against one explicit truth carrier;
- every court records the carrier hash.

### MIRA

Do not add MIRA-M now.

Keep the current GSA+skeleton backbone and point-query decoder.

The first coherence experiment is to query the **existing** MIRA field on whichever mechanical carrier is selected and compare against teacher truth compiled on that same carrier.

Only if the current point-query field fails after truth-carrier coherence is established may a topology-conditioned MIRA head be proposed.

## Compiler policy

Do not treat static topology quality as mechanical equivalence.

For every topology transition:

```text
T_old -> T_new
```

one of the following must hold:

1. `T_new` is the same mechanical truth carrier;
2. a typed `MechanicalTopologyEquivalenceReport` proves the transition preserves admitted deformation behavior;
3. rig/skin truth is recompiled/re-qualified on `T_new`, producing a new carrier hash.

No silent “same XYZ therefore same truth” transition.

## Carrier-selection court — required before E3

E3 is blocked.

First run `MECHANICAL_TRUTH_CARRIER_SELECTION_COURT_V1` with model prediction removed from the causal path.

Arms:

### C0 — source coherent control
Source topology + source/teacher rig + source/teacher skin + frozen probe bank.

Expected role: positive control.

### C1 — compact GSA face-provenance carrier
Use the exact compact-face provenance now preserved from the IRIS zero-surface/GSA lineage.
Compile/rebind teacher rig/skin evidence to this carrier.
Run hard G3/G3B.

### C2 — Stage18 relation/holeless baseline carrier
Use the exact pre-model Stage18 mechanical candidate.
Compile/rebind the same teacher mechanical truth to this carrier.
Run hard G3/G3B.

### C3 — current repaired V9 carrier
Use the current final V9 candidate only as a diagnostic comparison, with truth explicitly bound to that carrier.

Required outputs:
- static quality;
- hard G3/G3B;
- deformation-field error versus source coherent control;
- correspondence coverage;
- topology/face lineage identity;
- whether failure arises before any learned model prediction.

## Decision

- If C1 is mechanically sound and C2 introduces failure:
  - make compact GSA face provenance the canonical mechanical truth carrier;
  - Stage18 may only perform mechanically-proven equivalent refinement/partitioning.

- If C1 is mechanically unsound but C2 is sound after mechanically compiled truth:
  - Stage18 becomes the canonical mechanical truth carrier;
  - keep ATLAS/MIRA input representation, but compile training/evaluation truth onto Stage18 carrier;
  - query current MIRA decoder directly at Stage18 points.

- If neither C1 nor C2 is mechanically sound with mechanically compiled teacher truth:
  - Compiler topology construction remains the primary owner;
  - solve carrier construction before retraining either model.

- Only after one carrier path is closed may E3 resume.

## E3 revised meaning

E3 is **not** “train/justify a MIRA-M head.”

Once the carrier is selected:

1. run existing MIRA point-query decoder on the exact carrier points;
2. compare current MIRA prediction with coherent carrier-bound teacher truth;
3. run hard G3/G3B;
4. optionally run direct weight optimization only as a ceiling diagnostic.

A new model head is authorized only if this coherent existing-field path has a demonstrated residual it can plausibly solve.

## Binding rules

- `explicit_product_mesh_topology_required_as_ATLAS_input = false`
- `explicit_product_mesh_topology_required_as_MIRA_input = false`
- `gsa_relation_graph_is_not_stage18_mesh_topology = true`
- `mira_existing_decoder_is_point_query_capable = true`
- `mechanical_truth_carrier_hash_required = true`
- `teacher_weight_projection_alone_proves_equivalence = false`
- `static_mesh_quality_proves_equivalence = false`
- `mira_m_authorized = false`
- `e3_authorized = false`
- `next_gate = MECHANICAL_TRUTH_CARRIER_SELECTION_COURT_V1`
