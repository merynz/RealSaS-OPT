# RealSaS — Arachne Information-Preservation Audit V1 — Phase 5 Field-by-Field Boundary Matrix

**Date:** 2026-09-11  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_A1_IMPLEMENTATION_OR_TRAINING_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Continuation authority:** `main/CURRENT_STATE.md`  
**Running A0 4/8/16/32 experiment:** unchanged.

## 0. Purpose

This phase answers one narrow architecture question before any V7-native A1 code is written:

> For every shipping-legal field already present in `RiggingSurfaceIR S` and `QualifiedSkeletonIR[V2] G`, what survives the old A1 adapter, what is compressed or dropped, and what should the future A1 **boundary** retain independently of whether the learner consumes it in its first baseline?

The distinction is deliberate:

- **boundary retention** = keep the legal semantic information recoverable in typed/tensorized state;
- **neural admission** = decide by architecture/preregistered ablation whether the predictor actually consumes it.

No row below authorizes a neural feature merely because it exists.

## 1. Binding source facts

### 1.1 RiggingSurfaceIR public state

`SurfaceNode` publicly carries:

- `surface_id`;
- world position `P`;
- `support_views`;
- `provenance_refs`;
- `source_observation_ids`;
- per-view `raster_bindings`;
- `persistence_group_id`;
- `derived_normal`;
- `validity_flags`;
- metadata.

`SurfaceRelation` carries relation identity, endpoint surface IDs, relation kind, score and metadata. `RiggingSurfaceIR` adds geometry lineage, builder/schema and substrate metadata.

The promoted Geppetto tensorizer proves these can be exposed without flattening the substrate into the old 20/24D summary: it carries world/normalized P, normal+validity, 8-view support, per-view raster XY+validity, observed/completed validity, exact edge index, edge score/distance/type, unknown-crossing/bridge state and certificate/provenance hashes.

### 1.2 Qualified skeleton public state

`QualifiedJoint` currently carries:

- `canonical_joint_id`;
- position;
- `parent_canonical_id`;
- exact sparse `support_surface_ids`;
- `source_proposal_id`.

`QualifiedSkeletonIRV2` additionally carries:

- explicit `deform_root_ids`;
- `assembly_root_binding`;
- qualification report;
- skeleton lineage hash.

Current Compiler qualification uses proposal root/confidence/edge evidence to select the canonical graph, but the QualifiedJoint schema itself does not first-class retain selected-joint sigma/salience/confidence.

### 1.3 Old A1 public conditioning state

Old `ArachneConditioningBatchV2` contains:

- 20D per-surface summary;
- normalized surface positions;
- 8D per-joint summary;
- exact parent indices;
- 10D all-point×joint pair geometry;
- masks;
- lineage/operator/conditioning hashes.

The actual V2 predictor consumes only these tensors; it does not reach back into the richer IR objects.

---

## 2. Surface-node field matrix

| Upstream semantic | Legal class | Old A1 representation | Information delta | Future boundary ruling | First-baseline neural ruling |
|---|---|---|---|---|---|
| `surface_id` | canonical carrier identity | preserved as external row ID | identity preserved; raw string not neural | **PRESERVE EXACT for joins/output** | **DO NOT EMBED RAW ID**; serialization/permutation only |
| world `P` | product-legal geometry | normalized XYZ in 20D + separate normalized positions | geometric value preserved up to deterministic normalization | **PRESERVE world + normalized** | **BASELINE YES** |
| `support_views[0..7]` | product-legal observation support | 8 binary bits | identities preserved | **PRESERVE EXACT** | **BASELINE YES** |
| support count | deterministic summary of support bits | extra scalar | no new information; duplicate summary | do not require as independent boundary field | **DERIVE IF NEEDED**, not privileged feature |
| per-view `raster_bindings(view,xy)` | product-legal observation geometry | only `raster_count` | **XY and view-specific spatial structure lost** | **PRESERVE EXACT per-view XY + validity** | **BASELINE CANDIDATE; ablate use, not retention** |
| raster count | deterministic summary | scalar | on current Mage equals support count bitwise | derivable, no separate authority | **NO BASELINE NEED** |
| `derived_normal` | deterministic/product-legal local geometry | XYZ normal | preserved | **PRESERVE EXACT/normalized** | **BASELINE YES** |
| normal validity | typed validity | scalar | preserved; current Mage happens to be constant 1 | **PRESERVE** | **BASELINE YES or cheap mask** |
| `validity_flags` | typed qualification state | mostly collapsed to normal validity; observed/completed not exposed | **typed validity distinctions lost** | **PRESERVE vocabulary/bits** | **BASELINE LOW-COST CANDIDATE** |
| observed vs model-completed | typed scene-first state | not explicit in old A1 | **lost** | **PRESERVE** | **ABLATION CANDIDATE**; legal but usefulness unknown |
| `provenance_refs` | provenance-only | omitted from neural tensors; lineage hashes retained | correct separation | **PRESERVE certificate only** | **FORBID AS NEURAL SHORTCUT** |
| `source_observation_ids` | provenance/evidence linkage | omitted | current scene-first bridge intentionally leaves empty | preserve if meaningful in future producer | **NOT BASELINE** |
| `persistence_group_id` | provenance/association semantics | omitted | current scene-first relevance limited/producer-dependent | preserve typed/certificate state if non-null | **NOT BASELINE until semantic use exists** |
| node metadata | mixed | omitted except indirectly by operator hash | heterogeneous; some fields semantic, some provenance | **FIELD-WHITELIST only; never blind metadata ingestion** | **NO generic metadata channel** |
| normalized radius `||P_norm||` | derived geometry | scalar in old 20D | deterministic duplicate of position | not independent boundary field | **DROP/DERIVE** |

### Surface conclusion

The highest-confidence old-A1 losses are not scalar “extra features”; they are **relational/view geometry**:

1. per-view raster XY;
2. exact validity/observed-completed state;
3. exact local-relation graph (next section).

The future boundary should resemble promoted Geppetto's fieldwise tensorization rather than widening the 20D vector.

---

## 3. Surface-relation field matrix

| Upstream semantic | Legal class | Old A1 representation | Information delta | Future boundary ruling | First-baseline neural ruling |
|---|---|---|---|---|---|
| exact `(a_surface_id,b_surface_id)` adjacency | product-legal topology | node degree summary only | **edge identity completely lost** | **PRESERVE EXACT edge index** | **BASELINE YES as structural attention/message graph** |
| relation kind | typed structural evidence | omitted | lost; current scene-first witness may have one kind but schema is not universally constant | **PRESERVE vocab/index** | **ABLATE if single-kind; required when multi-kind appears** |
| relation score | product-legal relation evidence | per-node mean score | edge-level association lost | **PRESERVE per edge** | current Mage appears weak/constant; **ABLATION CANDIDATE** |
| normalized edge distance | deterministic relation geometry | not retained explicitly; indirectly inferable from P if edge known | edge identity loss prevents direct use | **PRESERVE/DERIVE on exact edge** | **BASELINE CANDIDATE** |
| `crosses_unknown` | typed relation diagnostic | omitted | lost | **PRESERVE** | **ABLATION CANDIDATE** |
| `unknown_bridge` | typed relation diagnostic | omitted | lost | **PRESERVE** | **ABLATION CANDIDATE** |
| relation metadata/provenance | mixed | omitted | expected | certificate + field whitelist only | **FORBID generic metadata ingestion** |
| degree | deterministic topology summary | normalized scalar | preserved | derivable from edge index | **DERIVE, not authority** |
| mean relation score | deterministic summary | scalar | current Mage constant 1; loses edge localization | derivable | **NOT BASELINE NEED** |

### Relation conclusion

The P0 issue is **not** “degree is a bad number.” It is that an exact 2813-edge compact topology was replaced by two per-node summaries before A1. The old summaries may remain useful diagnostics, but they are not a substitute for the graph.

---

## 4. Qualified-skeleton field matrix

| Upstream semantic | Legal/authority class | Old A1 representation | Information delta | Future boundary ruling | First-baseline neural ruling |
|---|---|---|---|---|---|
| `canonical_joint_id` | Compiler authority identity | external row ID | preserved | **PRESERVE for joins/output** | **DO NOT EMBED RAW HASH/ID** |
| qualified joint position | Compiler-authoritative geometry | normalized XYZ | preserved | **PRESERVE world + normalized** | **BASELINE YES** |
| exact parent relation | Compiler-authoritative tree | `parent_indices` exact | preserved | **PRESERVE EXACT graph** | **BASELINE YES** |
| parent-present bit | deterministic summary | scalar | duplicate of parent index | derive if needed | **NO independent need** |
| tree depth | deterministic graph feature | normalized scalar | preserved as one coarse summary | derive from exact graph | **DERIVE; optional positional/graph feature** |
| deform-root membership | Compiler authority | root bit | preserved in old adapter via `deform_root_ids` when V2 present | **PRESERVE EXACT SET** | **BASELINE YES / structural token type** |
| exact `support_surface_ids` | qualified sparse mechanical-support anchors | only count `len/16` | **176 identity-bearing links collapse to 22 scalars; on Mage every scalar=0.5** | **PRESERVE EXACT bipartite mapping** | **ABLATION CANDIDATE as soft positive bias; NEVER hard skin mask** |
| support count | deterministic summary | scalar | on Mage zero discriminative information | derive from mapping | **NO independent need** |
| `source_proposal_id` | provenance/join key | not neural | correct | **PRESERVE provenance** | **FORBID raw ID embedding** |
| `assembly_root_binding` | Compiler product authority | old A1 does not expose | lost/unimplemented on current Mage because `{}` | **PRESERVE when non-empty** | future attachment/assembly-aware consumer only |
| qualification report | Compiler evidence/provenance | omitted from neural input | expected in general; report is not wholesale neural evidence | **PRESERVE certificate/report** | **NO generic report ingestion** |
| skeleton lineage hash | provenance/authority binding | preserved as hash | correct | **PRESERVE** | **FORBID as neural shortcut** |
| trailing constant 1 in old 8D joint row | none | constant | no semantic information | **REMOVE from new semantic contract** | **NO** |

### Skeleton conclusion

The old A1 retained the **canonical tree and positions**, which is important and means it was not a dummy placeholder. The material loss is concentrated in **qualified surface-support mapping and richer qualified evidence**, not in root/parent geometry itself.

---

## 5. Pairwise deterministic relation matrix

Old A1 V2 introduced a 10D relation for every legal surface×joint pair:

1. point→control `dx,dy,dz`;
2. point→control distance;
3. point→parent-segment distance;
4. clamped segment parameter `t`;
5. segment length;
6. `abs(normal·bone_axis)`;
7. normal-axis validity;
8. parent-exists.

For Mage the pair mask is active for all `950×22 = 20,900` pairs.

**Ruling:** `PRESERVE_AS_DERIVED_PRODUCT_LEGAL_INDUCTIVE_BIAS_CANDIDATE`.

Reasons:

- it is teacher-free and derived entirely from admitted S+G;
- it directly parameterizes the skinning-relevant surface↔bone relation;
- unlike the support-count summary, it retains pair identity for every pair;
- it does not override Compiler authority.

A V7-native A1 may compute this on demand from exact S+G rather than serialize 20,900×10 values as primary authority. The semantic contract matters more than the storage layout.

---

## 6. Geppetto proposal evidence that does **not** belong in S+G baseline by default

Phase 3 prevents the audit from becoming “copy every upstream tensor downstream.”

| Candidate evidence | Current scientific status | Boundary ruling |
|---|---|---|
| `mechanical_salience_probability` | FIT1 all-positive target, nearly saturated; magnitude not calibrated | **DO NOT baseline-admit** |
| selected joint `confidence` | legal proposal evidence but narrow Mage range; already consumed by Compiler graph optimizer | **not first-baseline input; sidecar candidate only if justified** |
| `position_sigma_normalized` | trained through heteroscedastic position NLL; nonconstant | **best sidecar candidate, calibration still required** |
| full dense support logits | trained nearest-8 mechanical support evidence, not skin ownership | **do not expose blindly; sparse qualified anchors already survive** |
| raw all-pair parent/root alternatives | decision evidence consumed by Compiler | **FORBID as competing downstream authority** |
| model hidden states/surface attention | model-private latent | **DO NOT PUBLICIZE by default** |

If downstream structural confidence is eventually useful, prefer a Compiler-qualified sidecar tied to the accepted graph, not raw proposal alternatives.

---

## 7. V7 A0 codec interface vs V7-native A1 conditioning boundary

These must remain separate.

The current V7 codec has a deliberately narrow representation contract:

- geometry query = `xyz + normal + normal_valid` (7D);
- A0 field observation = that geometry + teacher scalar W (8D);
- `encode_field(W)` -> `K×512` continuous field tokens;
- condition encoder uses geometry only;
- decoder uses field tokens + condition tokens + query geometry.

The future A1 shipping predictor does **not** receive teacher W. Its legal job is conceptually:

`rich S + Qualified G -> predicted K×512 field-token set -> frozen A0 decoder -> W proposal`.

Therefore the fact that the frozen decoder only needs geometry7 at its query interface is **not a reason to reduce the A1 predictor input to geometry7**.

`K` remains pending the live 4/8/16/32 experiment and subsequent frozen-interface audit.

---

## 8. Proposed future A1 boundary classes — audit result, not architecture implementation

The Phase-5 matrix yields four classes.

### Class A — retain and baseline-consume

High-confidence task-relevant legal structure:

- surface normalized/world position;
- robust normal + validity;
- exact 8-view support;
- exact GSA edge topology;
- qualified joint positions;
- exact Compiler parent/deform-root graph;
- deterministic surface×joint/parent-segment geometry.

### Class B — retain at boundary, neural use by ablation

Legal evidence whose incremental predictive value is not yet proven:

- per-view raster XY + validity;
- observed/completed + detailed validity bits;
- relation kind/score/distance/unknown flags;
- exact joint→surface sparse support-anchor mapping;
- any future calibrated qualified position-confidence sidecar.

**Retention is not conditional on the ablation.** Only neural consumption is.

### Class C — retain for provenance/authority but never use as raw neural shortcut

- surface/joint raw IDs/hashes;
- source proposal IDs;
- provenance refs;
- operator/certificate hashes;
- generic qualification reports and arbitrary metadata.

### Class D — do not promote as evidence under current science

- current IRIS V3 `log_uncertainty` scalar, because its promoted lineage does not calibrate it as uncertainty;
- saturated FIT1 mechanical-salience magnitude;
- raw model-private activations;
- raw parent/root alternatives after Compiler qualification.

---

## 9. Minimal future tensor-boundary shape implied by the audit

This is **not** the predictor architecture. It is the information-preserving input envelope the predictor can be built against.

Conceptually:

```text
ArachneQualifiedConditioningEnvelope
  surface:
    ids/provenance                 [non-neural]
    P_world, P_normalized          [N,3]
    normal, normal_valid           [N,3], [N]
    view_support                   [N,8]
    raster_xy, raster_valid        [N,8,2], [N,8]
    validity_bits + vocabulary     [N,V]
    observed/completed             [N], [N]
    edge_index                     [E,2]
    edge_kind/score/distance/...   [E,*]
  skeleton:
    canonical IDs                  [non-neural identity]
    P_world, P_normalized          [J,3]
    parent_index                   [J]
    deform_root_mask               [J]
    support_anchor_edges           sparse (joint,surface)
    assembly binding               typed, if present
    optional qualified confidence  sidecar, if separately admitted
  derived:
    point_joint_segment_geometry   [N,J,10] or computed on demand
  lineage:
    surface/skeleton/operator hashes [non-neural]
```

This envelope is deliberately richer than the model must consume. It prevents architecture experiments from requiring another lossy IR rewrite.

---

## 10. Phase-5 verdicts

### P0-F05-A — confirmed

`Qualified S+G -> old A1` performs material, task-relevant information compression. The clearest losses are exact GSA topology, per-view raster coordinates and exact joint→surface support-anchor identity.

### P0-F05-B — important correction

The old A1 is **not empty/placeholder-only**. It correctly retains positions, exact parent indices, root/depth summaries and a rich 10D all-pair geometry. Its conceptual lineage remains useful.

### P0-F05-C — binding design rule

Do **not** repair the future A1 by simply increasing `surface_feature_dim=20` and `joint_feature_dim=8`. The lost information is typed/set/graph relational structure that should stay factorized.

### P1-F05-D — feature-admission discipline

Do not make every preserved field a baseline feature. Boundary preservation and neural admission remain separate, with controlled ablations for Class-B channels.

### INFO-F05-E — Compiler authority preserved

The future richer A1 boundary must not reopen canonical root/parent/tree decisions or interpret Geppetto support anchors as hard skin ownership.

---

## 11. Next phase

Phase 6 will specify a **no-training, no-implementation** `QualifiedSkeletonEvidence` sidecar candidate and decide which Geppetto/Compiler evidence can legally survive qualification without creating a second skeleton authority.

Phase 7 remains the independent rigid-attachment/assembly census. Phase 8 remains blocked on completion of the current A0 token arms.

**No A1 code, optimizer run, current A0 mutation, token-count decision or product PASS is authorized by Phase 5.**
