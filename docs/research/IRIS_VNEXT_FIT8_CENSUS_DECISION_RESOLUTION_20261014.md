# IRIS-vNext FIT8 census decision resolution

**Date:** 2026-10-14  
**Status:** architecture-decision evidence; partial leaf resolution only  
**Scope:** exact-byte FIT8 source census -> IRIS-vNext / teacher-schema decisions  
**Does not claim:** scientific PASS, product PASS, import admission, final model freeze

This document records only the architecture leaves that are already resolved by the exact eight-subject FIT8 source census. It deliberately leaves capacity, extraction resolution, retopology family, dynamic sampling density and CAO fragment granularity open until rendered/ray-hit/deformation evidence exists.

The source census is `docs/research/FIT8_SOURCE_STRUCTURE_CENSUS_20261014.md`. The open-surface representation audit is `docs/research/IRIS_VNEXT_OPEN_SURFACE_REPRESENTATION_AUDIT_20261014.md`.

---

## 0. Census facts that force decisions

| Subject | Source family | Geometric component evidence | Material regions | UV/texture | Max observed skin influences | Surface-class note |
|---|---|---:|---:|---|---:|---|
| Snake | FBX | 3 closed source components | 7 | no UV | 4 | one Geometry object != one component/material |
| Bat | glTF | 12 position-weld diagnostic components | 1 | 512x512 texture | 4 | open/non-manifold components exist |
| T-Rex | FBX | 1 closed source component | 5 | no UV | **7** | raw weights not unit-normalized |
| Horse | glTF | 1 closed welded component | 8 | no texture/UV | 4 | strongly smooth-deforming surface |
| Spider | FBX | 15 closed source components | 2 | no UV | **6** | many geometric components inside one Geometry object |
| GreenBlob | glTF | 2 closed welded components | 1 | 32x32 texture | 4 | smooth main body + rigid-like secondary component |
| Shark | FBX | 1 closed source component | 2 | no UV | 4 | axial morphology, no ground-contact assumption |
| George | glTF | 61 welded diagnostic components | 1 | 2048x2048 texture | 3 | 40 closed + 21 open diagnostic components; many rigid bone-owned parts |

Additional measured facts:

- all four inspected FBX sources preserve raw cluster weights whose per-vertex sums are not universally 1.0;
- glTF exports normalize `WEIGHTS_0` to numerical precision, so export-format normalization itself is provenance;
- Horse raw glTF indexing is heavily fragmented by render seams while position-weld evidence resolves one closed surface;
- Bat remains open/non-manifold even after diagnostic position welding;
- George is one mesh and one material but resolves many mechanically/geometrically distinct parts;
- native animation exists for all eight subjects, from 1 clip (Shark) to 20 clips (George).

---

## 1. RESOLVED — source/render topology is not mechanical truth

**State:** `REJECTED_BY_CONTRACT` for `RAW_RENDER_TOPOLOGY == CANONICAL_MECHANICAL_TOPOLOGY`.

Reason:

- glTF render vertices are duplicated by normal/UV/material seams;
- Horse demonstrates severe raw-index fragmentation that disappears under geometric equivalence;
- diagnostic welding is itself not automatically teacher authority.

Required architecture:

```text
RAW SOURCE / RENDER TOPOLOGY
        |
        +--> preserved exact identity
        |
        +--> qualified geometric-equivalence evidence
                     |
                     +--> teacher/reference surface
                     +--> later F/T/Compiler decisions
```

Any welding, triangulation, seam merge or topology normalization must carry explicit provenance and source binding.

---

## 2. RESOLVED — object, component, material, owner and attachment are independent axes

**State:** `REJECTED_BY_CONTRACT` for any equality assumption among:

```text
source object
geometric connected component
material region
mechanical owner
presentation owner
attachment relation
```

Evidence:

- Snake: one Geometry object, three disconnected closed components, seven materials;
- Spider: one Geometry object, fifteen disconnected closed components, two materials;
- George: one mesh/material, dozens of geometric/mechanical components;
- GreenBlob: one semantic subject with two geometric components exhibiting different deformation behavior.

Consequences:

- S cannot be a flat human-part class head;
- CAO cannot use material slot as fragment/owner authority;
- rigid bone ownership does not imply detachable/held attachment;
- teacher schema must preserve raw identities and normalized/qualified relations separately.

---

## 3. RESOLVED — fixed four-influence teacher truth is invalid

**State:** `REJECTED_BY_CONTRACT` for a universal `K=4` raw-skin contract.

Evidence:

- T-Rex reaches 7 raw FBX influences;
- Spider reaches 6;
- glTF happens to store at most four in the inspected exports, but that is an export representation, not a corpus invariant.

Required contract:

```text
RAW_AUTHORED_SKIN: variable/ragged influence count + exact provenance
DERIVED_NORMALIZED_SKIN: explicit deterministic/qualified transform
RUNTIME_SKIN: separately chosen compact policy after fidelity/mechanics court
```

The runtime may eventually choose a compact influence cap, but it must be justified by a measured approximation curve rather than silently rewriting teacher truth.

---

## 4. RESOLVED — raw skin and normalized mechanical teacher skin are different artifacts

**State:** `LOCKED_BY_FIT8_MEASUREMENT`.

FBX raw per-vertex weight sums are not universally one. glTF exports are normalized. Therefore normalization is a transformation with provenance, not source truth.

Required teacher lineage:

```text
RAW_AUTHOR/EXPORT WEIGHTS
        |
        +--> exact preservation
        |
        +--> deterministic normalization policy
                    |
                    +--> normalized mechanical teacher target
```

No consumer may relabel the normalized result as raw authored evidence.

---

## 5. RESOLVED — UV is validity-gated, not universally required

**State:** `REJECTED_BY_CONTRACT` for `UV_REQUIRED == TRUE`.

FIT8 contains:

- material-color FBX subjects with no UV layer;
- Horse with eight materials but no texture/UV;
- textured Bat/GreenBlob/George at 512, 32 and 2048 pixels respectively.

Required A/CAO truth supports both:

```text
surface material evaluation
and
UV/texture evaluation where authored
```

Continuous surface appearance addressing is the shared contract; source UV is one possible authored source address, not the authority domain itself.

---

## 6. RESOLVED — one universally valid global metric SDF is not a corpus invariant

**State:** `REJECTED_BY_CONTRACT` for `GLOBAL_METRIC_SDF_VALID_EVERYWHERE`.

Bat and George contain significant diagnostic evidence of open surface components. A generic-articulation system cannot equate all authored surfaces with boundaries of closed volumes.

Required geometry truth factorization:

```text
u(x) = unsigned/proximity truth                  broadly available
f(x) = metric signed distance where qualified   validity-gated
q(x) = sign/orientation validity                 explicit
s(x) = surface-existence/support truth           explicit
```

Closed qualified components may retain metric SDF supervision. Open orientable surfaces may use local/oriented level-set evidence. Ambiguous/non-manifold regions must permit abstention rather than fabricated sign.

This decision preserves legacy IRIS direct field supervision where mathematically justified without making closed-solid topology a genericity assumption.

---

## 7. RESOLVED — F must have explicit surface-support/existence evidence

**State:** `LOCKED_BY_FIT8_MEASUREMENT` at the abstraction level; implementation family remains open.

Because open surfaces terminate without a volumetric inside/outside event, F cannot rely solely on negative-SDF occupancy to decide where extraction exists.

Required abstraction:

```text
surface-support / active-domain evidence
AND
level-set / distance evidence
```

Current research ranking remains:

1. sparse active-cell / surface-support extraction + local signed field;
2. level-set/SDF + pointwise validity;
3. global UDF + dedicated UDF mesher.

The ranking is not a freeze of SparseFlex, NeAT or any specific implementation.

---

## 8. RESOLVED — component class and deformation mode are independent

**State:** `LOCKED_BY_FIT8_MEASUREMENT`.

Teacher/runtime reasoning must distinguish at least:

```text
closed smooth-deforming surface
open/two-sided smooth-deforming surface
rigid bone-owned open/closed part
explicit attachment
```

Evidence includes:

- Horse: one closed component with broad smooth multi-joint deformation;
- George: many one-hot rigid bone-owned components inside one mesh/material;
- GreenBlob: smooth main component plus one-hot secondary component;
- Bat: mixed thin/open geometry and mixed deformation behavior.

T must not "repair" all open parts into closed solids merely to satisfy a topology convention.

---

## 9. RESOLVED — semantic targets are subject-local/relational, not global anatomical IDs

**State:** `LOCKED_BY_FIT8_MEASUREMENT` for representation principle.

The cohort contains snake, bat, dinosaur, horse, spider, blob, shark and mech morphology. A global human/anime class simplex would either be incomplete or force semantically unrelated structures into arbitrary labels.

Primary S evidence therefore remains factored/relational:

```text
same-owner affinity / owner descriptor
owner boundary
same-material affinity / material descriptor
deformation mode
mechanical owner relation
attachment relation
amodal support
overlap/occlusion relation
uncertainty / abstention
```

Optional coarse roles may exist as secondary metadata, but do not own the core scene partition.

---

## 10. RESOLVED — dynamic teacher evidence is available, but dynamic semantics remain provenance-sensitive

**State:** `LOCKED_BY_FIT8_MEASUREMENT` for using authored motion as a teacher source; `OPEN_ABLATION_REQUIRED` for pose/frame sampling density.

All eight subjects contain native animation. This is sufficient to build dynamic consequence courts during FIT8 closure without scaling the corpus first.

However:

```text
authored animation
!= authored contact label
!= authored stylized draw-order label
```

Derived contact, overlap and ray-hit order must retain `DERIVED_FROM_AUTHORED_MOTION` provenance unless an explicit authored constraint/socket/relationship exists.

---

## 11. RESOLVED — appearance/presentation context is surface x context, not a second visual mesh

**State:** `LOCKED_BY_REFERENCE_AND_MATCHED_REQUIREMENT` + reinforced by FIT8 census.

Preserve the existing RealSaS invariant:

```text
canonical mechanical surface M
x
presentation context (direction; later pose/time)
```

Appearance/semantic/presentation evidence is addressed on the same canonical surface. A separate visual-mesh authority is not reintroduced.

George proves why material ID alone cannot carry presentation decomposition; Bat proves why two-sided/open support must be explicit; Horse proves why render seam identity cannot be used as mechanical identity.

---

## 12. OPEN — do not freeze these from the structural census alone

The following remain unresolved by design:

| Leaf | Current state | Required next evidence |
|---|---|---|
| image backbone exact variant / input resolution | `OPEN_ABLATION_REQUIRED` | matched encoder/scene-budget experiments |
| early vs late camera conditioning | `OPEN_ABLATION_REQUIRED` | cross-view geometry/owner/material correspondence |
| scene transformer width/depth | `OPEN_ABLATION_REQUIRED` | capacity/compute curve |
| triplane resolution/channels | `OPEN_ABLATION_REQUIRED` | field/boundary/thin-structure curve |
| owner/material query/slot capacity | `OPEN_ABLATION_REQUIRED` | rendered owner/material/component cardinality distribution |
| component merge/split normalizer | `OPEN_ABLATION_REQUIRED` | canonical-view + dynamic visibility graph measurements |
| sparse support grid / F pitch | `OPEN_ABLATION_REQUIRED` | zero-false-negative support recall on all FIT8 subjects |
| local sign/orientation policy | `OPEN_ABLATION_REQUIRED` | source-normal/orientation audit on important open components |
| semantic-equivalent extraction mechanics | `OPEN_ABLATION_REQUIRED` | decomposition-state extraction stability |
| T decoder family | `OPEN_ABLATION_REQUIRED` | matched-budget mechanical + fidelity court |
| T runtime triangle budget | `OPEN_ABLATION_REQUIRED` | Pareto curve: static fidelity vs mechanics vs mobile cost |
| dynamic pose/frame sampling count | `OPEN_ABLATION_REQUIRED` | clip diversity / event coverage curve |
| CAO fragment granularity | `OPEN_ABLATION_REQUIRED` | ray-hit/overlap/contact/presentation graph cardinality |
| slot-level vs fragment-level draw-order primitive | `OPEN_ABLATION_REQUIRED` | dynamic overlap graph complexity |
| learned 1->8 prior | `DEFERRED_PRODUCT_PHASE` | only after eight-view reconstruction closure |

---

## 13. Next closure experiment

The next data-producing step is not another architecture guess. It is a deterministic FIT8 rendered/mechanical census using the existing full-truth schema:

```text
for each subject
  canonical 8 cameras
  source-exact RGBA/depth/normal
  owner/material/component IDs
  all-hit ray stacks
  valid decomposition-state renders
  selected authored motion frames
  posed geometry / skin consequence
  overlap graph / amodal support
  open-boundary + support-cell truth
```

This experiment should resolve the next group of leaves:

- subject-local owner/material region cardinality;
- decomposition-state count;
- ray-hit stack depth and overlap graph cardinality;
- visible significance of Bat/George open components;
- minimum support-grid pitch for zero false-negative holes;
- dynamic draw-order primitive granularity;
- candidate CAO fragment granularity.

Only after those measurements should the first full IRIS-vNext training configuration be frozen.

---

## 14. Decision summary

The exact FIT8 source structure has already falsified several convenient assumptions. That is a successful closure result, not a setback.

The architecture now has stronger generic invariants:

```text
raw truth != normalized truth
render topology != mechanical topology
material != owner
component != attachment
rigid ownership != attachment
UV is optional
raw skin influence count is variable
global SDF sign is validity-gated
surface existence/support is explicit
open and closed surfaces are both first-class
semantic identity is relational/subject-local
one canonical surface remains the cross-domain address authority
```

Everything not supported by those facts remains explicitly open.