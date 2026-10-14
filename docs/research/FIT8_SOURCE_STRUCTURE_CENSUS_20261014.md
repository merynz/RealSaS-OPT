# FIT8 exact-byte source structure census

**Date:** 2026-10-14  
**Status:** research census / schema input — NOT import admission, NOT teacher authority, NOT scientific PASS  
**Scope:** the exact eight files selected by `canonical/FIT8_DIVERSITY_SELECTION_V1.json`

This census exists before the FIT8 full-truth schema is frozen. Its purpose is to prevent human-character assumptions, export-format assumptions, or convenient tensor shapes from silently becoming RealSaS truth contracts.

## 0. Method and authority boundary

All eight source files were read from the sealed source locators and their raw SHA-256 digests were checked against `FIT8_DIVERSITY_SELECTION_V1.json`. All eight matched exactly.

For FBX, the diagnostic census reads binary FBX v7400 structure, geometry, materials, deformers/skin clusters and animation stacks without mutating source data.

For glTF, the diagnostic census reads embedded buffer/accessor data directly. Two topology views are intentionally distinguished:

```text
RAW RENDER TOPOLOGY
    exact glTF indexed vertices/faces as authored/exported

POSITION-WELDED DIAGNOSTIC TOPOLOGY
    exact-position-like grouping (1e-7 decimal diagnostic tolerance)
    used only to identify exporter seam duplication
```

The welded topology is **not** automatically teacher authority. It is diagnostic evidence that raw glTF vertex identity may encode UV/normal/material seams rather than geometric identity.

No source file was modified. No Compiler admission or training Attempt is implied.

---

## 1. Exact source identity check

| Subject | Format | Bytes | SHA-256 selection match |
|---|---:|---:|---|
| Snake | FBX | 422,252 | YES |
| Bat | glTF | 515,943 | YES |
| T-Rex | FBX | 1,074,060 | YES |
| Horse | glTF | 3,610,082 | YES |
| Spider | FBX | 1,233,484 | YES |
| GreenBlob | glTF | 148,007 | YES |
| Shark | FBX | 101,788 | YES |
| George | glTF | 5,152,038 | YES |

---

## 2. Geometry / topology census

### 2.1 FBX subjects

| Subject | Source vertices | Source polygons | Triangulated faces | Connected components | Boundary edges | Non-manifold edges | Material regions | UV layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Snake | 815 | 822 | 1,618 | 3 | 0 | 0 | 7 | 0 |
| T-Rex | 912 | 838 | 1,820 | 1 | 0 | 0 | 5 | 0 |
| Spider | 1,386 | 1,380 | 2,712 | 15 | 0 | 0 | 2 | 0 |
| Shark | 324 | 328 | 644 | 1 | 0 | 0 | 2 | 0 |

All four inspected FBX geometries are edge-manifold and closed under the source polygon connectivity diagnostic. They still require orientation/nesting/sign qualification before any global metric signed-distance claim.

Important examples:

- Snake is **one authored Geometry object but three disconnected surfaces and seven material regions**.
- Spider is **one authored Geometry object but fifteen disconnected surfaces**.
- Therefore `object == owner == connected component == material` is invalid even within this small cohort.

### 2.2 glTF subjects

| Subject | Render vertices | Triangles | Materials | UV | Texture | Position-weld vertices | Weld components | Closed weld components | Open weld components | Non-manifold weld components |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| Bat | 1,008 | 1,200 | 1 | YES | 512x512 | 650 | 12 | 3 | 9 | 2 |
| Horse | 4,400 | 2,182 | 8 | NO | none | 1,093 | 1 | 1 | 0 | 0 |
| GreenBlob | 878 | 1,520 | 1 | YES | 32x32 | 764 | 2 | 2 | 0 | 0 |
| George | 7,755 | 7,864 | 1 | YES | 2048x2048 | 4,124 | 61 | 40 | 21 | 0 |

Critical export observations:

- Horse looks extremely fragmented under raw indexed glTF connectivity because different material/normal seams duplicate vertices. Position-weld evidence reconstructs one closed manifold surface. Raw render indices are therefore not a safe mechanical topology authority.
- Bat has 1,008 render vertices but only 650 diagnostic unique positions. Even after position welding, multiple components remain open/non-manifold. A universal closed-solid/SDF assumption would be false.
- GreenBlob consists of two closed welded components despite being an amorphous subject.
- George is one glTF mesh with one texture material, yet diagnostic welding reveals **61 geometric components**, including both closed and open parts. Material/object count cannot substitute for mechanical/presentation ownership.

---

## 3. Appearance representation census

The FIT8 cohort already spans materially different appearance sources:

```text
FBX Snake/T-Rex/Spider/Shark
    authored material colors
    no UV layer detected
    no texture objects detected

Horse glTF
    8 authored base-color materials
    no texture / no TEXCOORD_0

Bat glTF
    1 UV texture, 512x512
    KHR_materials_unlit

GreenBlob glTF
    1 UV atlas texture, 32x32

George glTF
    1 UV texture, 2048x2048
    KHR_materials_unlit
```

Consequences:

1. UV is a validity-gated authored channel, not a required universal truth.
2. Appearance truth must support both continuous material-color surfaces and textured surfaces.
3. Material cardinality is not an ownership cardinality proxy: George has one material but many mechanical/geometric parts.
4. Texture resolution varies by 64x per axis in the textured subset (32 -> 2048); A/CAO teacher sampling must be surface-addressed rather than assuming fixed source texel density.

---

## 4. Skin / articulation census

### 4.1 FBX raw authored weights

| Subject | Nonempty clusters | Weighted vertices | Max influences | Raw weight-sum min | max | mean | vertices with `abs(sum-1)>1e-3` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snake | 12 | 815 | 4 | 0.2000 | 1.8108 | 0.9853 | 255 |
| T-Rex | 28 | 912 | **7** | 0.8501 | 1.0160 | 0.9860 | 758 |
| Spider | 35 | 1,386 | **6** | 0.9044 | 1.0122 | 0.9945 | 506 |
| Shark | 8 | 324 | 4 | 0.8488 | 1.0000 | 0.9848 | 289 |

These are raw authored FBX cluster weights. RealSaS must preserve them as authored evidence. If a mechanical training target requires normalized per-vertex weights, the normalized tensor is a **derived artifact with explicit provenance**, not a silent rewrite of raw truth.

A hard-coded 4-influence teacher contract is invalid: T-Rex reaches 7 and Spider reaches 6.

### 4.2 glTF weights

The four glTF exports use normalized `WEIGHTS_0` to numerical precision and max four stored influences in the inspected exports:

- Bat: 15 joints;
- Horse: 50 joints;
- GreenBlob: 4 joints;
- George: 47 joints.

The difference between FBX raw cluster weights and glTF normalized export weights is itself provenance. Format normalization must not be mistaken for original authoring truth.

---

## 5. Connected component versus deformation mode

A major finding is that geometric component identity and deformation behavior are related but not equivalent.

### George

Diagnostic welded components: 61.

- 53 / 61 components are dominated by exactly one joint and are 100% one-hot weighted in the inspected glTF export.
- Examples include components owned rigidly by `UpperArm.L`, `LowerArm.L`, `UpperArm.R`, `LowerArm.R`, `UpperLeg.L/R`, `Shoulder.L/R`, `Head`, `Neck`, `Chest`.
- The asset is still one mesh / one material.

This provides unusually clean deterministic teacher evidence for **rigid bone-owned surface components** without requiring object/material names.

### Horse

One welded geometric component with 45 observed dominant joints across the surface.

- mean dominant normalized weight ~0.790;
- minimum dominant normalized weight ~0.251;
- only ~20.5% of render-vertex references exceed 0.99 dominant weight.

Horse is a canonical smooth-deformation case rather than a rigid-component case.

### GreenBlob

Two welded components:

- main 1,152-face component is smoothly blended across four joints, mean dominant weight ~0.632;
- secondary 368-face component is fully one-hot to `Head`.

One subject therefore naturally mixes smooth-deforming and rigid-like surface components.

### Bat

Twelve welded components. Most small head/detail components are one-hot, while the wing/body structures include blended multi-joint support. This gives a useful thin-structure + mixed-deformation stress case.

### Contract consequence

Keep these axes separate:

```text
geometric connected component
owner / presentation region
material region
deformation mode
mechanical owner joint
attachment relation
```

In particular:

> `RIGID_BONE_OWNED` is not the same statement as `ATTACHMENT`.

An attachment relation requires authored parent/constraint/socket evidence or separately qualified relational evidence. A one-hot body panel must not be mislabeled as a detachable/held attachment.

---

## 6. Animation corpus present in FIT8

Native animation coverage is already heterogeneous:

- Snake: 4 clips;
- Bat: 4 clips;
- T-Rex: 6 clips;
- Horse: 13 clips;
- Spider: 5 clips;
- GreenBlob: 9 clips;
- Shark: 1 clip;
- George: 20 clips.

This supports dynamic teacher sampling during FIT8 architecture closure without expanding to the 139-subject corpus yet.

However:

```text
animation exists
!= authored contact labels exist
!= authored draw-order labels exist
```

Dynamic contact/overlap generated from geometry and motion must retain `DERIVED_FROM_AUTHORED_MOTION` provenance. Explicit authored constraints/sockets, when present, can carry stronger authority.

---

## 7. Truth-schema consequences before freeze

The current FIT8 full-truth schema must represent at least the following distinct source/derived domains.

### Geometry identity

```text
RAW_RENDER_TOPOLOGY
RAW_SOURCE_POLYGON_TOPOLOGY
GEOMETRIC_EQUIVALENCE / WELD EVIDENCE
COMPONENT_TOPOLOGY_CLASS
SIGN_VALIDITY
TRIANGULATION_PROVENANCE
```

Do not silently turn glTF render indices or diagnostic welding into canonical mechanical topology.

### Geometry field truth

A universal metric global signed distance is not guaranteed by FIT8.

Teacher queries should distinguish:

```text
unsigned distance                  broadly valid
local oriented/sign evidence       validity-gated
closed-component signed distance   component-validity-gated
global sign / nesting              separately qualified
```

This preserves legacy IRIS signed-field strength where justified without fabricating sign truth for open/non-manifold authored surfaces.

### Skin truth

```text
RAW_AUTHORED_SKIN
DERIVED_NORMALIZED_SKIN
DEFORMATION_MODE
MECHANICAL_OWNER_JOINT
```

Each has separate provenance/validity.

### Appearance truth

```text
material color / material parameters
UV + texture when authored
continuous surface RGBA
source-view exact samples
hidden/decomposition-state RGBA
```

### Semantic / ownership truth

Primary truth is factored, not one flat global class ID:

```text
source object identity
geometric component identity
material identity
mechanical ownership
deformation mode
explicit attachment relation
subject-local owner/material decomposition
```

### Dynamic truth

```text
posed geometry
deformation consequences
overlap / ray-hit order
contact evidence with provenance
appearance/presentation context
```

---

## 8. Immediate architecture implications

1. **Generic semantics are justified by actual data, not philosophy alone.** Mech, blob, animal and thin-wing subjects all require the same factored contract.
2. **S cannot be a human-class segmentation head.** Owner/material/deformation relations have independent cardinalities.
3. **G and F need different notions of field validity.** Metric/sign geometry truth and extraction level-set behavior must not be conflated.
4. **T needs authored dynamic consequence supervision.** Source topology is not automatically the desired runtime carrier, but source motion provides teacher consequences.
5. **CAO needs appearance decomposition beyond material slots.** George proves one material may cover dozens of mechanically distinct parts.
6. **Raw and derived truth must coexist.** Import normalization/welding/triangulation/skin normalization are explicit transformations, never silent teacher facts.

---

## 9. Still open after this census

The census deliberately does not freeze:

- owner/material region slot capacity;
- component merge/split rules;
- global sign authority algorithm;
- F coarse/fine extraction resolution;
- T retopology decoder family;
- dynamic pose sampling count;
- contact-capability ontology;
- CAO fragment granularity.

Those decisions should consume this census plus rendered/ray-hit/deformation measurements rather than being guessed now.

No scientific or product PASS is claimed.