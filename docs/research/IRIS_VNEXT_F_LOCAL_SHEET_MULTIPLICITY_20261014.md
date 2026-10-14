# IRIS-vNext F local-sheet multiplicity and support-query audit

**Date:** 2026-10-14  
**Status:** active architecture refinement — no final representation freeze  
**Depends on:** `IRIS_VNEXT_F_SUPPORT_GRID_COLLISION_AUDIT_20261014.md`

The support-grid collision audit established two facts at once:

1. surface support is extremely sparse at high spatial resolution;
2. some active cells contain multiple genuinely near/contacting authored components even at very high grid resolution.

This document asks what that means for the actual learned representation.

---

## 1. Persistent collision pairs are geometrically real

The corrected common-grid SAT audit identified persistent Bat/George component pairs that continue to occupy common support cells at high resolution.

A triangle-level distance diagnostic was then run on those pairs using the exact normalized source geometry.

### Bat

13 persistent component pairs were examined.

Diagnostic distance classes:

- 3 / 13: `INTERSECT_OR_EXACT_CONTACT`;
- 8 / 13: separated by less than half of a 512-cell width;
- 2 / 13: separated by less than one 256-cell width.

Therefore:

```text
13 / 13
```

persistent Bat collision pairs are geometrically closer than one 256-cell width, and most are substantially closer.

### George

82 persistent N=256 collision pairs were examined.

Diagnostic distance classes:

- 12 / 82: `INTERSECT_OR_EXACT_CONTACT`;
- 62 / 82: separated by less than half of a 512-cell width;
- 6 / 82: separated by less than one 256-cell width;
- 2 / 82: larger separation but extended triangles still occupy common cells.

Thus:

```text
74 / 82 ~= 90.2%
```

are exact/contact or closer than half a 512-cell width, and

```text
80 / 82 ~= 97.6%
```

are closer than one 256-cell width.

The current collision problem is therefore not primarily a voxel-grid alignment artefact.

---

## 2. Consequence: resolution cannot separate exact contact/intersection

Increasing N helps when two surfaces are merely close.

It cannot make the following cease to coexist locally:

```text
exact contact
true intersection
coincident/near-coincident layered surfaces
```

If those surfaces must retain different owner/material/mechanical identities, a representation that relies only on one unconditioned scalar field must prove that it can preserve them.

This is especially relevant to RealSaS because contact/near-contact zones are precisely where downstream failures matter most:

- grip;
- elbow/chest or limb/body overlap;
- armor/body layering;
- wing/body junctions;
- rigid-panel articulation;
- source-exact appearance ownership;
- draw-order switching under motion.

---

## 3. Correcting the earlier `s(x)` abstraction

Earlier research notes used a conceptual field:

```text
s(x) = surface existence/support
```

That notation is useful conceptually but is not the best teacher target.

A mathematical surface has measure zero in 3D. A binary pointwise label `surface exists exactly at x` is therefore not a stable dense learning target without introducing an arbitrary support radius.

The exact authored truth available to RealSaS is naturally **scale-aware cell support**:

```text
A(c, h) = 1
iff the authored/qualified surface intersects cell c
with half-width / pitch h
```

This is deterministic from source triangles and does not require a watertight inside/outside notion.

Therefore the cleaner factorization is now:

```text
G continuous point field
    u(x)  unsigned/proximity evidence
    f(x)  signed or locally-oriented level-set evidence where valid
    q(x)  sign/orientation validity
    z_g(x) geometry feature / uncertainty

F sparse support proposal
    A(c,h) surface-intersects-cell probability
    R(c,h) region/owner/material affinity evidence
    M(c,h) local sheet-multiplicity risk

F local extraction field
    scalar / deformation / FlexiCubes configuration
    queried only inside admitted sparse cells
```

This avoids forcing a scale-dependent active-cell question into a scale-free point-field contract.

---

## 4. Teacher truth can supervise A(c,h) exactly

Unlike many image-to-3D systems, RealSaS owns authored source geometry during teacher construction.

For any sampled cell:

```text
cell center
cell pitch / half extent
exact triangle-cell SAT intersection
source triangle addresses
geometric component IDs
owner/material region IDs
surface class
local normal/orientation validity
```

can be generated deterministically or under explicit qualification.

This enables multi-resolution training pairs such as:

```text
(cell, pitch=2/64)  -> support + region set
(cell, pitch=2/128) -> support + region set
(cell, pitch=2/256) -> support + region set
(cell, pitch=2/512) -> support + region set
```

without making one grid resolution a permanent teacher ontology.

---

## 5. Local multiplicity is not identical to component count

A cell containing two source component IDs is evidence of possible local sheet multiplicity, but the two concepts must not be equated.

Examples:

- two components may meet at one intentional contact seam and form one local geometric sheet;
- one component may fold so that two separate parts of the same surface enter one cell;
- two components may be parallel layered surfaces;
- two components may truly intersect;
- render-seam duplication may create identity multiplicity without geometric multiplicity if equivalence qualification is incomplete.

Therefore the final teacher target should not simply be:

```text
M = number_of_component_ids_in_cell
```

Instead Phase 2 should derive a qualified local sheet relation from triangle geometry plus source identity.

Candidate outputs:

```text
sheet_count_lower_bound
multi_sheet_risk
region_set
pair relation:
    CONTACT
    INTERSECT
    NEAR_PARALLEL
    CONTIGUOUS_SEAM
    UNKNOWN
```

---

## 6. Representation families after the geometry-distance audit

### A. Resolution-only single field

The audit weakens this family.

Reason:

- exact contact/intersection cannot be removed by increasing resolution;
- 512-class support still contains multi-component cells;
- contact-critical areas are product-critical rather than negligible.

Keep as a baseline, not current preferred solution.

### B. Global sparse support + subject-local region-conditioned extraction

This is now the strongest research candidate.

Conceptually:

```text
shared scene / triplane memory
       |
       +--> G point field
       |
       +--> sparse support A(c,h)
       |
       +--> S region/affinity evidence
                    |
                    v
        collision-risk cell
          /        |       \
      region a  region b  ...
          |        |
       local F   local F
          \        /
          Compiler qualification
```

Why it fits RealSaS:

- does not require global anatomical classes;
- uses subject-local owner/material/component evidence;
- preserves one shared scene field;
- retains FlexiCubes-style local extraction where possible;
- only pays decomposition cost where geometry/semantic evidence requires it;
- Compiler can reject/merge/stitch region proposals with provenance.

### C. Multi-sheet local primitive prediction

Still a valid research alternative, especially where region partition is uncertain.

Possible cell output:

```text
K local surface branches
+ branch confidence
+ local orientation
+ region embedding
```

But this creates a harder variable-cardinality/permutation problem and is a larger departure from the InstantMesh/StdGEN substrate.

### D. Adaptive octree/refinement only

Useful but insufficient as the sole answer to exact contact/intersection.

Adaptive refinement can resolve near surfaces; it cannot separate genuinely coincident/intersecting semantic surfaces by resolution alone.

---

## 7. A likely hybrid worth testing

A promising architecture court is:

```text
1. moderate sparse global support proposal
2. high-resolution subdivision around admitted surface cells
3. multi-sheet-risk prediction from G + S evidence
4. ordinary cells -> one shared FlexiCubes-style extraction
5. risk cells -> subject-local region-conditioned extraction
6. exact binding of every extracted patch back to shared field/source evidence
7. Compiler qualification / merge / stitch
```

This keeps the common case simple while giving contact/layered zones additional representational capacity.

It is important that step 5 is **not** a hidden second visual mesh system. Every accepted patch remains part of the one canonical geometry proposal and carries exact shared-field/source bindings.

---

## 8. Consequence for the teacher schema

The current full-truth schema contains point queries for unsigned/signed distance but does not yet explicitly encode scale-aware extraction-cell support or local region multiplicity.

Before first IRIS-vNext training, the schema should be reviewed for channels equivalent to:

```text
extraction_cell_query
    center xyz
    pitch / half extent

extraction_cell_surface_support
    exact triangle-cell intersection truth

extraction_cell_source_addresses
    source faces/triangles intersecting the cell

extraction_cell_region_set
    qualified owner/material/component evidence in the cell

extraction_cell_local_sheet_relation
    qualified multiplicity/contact/intersection relation
```

The exact naming and authority class remain open until the local-sheet qualification algorithm is specified.

Do not prematurely add `field_surface_exists: bool[Q]` as if surface support were a scale-free pointwise truth.

---

## 9. Consequence for S and CAO

The same cells that are difficult for F are high-value for presentation.

A multi-sheet-risk cell often implies one or more of:

- owner boundary;
- potential draw-order relation;
- contact/attachment proximity;
- source appearance ownership boundary;
- two-sided/open support;
- mechanical ownership transition.

Therefore support/multiplicity evidence should be available to S/CAO qualification rather than discarded after mesh extraction.

This is another reason geometry, semantics and appearance should remain co-addressed in the shared scene domain.

---

## 10. Next exact court

Before promoting region-conditioned extraction to the locked architecture, persistent collision pairs should be labeled using:

```text
triangle-triangle intersection / minimum distance
+ source object/component/material identity
+ normalized mechanical owner
+ authored/dynamic motion persistence
+ canonical all-hit ordering
+ front/back orientation
```

The decisive comparison should then train/evaluate under matched compute:

```text
F-A: sparse single-field
F-B: sparse + region-conditioned extraction
F-C: sparse + multi-sheet local branches
```

Metrics:

- source-surface recall;
- false bridges;
- missing sheets;
- open-boundary fidelity;
- semantic/owner boundary fidelity;
- static 8-view error;
- dynamic mechanical consequence;
- CAO visible coherence;
- query count / peak memory.

No final F family is frozen yet.
