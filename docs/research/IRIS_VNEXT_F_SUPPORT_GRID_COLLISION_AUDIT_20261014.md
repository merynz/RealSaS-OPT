# IRIS-vNext F support-grid collision audit

**Date:** 2026-10-14  
**Status:** active representation research — NOT final F freeze, NOT scientific PASS  
**Trigger:** Phase-1 rendered census showed visually significant open/two-sided support in Bat and George.

This audit drills one level below the high-level `SDF vs UDF vs sparse support` question.

The concrete question is:

> If F uses a sparse active support grid plus one local scalar/level-set field, what grid pitch is required before separate authored surface sheets stop collapsing into the same extraction cells?

This is intentionally separated from the final extraction-resolution question. A grid can preserve surface continuity yet still contain multiple distinct source sheets/components in one active cell.

---

## 1. Method

The first prototype used independent per-component voxelizers and produced non-monotonic fake component splits at some resolutions. That method was rejected.

The corrected diagnostic uses one **common canonical grid per subject** and exact triangle-versus-axis-aligned-cell SAT intersection tests.

Pipeline:

```text
exact source / diagnostic-equivalence triangles
        |
canonical RealSaS normalization
        |
common N x N x N grid over [-1,1]^3
        |
triangle-cell SAT intersection
        |
per-cell set of source component IDs
```

For every candidate N the audit measures:

- union surface-support cell count;
- active-cell fraction versus dense N^3;
- source-component connectivity under 26-neighbor support;
- cells containing more than one source component;
- maximum source components in one active cell;
- persistence of colliding component pairs as N increases.

The glTF component identities remain diagnostic position-equivalence components until full geometric-equivalence qualification closes. The grid-intersection calculation itself is deterministic.

---

## 2. Continuity result

With the corrected common-grid SAT method, every tested source component remains one connected support-cell component at every tested resolution.

Tested subjects include Bat, George, Spider and Snake across 48, 64, 96 and 128.

Therefore the earlier apparent 48/96 component splits were voxelizer-grid artefacts, not source geometry facts.

This is important operationally:

```text
support continuity
!=
component separation
```

The exact active support can stay connected while distinct nearby surfaces still share extraction cells.

---

## 3. Bat: support sparsity versus multi-component collision

| Grid N | Dense cells N^3 | Active support cells | Active fraction | Multi-component active cells |
|---:|---:|---:|---:|---:|
| 48 | 110,592 | 2,725 | 2.464% | 12.697% |
| 64 | 262,144 | 4,931 | 1.881% | 9.653% |
| 96 | 884,736 | 11,288 | 1.276% | 7.380% |
| 128 | 2,097,152 | 20,232 | 0.965% | 6.144% |
| 160 | 4,096,000 | 32,113 | 0.784% | 5.300% |
| 192 | 7,077,888 | 46,542 | 0.658% | 4.585% |
| 256 | 16,777,216 | 83,121 | 0.495% | 3.674% |
| 384 | 56,623,104 | 189,126 | 0.334% | 2.644% |
| 512 | 134,217,728 | 338,085 | **0.252%** | **2.084%** |

At every tested resolution the maximum observed source-component count in one active cell is 3.

The same 13 colliding component pairs seen at N=128 are still present at N=256 and N=512.

This makes a pure raster/grid-noise explanation unlikely. Some authored Bat structures remain genuinely near/contacting/intersecting at scales much smaller than a 128-class extraction cell.

---

## 4. George: support sparsity versus multi-component collision

| Grid N | Dense cells N^3 | Active support cells | Active fraction | Multi-component active cells |
|---:|---:|---:|---:|---:|
| 48 | 110,592 | 2,188 | 1.978% | 30.165% |
| 64 | 262,144 | 4,142 | 1.580% | 26.026% |
| 96 | 884,736 | 10,104 | 1.142% | 16.766% |
| 128 | 2,097,152 | 18,613 | 0.888% | 13.689% |
| 160 | 4,096,000 | 29,769 | 0.727% | 11.119% |
| 192 | 7,077,888 | 43,651 | 0.617% | 9.361% |
| 256 | 16,777,216 | 80,223 | 0.478% | 6.401% |
| 384 | 56,623,104 | 184,532 | 0.326% | 4.343% |
| 512 | 134,217,728 | 331,958 | **0.247%** | **3.036%** |

The maximum source-component count in one active cell falls from 4-5 at coarse grids to 3 at high resolution, but does not reach 1.

Collision-pair persistence:

- N=128: 91 distinct component pairs share at least one support cell;
- N=256: 82 pairs;
- N=512: 74 pairs;
- 74 / 91 = approximately **81.3%** of N=128 collision pairs persist even at N=512;
- approximately **90.2%** of N=256 collision pairs persist at N=512.

This is strong evidence that the issue is not solved by a modest global resolution increase.

---

## 5. Why this matters for a single scalar field

A support grid answers:

```text
where may surface exist?
```

A scalar/level-set field answers something closer to:

```text
where inside this local domain is an isosurface crossing?
```

If one active cell contains multiple distinct authored sheets/components, a single corner-sampled scalar field may not have enough local degrees of freedom to reconstruct all of them without aliasing, merging or losing one sheet.

This is especially dangerous for RealSaS because those nearby sheets are often not cosmetic duplicates. They may have different:

- mechanical owners;
- deformation modes;
- appearance regions;
- draw-order relations;
- attachment semantics;
- front/back visibility behavior.

Therefore:

> `explicit surface support` is necessary but may not be sufficient.

The next representation question is **local sheet multiplicity**.

---

## 6. The 64 vs 96 vs 128 question changes shape

The evidence does not support choosing one of 64/96/128 in isolation.

For George:

```text
64  -> 26.0% multi-component support cells
96  -> 16.8%
128 -> 13.7%
256 ->  6.4%
512 ->  3.0%
```

For Bat:

```text
64  -> 9.7%
96  -> 7.4%
128 -> 6.1%
256 -> 3.7%
512 -> 2.1%
```

A 128-class grid is therefore not obviously a sufficient generic solution if F is represented by one unconditioned local field.

But the sparse occupancy result simultaneously shows why a much higher **support** resolution can be computationally plausible:

- N=512 dense domain: ~134.2 million cells;
- Bat active surface cells: ~338k;
- George active surface cells: ~332k;
- active fraction: ~0.25%.

This strongly supports sparse high-resolution querying rather than dense 512-class evaluation.

---

## 7. F representation subtree opened by the result

The current F decision should branch into at least four explicit families.

### F-A — one sparse support grid + one shared local scalar field

Advantages:

- closest to InstantMesh / StdGEN / FlexiCubes substrate;
- simplest shared extraction implementation;
- high-resolution sparse domain is computationally attractive.

Failure risk:

- multiple nearby surface sheets may alias inside one cell.

Freeze requirement:

- demonstrate that N/pitch can reduce multi-sheet reconstruction failure to an acceptable level without impractical query cost.

### F-B — region-conditioned semantic-equivalent extraction

Shared G/S field, but extraction is conditioned on a subject-local owner/material/component region hypothesis.

Conceptually:

```text
shared surface support
+ shared local geometry field
+ region membership / affinity
        |
        +--> F_region_0
        +--> F_region_1
        +--> ... subject-local queries
        |
qualified merge / binding
```

Potential advantage:

Two nearby sheets can be represented separately even when they occupy the same coarse support cell because the extraction query is conditioned on different region evidence.

This is a generic translation of StdGEN's semantic-equivalent extraction rather than a copy of fixed hair/cloth/body categories.

Risk:

- region partition errors can create duplicate/missing geometry;
- merge/stitch authority must remain Compiler-owned;
- variable cardinality must stay subject-local.

### F-C — multi-sheet local cell representation

Predict more than one local surface primitive / level-set branch per active cell.

Potential advantage:

- directly represents sheet multiplicity without requiring a semantic partition first.

Costs:

- much larger architecture departure from FlexiCubes;
- permutation/cardinality problem per cell;
- harder differentiable extraction and topology assembly.

### F-D — adaptive local refinement around collision-risk cells

Use moderate sparse global support, then recursively refine cells with evidence for:

- multiple owners/components;
- high geometry uncertainty;
- thin/open surfaces;
- contact/overlap criticality.

This may be combined with F-A or F-B.

---

## 8. Current research ordering

No final freeze is made, but the evidence changes the priority ordering.

Current strongest research path:

```text
sparse high-resolution support
        +
local signed/oriented field where valid
        +
region-conditioned extraction or collision-risk refinement
```

This is more specific than the earlier generic `SparseFlex-inspired` statement.

The current evidence argues **against** both extremes:

```text
"just use one dense SDF grid"
```

and

```text
"replace everything with a global UDF"
```

The problem appears to be factored:

```text
1. where surface exists
2. what local distance/orientation evidence is valid
3. how many distinguishable surface sheets occupy the local extraction domain
4. which owner/material relation each sheet belongs to
```

---

## 9. Required next measurement before F freeze

The collision statistic alone cannot distinguish intentional contact from unresolved parallel sheets.

For every persistent collision pair, Phase 2 should classify:

```text
EXACT_INTERSECTION
AUTHORED_CONTACT / NEAR-CONTACT
PARALLEL_LAYERED_SURFACES
SEAM / EXPORT DUPLICATION
SEMANTICALLY_DISTINCT_BUT_GEOMETRICALLY_CONTIGUOUS
UNKNOWN
```

Required evidence:

- exact source triangle-triangle minimum distance/intersection;
- source object/component/material identities;
- mechanical owner relation;
- dynamic persistence under authored motion;
- canonical-view all-hit relation;
- front/back normal orientation;
- whether region-conditioned extraction separates the sheets at lower N.

Only after that classification should RealSaS decide whether the final solution is mainly:

- higher sparse resolution;
- semantic/owner-conditioned extraction;
- local adaptive refinement;
- or a genuinely multi-sheet cell representation.

---

## 10. Current conclusion

The support-grid audit gives two simultaneous results:

1. **Sparse high-resolution F is strongly justified by measured support sparsity.**
2. **Resolution alone does not remove local multi-sheet ambiguity fast enough to be assumed sufficient.**

The design target therefore becomes more precise:

```text
not merely:
    high-res SDF

but:
    high-res sparse surface support
    + validity-aware local geometry field
    + explicit handling of local sheet multiplicity
```

This remains an open architecture branch, not a product or scientific PASS.
