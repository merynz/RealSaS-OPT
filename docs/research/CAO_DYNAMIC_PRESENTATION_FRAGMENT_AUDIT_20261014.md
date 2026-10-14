# CAO dynamic presentation-fragment audit

**Date:** 2026-10-14  
**Status:** active presentation architecture research — no runtime-format freeze, no product PASS  
**Scope:** FIT8 collision-pair dynamic overlap evidence -> CAO draw-order granularity

This audit asks a narrower question than `draw order is dynamic`:

> Is one draw-order relation per material, owner, component or slot sufficient, or does CAO need multiple canonical-M-addressed presentation fragments inside the same relation pair?

The answer is measured from authored motion and canonical camera evidence on Bat and George collision-critical component pairs.

The diagnostic does **not** promote source connected components to final CAO owners. They are used only as an available exact/diagnostic partition for measuring whether coarser order primitives can represent the observed projection.

---

## 1. Why pair-level front/back is not enough

A conventional slot/order representation assumes something like:

```text
for context (pose, view):
    A FRONT_OF B
or
    B FRONT_OF A
```

That is sufficient only if the overlap between A and B has one consistent ordering over the entire projected overlap region.

The critical counterexample is:

```text
same pose
same canonical view
same pair A/B

region 1: A in front
region 2: B in front
```

No single pairwise order bit or slot permutation can reproduce that image without splitting at least one side into presentation fragments.

This is independent of whether the geometry is correct in 3D.

---

## 2. Dynamic pair court

Prototype context:

### Bat

- 13 persistent collision-critical component pairs;
- 4 native clips;
- sampled fractions: 0, 0.5, 1.0;
- 8 canonical views;
- 12 pose contexts × 8 views = 96 view/pose contexts per pair;
- diagnostic pair raster court at 64x64.

### George

- 82 persistent N=256 collision-critical component pairs;
- selected authored clips/fractions for the pair-local court;
- 8 canonical views;
- diagnostic pair raster court at 64x64.

The court records, within each projected pair overlap:

```text
A-front pixels
B-front pixels
coincident/ambiguous pixels
connected regions of the A-front/B-front sign map
```

`same dominant joint` is diagnostic mechanical-owner evidence only; it is not attachment or presentation authority.

---

## 3. Owner/material cannot explain the ordering complexity

### Material

Bat and George each use one inspected material set for all persistent pair rows in this court.

Therefore material identity contains no information capable of distinguishing the observed local ordering changes.

A material-level draw-order primitive is rejected for these cases.

### Dominant mechanical owner

Bat pair buckets:

```text
3  exact/contact + different owner + order flip
8  near + different owner + order flip
2  near + same owner + order flip
```

George pair buckets:

```text
7   exact/contact + different owner + order flip
28  near + different owner + order flip
5   exact/contact + same owner + order flip
42  near + same owner + order flip
```

Thus George has:

```text
47 / 82
```

persistent collision pairs whose two source components share the same dominant mechanical owner.

This means:

> mechanical owner is not fine enough to define geometric local sheet identity or presentation order identity.

Owner remains useful evidence, but CAO cannot equate `owner == drawable/order fragment`.

---

## 4. Same-view mixed ordering is common

The stronger test is not whether A is front in one view and B in another. That is ordinary view-dependent occlusion.

The stronger test is whether both directions occur **inside one view/pose overlap map**.

Measured diagnostic result:

### Bat

```text
13 / 13 persistent pairs
```

show at least one same-view/pose context containing both A-front and B-front regions.

Different-owner subset:

```text
11 / 11
```

also exhibits same-context mixed ordering.

### George

```text
69 / 82 persistent pairs
```

show same-view/pose mixed ordering.

Different-owner subset:

```text
33 / 35 ~= 94%
```

shows same-view/pose mixed ordering.

These are not rare one-pixel reversals. Across different-owner rows the minority ordering direction occupies a substantial fraction of overlap pixels; diagnostic medians are roughly one third of overlap support.

Therefore a single order relation per owner-pair is structurally insufficient for these cases.

---

## 5. Resolution sweep rejects a simple raster-alias explanation

Selected pairs were re-evaluated at 64, 128 and 256 resolution.

Representative examples:

### Bat Wing1.R / Head near pair

```text
minority direction fraction
64:  ~34.1%
128: ~33.1%
256: ~33.2%
```

Mixed ordering persists across five canonical views in the high-resolution court.

### Bat Wing1.R / Head exact/contact pair

```text
~37.8% -> ~37.4% -> ~37.4%
```

### George Neck / Chest exact pair

```text
~24.5% -> ~24.1% -> ~23.6%
```

### George Shoulder.R / Neck near pair

At 64 the coarse raster fails to expose all mixed contexts. At 128/256 the mixed relation becomes visible and the minority fraction remains near 48%.

Thus lower raster resolution can actually **hide** ordering complexity rather than create it.

The fragment requirement cannot be dismissed as 64x64 aliasing.

---

## 6. Fragment cardinality is modest, not unbounded

Needing fragments does not imply that CAO should materialize hundreds of drawable pieces.

The diagnostic sign maps were decomposed into connected A-front/B-front regions. To reduce one-pixel noise, a robust count also ignores regions smaller than three pixels at the 64x64 court resolution.

### Bat

Across 13 persistent pairs:

```text
max robust order fragments in one pair/view/pose: 3
95th percentile of per-pair robust maxima:        3
```

Different-owner pairs also reach a robust maximum of 3.

### George

Across 82 persistent pairs:

```text
max robust order fragments in one pair/view/pose: 4
95th percentile of per-pair robust maxima:        3
```

Different-owner subset:

```text
max robust: 3
p95 of pair maxima: 2
```

Same-owner subset:

```text
max robust: 4
p95 of pair maxima: 3
```

This suggests a useful design target:

> CAO must be fragment-capable, but the meaningful local order complexity in this FIT8 diagnostic is usually small.

Do **not** freeze `K=4` as a universal runtime cap from this small cohort. The measurement only says that a bounded sparse fragment representation is plausible.

---

## 7. What a CAO presentation fragment is

A presentation fragment must **not** become a second visual mesh authority.

Proposed conceptual identity:

```text
PresentationFragment =
    subset of canonical-M surface addresses
    + appearance/owner evidence
    + context-conditioned order relations
```

It may be represented as:

- face subset;
- barycentric surface interval/set;
- compiler-qualified connected canonical-M region;
- compact runtime index set derived from one of the above.

The canonical geometry remains M.

A fragment is therefore closer to a **presentation ownership/address partition over M** than to an independent drawable mesh.

---

## 8. Context is at least view + pose

Static source observations already require direction-dependent appearance/order.

The dynamic court shows that pose changes overlap/order structure as well.

Therefore the order relation domain cannot be only:

```text
fragment_i x fragment_j
```

It needs a presentation context equivalent to:

```text
(fragment_i, fragment_j, view, pose/time)
```

The runtime does not necessarily store a full dense tensor over this domain. It may compile/simplify the relations into clip/frame intervals or rule-based transitions.

But the **authority/proof domain** must be capable of expressing view- and pose-conditioned precedence.

---

## 9. Proposed CAO order graph abstraction

Research candidate:

```text
canonical M
   |
Compiler/CAO-qualified presentation fragments
   |
per context sparse precedence graph
   |
fragment A --FRONT_OF--> fragment B
fragment C --FRONT_OF--> fragment A
...
```

Important properties:

1. graph nodes are M-addressed fragments, not independent geometry;
2. edges carry evidence/provenance/confidence;
3. raw camera depth is evidence, not sole authority;
4. source-exact ordering in observed source context has strong authority;
5. learned S relation evidence may propose hidden/dynamic ordering;
6. authored/posed geometry and all-hit rays independently check physical consistency;
7. cycles/contradictions are Compiler qualification failures or require explicit fragment refinement.

---

## 10. When coarse slots may still be used

Slots are not rejected. They become a **compiled compression** of the richer fragment relation domain.

A slot/owner pair can collapse to one order relation when CAO proves that over the relevant context range:

```text
all overlapping fragment support has the same precedence
AND
no significant mixed-order region exists
AND
fragment split is not needed for source-exact appearance ownership
```

This gives an important optimization rule:

```text
rich authority representation
        -> qualification
        -> collapse uniform regions/relations
        -> compact runtime slots/permutations where valid
```

Do not make the compact runtime representation the truth ontology.

---

## 11. Relation to generic appearance decomposition

The same presentation fragment may need to carry:

```text
owner descriptor
material descriptor
local sheet/geometry binding
source appearance provenance
amodal support
front/back side evidence
opacity/transparency evidence
attachment/contact relation
```

But these axes must remain factored.

A fragment is not defined merely because two pixels share one material, owner or bone.

This is particularly important for George:

```text
one material
many geometric/mechanical parts
same-owner local sheet collisions
mixed same-view order
```

Generic appearance decomposition therefore has to consume S/G/F relation evidence rather than being material-atlas segmentation.

---

## 12. Link to F local-sheet representation

The latest F audit shows that local geometric sheet identity is finer than global component/owner/material identity.

CAO should not independently rediscover this identity from raster output if F can carry exact local patch/source bindings forward.

Preferred lineage:

```text
teacher authored local patch/sheet evidence
        |
IRIS F high-fidelity surface with patch/source binding
        |
T -> canonical M while preserving relevant bindings
        |
CAO presentation-fragment qualification on M
```

This does **not** require F patch = CAO fragment.

They solve different problems:

```text
F patch:
    local geometric representability / sheet continuation

CAO fragment:
    local presentation ownership / order / appearance
```

But shared bindings allow CAO to split/merge intentionally instead of rematching geometry after the fact.

---

## 13. Visible-coherence consequence

The historical failure

```text
M mechanically/static qualified
but render has holes / fragmented alpha / peppering
```

should be impossible to hide behind a single draw-order permutation.

For every view/pose context CAO qualification should compare:

```text
qualified visible M support
vs
appearance-covered support
vs
presentation fragment coverage/order
```

A missing region must resolve to one explicit state:

```text
VISIBLE_WITH_QUALIFIED_APPEARANCE
OCCLUDED_BY_QUALIFIED_FRAGMENT
INTENTIONALLY_TRANSPARENT
INTENTIONALLY_MASKED
```

Anything else is unexplained visible-support loss and fails the CAO visible-coherence court.

---

## 14. Current decision state

### Resolved by diagnostic evidence

1. material-level order is insufficient.
2. mechanical-owner-level order is insufficient.
3. one pairwise order bit per source component pair is insufficient in many collision-critical contexts.
4. CAO authority must be capable of fragment-level ordering on canonical M.
5. view + pose are both relevant presentation context axes.
6. fragment complexity appears sparse/modest enough to justify a compact sparse graph rather than dense per-pixel runtime authority.

### Still open

1. exact fragment-construction algorithm;
2. whether fragment seeds should originate from F sheet bindings, S owner/affinity, all-hit sign regions, or a joint court;
3. minimum region-size/antialias qualification independent of raster resolution;
4. graph cycle-resolution policy;
5. clip/frame compression strategy;
6. exact runtime format;
7. whether learned S directly predicts pairwise order or predicts descriptors consumed by Compiler/CAO;
8. fragment cardinality capacity beyond FIT8.

---

## 15. Next court

Before freezing CAO fragment representation:

1. generate full-resolution all-hit truth on selected FIT8 static/dynamic contexts;
2. bind every order region back to source/F/M surface addresses rather than only raster connected components;
3. measure fragment persistence through neighboring views and motion frames;
4. compare fragment-seeding strategies;
5. run the same visible-coherence court on Knight to test whether the historical elbow/cape and hole/peppering failures map to the new representation.

The target is not maximum decomposition.

The target is the **minimum canonical-M-addressed fragment graph that exactly explains qualified 2D presentation while preserving one geometry authority.**
