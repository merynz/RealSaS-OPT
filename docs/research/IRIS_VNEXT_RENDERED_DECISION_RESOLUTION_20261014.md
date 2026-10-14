# IRIS-vNext rendered/local-patch decision resolution

**Date:** 2026-10-14  
**Status:** architecture-decision evidence; partial leaf resolution only  
**Depends on:** FIT8 Phase-1 rendered/mechanical census, F support/collision/local-patch audits, CAO presentation-fragment audit  
**Does not claim:** final F implementation freeze, training PASS, product PASS

This addendum records only the architecture leaves resolved after the exact source census. It deliberately separates resolved representation requirements from implementation families that still require matched-budget experiments.

---

## 1. RESOLVED — scale-aware surface support belongs to F, not a binary point field

**State:** `LOCKED_BY_FIT8_MEASUREMENT` for the abstraction.

Earlier notes used the shorthand:

```text
s(x) = surface existence
```

The exact teacher geometry makes a cleaner target possible:

```text
A(c,h) = 1 iff qualified authored surface intersects cell c at pitch h
```

A surface is measure-zero in 3D; binary pointwise existence requires an arbitrary support radius. Exact triangle-cell intersection is naturally scale-aware and can be supervised at multiple pitches.

Updated factorization:

```text
G continuous point field:
  u(x) proximity/UDF evidence
  f(x) signed/local level-set evidence where valid
  q(x) sign/orientation validity
  geometry features/uncertainty

F sparse extraction domain:
  A(c,h) active surface support
  local patch/sheet evidence
  extraction topology + geometry parameters
```

The old conceptual `s(x)` may remain explanatory language but is not the preferred canonical teacher tensor.

---

## 2. RESOLVED — local surface sheet identity is independent of component/owner/material

**State:** `LOCKED_BY_FIT8_MEASUREMENT` for representation principle.

At N=128, local multi-patch cells exist even in globally single-component subjects:

```text
T-Rex: 154 same-global-component multi-patch cells, max local patches 4
Horse:   42 same-global-component multi-patch cells, max local patches 2
Shark:  170 same-global-component multi-patch cells, max local patches 2
```

George additionally has many multi-component and same-component local multiplicities; Snake/Spider/Bat/Blob also exhibit local multi-patch cells.

Therefore no equality assumption is allowed among:

```text
local surface sheet / patch
source object
connected component
material region
mechanical owner
presentation fragment
```

A local patch/sheet relation is a geometric factor in its own right.

---

## 3. RESOLVED — resolution-only single-crossing extraction is not a generic primary solution

**State:** `REJECTED_BY_CONTRACT` for the claim that increasing global grid N is sufficient to make one unconditioned edge crossing generically valid.

Exact teacher segment-triangle audits find multiple distinct authored surface crossings on the same extraction-cell edge. At N=128, multi-crossing cells are common inside collision-critical support regions for Snake, Spider, Bat and George. Selected problematic pairs retain high multi-crossing rates through N=512.

Exact contact/intersection cannot be removed by arbitrary refinement, and near-parallel sheets can require impractically high N before separation.

This does **not** reject sparse high-resolution extraction. It rejects only:

```text
sparse high resolution
+
one unconditioned crossing per edge
+
assume resolution will solve all sheet multiplicity
```

---

## 4. RESOLVED — preserve DMC/FlexiCubes strengths; the bottleneck is earlier than dual-vertex grouping

**State:** `LOCKED_BY_REFERENCE_AND_FIT8_MEASUREMENT`.

FlexiCubes/DMC can emit multiple dual contour vertices in one cube. Therefore multiple local patches in a cell do not justify discarding DMC.

The measured risk is the edge representation: a standard corner-scalar pair yields one interpolated zero crossing per grid edge, while FIT8 truth can contain multiple distinct crossings on one edge.

Required F research principle:

> preserve useful sparse DMC/FlexiCubes topology/configuration machinery where possible, while giving collision-critical cells enough local sheet/crossing capacity.

---

## 5. OPEN — F-B vs F-C

**State:** `OPEN_ABLATION_REQUIRED`.

Two strongest families remain:

### F-B — local-sheet-conditioned FlexiCubes

```text
sparse active cell
  -> K local sheet queries
  -> per-sheet scalar/orientation/crossing evidence
  -> FlexiCubes-compatible extraction
  -> Compiler stitch/merge qualification
```

### F-C — explicit multi-crossing / Hermite-DMC

```text
sparse active cell
  -> variable local patch tokens
  -> explicit edge crossing positions + normals/orientation
  -> DMC/manifold assembly
  -> Compiler qualification
```

F-A standard single scalar remains a required baseline but is no longer the preferred generic representation.

No universal K is frozen. FIT8 N=128 lower-bound local patch maxima happen to reach four, but this cohort cannot establish a product cap.

---

## 6. RESOLVED — TopoMesh mechanisms are high-value, universal watertight conversion is not

**State:** mixed `KEEP / TRANSLATE / REJECT`.

### KEEP / TRANSLATE

TopoMesh provides strong evidence for:

- explicit topology/geometry parameter separation;
- direct mesh-level supervision instead of render-only supervision;
- Teacher Forcing with GT topology while learning geometry;
- GT-guided sparse pruning to prevent early holes;
- progressive resolution training;
- sparse local-centric decoding.

RealSaS translation should supervise authored local patch/crossing structure directly wherever exact teacher authority exists.

### REJECT AS UNIVERSAL PREPROCESS

TopoMesh obtains unified DMC correspondence by remeshing arbitrary input into a watertight DMC-compliant volume. FIT8 proves open/two-sided authored surfaces are first-class and visibly significant.

Therefore:

```text
FORCE_ALL_TRUTH_TO_WATERTIGHT_VOLUME = FORBIDDEN
```

TopoMesh-style watertight DMC targets may be studied for closed-qualified components but cannot overwrite generic raw/qualified surface truth.

---

## 7. RESOLVED — CAO must be fragment-capable

**State:** `LOCKED_BY_FIT8_MEASUREMENT` at the authority level.

Dynamic collision-pair courts show same-view/pose regions in which the same pair contains both ordering directions.

Diagnostic evidence:

```text
Bat:    13/13 persistent collision pairs show same-context mixed ordering
George: 69/82 show same-context mixed ordering
George different-owner subset: 33/35 show mixed ordering
```

Resolution sweeps at 64/128/256 preserve substantial minority-order regions; low resolution can hide rather than create the effect.

Therefore these are insufficient as universal presentation authority:

```text
one order per material pair
one order per mechanical-owner pair
one order per component pair
one global slot permutation without fragment refinement
```

CAO must be able to qualify **canonical-M-addressed presentation fragments** and sparse context-conditioned precedence relations.

---

## 8. RESOLVED — fragment authority is rich; runtime may collapse it when proven uniform

**State:** `LOCKED_BY_MATCHED_REQUIREMENT`.

The authority representation should support:

```text
M-addressed fragment nodes
+ view/pose context
+ FRONT_OF / BEHIND relations
+ provenance/confidence
```

A compact slot/owner order is allowed only as a compiled optimization after uniform precedence is proven over the relevant fragment support/context.

This preserves a rich truth/qualification domain without requiring an unnecessarily expensive runtime representation everywhere.

---

## 9. FIT8 suggests bounded fragment complexity but does not freeze capacity

**State:** evidence only; exact capacity remains `OPEN_ABLATION_REQUIRED`.

At the current 64x64 diagnostic court, ignoring connected order regions smaller than three pixels:

```text
Bat:
  max robust fragments = 3
  p95 of per-pair robust maxima = 3

George:
  max robust fragments = 4
  p95 = 3
  different-owner max = 3
  same-owner max = 4
```

This makes a sparse bounded representation plausible.

It does **not** authorize a universal `K=4` fragment cap.

---

## 10. RESOLVED — F patch and CAO fragment are distinct but must remain co-addressed

**State:** `LOCKED_BY_ARCHITECTURE_REQUIREMENT`.

```text
F local patch:
  geometric local sheet identity / representability

CAO presentation fragment:
  appearance ownership / visibility / order identity
```

They are not required to be one-to-one.

But both must preserve exact/qualified lineage to the same shared surface/canonical M so CAO does not perform an anonymous post-hoc rematch.

Preferred lineage:

```text
source/teacher patch evidence
  -> IRIS F bindings
  -> T / Compiler canonical M bindings
  -> CAO split/merge presentation fragments on M
```

No second visual-mesh authority is introduced.

---

## 11. Updated first-F training philosophy

The first serious F experiment should exploit stronger teacher truth than typical image-to-3D systems.

Direct targets should be evaluated for:

```text
active-cell support
local patch cardinality/incidence
source triangle bindings
edge-crossing positions
crossing normals/orientation validity
neighbor patch continuation
owner/material/mechanical descriptors
```

alongside existing:

```text
RGB / alpha
silhouette
depth
normal
continuous field supervision
FlexiCubes/deformation regularization
```

The purpose is not to add every loss. The purpose is to test whether explicit teacher correspondence closes the exact failure classes observed in FIT8.

---

## 12. Open leaves after the rendered/local-patch phase

| Leaf | State | Required evidence |
|---|---|---|
| F-B conditioned scalar vs F-C explicit Hermite/multi-crossing | `OPEN_ABLATION_REQUIRED` | matched-budget static representability court |
| local patch query/cardinality architecture | `OPEN_ABLATION_REQUIRED` | FIT8 teacher target construction + permutation/continuation tests |
| topology/geometry decoupling exact parameterization | `OPEN_ABLATION_REQUIRED` | TopoMesh-inspired controlled overfit/closure court |
| sparse support pitch/resolution ladder | `OPEN_ABLATION_REQUIRED` | support recall + compute + local crossing curve |
| cross-cell patch continuation | `OPEN_ABLATION_REQUIRED` | topology/bridge/hole metrics |
| CAO fragment seeding algorithm | `OPEN_ABLATION_REQUIRED` | M-addressed full-resolution all-hit court |
| CAO fragment capacity | `OPEN_ABLATION_REQUIRED` | full FIT8 + Knight dynamic distribution |
| CAO runtime compression | `OPEN_ABLATION_REQUIRED` | graph->slot interval equivalence tests |
| T decoder family | `OPEN_ABLATION_REQUIRED` | F survivor + dynamic mechanics court |

---

## 13. Next architecture court

The next implementation should not yet be a full IRIS training campaign.

Build a **teacher-local patch/crossing extractor** for FIT8, then run small controlled representation overfits under fixed scene/teacher evidence:

```text
F-A standard sparse FlexiCubes
F-B sheet-conditioned FlexiCubes
F-C explicit patch/crossing DMC
```

The first court can operate from teacher spatial features rather than image reconstruction to isolate representation capacity.

Only representation families that can reconstruct authored FIT8 geometry without false bridges/missing sheets should be wired behind the full image-to-scene IRIS trunk.

This prevents spending expensive scene-model training budget on an extraction representation that is already falsified by exact teacher geometry.
