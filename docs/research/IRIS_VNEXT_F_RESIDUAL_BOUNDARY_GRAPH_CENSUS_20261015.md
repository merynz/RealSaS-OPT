# IRIS-vNext F residual boundary-graph census

**Date:** 2026-10-15  
**Status:** architecture-decision evidence — boundary topology only; no interior/candidate-surface/model/product PASS  
**Grid:** N=128 current qualified FIT8 teacher carriers

This court moves one level above primitive residual-type coverage.

Previous FIT8 evidence showed that the local primitive alphabet is highly reusable across morphology:

```text
variable EdgeCrossingSet
FaceStrand kinds:
  BOUNDARY_TO_BOUNDARY_ARC
  CLOSED_LOOP
  OPEN_INTERIOR_ENDPOINT
  BRANCH_OR_NONMANIFOLD
patch <-> strand incidence
```

Primitive-type leave-one-subject-out coverage was 100% on FIT8. That does **not** prove that complete local residual compositions form a small generic whole-template vocabulary.

This census therefore canonicalizes the **cell-boundary residual incidence graph** under:

```text
24 orientation-preserving cube rotations
x
all exact permutations of cell-local patch labels
```

Reflections are deliberately not admitted. Handedness / side / orientation can matter to later mechanical and appearance semantics.

The signature records topology/combinatorics, not continuous geometry:

```text
residual edge:
  cube-edge identity
  unlabeled local-patch identity
  crossing multiplicity

residual face:
  cube-face identity
  residual reason set
  strand topology profile
  patch incidence
  boundary-endpoint -> cube-edge incidence
```

The court explicitly does **not** claim cell-interior/tunnel topology closure.

---

## 1. Subject-level results

| Subject | active cells | residual cells | canonical boundary-graph signatures | edge-only | face-only | edge+face | exact-canonicalization abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snake | 22,654 | 1,169 | 39 | 431 | 719 | 19 | 0 |
| Bat | 20,232 | 1,204 | **172** | 309 | 826 | 69 | 0 |
| T-Rex | 9,610 | 470 | 48 | 400 | 42 | 28 | 0 |
| Horse | 17,531 | 983 | 30 | 473 | 487 | 23 | 0 |
| Spider | 13,524 | 752 | 29 | 708 | 24 | 20 | 0 |
| GreenBlob | 56,486 | 386 | **7** | 384 | 0 | 2 | 0 |
| Shark | 7,810 | 694 | 32 | 254 | 424 | 16 | 0 |
| George | 18,613 | 1,711 | **205** | 905 | 740 | 66 | 0 |

The variation is large and meaningful.

Examples:

```text
GreenBlob:
  huge active domain
  few residual cells
  only 7 boundary compositions

Snake / Horse / Spider:
  many residual events
  strongly repetitive boundary language

Bat / George:
  layered / open / part-heavy local structure
  much wider composition vocabulary
```

Therefore two quantities must remain separate:

```text
RESIDUAL EVENT SPARSITY
!=
RESIDUAL VOCABULARY COMPLEXITY
```

---

## 2. Global whole-graph template coverage

Across all eight subjects:

```text
residual boundary events = 7,369
unique rotation-canonical graph signatures = 401
```

Coverage by the globally most frequent whole-graph signatures:

| template count | global event coverage |
|---:|---:|
| 1 | 37.78% |
| 4 | 69.78% |
| 8 | 77.68% |
| 16 | 86.08% |
| 32 | 90.96% |
| 64 | 93.53% |
| 128 | 95.58% |
| 256 | 98.03% |
| 401 | 100% |

This is not an impossibly large vocabulary, but it is far wider than the primitive-type census suggested.

A fixed small whole-graph table would either:

- miss meaningful authored topology; or
- require an `OTHER` approximation class, which is incompatible with fail-closed topology authority.

---

## 3. Whole-graph 7->1 LOFO genericity

The stronger test derives the complete whole-graph vocabulary from seven subjects and asks what fraction of residual events in the held-out morphology use a previously seen graph signature.

| held-out subject | residual events | held-out unique signatures | event coverage by all 7-subject signatures | unseen unique signatures | unseen residual events |
|---|---:|---:|---:|---:|---:|
| Snake | 1,169 | 39 | **99.40%** | 7 | 7 |
| Bat | 1,204 | 172 | **76.66%** | 133 | 281 |
| T-Rex | 470 | 48 | **92.34%** | 24 | 36 |
| Horse | 983 | 30 | **99.59%** | 2 | 4 |
| Spider | 752 | 29 | **98.94%** | 6 | 8 |
| GreenBlob | 386 | 7 | **100.00%** | 0 | 0 |
| Shark | 694 | 32 | **97.98%** | 8 | 14 |
| George | 1,711 | 205 | **88.43%** | 139 | 198 |

This is the key result of the court.

Primitive types transfer extremely well across FIT8, but **whole residual compositions do not transfer perfectly**.

Bat is the strongest falsifier:

```text
all whole-graph templates from the other seven subjects
still leave ~23.34% of Bat residual events unseen.
```

George also contains a substantial unseen-composition tail.

Thus:

> generic local topology should not use a whole-template ID as its only representational language.

---

## 4. Top-K templates trained on the other seven subjects

Using only the most frequent graph signatures in the seven-subject training side makes the morphology difference even clearer.

Representative held-out event coverage:

### Bat

```text
top-8    ~45.68%
top-16   ~56.56%
top-32   ~72.43%
top-64   ~72.76%
top-128  ~75.25%
all seen ~76.66%
```

### George

```text
top-8    ~57.39%
top-16   ~73.93%
top-32   ~79.49%
top-64   ~84.86%
top-128  ~85.80%
all seen ~88.43%
```

### GreenBlob

```text
top-8    ~98.45%
top-32   ~99.48%
top-128 100.00%
```

A template proposal path can therefore be extremely effective on common/simple residual structure while remaining incomplete on hard morphology.

---

## 5. Architecture consequence — compositional incidence should be primary truth

The strongest current conclusion is:

```text
PRIMITIVE LANGUAGE
  generic / reusable

WHOLE-GRAPH TEMPLATE IDS
  useful compression / proposal
  but not generic truth ontology
```

The residual authority should therefore remain compositional:

```text
local patch nodes
edge crossing-set nodes
face strand nodes
patch <-> crossing incidence
patch <-> strand incidence
strand <-> cube-edge boundary incidence
neighbor continuation incidence
```

A common composition may be assigned a compact template ID for fast prediction/decoding.

But:

```text
TEMPLATE ID != TRUTH
```

If no qualified template exists:

```text
explicit incidence prediction / decode
or
uncertainty / abstain
or
Compiler rejection
```

Never silently map unseen topology to the nearest whole-template case.

---

## 6. This supports a two-path residual decoder, not two geometry authorities

Candidate learned implementation:

```text
shared residual feature
      |
      +--> template proposal head
      |       common graph code + confidence
      |
      +--> compositional incidence head
              primitive set / incidence / cardinality
      |
      v
Compiler qualification
      |
if template and graph agree -> compact decode
if template absent/unseen    -> explicit graph decode
if unsupported/uncertain     -> abstain/fail
```

Both paths describe the **same local residual topology** attached to the same sparse F extraction complex.

They are not separate meshes.

---

## 7. Why Bat matters disproportionately

Bat is not just one bad outlier to optimize away.

It stresses exactly the qualities RealSaS claims to support generically:

- open/two-sided support;
- thin structures;
- nearby layered sheets;
- articulation;
- mixed local topology;
- appearance-side ownership.

A fixed template ontology that works on smooth closed characters but fails on Bat would violate the project goal more than a small average metric drop would suggest.

Therefore Bat's LOFO tail is architecture evidence, not noise.

---

## 8. Important scope limit

This signature is **boundary topology**, not full cell topology.

It captures:

```text
edge residuals
face residuals
patch/strand/boundary incidence
```

It does not yet prove:

```text
cell-interior tunnel connectivity
interior contour-component pairing
actual candidate-surface assembly
primal Euler/boundary/genus closure
geometry accuracy
```

Calling these 401 signatures `complete cube topology cases` would be an overclaim.

---

## 9. Decision state

### Strongly supported

1. proper-rotation canonicalization is necessary and sufficient for the current template census; reflection equivalence remains intentionally unproven;
2. patch-local numeric IDs must be quotiented out of template identity;
3. boundary endpoint -> cube-edge incidence is part of topology identity and cannot be reduced to endpoint count;
4. a small primitive alphabet generalizes better than complete residual graph templates;
5. whole-template-only prediction is rejected as generic primary topology authority;
6. template proposal + compositional explicit incidence fallback is now the strongest residual-decoder direction;
7. residual compute/vocabulary should be morphology-adaptive rather than globally fixed.

### Still open

1. cell-interior/tunnel incidence authority;
2. whether a common template head materially improves optimization versus predicting only compositional incidence;
3. exact set/graph decoder architecture;
4. permutation matching / canonical query assignment;
5. false-negative residual activation policy under image-derived features;
6. differentiable assembly of explicit incidence into F geometry;
7. reflection/chirality equivalence rules after semantic/mechanical evidence is considered;
8. learned LOFO closure on RGB-derived scene features rather than teacher oracle topology.

---

## 10. Next court

Do **not** freeze F-R yet.

The next representation court should add the missing **cell-interior incidence layer** and ask:

```text
Given exact edge crossing sets + face strand sets,
what additional discrete state is required to connect boundary contour components through one cell interior?
```

The court should distinguish:

- topology determined uniquely from boundary incidence;
- finite ambiguous interior pairing/tunnel state;
- authored non-manifold/intersection state;
- insufficient / abstain.

Only after boundary + interior incidence form a complete local topology graph should F-R and full F-C be compared through actual oracle surface assembly and the primal topology comparator.
