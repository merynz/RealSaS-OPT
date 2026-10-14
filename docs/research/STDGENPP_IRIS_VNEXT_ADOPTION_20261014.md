# StdGEN / StdGEN++ -> RealSaS IRIS-vNext adoption

**Date:** 2026-10-14  
**Status:** design/implementation authority for the current IRIS-vNext intervention  
**Scope:** architecture translation, not code copying or product qualification

References:

- StdGEN: https://arxiv.org/abs/2411.05738
- StdGEN++: https://arxiv.org/abs/2601.07660
- released StdGEN code: https://github.com/hyz317/StdGEN

## 1. Reference decision

StdGEN++ supersedes StdGEN as the primary external architecture reference for the
current IRIS-vNext redesign. StdGEN remains important because its released code
makes the original S-LRM implementation concrete; StdGEN++ is the stronger
scientific reference because it documents why the authors changed the single-
branch design, dense extraction and monolithic texture representation.

RealSaS must not copy human-specific ontology or implementation bodies blindly.
The goal is to preserve the principles that generalize to arbitrary articulated
subjects and to replace human-specific specialization with generic evidence-
driven specialization.

## 2. What StdGEN++ confirms

The following design choices are directly aligned with the new RealSaS path:

1. one high-capacity image-to-triplane scene representation can jointly support
   geometry, color and semantics;
2. SDF and semantic fields should be coupled during extraction rather than
   treating semantics as a late 2D segmentation product;
3. a high-quality explicit surface benefits from bounded learned extraction
   deformation/configuration, not only fixed marching geometry;
4. semantic training should be staged: first visible/surface semantics, then
   occluded/internal semantic+color structure, then explicit mesh supervision;
5. high-resolution extraction should be sparse/coarse-to-fine rather than a
   blindly dense high-resolution grid;
6. thin semantic structures need explicit continuity/hole regularization;
7. appearance decomposition is a production concern, not merely a texture-file
   convenience;
8. a single shared branch can become a detail bottleneck, so selective
   specialization/refinement is justified when driven by a clear region need.

## 3. What RealSaS copies at the architecture level

### 3.1 Shared triplane scene

First full-capacity attempt floor:

```text
DINO-B/16-class encoder / 768-D image feature class
       ->
1024-D, 16-layer, 16-head scene transformer class
       ->
3 x 64 x 64 x >=80 triplane
```

The exact encoder may change after measured comparison. The capacity class should
not be reduced merely to preserve historical IRIS-v4 size.

### 3.2 Field heads

The shared spatial feature is decoded into at least:

```text
G: SDF + geometry uncertainty
A: appearance field + appearance confidence
S: owner/material/deformation/attachment semantic evidence
F: bounded extraction delta + cell/configuration evidence
T: animation-oriented explicit topology proposal
```

StdGEN/StdGEN++ demonstrate the viability of SDF + RGB + semantic + extraction
heads from a shared triplane. RealSaS extends the semantic field beyond four
human classes and integrates the old TESSA topology idea into the same scene.

### 3.3 Semantic-equivalent SDF

For a normalized semantic subset P, RealSaS adopts the equivalent-field idea:

```text
f_P(x) = max(
    f(x),
    max_{s not in P} p_s(x) - max_{s in P} p_s(x)
)
```

This is implemented clean-room in `models/iris/vnext/semantic_equivalent_v1.py`.

Important boundary: subject-local owner embeddings are relational descriptors,
not globally meaningful categorical labels. They must not be forced through this
formula. Equivalent SDF extraction is for explicit normalized semantic subsets,
material groups or compiler-resolved categorical regions where the probability
simplex has a clear meaning.

### 3.4 Sparse coarse-to-fine extraction

StdGEN++ evaluates a coarse SDF, dilates the inside region and upsamples that
proposal to restrict expensive high-resolution queries. RealSaS adopts the
geometry-safe core but extends the active set:

```text
active = geometry_surface_proposal
       UNION owner/material boundary proposal
       UNION high-uncertainty proposal
       UNION contact/overlap-critical proposal where available
```

This keeps the method generic. Human-face specialization is not assumed.

### 3.5 Hole/continuity regularization

StdGEN++ explicitly addresses fractured thin structures during semantic-aware
SDF extraction. RealSaS requires the equivalent capability, but the target is
broader than cloth:

- thin wings/fins;
- tails/tendrils;
- armor or panel edges;
- clothing/capes;
- semantic owner boundaries;
- any region where canonical M has valid geometric support but the extracted or
  presented appearance creates unexplained holes/peppering.

This becomes both a training concern and a CAO qualification concern.

## 4. What RealSaS deliberately does not copy

### Human face branch

StdGEN++ uses full-body and facial LoRA branches. RealSaS FIT8 contains
serpentine, flying, dinosaur, quadruped, arthropod, amorphous, aquatic and
mechanical subjects. A hard-coded face branch would encode the wrong ontology.

RealSaS instead uses **generic selective refinement**. Candidate detail regions
are proposed from semantic/material boundaries, thin structures, high geometry
or semantic uncertainty, high projected frequency/detail and contact/overlap
importance. A future learned router may allocate a specialized adapter to these
regions without assuming human anatomy.

### Hair/cloth/body ontology

StdGEN's categorical semantics are not the RealSaS core ontology. RealSaS
separates:

```text
owner identity / affinity
material identity / affinity
deformation mode
attachment relation
amodal support
overlap relation
optional coarse role
```

This is intentionally factored because different downstream compiler decisions
consume different relations.

### Monolithic or human facial texture decomposition

StdGEN++'s video-diffusion facial texture decomposition is an important product
lesson, but eyebrow/skin/iris layers are not generic enough for RealSaS.
RealSaS translates the principle into CAO generic appearance decomposition bound
to canonical M ownership/material regions.

## 5. Input-cardinality staging

Final product input remains one PNG.

The first Unified IRIS FIT8 architecture court uses eight authored canonical RGB
views. This intentionally separates reconstruction architecture closure from the
much harder learned one-to-eight prior.

Unified IRIS receives a stable `CanonicalViewEvidenceSet` with per-view
provenance and confidence. FIT8 Phase A uses:

```text
8 x AUTHORED_CANONICAL
```

A later product phase can use:

```text
1 x SOURCE_OBSERVED
7 x GENERATED_CANONICAL
```

without changing G/A/S/F/T output contracts or CAO authority. Generated views
remain hypotheses and are never silently promoted to exact source authority.

## 6. Training translation

The first FIT8 training program should preserve the staged insight from
StdGEN/StdGEN++ while replacing human-specific layer ordering with generic
component supervision.

### U1 — visible field formation

Train shared G/A/S representation with source RGB, mask, depth/normal where
available, signed-field supervision, visible owner/material semantics and
uncertainty.

### U2 — amodal/internal ownership and appearance

Use authored component isolation and component subsets to reveal hidden regions.
Unlike outside-in hair/cloth/body-only training, subset selection is generic and
subject-local. The target is a coherent continuous field for owner/material/RGB
under occlusion.

### U3 — explicit high-resolution surface

Enable F extraction supervision with mask, depth, normal, RGB, semantic boundary,
deviation/anti-flip and thin-structure continuity losses. Use sparse active-domain
queries rather than dense high-resolution evaluation.

### U4 — topology proposal

Train integrated T topology decoding against authored topology/mechanics evidence
where admissible. T consumes shared scene memory directly; the legacy 17-D GSA
bottleneck is not its scene representation.

## 7. CAO — Canonical Appearance Authority

CAA is renamed/reconceived as **CAO: Canonical Appearance Authority**.

CAO owns final appearance/presentation materialization on exact canonical M. It
is not merely a hole-filling texture compiler.

Inputs include:

```text
canonical M + SurfaceAddressing
exact observed source evidence
IRIS appearance field A
IRIS semantic/ownership field S
compiler-resolved appearance decomposition
pose/view overlap evidence
```

Authority order begins with:

```text
SOURCE_EXACT
SOURCE_REPROJECTED
LEARNED_SOURCE_CONSISTENT
LEARNED_CROSSVIEW_SUPPORTED
LEARNED_HIDDEN_COMPLETION
COMPILER_BOUNDED_FILL
UNSUPPORTED_ABSTAIN
```

Generated canonical views may support learned evidence but cannot masquerade as
`SOURCE_EXACT` or `SOURCE_REPROJECTED`.

## 8. Generic appearance decomposition

CAO decomposes final appearance into subject-local fragments bound to canonical
M. A fragment carries at least:

```text
owner descriptor
material descriptor
mechanical owner / attachment relation where available
amodal support
semantic confidence
```

Pairwise view/pose relations such as `FRONT_OF` and `BEHIND` are first-class.
This makes appearance decomposition directly useful to 2D draw-order and overlap
resolution instead of treating depth as the only ordering authority.

The decomposition is not required to equal source object boundaries exactly.
Compiler may split/merge learned/authored evidence when independent courts prove
that doing so gives the canonical editable presentation domain.

## 9. Qualified-M versus visible-result coherence

The previous single-M cutover proved an important negative lesson: using the same
canonical M for mechanics and rendering does not by itself guarantee a coherent
visible result. Holes/white peppering/fragmented alpha can still arise from
appearance support, raster ownership, clipping, semantic fragment boundaries or
presentation ordering even when M is mechanically qualified.

IRIS-vNext + CAO makes this mismatch explicit and testable.

Required invariant:

> Every geometrically visible canonical-M sample/pixel must have a qualified
> appearance/presentation explanation, unless it is explicitly classified as
> transparent, occluded or intentionally absent.

Therefore CAO must emit a **Visible Surface Coherence Report** comparing:

```text
geometric visible support
vs
final appearance alpha/support
```

and measuring at least:

```text
unexplained missing pixels
fragmented island pixels
maximum unexplained hole area
qualified transparent coverage
qualified occluded coverage
```

An unexplained visible hole is a CAO/product failure even if static mesh and
mechanics courts pass.

This is the direct closure path for the historical condition:

```text
canonical M PASS
but rendered character still contains gaps / fragmented pixel regions
```

## 10. Current implementation landing

This foundation change introduces:

- `models/iris/vnext/contracts_v1.py`
- `models/iris/vnext/semantic_equivalent_v1.py`
- `compiler/realsas_compiler_core/cao_authority_v1.py`
- `tools/corpus/fit8_full_truth_schema_v1.py`

No scientific PASS is claimed. No new training Attempt has been started. These
contracts are the prerequisite for the FIT8 full-truth extractor and the first
full-capacity Unified IRIS implementation.
