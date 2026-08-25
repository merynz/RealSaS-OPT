# RealSaS — Alternative Downstream Architecture Hypotheses — 2026-08-25

**Status:** `ARCHITECTURE_HYPOTHESIS__NOT_CANONICAL_AUTHORITY__NOT_IMPLEMENTATION_AUTHORIZATION`

This note preserves a set of architecture ideas that emerged while auditing the observable-geometry IRIS plan against RigAnything and SkinTokens. It does **not** supersede `canonical/PRODUCT_CONTRACT_V1.md`. The current canonical architecture remains in force until controlled downstream evidence supports an explicit contract revision.

## Why this note exists

The current product contract is intentionally linear at the responsibility level:

```text
IRIS -> Geppetto -> Arachne -> Compiler -> verified puppet
```

That diagram is useful for ownership, but it may be too literal as an execution/control-flow diagram. Three additional observations motivate a future controlled comparison:

1. RigAnything-like skeleton generation consumes an oriented surface representation centered on sampled positions and normals rather than authored mechanical truth.
2. SkinTokens-style skinning also conditions its neural geometry path on sampled vertices/positions and normals, and its `TokenRig.generate(...)` interface can accept externally supplied skeleton tokens.
3. RealSaS already has a compiler philosophy in which neural stages should propose evidence/structure while deterministic logic owns canonical validity, repair, proof and fail-closed behavior.

Together these suggest that the eventual downstream system may be better represented as a learned proposal pipeline embedded inside a cross-cutting Compiler authority plane.

---

## Hypothesis A — IRIS substrate remains observable geometry

No proposal in this note returns hidden mechanical ontology, source-rig identity or mandatory GFDR to IRIS.

Candidate IRIS external state remains approximately:

```text
P  common/object-frame surface position
N  local orientation / normal evidence
V  visibility / observational support
U  uncertainty / risk
+ cross-view persistence / provenance
+ optional learned surface-relation evidence if S0 proves it necessary
```

The key downstream question is not whether IRIS reproduces the source mesh or source rig. It is whether an IRIS-derived rigging substrate is downstream-noninferior to a ground-truth observation-equivalent surface substrate.

---

## Hypothesis B — replace a purely heuristic SurfaceBuilder with learned relational evidence + deterministic topology authority

`SurfaceBuilder` is a placeholder name and may not survive a future naming pass.

A purely deterministic `P/N -> adjacency` construction may be fragile around touching-but-disconnected or layered surfaces. A stronger candidate is:

```text
IRIS surfels
P / N / V / U
    +
learned pairwise surface-relation evidence
    |
    v
constrained deterministic surface solver
    |
    v
canonical RiggingSurface / sheet graph / editable topology
```

Possible learned relational evidence:

```text
p(same_sheet | i,j)
p(surface_adjacent | i,j)
p(boundary | i,j)
p(occlusion_only | i,j)
```

The model would not own irreversible global topology. It would emit local/relational evidence; a deterministic solver would own globally admissible components, sheet separation, topology validity and triangulation.

This preserves an important prior RealSaS lesson:

> learned evidence may be uncertain and relational; canonical identity/topology decisions should remain constrained and globally validated.

### Evidence required before promotion

Compare, on the same frozen downstream tasks:

```text
A. P/N-only deterministic adjacency
B. P/N + learned relation evidence + constrained solver
C. exact source-mesh adjacency oracle
```

Promotion requires a preregistered downstream deformation/rigging advantage, not intuition.

---

## Hypothesis C — Geppetto may be RigAnything-inspired while Arachne is SkinTokens-inspired

A promising division of labor is:

```text
qualified RiggingSurface
        |
        v
Geppetto
RigAnything-inspired skeleton proposal
(joints + hierarchy/parents)
        |
        v
canonical skeleton G
        |
        v
Arachne
SkinTokens-inspired skin representation/generation
        |
        v
weights W
```

### Why this hybrid is technically plausible

RigAnything provides a strong precedent for template-free skeleton generation from sampled oriented surface geometry.

SkinTokens provides a newer representation for skinning and, in its released code, neural geometry conditioning is built from sampled `vertices + normals`. Its `TokenRig.generate(...)` interface also accepts `skeleton_tokens`, so an externally produced skeleton can be supplied as the sequence prefix rather than requiring TokenRig to generate the skeleton itself.

A deterministic adapter could map Geppetto output:

```text
joints (J,3) + parents (J)
```

into the ordering/token convention expected by a SkinTokens-style Arachne.

### This is NOT yet a claim that the hybrid is superior

SkinTokens/TokenRig jointly models skeleton and skinning dependencies. Therefore the hybrid must be compared against both parent systems rather than assumed superior.

Recommended controlled downstream comparison:

```text
A. RigAnything-style skeleton + RigAnything-style skinning
B. RigAnything-style skeleton + SkinTokens-style skinning
C. TokenRig-style skeleton + SkinTokens-style skinning
```

Use the same surface substrate, families, functional deformation evaluator and compiler policy.

---

## Hypothesis D — Compiler is a control plane, not merely the final box

The current canonical diagram places `Compiler` after Arachne. That remains the responsibility-level authority today, but a more faithful eventual execution model may be:

```text
                     COMPILER AUTHORITY PLANE
        +------------------------------------------------+
        |                                                |
images -> IRIS -> qualified S -> Geppetto -> qualified G -> Arachne -> qualified W
        |                                                |
        +------ validate / normalize / repair / veto -----+
                                      |
                                      v
                                Motion Proof
                                      |
                                      v
                              verified editable puppet
```

More explicitly:

```text
IRIS
  |
  v
Compiler Geometry Gate
- cross-view fusion / consistency
- uncertainty rejection
- canonical coordinates
- surface/topology validity where applicable
  |
  v
Geppetto
  |
  v
Compiler Rig Gate
- root uniqueness
- acyclicity / parent validity
- duplicate/helper cleanup
- bounded structural completion
- surface support sanity
  |
  v
Arachne
  |
  v
Compiler Skin Gate
- normalization / sparsity constraints
- component leakage control
- topology-aware cleanup
- influence sanity
  |
  v
Compiler Motion Proof
- deformation probes
- failure signatures
- owner attribution
- repair directives / bounded retries
  |
  v
Final Compile / Export
```

Under this view, `Compiler` has two meanings that should be distinguished if the hypothesis is promoted:

1. **Compiler Core / Authority Plane** — cross-cutting deterministic canonicalization, validation, repair and fail-closed control between learned stages.
2. **Final Compiler / Export** — converts qualified `S + G + W` into the editable runtime puppet and performs final product proof/export.

### Why this fits existing RealSaS principles

The model stages remain proposal/evidence producers. Canonical truth remains outside opaque neural output. This is compatible with the existing motion-proof chain:

```text
MotionProof -> FailureSignature -> OwnerAttribution -> RepairDirective
```

and makes backward repair routing explicit:

```text
motion failure
  |- skin repair
  |- rig repair
  `- geometry/evidence re-evaluation
```

### Evidence required before promotion

The proposal should not be promoted merely because it is architecturally neat. A controlled implementation must show that intermediate qualification/repair improves downstream functional quality, safety or debuggability without destroying useful uncertainty/evidence.

---

## Candidate combined architecture — NOT CANONICAL

```text
8 ordered neutral-pose views
        |
        v
      IRIS
P / N / V / U / persistence
(+ relation evidence only if proven necessary)
        |
        v
Compiler Geometry / Surface Authority
        |
        v
qualified RiggingSurface S
        |
        v
     Geppetto
RigAnything-inspired skeleton proposal
        |
        v
Compiler Rig Authority
        |
        v
qualified skeleton G
        |
        v
      Arachne
SkinTokens-inspired skin proposal
        |
        v
Compiler Skin Authority
        |
        v
qualified weights W
        |
        v
Motion Proof / Attribution / Repair
        |
        v
Final Compile + Runtime Export
        |
        v
verified editable puppet
```

This diagram is a **candidate research architecture only**.

---

## Required sequence before downstream implementation

Do not start implementing Geppetto/Arachne or this alternative architecture merely because the idea has been recorded.

The intended order is:

```text
1. close/qualify IRIS against its observable-substrate contract
2. audit the most canonical historical Compiler + runtime code from Drive
3. recover valuable canonical code into the single GitHub authority
4. inventory which old validation / proof / repair systems remain reusable
5. freeze downstream S0 / skeleton / skinning comparison gates
6. only then choose/promote Geppetto, Arachne, surface-topology and Compiler-control-plane designs
```

The Drive audit matters because RealSaS has substantial historical compiler/runtime implementation that must not be silently abandoned or reimplemented from memory.

---

## Promotion rule

This note may become architecture authority only through an explicit revision of the canonical product contract tied to controlled evidence. Until then:

- current IRIS experiments are unchanged;
- current P/N/V/U observable-geometry direction is unchanged;
- PatchMatch-style intervention remains evidence-gated;
- Geppetto/Arachne physical implementation remains unfrozen;
- the existing canonical architecture diagram remains the default authority;
- this document is preserved as a linked alternative: **"could the downstream architecture instead be this?"**
