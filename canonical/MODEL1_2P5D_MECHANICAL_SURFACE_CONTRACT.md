# RealSaS Model-1 — 2.5D Mechanical Surface Contract

**Date:** 2026-08-21  
**Status:** `CANONICAL_PRODUCT_REPRESENTATION_OBJECTIVE`

## Product objective

Model-1 is not a generic 3D reconstruction system and is not required to recover the exact latent 3D asset, teacher rig, or teacher mechanical quotient.

Its job is to convert paired multi-view raster observations into an **observation-native, functionally sufficient 2.5D mechanical surface** that gives downstream skeleton, skinning, rig compilation, and deformation logic the kind of rich geometric/mechanical substrate that 3D auto-riggers receive directly from a mesh.

The reference asymmetry is:

```text
3D auto-rigger / RigAnything-like system
    receives explicit 3D shape / surface / normals / spatial relations
    -> skeleton + skinning become a downstream geometry/mechanics problem

RealSaS Model-1
    receives paired 8-view raster observations
    -> must construct an equivalent riggable mechanical substrate from observations
    -> compiler / later models consume that substrate for skeleton + skinning
```

The goal is therefore **functional equivalence of the useful input substrate**, not literal 3D scene reconstruction.

## What "2.5D mechanical surface" means

The construct should preserve enough observation-grounded structure to support high-quality rigging decisions across the visible object:

- stable surface/carrier identity across views and poses;
- calibrated multi-view spatial compatibility;
- surface position / feasible position support where observable;
- local orientation / normal-like evidence where recoverable;
- visibility / occlusion support;
- persistent appearance correspondence;
- differential motion / mechanical response evidence across poses;
- local and global relational compatibility between carriers;
- explicit uncertainty or set-valued support where the observation does not justify a singleton.

It may be represented by carriers, proposals, feasible sets, fields, relations, normals, visibility, mechanical evidence, or other observation-native quantities. It does **not** have to be a conventional watertight 3D mesh.

The acceptance criterion is downstream utility: can the compiler and subsequent skeleton/skinning stages construct a clean, editable, functionally correct rig and deformation model from it?

## Frontend design requirement

The paired `8 + 8` views are the primary evidence source from which this substrate is built. The frontend must therefore be treated as an evidence-construction system, not merely as a candidate generator.

A result such as:

```text
"the correct endpoint is usually somewhere in top-4"
```

is necessary but not sufficient.

If the correct candidate is present but the observable evidence cannot reliably distinguish it from alternatives, the frontend has not yet completed its job. Selection must not be left to accidental solver behavior or weak tie-breaking.

The desired contract is:

```text
raster observations
    -> rich, independent, observation-native evidence
    -> strongly discriminative proposal support / ranking
    -> globally consistent mechanical surface
    -> skeleton + skinning + deformation compilation
```

For difficult rows, proposal evidence should accumulate independent physical/observational constraints until one of two outcomes is justified:

1. a clearly preferred candidate / configuration; or
2. an explicit unresolved set / uncertainty state that downstream logic can handle safely.

Silent arbitrary collapse is forbidden.

## Why the current geometry-closure line exists

The recurring hard tail showed that candidate **coverage** alone was not the remaining problem. Correct alternatives were often retained, yet the system still lacked enough discriminative evidence to select the right one reliably.

Because the target product is explicitly geometry/mechanics grounded, the appropriate response is to fully exploit relevant observable geometry and correspondence research before accepting this ambiguity as irreducible.

The current controlled research ladder is therefore part of the Model-1 substrate construction itself:

```text
V5 descriptor + global relational baseline
    ↓
G1 — GGPT-like explicit multiview geometry authority
     triangulation / reprojection / conditioning exposed at proposal level
    ↓
G2 — MASt3R-like reciprocal association + cycle consistency
     independent reverse-correspondence evidence
    ↓
if still required:
MV-TAP / V-DPM-style camera/time-aware learned refinement
     while retaining explicit geometry as deterministic authority
    ↓
only after correspondence is reliable:
optional learned 3D refinement / richer surface consolidation
```

Each treatment must be introduced and qualified separately so its causal contribution is measurable. Multiple literature mechanisms must not be merged into one opaque treatment merely to maximize a benchmark score.

## Hard-tail interpretation rule

A persistent hard tail is not automatically evidence that the target is unobservable.

Before declaring irreducible ambiguity, test whether the frontend is failing to expose an observation-native discriminator that the paired views already contain, including:

- multiview reprojection consistency;
- camera-conditioned geometry;
- reciprocal correspondence;
- cycle consistency;
- cross-view persistence;
- cross-pose differential response;
- relational compatibility with neighboring carriers;
- visibility / occlusion consistency.

Only after the relevant observable channels have been tested prospectively may the remaining set be classified as genuinely unresolved by the observation contract.

## Authority boundary

Model-1 may use only production-available evidence in forward construction:

- paired rasters;
- known calibrated camera/view contract;
- frozen learned observation heads;
- deterministic geometry derived from those observations;
- observation-derived candidate and relation evidence.

Teacher truth may evaluate or train designated components under a frozen protocol, but it may not silently become forward inference authority.

## Non-goals

Model-1 is **not** required to:

- recover the exact hidden 3D source asset;
- reconstruct a photogrammetric or watertight world model;
- reproduce the teacher skeleton exactly;
- reproduce teacher skinning weights exactly;
- force every carrier to a singleton when evidence remains genuinely ambiguous;
- solve skeleton generation or dense skinning inside the correspondence frontend itself.

Those would confuse the representation objective with downstream compilation targets.

## Canonical success statement

The intended end state is:

> From paired 8-view 2D observations, Model-1 constructs a rich, observation-grounded 2.5D mechanical surface that supplies the geometric, correspondence, visibility, orientation, and differential mechanical evidence needed for clean skeleton and skinning generation. It recreates the *functional input advantage* of 3D auto-rigging systems without requiring literal 3D reconstruction.

This statement is the parent rationale for the current geometry-grounding, reciprocity/cycle, and later camera/time-aware frontend qualification line.
