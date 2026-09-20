# RealSaS — Why V1 Became V2

**Date:** 2026-09-20  
**Status:** CURRENT ARCHITECTURE RATIONALE  
**Supersedes as current design rationale:** the 40-stage V1 product-mainline assumptions. V1 artifacts remain historical/scientific evidence.

## Executive reason

V2 is not a cosmetic renumbering and not a request to throw away working models. It is an authority correction.

V1 treated geometry and mechanics as first-class scientific products: they had typed IRs, provenance, frozen numerical gates, dynamic proofs, repair ownership and fail-closed behavior. Appearance did not receive the same treatment. It remained largely a set of source-donor bindings, view fallbacks and renderer rules attached after the mechanical state existed.

That asymmetry is incompatible with the product target. A RealSaS puppet is not successful merely because its mesh, rig and skin are numerically valid. The product target is **Spine-class authored 2D art under motion**. Therefore three product axes are co-equal:

1. **Geometry** — what renderable surface exists and where it is in canonical 3D.
2. **Mechanics** — how that surface is rigged, skinned, deformed and constrained.
3. **Appearance** — what authored 2D art is defined on every renderable surface state and how faithfully it is preserved.

Presentation and runtime consume those authorities. They are not allowed to invent missing truth.

## What V1 taught us

### 1. Geometry was carrying an appearance burden

Stage13-style rest reprojection pressure made dense source-mask agreement look like a final product objective. The STRIDE8 experiments exposed the mismatch: sparse signed supervision was being asked to satisfy dense 1024x1024 coverage. More importantly, even a perfect geometric carrier is not the same thing as faithful 2D appearance.

V2 keeps geometry qualification strict, but stops interpreting geometry as RGB/line-art authority. IRIS/GSA must produce a mechanically and visually adequate carrier; appearance has its own authority and proofs.

### 2. The product mesh producer and its gate were not proved mutually feasible

The product mesh is not a simple decimation of the dense marching-cubes surface. Scene-first GSA compacts dense geometry into nodes and a local relation graph; canonical relation triangles are then reconstructed as 3-cliques. CDT V1 refines each baseline triangle while preserving its boundary.

That means relation-parent quality is its own scientific object. If a parent corner is below the frozen 7.5 degree target, interior Steiner refinement without boundary splits cannot make every child triangle exceed 7.5 degrees. V2 therefore requires producer-versus-gate feasibility proofs upstream rather than waiting for a later mesh gate to fail.

### 3. Partial appearance was a structural error, not only a renderer bug

V1 appearance was effectively partial: some faces had direct-source donors, some used other-view donors, and some could remain UNSEEN. This created a large control-flow surface around missing appearance: donor search, UNSEEN behavior, rest-unseen motion exposure, underlays and fallback policy.

Eight rest views do not imply that every renderable mechanical surface has observed appearance. The Knight observability study made that premise untenable.

V2 replaces partial appearance with a **Complete Appearance Authority (CAA)**. Every supported renderable surface state has defined appearance before shipping. Missing source evidence remains visible in provenance; it is not allowed to remain an undefined runtime state.

### 4. Visibility and appearance were too easy to conflate

These are separate questions:

- **Visibility:** which canonical surface owns the pixel? Authority = posed canonical XYZ + camera + depth test.
- **Appearance:** what RGBA/line-art value does the visible surface carry? Authority = CAA.

Texture alpha is an art/coverage channel on the already selected surface. It is not allowed to become a hidden replacement for geometric depth ownership.

V2 therefore requires explicit visibility proofs independently from CAA totality.

### 5. Rest holes prove that motion was never the only problem

Rest-pose holes can arise before any animation:

- no rasterizable carrier at the pixel;
- degenerate or subpixel/sliver triangles;
- appearance undefined on an otherwise visible surface;
- alpha/UV sampling and atlas-edge failure;
- incorrect depth/equal-depth ownership;
- composition policy that hides the correct surface.

V2 does not assign all of these to CAA. It separates them into geometry coverage, triangle conditioning, visibility/depth, appearance definedness, sampling/alpha and native visual-integrity proofs.

### 6. Total appearance removes holes but can hide wrong inference

V1's missing information could appear as an obvious hole. CAA makes appearance total, so missing evidence will instead appear as a value. That value may be wrong without looking obviously broken.

Therefore holdout accuracy, provenance-boundary seam tests and compiled-unobserved dynamic exposure budgets are not optional quality metrics. They are the honesty mechanism of a total appearance system.

### 7. Sampling is part of product correctness

Straight-alpha bilinear filtering and missing atlas bleed can make a mathematically defined texture look transparent or dark at boundaries. V2 fixes the internal contract to premultiplied-alpha filtering/compositing plus mandatory island bleed, with at most one declared unpremultiply at an external export boundary.

### 8. Input observations and output presentation directions are different authorities

V1 structurally assumed exact eight source observations in too many places. The product may eventually consume 2, 4, 8 or another qualified observation set while still producing the fixed eight presentation directions V0..V7.

V2 seals source observation authority separately from output presentation-direction authority. The first Knight V2 witness will still use exact controlled eight-source input only to isolate the architecture change.

### 9. A linear executor did not match the actual architecture

The plan already contained dependencies, but V1 execution semantics were effectively linear: first non-PASS stage, ordinal ranges and first failure stopping later work.

V2 execution is a real DAG. Ordinals are documentation only. Readiness is determined by dependency closure. A failed appearance branch must not erase or prevent an independent mechanics branch from producing valid evidence, and vice versa.

## V2 authority model

```text
                         SOURCE OBSERVATIONS
                                |
                    +-----------+-----------+
                    |                       |
              GEOMETRY EVIDENCE       OUTPUT DIRECTIONS
             IRIS -> GSA surface          V0..V7
                    |                       |
                    +----------+------------+
                               |
                   CANONICAL MESH DOMAIN
       MeshCandidate + SurfaceAddressing + AppearanceDomain
                               |
                       STATIC MESH FREEZE
                               |
             +-----------------+------------------+
             |                                    |
      MECHANICS AUTHORITY                  APPEARANCE AUTHORITY
   skeleton -> skin -> dynamic         source-lock -> completion
        qualification                  -> bake -> qualification
             |                                    |
             +-----------------+------------------+
                               |
                    PRESENTATION STRUCTURE
                               |
                      SEALED COMPLETE PUPPET
                               |
                    MOTION + DYNAMIC PROOFS
                               |
                 DETERMINISTIC DEPTH RUNTIME
                               |
                  DYNAMIC VISUAL INTEGRITY
                               |
                       PRODUCT CLOSURE
```

Geometry, mechanics and appearance are deliberately shown as product authorities, not implementation details.

## Appearance is co-equal, not downstream polish

V2 adopts the following engineering rule:

> **A visually incorrect puppet is not a mechanically successful RealSaS product.**

Appearance therefore receives the same class of discipline already expected for geometry and mechanics:

- typed canonical IRs and binding hashes;
- explicit source authority and provenance;
- preregistered thresholds;
- subject-free calibration where possible;
- producer/gate feasibility checks;
- fail-closed qualification;
- adversarial synthetic tests;
- independent rest and dynamic visual proof;
- exact invalidation when mesh/cameras/addressing change;
- deterministic runtime consumption;
- no silent renderer-side correction.

The target is not a 3D model with textures. Mechanics may be 3D; presentation must remain 2D-authored in character. No PBR, relighting, normal-derived shading, generated character highlights or engine lighting may overwrite the art.

## What CAA does and does not solve

CAA structurally solves one class:

> canonical renderable surface exists, but appearance is undefined.

CAA does **not** prove:

- that geometry covers the intended silhouette;
- that triangles are numerically rasterizable;
- that z-buffer ownership is correct;
- that inferred unseen art is semantically correct;
- that sampling is free of bleed/pepper/dark fringes;
- that motion preserves visual continuity.

Those remain independent gates.

## Mesh and appearance meet at Stage18/19

V2 intentionally creates stable surface addressing when the canonical mesh candidate is born. Appearance is therefore compiled **on the mesh domain**, not pasted onto an unrelated later representation.

Any change to qualified mesh identity, surface addressing or bound camera/output-direction set invalidates dependent CAA assets. Package load re-verifies the binding and fails closed on stale pairings.

## Backend policy

The authority contract is independent from the appearance model/backend:

1. `DETERMINISTIC_V1` — first native E2E baseline; source-lock plus deterministic completion/propagation.
2. `IM2SURFTEX_RESEARCH_ONLY` — direct research benchmark, never product authority merely because it produces plausible texture.
3. `LEARNED_V2` — future learned completion backend if it satisfies the same CAA contract and gates.

This prevents model enthusiasm from changing product semantics.

## Why 40 stages became 46

The additional stages are not bureaucracy. They make previously implicit product truths explicit: output-direction authority, mesh/addressing birth, static mesh freeze, CAA preregistration/compile/bake/qualification/rest proof, dynamic visual integrity and full closure.

At the same time, V2 removes old concepts from the current path: runtime UNSEEN appearance, donor search as product authority, appearance-completion during playback, and the assumption that exact eight source observations are the same thing as eight output directions.

## Migration rule

- V1 source code and artifacts may remain in the repository as historical/scientific evidence.
- Current V2 plan/adapters/runtime may not depend on obsolete V1 product semantics merely for compatibility.
- Shared generic numerical/library helpers may be reused when their semantics are still valid.
- A compatibility path must be outside current product authority and may not silently participate in Stage46.
- Knight V2 execution is held until the full V2 implementation-readiness seal passes.

## Definition of V2 readiness

Knight may restart at Stage01 only after all of the following are green on current `main`:

- the 46-stage plan is a valid acyclic dependency graph;
- every stage has a bound V2-compatible adapter;
- IRIS/GSA geometry semantics have been re-audited against the new role;
- relation-parent/CDT mesh feasibility is instrumented and tested;
- CAA compile, bake, source-lock, holdout, seam and binding logic exists;
- visibility uses canonical posed XYZ/depth independently of appearance;
- premultiplied-alpha/bleed sampling contract is implemented end to end;
- runtime contains no UNSEEN/donor-generation product dependency;
- dynamic visual-integrity proof diagnoses geometry / visibility / appearance / sampling failures separately;
- closure and editable-authoring export bind the new authorities;
- adversarial module-audit/readiness tests pass.

Only then is Subject-2 Knight minted as the fresh V2 witness lineage.
