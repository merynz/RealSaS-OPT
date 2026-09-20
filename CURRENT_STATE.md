# RealSaS-OPT — Current State

**Date:** 2026-09-20  
**Canonical branch:** `main`  
**Current mode:** **V2 STAGE-LEVEL RED-TEAM HARDENING**  
**Current status token:** `V2_STAGE_REDTEAM_REOPENED__WITNESS_FORBIDDEN`  
**Canonical governance ledger ID:** `V2_IMPLEMENTATION_ASSEMBLY` — repository implementation governance only; not a witness run.  
**Witness execution:** **FORBIDDEN** — orchestration is now green, but the readiness seal remains revoked by stage-level product-claim red-team findings RT-37 and RT-45/46.  
**Next witness:** Subject-2 Knight only after RT-37 and RT-45/46 are resolved or deliberately narrowed, a fresh closure is green, readiness is resealed, and the user explicitly approves execution  
**Mainline:** 46-stage dependency DAG; ordinals are display order only

## Read first

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
4. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
5. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
6. `canonical/V2_ADVERSARIAL_MODULE_AUDIT_PROTOCOL_20260920.md`
7. this file

## Latest exact green evidence

- Functional head: `ca9d2d2e59bc517a139c743eb6a91df2e48ac761`.
- Self-hosted mainline CI run `35536134130`: PASS; 48 repository/governance + 51 learned/model + 281 compiler tests.
- Subject-free Stage01–08 orchestration/artifact dry-run `35536134136`: PASS.
- Model source gate `35536134125`: PASS.
- Pre-red-team implementation closure: `a46361fc2d9b989a2a4490829b73b835202521c5efed6916e44aab253cabf882`.
- The audit then reopened readiness; green implementation evidence is preserved, not erased.

## Current red-team blockers

1. **RT-37 — automatic presentation segmentation:** current V2 splits disconnected face islands inside mechanical components, but does not automatically split mechanically equivalent, topologically connected visual regions that need independent Spine-style addressing.
2. **RT-45/46 — dynamic appearance quality:** Stage45 exhaustively proves native/reference parity, provenance totality and compiled-unobserved exposure, but not dynamic 2D-art deformation quality such as screen-space texture/line distortion or temporal edge integrity. Stage46 therefore must not over-read contract closure as proof of universal Spine-class dynamic appearance.

Canonical detail: `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`.

## The V2 product has three co-equal quality authorities

### Geometry
Owns the renderable canonical surface: silhouette capacity, topology, stable addressability and rasterizable triangle conditioning. The qualified single product geometry identity is `QualifiedMeshIR`; stable surface addressing is carried by `SurfaceAddressingIR`.

### Mechanics
Owns skeleton, skin, deformation, contacts, motion and dynamic conditioning.

### Appearance
Owns source-faithful total 2D art: color, alpha, line character, provenance, unseen completion quality, provenance-boundary seams, sampling behavior and temporal visual continuity.

**A failure on any one axis is a product failure. A visually wrong puppet does not pass because its mesh, rig and skin are mechanically valid.**

The target remains Spine-class 2D art with internal 3D mechanics. Mechanics may be 3D; presentation must remain 2D-authored in character.

## Why V2 exists

V1 gave geometry and mechanics rigorous typed authorities and proofs, but appearance remained too close to donor, fallback and renderer logic. It also mixed independent failure classes: carrier coverage, visibility/depth, appearance definedness and texture sampling.

V2 separates them. See `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`.

Key corrections:
- real DAG execution rather than hidden linear stage semantics;
- source observations separated from output V0..V7 directions;
- IRIS/GSA treated as geometry/surface evidence, not final art authority;
- mesh producer/gate feasibility audited before downstream qualification;
- stable mesh addressing created with the mesh domain;
- Complete Appearance Authority makes supported appearance total before runtime;
- CAA cannot hide geometry, depth or sampling failures;
- visibility = posed canonical XYZ + z-buffer;
- art = sealed CAA;
- premultiplied-alpha filtering/compositing + atlas bleed;
- native Dynamic Visual Integrity is a product gate, not a cosmetic screenshot.

## Current audit blocks

Knight does **not** run yet. The prior readiness seal is not authoritative. Current main must first close:
- IRIS/GSA role and Stage13 geometry-floor redesign;
- relation-graph / 3-clique / CDT mesh-quality audit;
- CAA deterministic compile, bake, qualification, holdout and seam logic;
- source-lock confidence/correspondence rules;
- premultiplied alpha + bleed end to end;
- removal of current-path donor/UNSEEN runtime semantics;
- posed-XYZ visibility/equal-depth proof;
- mechanics branch revalidation against frozen mesh;
- presentation and complete puppet seal;
- motion dynamic exposure budget;
- Runtime CAA-only binding and package-load verification;
- Dynamic Visual Integrity attribution;
- Stage46 closure and editable-authoring export;
- complete adversarial module audit;
- witness orchestration dry-run: run-local ledger initialization, target-closure CLI, artifact schema/output verification and Stage01–08 contract without subject-result admission.

## Knight rule

Only after a **new** implementation-readiness seal is green do we mint a fresh Knight V2 run from Stage01. The previous seal is explicitly revoked and attempts triggered from it are inadmissible as scientific witness evidence. Exact controlled eight-source input remains the first witness apparatus to isolate architecture changes; it is not a permanent assertion that product input must always contain eight views.
