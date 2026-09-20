# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace. **Proprietary / all rights reserved.** See `LICENSE` and `THIRD_PARTY_NOTICES.md`.


## Current implementation status — V2 red-team hardening

The 46-stage V2 contract has passed self-hosted implementation/orchestration closure before red-team. The second-pass red-team found two proof-scope overclaims and the current hardening makes both explicit in machine-readable IR:

- Stage37 automatic presentation segmentation is limited to connected face islands within mechanical components; original artist-layer recovery and connected mechanically-equivalent visual-region auto-splitting are **not** claimed.
- Stage45 proves exhaustive native dynamic runtime integrity, provenance totality and compiled-unobserved exposure; it explicitly does **not** claim that runtime parity/mechanical conditioning proves professional perceptual 2D-art quality.
- Stage46 `product_pass` means **V2 executable-contract closure**. Spine-class perceptual quality remains a first-class product target and must be evaluated on the controlled witness; it is not pre-certified by deterministic parity.

The hardening is part of the exact implementation closure. Knight may run only after that exact closure is green, readiness is resealed, and the user explicitly approves execution.

## Current product architecture — RealSaS V2

RealSaS compiles qualified multi-view 2D artwork into an editable 2D/2.5D puppet with internal 3D mechanics.

The current product path is the 46-stage dependency DAG in:

`canonical/MAINLINE_EXECUTION_PLAN_V2.json`

The three co-equal product-quality authorities are:

- **Geometry** — canonical renderable surface, silhouette capacity, topology, stable addressing and rasterizable conditioning.
- **Mechanics** — skeleton, skin, deformation, contacts and full-3D motion.
- **Appearance** — complete source-preserving 2D art, provenance, holdout/seam quality, alpha/sampling and dynamic exposure.

A visually incorrect puppet does not pass because its mesh, rig and skin are mechanically valid.

## Core architecture

```text
Source observations
      │
Geometry / IRIS
      │
     GSA
      │
Canonical Mesh Domain
   /             \
Mechanics       Appearance / CAA
   \             /
     Complete Puppet
           │
         Motion
           │
Deterministic Runtime
           │
Dynamic Visual Integrity
           │
      Stage46 Closure
```

The canonical mesh is the shared mechanics/appearance address domain.

Visibility is owned by posed canonical XYZ + camera depth. Appearance does not choose the front surface.

Runtime performs no donor search, generative appearance correction, hidden retriangulation, PBR character relighting or normal-derived character shading.

## Current execution state

The previous V2 implementation-readiness seal was revoked after pre-Stage01 orchestration gaps were discovered.

Current subject-witness execution is forbidden until:

1. every required readiness proof is PASS;
2. the subject-free run-local Stage01–08 orchestration/artifact dry-run is PASS;
3. the exact implementation-closure fingerprint matches current main;
4. `canonical/V2_IMPLEMENTATION_READINESS.json` is resealed as `READY_FOR_WITNESS_EXECUTION`.

Subject-2 Knight is the next witness only after that gate. Earlier premature Knight attempts are not admissible scientific witness evidence.

## Execution authority

The repository ledger:

`canonical/ACTIVE_RUN_V2.json`

is implementation-governance state only.

Real executions use run-local ledgers:

`$REALSAS_AUTHORITY_ROOT/runs/<run_id>/ACTIVE_RUN_V2.json`

The orchestrator is:

`compiler/realsas_compiler_services/orchestrator/mainline.py`

Current execution semantics are dependency-DAG based; stage ordinals are human-readable order only.

## Appearance authority

Complete Appearance Authority is a first-class subsystem, not renderer polish.

Current implementation homes:

- `compiler/realsas_compiler_core/appearance_authority_v2.py`
- `compiler/realsas_compiler_core/appearance_compile_v2.py`
- `compiler/realsas_compiler_core/appearance_bake_v2.py`
- `compiler/realsas_compiler_core/appearance_quality_v2.py`
- `compiler/realsas_compiler_core/appearance_render_v2.py`
- `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
- `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`

CAA is total over supported renderable surface states, preserves explicit provenance, may not mutate canonical geometry, and uses premultiplied-alpha filtering/compositing with mandatory atlas bleed.

## Native runtime

The V2 runtime consumes exact qualified topology, posed XYZ, CAA UV/atlas/provenance and sealed cameras.

Current homes:

- `compiler/realsas_compiler_core/runtime_authority_v2.py`
- `compiler/realsas_compiler_core/runtime_package_v2.py`
- `runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp`

Native/reference byte parity and Dynamic Visual Integrity are executable runtime-integrity gates. They do not substitute for controlled-witness evaluation of Spine-class perceptual 2D-art quality.

## Start here

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `CURRENT_STATE.md`
4. `canonical/CONTEXT_STATE_V2.json`
5. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
6. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
7. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
8. `canonical/V2_ADVERSARIAL_MODULE_AUDIT_PROTOCOL_20260920.md`
9. `REPOSITORY_MAP.md`
10. `SYSTEM_INDEX.md`

## Historical evidence

Mage FIT1/FIT2, the old 40-stage V1 product path, donor-era appearance/runtime code and historical repair branches remain preserved scientific evidence.

They are not current continuation authority unless explicitly promoted by the V2 authority spine.

Promotion and supersession never delete scientific history; they only change what current main is allowed to claim and execute.

## Scientific boundary

Same-witness FIT evidence does not establish unseen-family generalization.

A product PASS requires exact Stage46 closure across Geometry, Mechanics, Appearance, Motion, Runtime and native Dynamic Visual Integrity.

## Repository governance

- `CONTRIBUTING.md` — scientific change and promotion rules.
- `SECURITY.md` — private reporting and artifact handling.
- `.github/CODEOWNERS` — authority/source ownership.
- `REPOSITORY_MAP.md` / `SYSTEM_INDEX.md` — current navigation.
- Current authority workflows run only on self-hosted RealSaS runner labels `[self-hosted, linux, x64, realsas]`.
