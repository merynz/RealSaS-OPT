# RealSaS-OPT

Canonical RealSaS research, compiler, proof and runtime workspace. **Proprietary / all rights reserved.** See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Current implementation status — V2 stage red-team hardened

The current RealSaS V2 implementation completed a subject-free 46-stage second-pass red-team. The two product-claim gaps found by that audit were repaired without using Knight:

- **RT-37 presentation partition:** Stage37 now emits a separate role-free `PresentationPartitionEvidenceIR.v2`. Mechanically equivalent, topologically connected regions can become independently addressable only when frozen **source-backed CAA appearance-boundary evidence** supports the split. Categorical object identity is not invented.
- **RT-45 dynamic appearance quality:** Stage45 now gates consequential visible faces with **rigid-motion-invariant intrinsic UV→surface conditioning**, rest-relative stretch/shear and adjacent-frame intrinsic stretch, in addition to native byte parity, provenance totality and compiled-unobserved exposure.
- **RT-46 closure:** Stage46 consumes both repaired authorities and exports presentation-partition evidence in the editable authoring bundle before `PASS_PRODUCT_V2`.

Functional hardening head `66d2e2a0cd60a6b3fcc462db15a71d9ad0c976d1` passed the self-hosted mainline gate: **49 repository/governance + 51 learned/model + 288 compiler tests**. The real subject-free Stage01–08 orchestration dry-run also passed.

This is implementation evidence, not Knight performance evidence. Subject-2 Knight may execute only when `canonical/V2_IMPLEMENTATION_READINESS.json` is exactly `READY_FOR_WITNESS_EXECUTION`, its implementation-closure fingerprint matches current `main`, and the user explicitly approves execution.

## Current product architecture — RealSaS V2

RealSaS compiles qualified multi-view 2D artwork into an editable 2D/2.5D puppet with internal 3D mechanics.

The current product path is the 46-stage dependency DAG in `canonical/MAINLINE_EXECUTION_PLAN_V2.json`.

The co-equal product-quality authorities are:

- **Geometry** — canonical renderable surface, silhouette capacity, topology, stable addressing and rasterizable conditioning.
- **Mechanics** — skeleton, skin, deformation, contacts and full-3D motion.
- **Appearance** — complete source-preserving 2D art, provenance, completion quality, alpha/sampling, seams and dynamic deformation conditioning.
- **Presentation** — role-free slots/attachments/grouping derived from observable evidence, never categorical identity invention.

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
 Presentation Partition
          │
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

The canonical mesh is the shared geometry/mechanics/appearance address domain. Visibility is owned by posed canonical XYZ + camera depth. Appearance does not choose the front surface.

Runtime performs no donor search, generative appearance correction, hidden retriangulation, PBR character relighting or normal-derived character shading.

## Appearance and presentation authority

Static CAA authority:
- `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
- `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`
- `compiler/realsas_compiler_core/appearance_authority_v2.py`
- `appearance_compile_v2.py`, `appearance_bake_v2.py`, `appearance_quality_v2.py`

Role-free presentation partition:
- `canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json`
- `compiler/realsas_compiler_core/presentation_partition_v2.py`

Dynamic appearance conditioning:
- `canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`
- `compiler/realsas_compiler_core/dynamic_appearance_conditioning_v2.py`

These authorities deliberately do not claim semantic object recognition or human aesthetic optimality.

## Execution authority

Real execution is run-local:

```text
$REALSAS_AUTHORITY_ROOT/runs/<run_id>/
    run_manifest.json
    ACTIVE_RUN_V2.json
    artifacts/<stage_id>/...
```

`canonical/ACTIVE_RUN_V2.json` is repository implementation-governance state only; it is never reused as a subject witness ledger.

Mainline execution is a dependency DAG. `depends_on` defines readiness; ordinal is display order only. Verified independent upstream outputs survive downstream failure under exact identity.

## Read first

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `CURRENT_STATE.md`
4. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
5. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
6. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
7. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
8. `canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json`
9. `canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`

## Claim boundary

Current V2 subject-free implementation evidence supports the exact controlled eight-view architecture and its fail-closed proof chain. It does **not** by itself prove Knight product performance, unseen-family generalization, arbitrary real-world view normalization, semantic object segmentation or universal perceptual/aesthetic optimality.

Historical Mage/FIT/V1 artifacts remain scientific provenance only unless explicitly promoted by the current V2 authority spine.
