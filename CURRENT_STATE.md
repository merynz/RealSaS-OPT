# RealSaS-OPT — Current State

**Date:** 2026-09-21  
**Canonical branch:** `main`  
**Current mode:** **V2 APPEARANCE FIDELITY RECLOSURE — WITNESS FORBIDDEN**  
**Implementation readiness:** `REOPENED_APPEARANCE_AND_DYNAMIC_VISUAL_FIDELITY_AUDIT__WITNESS_FORBIDDEN`  
**Current status token:** `APPEARANCE_FIDELITY_RECLOSURE__WITNESS_FORBIDDEN`  
**Rehydration focus token:** `REOPENED__WITNESS_FORBIDDEN`  
**Canonical governance ledger:** `V2_IMPLEMENTATION_ASSEMBLY` — implementation governance only; never a subject witness ledger.  
**Witness execution:** **FORBIDDEN** — post-seal appearance review reopened tile density, completion quality, cross-view RGB consistency and dynamic hole/speckle gates. User approval is necessary later but is not sufficient until these blockers close.  
**Mainline:** 46-stage dependency DAG; ordinal is display order only.

## Read first

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
4. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
5. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
6. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
7. `canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json`
8. `canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`

## Historical pre-reopen green evidence

- Documentation-inclusive implementation head tested: `4419509ee0f62a7e5ca088b87feb682b1a664105`.
- Documentation-inclusive implementation closure SHA-256: `c2ba1569285811ecbac448c54a4fbe07d57111f76ff4f7003c7482c7ce26306e`.
- Plan SHA-256: `b703b139d30987de0df8c7753eb0ecb8a69134d4c87fd0247ff5e478d1d6163f`.
- Self-hosted mainline CI run `35539075794`, job `106153297550`: **PASS**.
- Repository/governance tests: **49 PASS**.
- Learned/model ownership tests: **51 PASS**.
- Compiler regressions: **288 PASS**.
- Native V2 CAA player SHA-256: `b1440a6c421123620ec7052d83100bd0fbea1116843f987ac312d419de1422e1`.
- Model/source ownership run `35539075776`: **PASS**.
- Subject-free Stage01–08 orchestration run `35539075769`, job `106153297245`: **PASS**.
- Uploaded orchestration evidence ZIP SHA-256: `e5d994baccae4baee42edb9b00381504f972e619d97521694f10efdb0398c1f1`.
- Canonical assembly ledger remained unchanged during dry-run.
- Knight data used during red-team repair/calibration: **none**.

## 46-stage red-team result

The second-pass audit found two real product-quality gaps and both are now closed subject-free with scoped claims:

1. **RT-37 — presentation partition:** Stage37 emits hash-bound `PresentationPartitionEvidenceIR.v2`. Connected regions inside one mechanical component may split only when frozen source-backed CAA appearance-boundary evidence supports it. Categorical object identity is not minted.
2. **RT-45 — dynamic appearance:** Stage45 gates native parity, provenance and exposure **plus rigid-motion-invariant intrinsic UV→surface deformation conditioning**. Raw screen-space conditioning was explicitly rejected as a shipping gate because legitimate 3D foreshortening is not artwork deformation.
3. **RT-46 — product closure:** Stage46 requires both repaired authorities and exports presentation-partition evidence in the editable authoring bundle before `PASS_PRODUCT_V2`.

## Product authority

### Geometry
Owns the renderable canonical surface: silhouette capacity, topology, stable `SurfaceAddressing` and rasterizable conditioning. The single qualified product-geometry identity after mesh qualification is `QualifiedMeshIR`.

### Mechanics
Owns skeleton, skin, deformation, contacts and full-3D motion.

### Appearance
The **Complete Appearance Authority** owns source-faithful total 2D art, provenance, completion, seams, premultiplied-alpha sampling, exposure and bounded intrinsic deformation.

### Presentation
Owns role-free slots/attachments/editable grouping from observable evidence. It may not mint semantic object identity.

A failure on any one quality axis is a product failure. Mechanics/runtime correctness may not hide appearance or presentation failure.

## Claim boundary

The prior READY seal is historical evidence only and is currently revoked. The live authority is the reopened subject-free visual-fidelity/governance reclosure. Nothing in the current state authorizes Knight execution, proves Knight product performance, unseen generalization, semantic recognition, or universal human aesthetic optimality.

## Current reclosure blockers

1. **VF-11:** R256/R384 geometry feature-survival floors were insufficient; byte-identical slabbed apparatus is validated and the preregistered R512 same-physical-feature sweep is active.
2. **VF-23:** exact unmodified production-policy Stage20→46 same-context E2E plus representative subject-free semantic bank must pass on a current compatible implementation.
3. **Exact-head closure:** mainline, runtime/native, source ownership, native source seal and subject-free orchestration evidence must be green and bound to the final implementation closure.
4. **Explicit user approval:** even after technical readiness is resealed, named witness initialization is machine-blocked until `APPROVED_EXPLICITLY_BY_USER`.

## Next action

Complete R512, VF-23 and exact-head reclosure. Only after technical readiness is resealed should the user be asked for explicit Knight authorization. A fresh run-local Knight ledger may then be minted; canonical `ACTIVE_RUN_V2.json` remains implementation governance only.

## Post-seal appearance fidelity reopening — 2026-09-21

The prior READY seal was revoked before Knight execution. New review found that fixed CAA tile resolution 8 was capacity-derived rather than art-frequency qualified; unobserved completion still uses nearest-surface color copy; cross-view RGB compatibility is not an explicit authority; dynamic alpha holes are currently diagnostic rather than a hard Stage45 failure; and scattered/speckled holes are not independently bounded. Current learned-model fits remain scoped FIT evidence and are not claimed final or unseen-ready.


## Knight demo-only execution — 2026-09-24

A separate investor-demo execution lane is active and does **not** change product readiness or scientific geometry status.

- Demo run: `SUBJECT2_KNIGHT_DEMO_V2_20260924` / `DEMO_WITNESS`.
- IRIS Stage13 remains scientific **FAIL**; product `PASS_PRODUCT_V2` is forbidden on this lineage.
- Demo geometry is frozen C: checkpoint `222350d1d1fb37fffdf0b5cbef3ca941fe94dd440451ee9fd0e6f8029b51d2e0`; zero-surface `adbaf0a939eb414631736dabba1140245e88b14a41af0dc172adb7a25e5c084c`.
- Fresh demo Stage01–08 self-hosted preflight run `36022491562` passed, but its render bytes/numerics drift slightly from the historical frozen-C upstream and therefore are **not** being silently rebound to the checkpoint.
- Exact frozen-C Stage08 upstream remains GitHub artifact `10586270615` from run `35447124607`, digest `5cd95805c1096d7b1a1a2a433a8db90c2a9d8e26650fd3ac711069730111c6b7`.
- Durable continuation authority: `canonical/KNIGHT_DEMO_EXECUTION_STATE_20260924.json`.
