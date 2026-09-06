# New Chat Handoff — Pre-C Rigging Line Closed

**Date:** 2026-09-06  
**Branch:** `e2e/mage-scene-first-v1-20260905`  
**Parent before closure patch:** `53067213c4b05ab0b6b9c8c38aa2424daf4b7321`  
**Status:** `CONTINUE_WITH_NOTEBOOK_DESIGN_AUDIT_THEN_C_EXPERIMENTS`  

## Binding state

The non-experimental rigging infrastructure from scene-first GSA through product-authoritative `QualifiedSkeletonIRV2` is closed by the commit containing this handoff.

Read first:

- `canonical/RIGGING_LINE_PRE_C_CLOSURE_V1_20260906.md`
- `canonical/RIGANYTHING_CODE_LEVEL_REFERENCE_AUDIT_V1_ERRATUM_20260906.md`

Key code:

- `models/geppetto/v2/geppetto_conditioning_v2.py`
  - diagnostic adapter remains non-promoting
  - `GeppettoProductConditioningAdapterV2` is the strict product adapter
  - product conditioning certificate
  - preserved GSA local topology + topology neighbor helper
- `compiler/realsas_compiler_core/substrate/validation.py`
  - strict scene-first boundary validation
  - boundary audit hash
  - exact topology fingerprint
- `compiler/realsas_compiler_core/skeleton_admission_v1.py`
  - explicit pre-optimizer admission
  - no geometry-only fusion
  - no deform-node synthesis
  - product support requirement
- `compiler/realsas_compiler_core/rig.py`
  - compatibility qualifiers are non-promoting
  - `compile_scene_first_rigging_v1` is product authority
  - single connected deform-tree policy
- `tests/compiler/test_rigging_line_closure_v1.py`
- `.github/workflows/rigging_line_closure_v1.yml`

## Scientific state

Do not reopen IRIS/GSA unless new evidence points upstream.

Do not begin Arachne/SkinTokens yet.

Do not treat the current RigAnything challenger as already promoted.

The next work is exactly:

1. inspect how the successful IRIS and Mage FIT notebooks were structured;
2. design new C experiments on the strict product conditioning + compiler route;
3. keep the 41-control / 31-locus exact-FIT regression and proven phase-dependent optimizer policy;
4. compare C0/C1/C2 first;
5. add C3/C4 only if needed by evidence;
6. separately close native STOP/count under the winner;
7. promote/freeze the winning rigging architecture;
8. only then move to Arachne and use SkinTokens as the primary skinning reference.

## Product authority reminders

- historical compact stripped-raster Mage fixtures are diagnostics, not product-substrate authority;
- GSA topology is preserved but C0 still explicitly uses Euclidean KNN until an experiment promotes another locality prior;
- canonical graph optimizer owns root/parents, not native joint count;
- geometry-only duplicate fusion is forbidden because legitimate controls can share a locus;
- deform-node completion budget is zero until a separately proven completion owner exists;
- current product graph policy is one connected deform tree;
- `QualifiedSkeletonIRV2` remains semantically future-capable, but current product promotion does not claim multi-root forest support.

## Stop point

Skinning is out of scope. The next chat should continue directly with the notebook-design audit and C experiment construction.
