# RealSaS-OPT — Restoration State

> **SUPERSEDED FOR CONTINUATION / PRESERVED AS HISTORICAL EVIDENCE.**  
> This file records the 2026-09-04 restoration-era state and contains stop/go statements that were correct at that time but are **not current authority**.  
> For present work start from `canonical/REHYDRATION_PACKET.md` and `CURRENT_STATE.md`. Do not use this file to infer whether FIT1 or a current experiment is authorized.

**Historical snapshot date:** 2026-09-04  
**Source branch:** `restoration/compiler-runtime-promotion-v1-20260903`  
**Verified behavioral base:** `2b5d467186839401ab30f9566015d9e9d49a2a06`  
**Global architecture refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`  
**Real-family fit:** `NOT AUTHORIZED` **(historical restoration-era status; superseded for current continuation)**

## Mission

Promote still-valuable historical Compiler/runtime production knowledge **behind** the current V4 typed Compiler authority, while converting the repository into a readable research library with one obvious current home for each executable subsystem.

> We are not restoring the old Compiler as owner. We are restoring selected production mechanisms as subordinate services of the current Compiler.

> We are not sterilizing a research repository. We are establishing a visible current mainline that experiments can falsify and upgrade.

## Current restoration ledger

| Stage | State | Commit / evidence |
|---|---|---|
| Repository authority + navigation skeleton | **DONE** | `fe8bedfbe0f04519711432942f656aaf4735603e` |
| Exact native C++ runtime consumer | **DONE / CI PASS** | `25d810e272cc00a7b6fd4d682eabc16fef226223` |
| Diagnostic failure-signature semantic rebind | **DONE / CI PASS** | `5665ecacb77c02e53297340249b0972981a5c4bc` |
| Current learned-model source ownership promotion | **DONE IN CURRENT TREE** | `models/` + model READMEs + `SYSTEM_INDEX.md` |
| IRIS -> Compiler substrate ownership seam | **DONE** | `compiler/realsas_compiler_core/substrate/iris_v2.py` |
| Compiler substrate physical normalization | **DONE** | `bc443d063d4d8f0bd52981ec2db5b99a795985e1` |
| Compiler mesh physical normalization | **DONE** | `295b798b1fb40ccb1752afe9da9378dfe8734262` |
| Direct authored-motion evaluator promotion | **RETRACTED** | `canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json` |
| Qualification-owned motion bake + fail-closed proof seam | **DONE / CLOSURE PASS** | `proof/motion_bake.py`, `proof/motion_frame_metrics.py`, current `proof_engine.py` |
| Compiler-owned directional joint/view binding | **DONE / CLOSURE PASS** | `compiler/realsas_compiler_core/directional_binding.py` + restoration closure verdict |
| Qualified directional motion evaluator | **DONE / ROTATION-ONLY CURRENT PRESET / CLOSURE PASS** | typed binding + typed provider; unsupported semantics fail closed |
| Typed motion-provider authority | **DONE / FAIL-CLOSED / CLOSURE PASS** | arbitrary callbacks rejected; exact product/binding/policy/evaluator identity required |
| Controlled causal owner attribution | **DONE / REGRESSION PASS** | `compiler/realsas_compiler_services/proof/causal_attribution.py` |
| Bounded repair directive + mandatory same-probe re-proof contract | **DONE / REGRESSION PASS** | `compiler/realsas_compiler_services/proof/repair_loop.py` |
| Core repair operation registry + real child-state delta audit | **DONE / FAIL-CLOSED** | `12eef470348a4e9822dd927e2ac950e3fcd9dfb7` |
| Repair execution authority for first-fit | **DONE / 0 EXECUTORS ACCEPTED FAIL-CLOSED** | `canonical/REPAIR_EXECUTION_AUTHORITY_DISPOSITION_V1_20260904.json` |
| Historical rig/weight repair executors | **NOT PROMOTED** | future typed requalification seams required |
| Pure native runtime-v2 package writer | **VALID / KEEP** | `compiler/realsas_compiler_services/export/runtime_v2.py` |
| Current V4 proof/bake -> native-v2 projection | **DONE / CLOSURE PASS** | proof-owned rest/frame XY, exact local raster UV, no solver replay |
| Current V4 `.rss/.rsr` -> sealed C++ runtime interlock | **DONE / CI PASS** | run `33827594057`; `PASS_CURRENT_V4_NATIVE_PACKAGE_OPEN_RENDER` |
| Historical CDT / BBW-KKT / ARAP / XPBD/contact source diff | **DONE / NO CURRENT PROMOTION REQUIRED** | `canonical/HISTORICAL_NUMERICS_SOURCE_DIFF_DISPOSITION_V1_20260904.json` |
| Canonical-main-before-fit gate | **SEALED / MANDATORY** | `canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json` |
| Full behavioral + complete-E2E restoration closure | **DONE / PASS** | closed source `7324e6a63df54d999103a59835212cb2d1dc46ee`; run `33827594057`; verdict `canonical/RESTORATION_CLOSURE_VERDICT_V1_20260904.json` |
| Canonical `main` promotion | **REQUIRED REF EQUALITY BEFORE FREEZE** | the closure-qualified tree must be the canonical `main` ref before post-merge integrity can qualify a first-fit base |
| Post-merge `main` integrity check | **MANDATORY ON EXACT CANONICAL MAIN** | the exact `main` SHA must pass repository integrity before first-fit-base freeze |

P0/runtime source-closure evidence: `canonical/P0_DIRECTIONAL_BINDING_RUNTIME_INTERLOCK_CLOSURE_V1_20260904.json`.  
Historical numerics disposition: `canonical/HISTORICAL_NUMERICS_SOURCE_DIFF_DISPOSITION_V1_20260904.json`.  
Repair execution disposition: `canonical/REPAIR_EXECUTION_AUTHORITY_DISPOSITION_V1_20260904.json`.  
Canonical-main-before-fit gate: `canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json`.  
Restoration-wide closure verdict: `canonical/RESTORATION_CLOSURE_VERDICT_V1_20260904.json`.

## Repository organization contract

```text
mainline library = models/ + compiler/ + runtime/
labs             = experiments/
decision/evidence= canonical/
provenance reserve= historical/
```

`models/` contains the semantic homes for IRIS, Geppetto, SkinFieldCodec and Arachne. Models emit evidence/proposals only; Compiler qualification remains authoritative. Mainline must not permanently import dated experiment implementations.

## Directional motion authority

The previously promoted direct evaluator was retracted because it treated mechanical joint coordinates and directional mesh `P.xy` as one coordinate system. That shortcut remains forbidden.

Current directional authority is now:

1. `DirectionalJointViewBindingSetIR` derives view projection only from admitted mechanical-surface `P` <-> raster correspondences;
2. rank-3 planar projection is allowed only when every canonical joint is qualified on the admitted surface affine hull;
3. directional joint pivots are produced only through that qualified projection;
4. directional mesh rest raster coordinates come from each mesh vertex's admitted `SurfaceSupportBinding`, not from `mesh.P.xy`;
5. a typed `QualifiedDirectionalMotionBakeProviderV1` binds the exact product, directional binding, evaluator policy and evaluator semantic version;
6. current evaluator scope is the generic rotation-only preset lane; nonzero translation, non-unit scale, depth offset, and unqualified order/visibility animation fail closed;
7. `motion_bake.py` binds the exact frames to product + proof plan + clip + evaluator identity;
8. `motion_frame_metrics.py` measures deformation consequences;
9. export consumes those exact proof-owned frames and never replays the motion evaluator or a solver.

Missing qualified frame evidence still makes MOTION **ABSTAIN**.

## Native runtime authority

`RUNTIME_CONSUMPTION` inside `ProductProofBundleIR` is explicitly a **pre-export runtime-contract compatibility** domain. It does not claim that the native package was executed.

Post-export native authority is a separate interlock:

`PASS proof -> exact proof-owned bakes -> current V4/native-v2 projection -> .rss/.rsr -> sealed runtime/realsas_cpp open -> exact source/proof hash check -> 8-view sample -> software render`.

That post-export interlock is **executed and PASS** on closed source commit `7324e6a63df54d999103a59835212cb2d1dc46ee` in self-hosted Actions run `33827594057`. The emitted package bound `SOURCE_BINDING_SHA256=6ab523629162c12dd3f34fed826af68c04a1ec9e5bb66ddf406b04223d8cc12e` and `PROOF_BUNDLE_SHA256=7500b2bb96fef7d79ad33dc8e9cdacce945de25d6a872f92a11c5b628ea5166e`; the external sealed C++ probe returned `PASS_CURRENT_V4_NATIVE_PACKAGE_OPEN_RENDER`.

## Historical numerical backends

Historical v0.5 CDT, BBW/KKT, ARAP, XPBD and SDF-contact implementations were source-diffed against current V4.

They remain valuable future backends, but none is a current restoration dependency:

- CDT would need typed support lineage for generated/Steiner vertices before it could emit a current `MeshDiscretizationCandidateIR`;
- BBW/KKT would have to emit `SkinProposalIR` or retained candidate evidence and remain behind current skin qualification;
- ARAP would require a typed corrective-deformation contract that binds solver identity into proof-owned frames and export;
- XPBD/contact require a future secondary-dynamics/contact product and runtime contract.

Therefore **no numerical bulk restore is authorized**. Current LBS remains the only promoted numerical runtime/deformation dependency required by present V4.

## Repair execution policy

The current proof/repair architecture is complete as a fail-closed recovery contract, but has **0 executable repair operations**.

This does **not** block first-fit-base freezing because automatic repair is neither a required product capability nor a required proof domain. The operational rule is:

- PASS product proof -> may proceed to proof-gated runtime export;
- FAIL/ABSTAIN -> stop; no automatic repair credit and no current runtime export;
- future repair executors must be separately promoted through current proposal/requalification seams and distinct child-state re-proof.

Thus historical rig-parent and retained-weight repair code is not restored merely to satisfy a checklist.

## Canonical-main-before-fit rule

Real-family fit work may not begin directly from the restoration branch, a safety branch, or an experiment branch.

Required order:

1. close restoration source/behavioral/native gates — **DONE**;
2. record an explicit restoration verdict — **DONE**;
3. make the closure-qualified tree the canonical GitHub `main` ref;
4. run repository-integrity checks on that exact `main` SHA;
5. freeze that exact passing `main` SHA as the first-fit base;
6. only then may a later workflow authorize Family-1/FIT work.

The first fit must record the exact canonical `main` commit it descends from. This document intentionally records the invariant sequence rather than a time-sensitive branch-ref status, so the same qualified tree remains truthful across the fast-forward transition.

## Non-negotiable firewalls

- no historical front-brain/teacher-exact ownership path;
- no second canonical graph/ID authority;
- no historical solver or repair executor executes merely because its source exists;
- no mechanical joint Vec3 / directional mesh `P.xy` shared-frame shortcut;
- no MOTION PASS without a typed qualification-owned directional provider and exact frame evidence;
- no arbitrary callable may masquerade as motion-bake authority;
- no proof can bind a different product state than export/runtime;
- no export-time evaluator/solver replay;
- no cross-view/completion texture projection into native-v2 until its atlas authority is separately qualified;
- no historical numerical backend becomes current merely because it once produced strong metrics;
- no failure signature may invent causal ownership;
- no repair may mutate production state without a distinct child lineage and mandatory same-probe re-proof;
- no permanent current dependency on a dated experiment implementation;
- no family-specific constants during restoration;
- no FIT8 / Family-1 execution until canonical `main` promotion and post-merge integrity explicitly re-authorize it.

## Promotion / freeze protocol

1. explicit restoration verdict — **DONE**;
2. canonical `main` ref must equal the closure-qualified tree;
3. that exact `main` SHA must pass post-merge repository integrity;
4. that exact passing `main` SHA must be frozen under a dedicated first-fit-base ref;
5. stop before any Family-1/FIT execution.
