# RealSaS-OPT — Restoration State

**Date:** 2026-09-03  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`  
**Verified behavioral base:** `2b5d467186839401ab30f9566015d9e9d49a2a06`  
**Global architecture refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`  
**Real-family fit:** `NOT AUTHORIZED`

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
| Qualification-owned motion bake + fail-closed proof seam | **DONE** | `proof/motion_bake.py`, `proof/motion_frame_metrics.py`, current `proof_engine.py` |
| Directional joint/view binding | **P0 BLOCKER** | `CURRENT_DIRECTIONAL_JOINT_VIEW_BINDING_MISSING` |
| Controlled causal owner attribution | **DONE / LOCAL REGRESSION 4/4 PASS** | `1dd9a52dda4904c4910f47e66559974b0dc4ad72` |
| Bounded repair directive + mandatory same-probe re-proof contract | **DONE / LOCAL REGRESSION 5/5 PASS** | `9a60fa341055713c9df8b5d698c58c19a4098025` |
| Core repair operation registry + real child-state delta audit | **DONE / FAIL-CLOSED** | `12eef470348a4e9822dd927e2ac950e3fcd9dfb7` |
| Historical rig/weight repair executors | **NOT PROMOTED** | current typed seams do not authorize them |
| Pure native runtime-v2 package writer | **VALID / KEEP** | `compiler/realsas_compiler_services/export/runtime_v2.py` |
| Current V4 export -> native runtime interlock | **BLOCKED ON P0** | exact proof-owned bake required before projection/materialization |
| CDT / BBW-KKT / ARAP / XPBD/contact source diff | **PENDING** | no numerical bulk restore |
| Full behavioral + complete-E2E restoration closure | **PENDING** | required before refreeze decision |

## Repository organization contract

```text
mainline library = models/ + compiler/ + runtime/
labs             = experiments/
decision/evidence= canonical/
provenance reserve= historical/
```

`models/` contains the semantic homes for IRIS, Geppetto, SkinFieldCodec and Arachne. Models emit evidence/proposals only; Compiler qualification remains authoritative. Mainline must not permanently import dated experiment implementations.

## Motion-proof correction

Authored/requested motion still must be dynamically exercised before MOTION PASS. However, the Compiler core and proof binder are **not allowed to manufacture directional frames by treating mechanical `QualifiedJoint.position` and directional editable-mesh `P.xy` as the same coordinate frame**.

The previously promoted direct evaluator was therefore retracted. Its historical promotion record remains as provenance; `canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json` is the current disposition.

Current authority is:

1. a separately qualified directional evaluator produces frames using a typed Compiler-owned joint/view binding;
2. `motion_bake.py` binds those frames to the exact product state, proof plan, clip and evaluator identity;
3. `motion_frame_metrics.py` measures deformation consequences;
4. missing qualified bake => MOTION **ABSTAIN**;
5. export must consume the same bound frames; solver replay at export is forbidden.

The missing prerequisite is `CURRENT_DIRECTIONAL_JOINT_VIEW_BINDING_MISSING`.

## Causal proof and repair firewalls

Failure localization is not causal ownership. Controlled attribution requires a same-probe, single-owner bounded counterfactual with material improvement and no protected regression. Repair requires a distinct child state and mandatory same-probe re-proof.

Current executable repair-operation count: **0**. Historical source presence is not execution authority.

## Non-negotiable firewalls

- no historical front-brain/teacher-exact ownership path;
- no second canonical graph/ID authority;
- no historical solver or repair executor executes merely because its source exists;
- no mechanical joint Vec3 / directional mesh `P.xy` shared-frame shortcut;
- no MOTION PASS without qualification-owned directional frame evidence;
- no proof can bind a different product state than export/runtime;
- no failure signature may invent causal ownership;
- no repair may mutate production state without a distinct child lineage and mandatory same-probe re-proof;
- no permanent current dependency on a dated experiment implementation;
- no family-specific constants during restoration;
- no FIT8 / Family-1 execution until restoration closure explicitly re-authorizes it.

## Next execution order

1. solve and type the Compiler-owned directional joint/view binding;
2. implement/qualify the directional evaluator on that binding;
3. bind the exact same qualification-owned frames to proof and export with no replay;
4. close `.rss/.rsr` -> native C++ open / clip lookup / render interlock;
5. source-diff CDT / BBW-KKT / ARAP / XPBD/contact and promote only kernels required by current typed architecture;
6. decide whether any current proposal-requalification / retained-candidate seam is required for repair execution;
7. run restoration-wide source/regression/E2E/native gates;
8. only then decide refreeze and FIT authorization.
