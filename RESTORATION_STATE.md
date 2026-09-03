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
| IRIS -> Compiler substrate ownership seam | **DONE** | current `compiler/realsas_compiler_core/substrate/iris_v2.py` |
| Compiler substrate physical normalization | **DONE** | `bc443d063d4d8f0bd52981ec2db5b99a795985e1` |
| Compiler mesh physical normalization | **DONE** | `295b798b1fb40ccb1752afe9da9378dfe8734262` |
| Authored-motion dynamic probe / proof semantic rebind | **DONE / LOCAL REGRESSION 2/2 PASS** | `fbe80348351c81fa5bba7a56ae1898d5b09ca160`; `canonical/AUTHORED_MOTION_PROOF_PROMOTION_V1_20260903.json` |
| Causal owner attribution + bounded repair/re-proof | **NEXT** | must remain separate from failure localization and require controlled intervention evidence |
| Current V4 export -> native runtime interlock | **PENDING** | proof-state identity must be exact |
| CDT / BBW-KKT / ARAP / XPBD source diff | **PENDING** | no numerical bulk restore |
| Full behavioral + complete-E2E restoration closure | **PENDING** | required before refreeze decision |

## Repository organization contract

The repository is treated as four semantic zones:

```text
mainline library = models/ + compiler/ + runtime/
labs             = experiments/
decision/evidence= canonical/
provenance reserve= historical/
```

`models/` contains the semantic homes for the current learned stack: IRIS, Geppetto, SkinFieldCodec and Arachne. Models emit learned evidence/proposals only; current Compiler qualification remains product authority.

Mainline code must converge away from permanent imports of dated experiment packages. Experiments may import mainline; successful experiments are promoted into the relevant semantic home after closure/regression.

Compiler physical normalization follows the logical layer map in `compiler/README.md` through dependency-safe moves/re-exports. No mass aesthetic reshuffle is authorized.

## Historical byte authority

Restoration byte authority currently used:

- full v0.5 source archive: `RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip`
- verified SHA-256: `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`

The native `runtime/realsas_cpp/` subtree was restored byte-exactly from that verified archive and is sealed file-by-file in `canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`.

Historical records also mention a standalone C++ runtime archive (SHA-256 `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`). It is provenance/reference only for this restoration; it was not used as promoted subtree byte authority.

Other historical authorities remain candidate evidence until source-diffed and explicitly promoted:

- R5_3 semantic authority SHA-256 `6224661cb4323f78a9b808af10f68dd584431a422e28d26a69e87816a3b0ef80`
- v97_43 numerical/rig authority SHA-256 `09a94871f938b069ba5c8219f203355e724f2f58afa6e10dc5c6148d98b43efb`

## Completed semantic restorations

### Failure localization != causal ownership

Historical v0.5 failure diagnostics preserved an important firewall: a measured geometry/proof failure is diagnostic localization, **not causal owner attribution**. Current `failure_signatures.py` now preserves this separation. `owner_attribution` remains empty until a separate controlled mutation/fault experiment proves ownership. A failure signature cannot authorize repair.

### Authored motion must be dynamically exercised

Historical `realsas_deformation/motion_proof.py` contained another valuable invariant: requested/authored motion must itself be executed and measured; static existence of a clip/track is insufficient proof.

That invariant is now rebound to current V4 state through `compiler/realsas_compiler_services/proof/motion_probe.py` and `motion_probe_geometry.py`. The service consumes exact current `CanonicalPuppetGraph.v3` mechanical state, directional meshes/mesh-skin and puppet-local motion, evaluates actual LBS deformation over sampled clip time, and reports dynamic consequences such as effective motion, edge relative change, triangle area compression/expansion, degeneration, non-finite deformation and loop seam. Compiler `proof_engine.py` owns the final MOTION PASS/FAIL binding.

The promoted probe does not create canonical IDs, does not repair state, does not infer causal ownership, and does not claim full-3D reconstruction authority.

## Promotion rule

Every historical mechanism receives one of these dispositions before execution:

- `PROMOTE_EXACT_CONSUMER` — byte-exact non-authoritative runtime/consumer code;
- `PROMOTE_REBIND` — preserve semantics, rewrite against current typed authority;
- `SOURCE_DIFF_SELECT` — compare competing historical implementations and promote only the strongest bounded kernel;
- `ARCHIVAL_ONLY` — preserve provenance/evidence, never execute in current product path;
- `DO_NOT_PROMOTE_MONOLITH` — explicitly forbid resurrection as a current subsystem.

`compiler/realsas_orchestrator/pipeline.py` remains `DO_NOT_PROMOTE_MONOLITH`.

## Non-negotiable firewalls

- no historical front-brain/teacher-exact ownership path;
- no second canonical graph/ID authority;
- no historical solver executes merely because its source exists;
- no proof can bind a different product state than export/runtime;
- no failure signature may invent causal ownership;
- no repair may mutate production state without mandatory re-proof;
- no permanent current dependency on a dated experiment implementation;
- no family-specific constants during restoration;
- no FIT8 / Family-1 execution until restoration closure explicitly re-authorizes it.

## Next execution order

1. introduce explicit causal owner-attribution evidence based on controlled intervention/counterfactual measurement rather than failure labels;
2. define bounded repair directives that can target only proven owners and create a new product attempt/state;
3. enforce `prove -> diagnose -> attribute -> repair -> re-prove` as a state machine;
4. bind current exact-PASS product proof to `.rss/.rsr` export and native runtime consumption;
5. source-diff numerical kernels rather than restoring whole historical numerical stacks;
6. continue dependency-safe Compiler physical normalization where it improves authority clarity;
7. run restoration-wide behavioral/source/E2E/native gates;
8. only then decide whether to refreeze and reopen FIT selection.
