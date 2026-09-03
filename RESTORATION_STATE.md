# RealSaS-OPT — Restoration State

**Date:** 2026-09-03  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`  
**Verified behavioral base:** `2b5d467186839401ab30f9566015d9e9d49a2a06`  
**Global architecture refreeze:** `NOT PERFORMED`  
**Formal Family-1 selection:** `BLOCKED`  
**Real-family fit:** `NOT AUTHORIZED`

## Mission

Promote still-valuable historical Compiler/runtime production knowledge **behind** the current V4 typed Compiler authority, then normalize repository navigation and ownership without destroying provenance.

> We are not restoring the old Compiler as owner. We are restoring selected production mechanisms as subordinate services of the current Compiler.

## Current restoration ledger

| Stage | State | Commit / evidence |
|---|---|---|
| Repository authority + navigation skeleton | **DONE** | `fe8bedfbe0f04519711432942f656aaf4735603e` |
| Exact native C++ runtime consumer | **DONE / CI PASS** | `25d810e272cc00a7b6fd4d682eabc16fef226223` |
| Diagnostic failure-signature semantic rebind | **DONE / CI PASS** | `5665ecacb77c02e53297340249b0972981a5c4bc` |
| Motion probe / playback measurement rebind | **NEXT** | historical source audit in progress |
| Causal owner attribution + bounded repair/re-proof | **PENDING** | must remain separate from failure localization |
| Current V4 export -> native runtime interlock | **PENDING** | proof-state identity must be exact |
| CDT / BBW-KKT / ARAP / XPBD source diff | **PENDING** | no numerical bulk restore |
| Full behavioral + complete-E2E restoration closure | **PENDING** | required before refreeze decision |

## Historical byte authority

Restoration byte authority currently used:

- full v0.5 source archive: `RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip`
- verified SHA-256: `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`

The native `runtime/realsas_cpp/` subtree was restored byte-exactly from that verified archive and is sealed file-by-file in `canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`.

Historical records also mention a standalone C++ runtime archive (SHA-256 `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`). It is **provenance/reference only** for this restoration; it was not used as the promoted subtree byte authority.

Other historical authorities remain candidate evidence until source-diffed and explicitly promoted:

- R5_3 semantic authority SHA-256 `6224661cb4323f78a9b808af10f68dd584431a422e28d26a69e87816a3b0ef80`
- v97_43 numerical/rig authority SHA-256 `09a94871f938b069ba5c8219f203355e724f2f58afa6e10dc5c6148d98b43efb`

## Promotion rule

Every historical mechanism receives one of these dispositions before execution:

- `PROMOTE_EXACT_CONSUMER` — byte-exact non-authoritative runtime/consumer code;
- `PROMOTE_REBIND` — preserve semantics, rewrite against current typed authority;
- `SOURCE_DIFF_SELECT` — compare competing historical implementations and promote only the strongest bounded kernel;
- `ARCHIVAL_ONLY` — preserve provenance/evidence, never execute in current product path;
- `DO_NOT_PROMOTE_MONOLITH` — explicitly forbid resurrection as a current subsystem.

`compiler/realsas_orchestrator/pipeline.py` is `DO_NOT_PROMOTE_MONOLITH`.

## Completed semantic correction

Historical v0.5 failure diagnostics contained a valuable firewall: a measured geometry/proof failure is diagnostic localization, **not causal owner attribution**. Current proof code had collapsed these concepts by filling `owner_domain` directly from the failed proof domain.

That leak is now removed. `compiler/realsas_compiler_services/proof/failure_signatures.py` derives typed diagnostics while `owner_attribution` remains empty until a separate controlled mutation/fault experiment proves ownership. A failure signature cannot authorize repair.

## Non-negotiable firewalls

- no historical front-brain/teacher-exact ownership path;
- no second canonical graph/ID authority;
- no historical solver executes merely because its source exists;
- no proof can bind a different product state than export/runtime;
- no failure signature may invent causal ownership;
- no repair may mutate production state without mandatory re-proof;
- no family-specific constants during restoration;
- no FIT8 / Family-1 execution until restoration closure explicitly re-authorizes it.

## Next execution order

1. rebind historical motion probe and playback measurement semantics to current typed proof inputs;
2. introduce explicit causal owner-attribution evidence and bounded repair directives;
3. enforce `prove -> diagnose -> attribute -> repair -> re-prove` as a state machine;
4. bind current exact-PASS product proof to `.rss/.rsr` export and native runtime consumption;
5. source-diff numerical kernels rather than restoring whole historical numerical stacks;
6. run restoration-wide behavioral/source/E2E/native gates;
7. only then decide whether to refreeze and reopen FIT selection.
