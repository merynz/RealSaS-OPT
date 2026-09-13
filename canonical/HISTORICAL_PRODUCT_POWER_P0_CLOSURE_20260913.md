# RealSaS — Historical Product-Power P0 Closure

**Date:** 2026-09-13  
**Branch:** `audit/historical-product-power-rebind-20260913`  
**Base main:** `bbdde6ec851717dddfba52107a28343e91864d68`  
**Status:** `P0_CLOSED_PASS__CURRENT_TYPED_TRANSACTION_AUTHORITY__NO_HISTORICAL_OWNER_RESTORED`

## Decision

P0 is closed.

The historical Compiler/runtime source reserve was reacquired from the Library, SHA-verified, compared across the Aug-7, V19_30R1, V19_30R2 and v0.5 source families, and then used as implementation evidence for a **new current typed compile-transaction layer**. The historical monolithic orchestrator was not restored and no historical acceptance, decomposition, truth, repair or product owner was promoted.

The current Compiler remains the only canonical product owner.

This closure changes no active Geppetto, Arachne or mesh scientific gate and makes no `PRODUCT_PASS` claim.

## Reacquired historical authorities

| Source | SHA-256 | Bytes |
|---|---|---:|
| `RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip` | `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130` | 12,957,145 |
| `RealSaS_V19_27R5D3_ALL_SYSTEMS_ONLINE_ARCHITECTURE_SOURCE_20260807.zip` | `ccf27f2a4546c7a0998ad22bc34a6712b543c8dbfe308d01d354fde829e1da2b` | 12,616,430 |
| `RealSaS_V19_30R1_GLOBAL_PARTITION_IDENTITY_SOURCE_20260808.zip` | `535472ebcaed7d5ea687edef72b2701e04c5c8a26af9aa87ad99ca9b50885d07` | 12,653,141 |
| `RealSaS_V19_30R2_GLOBAL_PARTITION_REPAIR_SOURCE.zip` | `679b8d4d80f284b1dbce2244b425a65c341b7f99351ab0a6dfa627f86cdd0786` | 12,835,944 |
| `SaS_AnimationSystem_20260524_v97_39_final_package_singletruth_rebind_integrated_full.zip` | `3f01a2df96c25ca82890709b6d3746384d26b6494ecd9929e50e6e4ddb061fe5` | 80,309,879 |

## Exact source-diff result

For the audited orchestration/reference-runtime files, **Aug-7, V19_30R1, V19_30R2 and v0.5 are byte-identical**. This removes the earlier uncertainty over which late historical branch contains the relevant orchestration semantics.

The shared source includes:

- `attempt_executor.py` — `87f454466e906a00579a82e38960e70d94f9f0e785c4706336f708fc3d19dedc`, 5,834 bytes;
- `budget_adaptive_retry.py` — `8bf6050cad5978dfb7e6ab0f56dc6d85ed108303f5c032994a6dd675c21752a0`, 12,009 bytes;
- `feedback.py` — `1b70470d03b6db113df611e90689a799089ba1df573d04b07a5bb3c9f0518bac`, 19,207 bytes;
- `meaningful_proof_loop.py` — `a50b3429e17cbf8456a6ad8274e053084ce134f25be332acffed28535ec60348`, 40,505 bytes;
- `pipeline.py` — `d47f29f5c853a180a3f0de7347e2ab8861b7c29015fd98fd79b38c817c2be614`, 255,433 bytes;
- `policy.py` — `0f97f4b43f325aecd9a653fd4c1c4ee350b5bd6402fc3514c99ec0ea7fd36f40`, 8,821 bytes;
- `visual_issue_router.py` — `47949bd9dcc61dede1801cdddc9f17b3b77e7050f6166472537efa2ab4c052dc`, 16,321 bytes;
- reference-runtime `animation_state.py`, `deformation_evaluator.py` and `runtime.py`, also byte-identical across the four source families.

The old `pipeline.py` directly imports superseded authority families including `realsas_decomposition`, `realsas_truth`, `realsas_prior`, `realsas_hypothesis` and `realsas_reasoning`. Therefore the previous restoration decision remains correct: **wholesale import is forbidden**.

## What was actually recovered

Historical implementation was treated as a mechanism source, not an owner. The recovered semantics are:

1. immutable compile attempts;
2. explicit stage/artifact contracts;
3. optional process-isolated attempt execution;
4. bounded retry budgets;
5. explicit `FAIL` / `ABSTAIN` propagation;
6. monotonic/no-regression child selection as a control invariant, not as an old weighted acceptance score;
7. probe → measure → diagnose → repair → same-probe re-proof;
8. a repair/retry creates a distinct child lineage, never in-place mutation;
9. exact product/proof/runtime identity continuity through final package/export.

Rejected as current authority:

- the historical weighted attempt score;
- old decomposition/front-brain ownership;
- old truth/teacher ownership;
- implicit historical repair executors;
- stable or `latest` filesystem path as identity;
- the 255 KB monolithic pipeline as product owner.

## Current typed transaction authority

P0 introduced:

`compiler/realsas_compiler_core/compile_transaction.py`

with:

```text
CompileStageContractIR
CompileRequestIR
CompileArtifactBindingIR
CompileStageResultIR
CompileTransactionIR
CompileResultIR
```

The transaction freezes exact source evidence, cameras, Compiler semantic version, policy bundle and per-stage implementation/policy hashes before execution. Each stage consumes exact artifact binding hashes and a `PASS` must emit exact output bindings. A slot cannot silently change identity inside the same transaction.

A `FAIL` or `ABSTAIN` is terminal and requires an explicit blocker. Any retry/repair is a separate child transaction bound to the parent transaction hash and the same frozen request.

The current product binding helper records exact S/G/W plus every directional/component M and B identity, component identity, directional state, motion state and final product hash. Proof and runtime are then bound to that exact product. Result sealing refuses a proof/product mismatch or runtime/product/proof mismatch.

Literal mutable artifact aliases `latest`, `current` and `newest` are rejected as authority.

Canonical routes are now:

```text
orchestrator/compile_request_ir.json
orchestrator/compile_transaction_ir.json
orchestrator/compile_result_ir.json
```

## Implementation commits

- compile transaction authority — `7ee4845b2f699b404b02890e697096f5ec60c727`
- core export — `3dcf0ef557cf981c5c101926caec72192c096e75`
- `CompilerFacade` wiring — `7bcc3b094b0976c84797d554ea1742268ef1eece`
- canonical bundle routes — `2a4070024db90add82d4653de65314ea139b35c2`
- contract tests — `ee468c16725e61acf20824fecece77605f7889f6`
- self-hosted P0 contract workflow — `eaccc5b866bcf6be6a7cdc0bd66f4f7f310a7db1`

## Self-hosted verification

Workflow: `.github/workflows/historical_product_power_rebind_contract.yml`  
Run: `34762089212`  
Head: `eaccc5b866bcf6be6a7cdc0bd66f4f7f310a7db1`  
Runner: `realsas-wsl-1660ti` (`self-hosted, linux, x64, realsas`)  
Conclusion: **SUCCESS**

Evidence:

- current transaction/proof sources compile — PASS;
- compile-transaction contract — `9/9 PASS`;
- bounded child-attempt repair loop — `5/5 PASS`;
- existing V4 architecture regression — `10/10 PASS`;
- Living Compile product-shell identity regression — `4/4 PASS`.

Total behavior tests in this gate: **28/28 PASS**.

## Closed boundary

P0 proves the product pipeline now has a small current transaction/identity spine informed by the strongest historical orchestration mechanisms without resurrecting old ownership.

It does **not** claim that corrected real G/W/M/B, professional motion or final runtime product are finished. Those are real downstream inputs to this now-closed contract.

The next legal transition is P1:

```text
fresh corrected G
  -> fresh corrected W
  -> strict corrected M
  -> exact B = M <- W
  -> component/mechanical assembly
  -> bind all exact identities into CompileTransactionIR
```

The active scientific gates remain unchanged. Once those artifacts close, they plug into this transaction instead of requiring another product-orchestration redesign.
