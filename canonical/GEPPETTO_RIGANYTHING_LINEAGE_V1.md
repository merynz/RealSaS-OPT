# RealSaS — Geppetto / RigAnything Mechanism Lineage V1

**Date indexed:** 2026-09-07  
**Purpose:** prevent repeated rediscovery and semantic widening across chats/agents.

This file is a **semantic lineage map**, not continuation authority. `CURRENT_STATE.md` remains continuation authority.

## Core memory rule

RealSaS already performed a clean-room translation of the important RigAnything mechanism family into a research challenger. Do **not** propose reimplementing a "RigAnything-equivalent Geppetto" from scratch before reading this lineage and the existing source.

Fuller challenger creation commit:

`ba634955777479ee05a5b199742710b1736123b5`

Commit time: `2026-09-05T22:27:24Z`  
Commit message: `Add RigAnything mechanism challengers and functional rig quality framework.`

Source:

`models/geppetto/challengers/riganything_mechanisms_v1.py`

Blob SHA at creation commit:

`136d358572be6ff72ba15999474b6d3be600f5bc`

Architecture ID:

`RealSaS.GeppettoChallenger.RigAnythingMechanisms.v1`

## Mechanism decomposition — implementation vs execution

| Rung | Mechanism delta | Authoritative execution status |
|---|---|---|
| `R2` | corrected-raster matched historical baseline; xattn OFF; diffusion OFF; geometry feedback OFF | **RUN — stable FAIL** in Causal Repair V2 contract `233ec3bcd77e0f02` |
| `C1` | R2 + per-step full-surface cross-attention | **RUN — stable PASS** in Causal Repair V2; winner candidate `C1` |
| `C2` | C1 + conditional diffusion locus; geometry feedback still OFF | **IMPLEMENTED / PREREGISTERED BUT NOT RUN in authoritative Causal Repair V2** because the preregistered staircase stopped after C1 PASS |
| `C3` | C2 + current joint / parent geometry feedback token into subsequent generation | **SOURCE-CODED; no authoritative closed full-formulation run registered** |
| `C4` | C3 + sibling/BFS-equivalent-order augmentation | **SOURCE-CODED; no authoritative closed run registered** |

This distinction is binding: **C2 source/prereg existence is not C2 experimental evidence.**

## Causal Repair V2 — exact normalized result

Drive authority folder:

`RealSaS_MAGE_GEPPETTO_CAUSAL_REPAIR_V2_NO_TOKEN`

Contract tag:

`233ec3bcd77e0f02`

Comparison artifact created:

`2026-09-06T04:06:17.337Z`

Preregistered staircase:

`RUN_R2_FIRST -> IF_R2_FAIL RUN_C1 -> IF_C1_PASS STOP_MECHANISM_SEARCH -> C2 only if C1 fails`

The actual run folder contains only `R2_*` and `C1_*` result/checkpoint files. There is no C2 result artifact in that authoritative run.

### R2

- stable PASS: `false`
- first stable: `null`
- best step: `12800`
- best slot p95: `0.041321732103824615`
- final outside: `27`
- final occupancy L1: `27`
- final slot p95: `0.1146610826253891`

### C1

- stable PASS: `true`
- first stable: `4992`
- epsilon switch: `4736`
- best step: `6208`
- best slot p95: `0.004098494071513414`
- final outside: `0`
- final occupancy L1: `0`
- final slot p95: `0.005874851252883673`

Winner candidate:

`C1`

Recorded causal interpretation:

`R2_NO_STABLE_PASS_WITHIN_BUDGET__C1_XATTN_STABLE_PASS__XATTN_RESCUES_THIS_FROZEN_TRAINING_PROTOCOL__NOT_THEORETICAL_NECESSITY`

Promotion authority was `false`.

Therefore Causal Repair V2 provides **controlled component evidence for full-surface cross-attention under that frozen Mage FIT training protocol**. It does not provide a C2 diffusion result because C2 was never entered.

## Why the mechanism ladder was split

The Causal Repair V2 prereg explicitly defined:

- R2: no xattn, no diffusion, no geometry feedback;
- C1: xattn ON, diffusion OFF, geometry feedback OFF;
- C2: xattn ON, diffusion ON, geometry feedback OFF.

This separation was deliberate: preserve causal attribution instead of changing full-surface access, locus generation, generated mechanical state, and sibling ordering simultaneously.

Because C1 passed, the same causal-discipline rule **prevented C2 from running**. This is why later memory must say:

> C2 existed as an executable/preregistered rung, not as a closed result in this experiment.

## Why AR-01 existed despite C3 already existing

C3 already contained a broader joint+parent feedback mechanism, but bundled it with the fuller challenger formulation. AR-01 was created to isolate one question on a matched family:

> Does minimal joint+parent mechanical recurrent feedback help when lossless evidence, full-surface access, direct locus head, structural serialization, optimizer and evaluation are held fixed?

Therefore:

`AR-01 != C3/C4 full-formulation test`

AR-01 source/prereg lineage:

- experiment branch: `exp/geppetto-ar01-skeleton-causal-v1-20260907`
- repo prereg SHA-256: `78981b0215c88462d7a2be607274f638494dc1517f88ed882084a4e83d11fae1`
- AR-01 research source SHA-256: `6f52e168883c7385c62fb7705bf526c740ad980a5fb0eed79356cdfde65d540d`
- executed contract tag: `41055bd073538d6b`
- terminal verdict: `AR01_NO_TERMINAL_CLOSURE`
- result authority: `canonical/AR01_RESULT_20260907.md`

## What AR-01 added to the lineage

AR0 and AR1 both failed the preregistered terminal 48-check stability gate.

Important differentiated evidence:

- AR0 showed strong same-witness reachability and a 61-check exact structural streak, but later collapsed and ended with terminal streak 1.
- AR1 never achieved a free-running structural PASS, despite the final teacher-forced model being structurally exact with parent accuracy 1.0.
- AR1 therefore exposes a strong teacher-forced/free-running distribution-gap / recurrent-error-amplification failure mode in the **minimal residual-feedback implementation**.

This does **not** erase the C3/C4 challenger or convert AR-01 into a verdict on it.

## Current unresolved decomposition after AR-01

Three questions must remain separate:

1. **AR0 residual stability:** Why can a no-feedback model sustain long exact streaks and then catastrophically leave the region under the current optimizer/trajectory?
2. **AR1 exposure/recovery:** Why does teacher-forced structural closure fail under generated joint + hard predicted-parent feedback, and what feedback/training policy would make free-running state robust?
3. **Fuller formulation:** Does the already-existing C3/C4 combination — including conditional diffusion and tokenized mechanical feedback/order handling — materially change the outcome under a properly preregistered formulation-level test?

A future experiment must state which one it tests. Do not bundle all three and later infer causality post hoc.

## Anti-conflation rules

- `C2 implemented/preregistered` != `C2 run`.
- `source-coded C3/C4` != `C3/C4 scientifically closed`.
- `C1 PASS + C2 source + AR-01 result` != `full C3/C4 verdict`.
- `AR1 exposure collapse` != `all structural AR falsified`.
- `AR0 long exact streak` != `terminal stability PASS`.
- `FIT1` != `generalization`.
- Detached/generated drafts never outrank the repo prereg/result hashes.

## Bootstrap audit status

The highest-risk RigAnything/Geppetto semantic gap is now materially reduced:

- fuller challenger existence is commit/blob bound;
- Causal Repair V2 R2/C1 execution is Drive-result bound;
- C2 non-execution is explicit;
- C3/C4 source-only status is explicit;
- AR-01 scope/result is explicit.

Older Geppetto numerical/result pointers and other modules remain in the broader audit-of-audits queue until the coverage audit marks them semantically reconciled.