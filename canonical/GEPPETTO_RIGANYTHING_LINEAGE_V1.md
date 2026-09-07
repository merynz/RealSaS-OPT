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

## Mechanism decomposition

The challenger preserves a mechanism ladder rather than treating RigAnything as one indivisible black box.

| Rung | Mechanism delta | Scientific meaning / status |
|---|---|---|
| `R2/C0-family baseline` | matched historical/corrected Geppetto baseline | control lineage; exact historical/result pointer still undergoing bootstrap normalization |
| `C1` | per-step full-surface cross-attention | component evidence exists; does not establish diffusion or skeleton-feedback necessity |
| `C2` | C1 + conditional diffusion locus | component evidence lineage exists; joint/parent geometry feedback explicitly absent |
| `C3` | C2 + current joint / parent geometry feedback token into subsequent generation | source-coded fuller skeleton-causal formulation; **not registered as an authoritative isolated/full-formulation closed run** |
| `C4` | C3 + sibling/BFS-equivalent-order augmentation | source-coded order-handling extension; **not registered as an authoritative closed causal result** |

The prior Causal Repair V2 prereg explicitly defined:

- R2: no xattn, no diffusion, no geometry feedback;
- C1: xattn ON, diffusion OFF, geometry feedback OFF;
- C2: xattn ON, diffusion ON, geometry feedback OFF.

This separation was deliberate: it preserved causal attribution instead of changing full-surface access, locus generation, generated mechanical state, and sibling ordering simultaneously.

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

- `source-coded C3/C4` != `C3/C4 scientifically closed`.
- `C1 + C2 + AR-01 component evidence` != `full C3/C4 verdict`.
- `AR1 exposure collapse` != `all structural AR falsified`.
- `AR0 long exact streak` != `terminal stability PASS`.
- `FIT1` != `generalization`.
- Detached/generated drafts never outrank the repo prereg/result hashes.

## Bootstrap audit status

This lineage closes the highest-risk semantic gap that caused repeated "we already built that / why is it missing?" rediscovery. Exact historical numeric/result-pointer normalization for all C1/C2 and older Geppetto experiments remains part of the broader audit-of-audits queue until the coverage audit marks those artifacts semantically reconciled.