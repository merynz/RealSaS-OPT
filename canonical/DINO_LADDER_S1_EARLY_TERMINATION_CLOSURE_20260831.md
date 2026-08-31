# RealSaS — DINO Ladder S1 Early-Termination Closure

**Date:** 2026-08-31  
**Status:** `EARLY_TERMINATED_POST_HOC_SHARED_APPARATUS_CONFOUND`  
**Original planned order:** `S -> B -> L -> g`  
**Termination decision:** after the completed B `PRIMARY_32768` result and subsequent code-level audit  
**Machine-readable closure SHA-256:** `773483ad80c2adc1421da41154d2fe4312c686bfe209265a165786e1f0aa76f0`

## 1. What completed

| Rung | State | FIT_PROXY32 primary | TRAIN_DIAG32 primary | Hard route failures | Accessibility |
|---|---|---:|---:|---:|---|
| S | complete @32768 | 10/54 | 17/56 | 0 | FAIL |
| B | complete @32768 | 10/54 | 18/56 | 0 | FAIL |
| L | stopped partial | not evaluated | not evaluated | n/a | no claim |
| g | not run | n/a | n/a | n/a | no claim |

The persisted L history reaches step `192`. No L primary evaluation was opened. The partial L trace is provenance only and must not be interpreted scientifically.

DEV32 remained closed.

## 2. Why S1 was stopped

S1 was preregistered to isolate frozen-backbone representation scale under one byte/config-identical shared learner. That comparison remains valid for the completed S/B pair.

After the S/B results, the previously frozen learner/loss/access path was audited at code level. The audit exposed a material shared accessibility ceiling/confound:

- local DINO features are retained at `37x37`; therefore the earlier informal claim that all DINO information was reduced to `16x16` is explicitly rejected;
- cross-view context is pooled to `16x16` and bilinearly returned to `37x37`;
- every rung passes through the same `1536 -> 256` shared stem;
- native-1024 detail access is only a sparse 25-RGBA-sample stencil plus a small MLP, not a dense learned high-resolution spatial field;
- the final raster-query prediction is pointwise after sampling the shared encoded field;
- the objective is uniform mean SmoothL1 with `beta=0.01`, with no direct tail/derivative/consumer-aligned term.

These facts do **not** prove that any one component caused the S/B plateau. They do mean that finishing L/g under the same apparatus has substantially reduced expected information value relative to its compute cost.

This is a post-hoc termination. It must never be rewritten as a completed four-rung ladder.

## 3. Permitted claims

1. Under `SharedLearnerV1` and the frozen S1 apparatus, S->B scaling did not materially improve downstream accessibility at `PRIMARY_32768`.
2. S and B both failed the frozen primary accessibility criterion with zero hard-route failures.
3. The S1 run does not establish shared learner adequacy.

## 4. Forbidden claims

The S1 run does **not** support any of the following:

- L would fail;
- g would fail;
- DINO-L or DINO-g representation is insufficient;
- backbone scale is globally irrelevant;
- S->B would remain nearly flat under a different decoder, loss, native-detail path, sampling policy or representation interface;
- the partial L history predicts L primary performance.

That fifth prohibition is binding: the measuring apparatus changed conceptually after the audit, so S->B flatness is an apparatus-conditional result only.

## 5. Next causal sequence

No new four-rung ladder is authorized.

The next work is:

1. **Frozen-component code audit** before any new training.
2. **Single-rung decoder/access ceiling**, using B as the default diagnostic rung unless the audit invalidates that choice. Keep backbone tokens, membership, sample stream, evaluator and current loss fixed; change only the spatial decoder/native access path.
3. Run asymmetric ablations so a strong native path is allowed to falsify the need for the foundation prior:
   - DINO ON / native ON,
   - DINO OFF / native ON,
   - DINO ON / native OFF.
4. Only after the decoder experiment is interpreted, run loss causality in separate stages:
   - beta-only change;
   - then tail/CVaR-only addition with beta held fixed.
5. Reopen an S/B/L/g ladder only if the revised apparatus first demonstrates adequate single-rung accessibility.

If `native ON / DINO OFF` solves most of the task, that is a legitimate result, not an ablation failure.

## 6. Program-level lesson

A repeated failure mode is now explicit: components treated as fixed controls can themselves determine the measured boundary. Future preregistration must therefore treat **every frozen component as suspect until audited**. Learned modules and deterministic/frozen operators receive the same code-level scrutiny before sealing.

Machine-readable authority:
`experiments/iris_dino_controlled_20260829/DINO_LADDER_S1_EARLY_TERMINATION_V1.json`.
