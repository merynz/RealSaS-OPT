# RealSaS — Arachne A0 V7 K=4 Closure Freeze

**Date:** 2026-09-11  
**Status:** `A0_MAGE_GSA_CLOSED__K4_INTERFACE_FROZEN__A1_PREREG_NEXT`  
**Branch:** `exp/arachne-a1-v7-native-fit1-20260911`  
**Product PASS:** not claimed.  
**Unseen/generalization PASS:** not claimed.

## 1. Closure artifact

The selected A0 K=4 interface was regenerated and sealed after the completed 4/8/16/32 token-capacity experiment failed to show a meaningful capacity gain beyond K=4.

Frozen checkpoint:

- file: `A0_K4_CLOSURE_MODEL_ONLY_FP32.pt`
- SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`
- optimizer steps: `4096`
- field tokens: `4`
- latent channels: `512`
- condition tokens: `384`
- V7 config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`

A1 supervision/evaluation bank:

- file: `A0_K4_A1_SUPERVISION_BANK.npz`
- SHA-256: `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`

Closure preregistration SHA-256:

- `3ffd304691f836926f2e7787d22b6608ec1a249206f6c2535177f2fd9f4fc857`

## 2. FIT1 closure metrics

Final GSA950:

- row-L1 p95: `0.04491063521144626`
- deformation error ratio: `0.018186409026384354`
- row-L1 mean: `0.013207706702623423`
- dominant accuracy: `0.9882226980728052`
- teacher dominant top-3 inclusion: `1.0`
- evaluated rows: `934 / 950`

The final three required observations all pass the frozen `.05/.05` gate:

- step 3584: p95 `0.04870356093865557`, deformation `0.01905974932014942`
- step 3840: p95 `0.04750191859206554`, deformation `0.018206628039479256`
- step 4096: p95 `0.04491063521144626`, deformation `0.018186409026384354`

Therefore:

`A0_MAGE_GSA_FIT1_STABLE_LAST3 = TRUE`.

## 3. Disjoint-surface diagnostic

Final same-character disjoint-surface holdout:

- row-L1 p95: `0.10624249461034009`
- deformation error ratio: `0.14126436412334442`
- row-L1 mean: `0.04270809743583189`
- dominant accuracy: `0.980469715698393`
- evaluated rows: `8090`

This plane remains diagnostic only. It is not an A1-transition gate, product evaluation, or unseen-character/family claim.

The completed token-capacity study did not establish a meaningful practical gain for 8/16/32 tokens over K=4. K=4 is therefore selected manually for parsimony, continuity, and minimum interface complexity. No claim is made that K=4 was the numerical winner of the completed multi-arm experiment.

## 4. Token-interface semantic audit

The first post-fit guard incorrectly treated a BF16 permutation delta as semantic token-order evidence. No optimizer step occurred after that failure and no checkpoint parameter was modified.

Re-audit:

- FP32 full-set permutation max absolute logit delta: `5.7220458984375e-06`
- FP32 tolerance: `1e-05`
- BF16 full-set permutation max absolute logit delta: `0.0016632080078125`
- original BF16 threshold: `5e-4`

Verdict:

`FP32_FULL_SET_PERMUTATION_INVARIANT = TRUE`

The BF16 delta is treated as finite-precision attention/reduction-order drift, not learned token-rank semantics.

Consequences for A1:

- ordered token-1 -> token-1 latent alignment is **not** authorized as semantic authority;
- A1 primary supervision must be decoded field behavior plus normalized joint-row behavior;
- latent-set alignment, if ever added, must be permutation-aware or otherwise justified by a separate contract.

## 5. Transition authority

A0 representation/decode closure for the Mage/GSA FIT1 witness is complete at K=4.

This record authorizes:

- freezing the exact K=4 decoder/interface for A1;
- completing the V7-native A1 source/preregistration transaction;
- using teacher W only in the objective/evaluation lane;
- using the supervision bank above as bound A1 training/evaluation target evidence.

This record does **not** by itself authorize an A1 optimizer step until the exact V7-native A1 source/config/objective/evaluation preregistration is sealed.

Current transition:

`A0 K4 CLOSED -> V7-NATIVE A1 SOURCE + EXACT PREREG -> A1 FIT1`.
