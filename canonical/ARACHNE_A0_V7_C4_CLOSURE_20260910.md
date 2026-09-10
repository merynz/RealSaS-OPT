# RealSaS — Arachne Mage A0 FIT1 — V7-C4 Closure

**Date:** 2026-09-10  
**Experiment:** `V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION`  
**Scientific scope:** **Mage A0 SkinFieldCodec representation/shared-decoder only**  
**Categorical verdict:** `CLOSED_FAIL__REJECT_THIS_SAMPLING_FAMILY__DO_NOT_AUTO_STACK_CUSTOM_BLEND_RATIO_LOSS`  
**A0 terminal closure:** `FALSE`  
**A1 authorization:** `FALSE`  
**Product evaluation:** `NOT PERFORMED`

This closure narrows older notebook/prereg shorthand that says “Arachne FIT1”: for current architecture authority, C4 is an **A0 Codec FIT1** experiment only. It cannot close A1, learned skinning end to end, generalization, or product acceptance.

## Frozen provenance

- research branch before evidence commit: `2cba8bcafefe7c5e3983fe5693dac8e17a548e01`
- research-branch evidence/closure commit: `d0b666725e79f3beb0ea001f456375421cc14375`
- preregistration commit: `372cb2dd510183800460b496a7ecfbeb1204369f`
- preregistration blob: `4ada6333afa1dcfe40d08eac286a88d3acc158ad`
- source-contract commit: `cb1bea3d457fffd46f9eb69733ae9b66d6d13f22`
- source-contract blob: `cf4401f6ef3d91a32c6615b3325d9d9fbe67813b`
- embedded scientific source SHA-256: `89be3ca449dce1827d3e52a1bc949ee601136f769eb1e9cadc0c04af81cde7c5`
- C2 base model SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
- normalized mesh/source SHA-256: `528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f`
- reference A/B row schedule SHA-256: `e7a257f20a447e5b34ff850658851e937691fc60aed4c6286d6a8043b071cef2`
- common global/prefix SHA-256: `0875c257bb53f06439bb3de9eb28d4bc642cfb8807cff1c703a3c3932d74268e`
- C face-bary schedule SHA-256: `d6c9b895233414caffb0449902bcd3dda929bc6296ce274721a0aae57566c842`

A100 preflight passed and revalidated the frozen C2 baseline exactly before scientific optimization. The A/B query schedule and prefix are exact matches; C uses the preregistered source-face barycentric schedule.

## Exact result evidence

Drive root: `ARACHNE_MAGE_A0_V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION_20260909` (`1fgrLmOmmWRhPhg3q0CZ-UhyYWX0Qt89i`)  
Drive run folder: `MAIN` (`12AWAj05O9GuZ43x1s-RJOnlT7c3S0lLO`)

- A100 preflight JSON SHA-256: `6f5f1487d8865f9c737fe79b5d0db098819b38a6357441a9a3dc520eca51e21d`
- Arm A result JSON SHA-256: `d56b54f6e529e5308d96f2af6fdd336fe7ae1555302fca836df0ac9010fc8b88`
- Arm B result JSON SHA-256: `56d7673dc8050d1140a3fd08bf8637310c011d0f5e57d7d39811aa5886826ee8`
- Arm C result JSON SHA-256: `81331db01815cb5fa4681d14caa1685ca884894dfd3b2840e012a7b398bcd793`
- aggregate result JSON SHA-256: `439583ac60eca855dc2b55efabf5b4a6df4de5f9617e9e35583cf136b4cb55d9`
- Arm B final model SHA-256 recorded by result: `c13453093f7713bb2d7f6fade49da73aa6a4ce0e9b09faf50a46ce350d36c71e`
- Arm C final model SHA-256 recorded by result: `7909a5a878eaee6094b28b274e23347f031fd37fd8368d15d22dd53cb1f562cc`
- Arm A result emitted no separate final-model SHA; none is invented by this closure.

The exact external JSON/model identities are also bound in `canonical/ARACHNE_A0_V7_C4_EVIDENCE_MANIFEST_V1.json`.

## Terminal gate table

| Arm | raw p95 | qualified p95 | raw deform | qualified deform | dominant acc. | terminal streak | FIT1 gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A — importance-corrected active-only | 0.1817569972 | 0.1817569988 | 0.0611890741 | 0.0611890741 | 0.9935760171 | 0 | FAIL |
| B — biased active-only | 0.2705708147 | 0.2705707853 | 0.0908120275 | 0.0908120424 | 0.9935760171 | 0 | FAIL |
| C — face+bary biased | 1.4804241555 | 1.4804241008 | 0.5461477637 | 0.5461477637 | 0.7655246253 | 0 | FAIL |

The frozen acceptance limits remain p95 `<= 0.05` and deformation ratio `<= 0.05`, with a terminal passing streak of at least three observations.

All three arms retained the non-accuracy legality/numerical guards: `950` qualified rows; finite/nonnegative outputs; raw and qualified simplex residuals within `1e-6`; Compiler total/mean/max correction within the preregistered caps; zero sparsification discarded mass. Therefore this is a **scientific failure**, not an infrastructure failure.

## Causal interpretation

### A -> B: remove `u/q`

B worsened relative to A:

- raw p95: `+0.0888138175`
- raw deformation ratio: `+0.0296229534`
- dominant-joint accuracy: unchanged at `0.9935760171`
- top-3 inclusion: unchanged at `1.0`
- top-4 false displacement rows: `48 -> 62`

Therefore **objective-bias cancellation by importance correction is not supported as the blocker**. Removing `u/q` on the exact same sampled GSA queries made the primary metrics worse.

### B -> C: add source-faithful face/topology/barycentric dense supervision

C worsened sharply relative to B:

- raw p95: `+1.2098533407`
- raw deformation ratio: `+0.4553357363`
- dominant-joint accuracy: `0.9935760171 -> 0.7655246253`
- top-3 inclusion: `1.0 -> 0.9582441113`
- top-4 false displacement rows: `62 -> 202`

Therefore **the preregistered face/topology/barycentric dense-supervision port is not supported for this FIT1 target**. It also degraded ownership, so the secondary preregistered support condition fails.

## Closed claim boundary

C4 establishes only:

1. the exact A0 fixed-GSA biased-objective intervention B does not improve over A;
2. the exact A0 source-face/barycentric intervention C does not improve over B and severely degrades the measured fit;
3. none of A/B/C closes the frozen A0 Codec FIT1 gate.

C4 does **not** establish that blend-boundary localization was false, that every possible topology-aware method is impossible, that A1 fails, or that RealSaS product skinning fails.

## Required next fork

The preregistration binds the next action when neither B nor C improves over A: **reject this sampling family and reopen a separately preregistered within-support blend-ratio/calibration diagnosis**. Do not automatically stack a custom blend-ratio loss and do not weaken the `0.05` gates.

Until a future A0 experiment actually terminal-passes:

`A0 = OPEN`  
`A1 = BLOCKED_NOT_AUTHORIZED`  
`PRODUCT_PASS = NOT EVALUATED`
