# V2 Source Recovery V1 — Frozen Behavioral Parity Preregistration

**Date:** 2026-08-20  
**Status before execution:** `FROZEN__NOT_YET_EXECUTED`  
**Parent scientific authority:** `08620737ce526b0c0cc6c551f53910e34a1107fe`  
**Source persistence manifest commit:** `641b8a60011fa4b98e5c5e88f8e6ff5a2f81ce4c`  
**Recovered source SHA-256:** `578fdddcd17d922603fb463ee1f2cae5b88e0e2c96a850900716abef5e7588be`  
**SOURCE_PERSISTENCE_GATE:** `PASS`

## Question

Can the missing/corrupted historical V2 execution source be recovered by the already-frozen Recovery V1 bytes without post-outcome tuning?

This is a **source recovery parity experiment**, not a new scientific solver experiment.

## Frozen reconstruction choices

No choice below may change after the first historical parity execution:

1. Intact persisted V2 prefix/suffix are retained.
2. Problem-A visual-hull/DIS front door uses the preserved Stage-A recovery behavior.
3. F16 refinement uses `coarse_k=8`, 5×5 offsets `(-4,-2,0,2,4)^2`, stable score ordering, final K=16.
4. Pairwise triangulation is the exact adaptive-retention `make_H` mathematics, with cached pair pseudoinverses only as an execution optimization.
5. `bilinear_flow_samples` is frozen as raw-pixel bilinear sampling of DIS flow.
6. `local_scale(P,k=8)` is frozen as the median of the 8 nearest non-self Euclidean surface distances.
7. No numerical threshold, candidate ordering, tie-break, graph rule, pair factor, min-sum rule, eligibility rule, or evaluation rule may be altered after execution begins.
8. No historical metric may be used to select among alternate helper definitions. There is exactly one Recovery V1 attempt.

## Data scope

Only the already-open eight-family e00 panel is allowed:

`09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290`

Canonical source is the `RealSaS_IRIS_SEES_N1B_MULTIINTERVENTION_20260817/multiintervention_corpus_v1` Drive lineage, Pose-A plus `episodes/e00/poseB` plus e00 `observation_sidecar.npz`.

`sealed21 = CLOSED`  
`external10 = CLOSED`

## Frozen checkpoint

SHA-256 must equal:

`0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`

Any mismatch => `RECOVERY_INVALID_INPUT`.

## Gate A — full F16 parity

Exact expected per-family 2× target-near coverage:

- 09908: `1.0`
- 11032: `0.9491525424`
- 12772: `1.0`
- 13203: `0.9508196721`
- 14404: `0.9682539683`
- 14702: `1.0`
- 14758: `0.9838709677`
- 15290: `1.0`

Each must match within the historical source tolerance `1e-8`.

## Gate B — M256 preflight parity

Required exact historical values:

- reliable denominator: `492`
- pooled contain2: `0.975609756097561`
- worst-family contain2: `0.9322033898305084`
- families ≥ .90: `8`
- per-family contain2:
  - 09908 `1.0`
  - 11032 `0.9322033898305084`
  - 12772 `1.0`
  - 13203 `0.9344262295081968`
  - 14404 `0.9682539682539683`
  - 14702 `1.0`
  - 14758 `0.9838709677419355`
  - 15290 `0.9836065573770492`
- per-family contain1:
  - 09908 `0.9841269841269841`
  - 11032 `0.8135593220338984`
  - 12772 `1.0`
  - 13203 `0.8524590163934426`
  - 14404 `0.9047619047619048`
  - 14702 `1.0`
  - 14758 `0.9516129032258065`
  - 15290 `0.8852459016393442`

Integer/count gates must be exact. Floating gates use absolute tolerance `1e-12` unless a historical value was explicitly rounded (F16 above).

## Gate C — V2 behavioral parity

Primary:
- `n = 261`
- U_ONLY contain1 `0.3716475095785441`
- U_ONLY contain2 `0.6781609195402298`
- U_ONLY worst contain2 `0.28`
- U_ONLY median normalized error `1.3179303569635419`
- U_ONLY p90 normalized error `4.1520870275051776`
- G_REL_DIS contain1 `0.4521072796934866`
- G_REL_DIS contain2 `0.8084291187739464`
- G_REL_DIS worst contain2 `0.44`
- G_REL_DIS median normalized error `1.142215140124115`
- G_REL_DIS p90 normalized error `3.045158138805841`

Per-family primary contain2:
- 09908: U `0.8333333333333334`, G `0.9`
- 11032: U `0.28`, G `0.44`
- 12772: U `0.8181818181818182`, G `0.7272727272727273`
- 13203: U `0.6111111111111112`, G `0.6944444444444444`
- 14404: U `0.5`, G `0.71875`
- 14702: U `0.8181818181818182`, G `0.9772727272727273`
- 14758: U `0.8181818181818182`, G `1.0`
- 15290: U `0.7`, G `0.82`

Secondary:
- `n = 480`
- U_ONLY contain2 `0.7958333333333333`
- G_REL_DIS contain2 `0.8791666666666667`

Historical V2 verdict must be:
`RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`.

All values above use absolute tolerance `1e-12`.

## Gate D — descendant P1 parity

The recovered family state is then consumed by the already-persisted P1 source. Required:

- P1 M256 denominator `492`
- P1 U_ONLY contain2 `0.6781609195402298`
- round-1 candidate-index equality to persisted one-pass solver: `512 / 512`
- round1 contain1 `0.47126436781609193`
- round1 contain2 `0.7777777777777778`
- round1 worst-family contain2 `0.4`
- round1 median normalized error `1.0715530716411676`
- round1 secondary contain2 `0.8666666666666667`
- round2 contain1 `0.45977011494252873`
- round2 contain2 `0.8160919540229885`
- round2 worst-family contain2 `0.52`
- round2 median normalized error `1.112911946142779`
- round2 p90 normalized error `3.045158138805841`
- round2 secondary contain2 `0.8875`

Round2 per-family contain2:
- 09908 `0.9`
- 11032 `0.52`
- 12772 `0.7272727272727273`
- 13203 `0.6666666666666666`
- 14404 `0.75`
- 14702 `0.9772727272727273`
- 14758 `1.0`
- 15290 `0.82`

Counts/index equality exact; floating values tolerance `1e-12`.

## Decision

`SOURCE_RECOVERY_PASS` iff Gates A+B+C+D all pass and source/checkpoint SHA guards pass.

Any failure => `SOURCE_RECOVERY_V1_FAIL__DO_NOT_TUNE__DO_NOT_RUN_CORRECTED_0_OF_48`.

A failed V1 may not be repaired after seeing the failed metric. A different recovery hypothesis would require a new source authority, new three-store persistence manifest, and a new preregistration.

The corrected observable-geometry 0/48 audit remains forbidden until `SOURCE_RECOVERY_PASS`.
