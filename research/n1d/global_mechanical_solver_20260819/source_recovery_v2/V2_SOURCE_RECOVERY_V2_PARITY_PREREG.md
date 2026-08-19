# V2 Source Recovery V2 — Frozen Behavioral Parity Preregistration

**Date:** 2026-08-20  
**Status before execution:** `FROZEN__NOT_YET_EXECUTED`  
**Parent scientific authority:** `08620737ce526b0c0cc6c551f53910e34a1107fe`  
**Source persistence manifest commit:** `a56e53f779ee1284800bc4282112e99c2b0ccc0e`  
**Recovered source SHA-256:** `0747bf3754443053fa5a992f03df4cacddc95f75969572fe4de11253f1e441b4`  
**SOURCE_PERSISTENCE_GATE:** `PASS`

## Recovery hypothesis

Recovery V2 differs from failed Recovery V1 only in `local_scale`, using independent pre-existing `realsas_gfdr_v2.py` authority created 2026-08-17 before the historical V2 experiments:

- evidence SHA-256 `954302fa7962841179521a49e8c989e44535138f0d733f7cd4d76746062593b0`;
- `k_neighbors = 12`;
- self excluded;
- per-point local scale = median kNN distance;
- non-positive/degenerate scale replaced by median positive scale, fallback 1.0.

V1→V2 unified-diff SHA-256 is frozen as `8fc54ca574dcaf0b103ccac9c0a2073aca9af8781ebbb60be2f83970855e9a0e`. No other scientific logic changed.

No V1 failure metric was used to choose `k=12`; this value comes from the pre-existing exact GFDR source.

## Data and checkpoint

Allowed open panel only: `09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290`, e00 from canonical N1B multi-intervention corpus.

Checkpoint SHA-256 must equal `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`.

`sealed21 = CLOSED`  
`external10 = CLOSED`

Execution hygiene: all transient `global_rel_solver_v2/cache/*.pkl` files produced under Recovery V1 must be deleted before the first Recovery V2 run. Recovery V2 must build all eight family caches from the frozen source and canonical e00 inputs; cross-authority cache reuse is forbidden.

## Gate A — exact full-F16 parity

Expected 2× target-near coverage (historical rounded values, tolerance `1e-8`):

- 09908 `1.0`
- 11032 `0.9491525424`
- 12772 `1.0`
- 13203 `0.9508196721`
- 14404 `0.9682539683`
- 14702 `1.0`
- 14758 `0.9838709677`
- 15290 `1.0`

All eight must pass.

## Gate B — exact M256 preflight parity

Counts exact; floats absolute tolerance `1e-12`.

- reliable denominator `492`
- pooled contain2 `0.975609756097561`
- worst-family contain2 `0.9322033898305084`
- families >= .90 `8`

Per-family contain2:
- 09908 `1.0`
- 11032 `0.9322033898305084`
- 12772 `1.0`
- 13203 `0.9344262295081968`
- 14404 `0.9682539682539683`
- 14702 `1.0`
- 14758 `0.9838709677419355`
- 15290 `0.9836065573770492`

Per-family contain1:
- 09908 `0.9841269841269841`
- 11032 `0.8135593220338984`
- 12772 `1.0`
- 13203 `0.8524590163934426`
- 14404 `0.9047619047619048`
- 14702 `1.0`
- 14758 `0.9516129032258065`
- 15290 `0.8852459016393442`

## Gate C — historical V2 behavioral parity

Primary `n = 261`.

U_ONLY:
- contain1 `0.3716475095785441`
- contain2 `0.6781609195402298`
- worst-family contain2 `0.28`
- median normalized error `1.3179303569635419`
- p90 normalized error `4.1520870275051776`

G_REL_DIS:
- contain1 `0.4521072796934866`
- contain2 `0.8084291187739464`
- worst-family contain2 `0.44`
- median normalized error `1.142215140124115`
- p90 normalized error `3.045158138805841`

Primary contain2 by family, U/G:
- 09908 `.8333333333333334 / .9`
- 11032 `.28 / .44`
- 12772 `.8181818181818182 / .7272727272727273`
- 13203 `.6111111111111112 / .6944444444444444`
- 14404 `.5 / .71875`
- 14702 `.8181818181818182 / .9772727272727273`
- 14758 `.8181818181818182 / 1.0`
- 15290 `.7 / .82`

Secondary:
- `n = 480`
- U_ONLY contain2 `0.7958333333333333`
- G_REL_DIS contain2 `0.8791666666666667`

Historical V2 verdict must equal `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`.

All floats tolerance `1e-12`.

## Gate D — persisted P1 descendant parity

Counts/index equality exact; floats tolerance `1e-12`.

- M256 denominator `492`
- U_ONLY contain2 `0.6781609195402298`
- round1 candidate-index equality `512 / 512`
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
- 09908 `.9`
- 11032 `.52`
- 12772 `.7272727272727273`
- 13203 `.6666666666666666`
- 14404 `.75`
- 14702 `.9772727272727273`
- 14758 `1.0`
- 15290 `.82`

## Decision and no-tuning rule

`SOURCE_RECOVERY_V2_PASS` iff source/checkpoint guards and Gates A+B+C+D all pass.

Any mismatch => `SOURCE_RECOVERY_V2_FAIL__DO_NOT_TUNE__DO_NOT_RUN_CORRECTED_0_OF_48`.

No helper, k, threshold, candidate rule, ordering, evaluator, or tolerance may be changed after the first historical parity outcome is observed. A further recovery hypothesis would require new independent source evidence, new source bytes, new three-store persistence, and a new preregistration.

Corrected observable-geometry 0/48 remains forbidden until `SOURCE_RECOVERY_V2_PASS`.
