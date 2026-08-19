# RealSaS N1D — Post-Stage-B Scalar-Motion Factorization Ceiling V1

**Date:** 2026-08-19  
**Status:** TRUTH-OPEN DEVELOPMENT DIAGNOSTIC — NOT A NEW QUALIFICATION  
**Frozen authority:** Hybrid V11 Stage-B remains `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`.

## 1. Question

Hybrid V11 failed untouched Stage-B only on the primary motion-activity ordering gate while preserving strong geometry/mechanics. This diagnostic asks a narrow causal question:

> If V11's displacement directions are kept frozen, how much of the failure can be removed by changing only scalar displacement magnitude/activity ordering?

It intentionally does **not** train or retune V11 and does **not** claim that the resulting endpoints lie inside the original V11 feasible candidate basin. The frozen prediction files do not persist the full candidate sets `H_i`, so candidate-basin-safe composition is a separate required experiment.

## 2. Provenance and fail-closed guards

The diagnostic consumes only the already-frozen Stage-B V11 predictions plus the now-open Stage-B truth sidecars and exact canonical N1D evaluator source.

Guards passed before any counterfactual result was accepted:

- frozen Stage-B result SHA-256: `33d5f55e9f2f8e8619d0e76638283e8bd2388dd3ae4db9974d85ce8952c1d472`
- prediction-freeze ledger SHA-256: `f05ac9d48029a4bd1d62a9ec10967978e524213cb20f2b25a757fad351b48c19`
- Stage-B prereg SHA-256: `19eca4d653628ed451f5f2ea1d1ad6b4c5ca4ed29c73cead85f5d72f983fc4c5`
- truth-open ledger SHA-256: `80b2b6b371a199b8023718a8ae83fa96084592399c7296255394e5093fa392c2`
- exact GFDR source SHA-256: `ca22e3fd42e9c812632eb372544e2319ecf720f6dede1f4a60c8da14ff0cb8fa`
- exact target builder SHA-256: `365cdf52cf1cfc2c7feebbf921a5702b0e8b78338be5692bc107284c4aff534f`
- truth sidecars: **12/12 SHA-256 PASS**
- frozen predictions: **12/12 SHA-256 PASS**
- canonical source tests: **39/39 PASS**
- frozen evaluator baseline parity: **12/12 episodes, max numeric diff = 0.0**

The canonical checkpoint SHA `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18` was independently recovered as context, but this diagnostic does not consume the checkpoint because it operates on the already-frozen V11 prediction artifacts.

## 3. Counterfactual arms

### A. Baseline

Frozen V11 Stage-B prediction unchanged.

### B. Exact-amplitude / frozen-direction ceiling

For each carrier with a nonzero V11 displacement direction:

```text
u_i = normalize(P_B^V11(i) - P_A^V11(i))
P_B'(i) = P_A^V11(i) + ||Delta P_truth(i)|| * u_i
```

If V11 supplied zero displacement, the diagnostic keeps the displacement zero rather than leaking a truth direction.

This is an upper-bound diagnosis; it changes magnitudes and therefore does not preserve the V11 magnitude distribution.

### C. Truth-rank / frozen-distribution / frozen-direction ceiling

This is the key factorization arm. It preserves the V11 displacement-magnitude multiset exactly and changes only which nonzero-direction carrier receives which frozen V11 magnitude:

1. freeze every V11 displacement direction;
2. keep zero-direction carriers fixed at zero;
3. sort the remaining carriers by truth displacement magnitude;
4. sort the **existing V11 magnitudes**;
5. assign those existing magnitudes in truth rank order.

Therefore this arm does **not** make the predicted motion globally larger or smaller. It changes scalar motion **ordering** while retaining the original magnitude distribution.

Measured whole-panel multiset preservation error: **2.78e-17**.

### D. Active-only truth-rank ceiling

Same principle as C, but the rank permutation is restricted to the truth-active, frozen-nonzero-direction support. This is secondary because truth activity membership itself is oracle information.

## 4. Primary result

| Arm | F activity ↑ | F kernel ↑ | D ↓ | R ↓ | G direction ↑ | G line ↓ | 6/6 quality gates |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen V11 baseline | 0.433160 | 0.634387 | 0.061331 | 0.042740 | 0.899471 | 0.088411 | **FAIL** |
| Exact amplitude, frozen direction | 1.000000 | 0.854414 | 0.025659 | 0.013112 | 0.956101 | 0.057649 | **PASS** |
| **Truth rank, frozen magnitude distribution, frozen direction** | **1.000000** | **0.838823** | **0.068851** | **0.043841** | **0.897785** | **0.076640** | **PASS** |
| Active-only truth rank | 1.000000 | 0.858687 | 0.061331 | 0.034158 | 0.933018 | 0.079173 | **PASS** |

The strongest causal observation is the third row. `F_activity` is itself rank-based, so an oracle truth-rank arm reaching 1.0 is definition-aligned and is not by itself the scientific result. The important result is that the same rank correction can be applied while:

- keeping every defined V11 direction frozen;
- preserving the complete V11 displacement-magnitude distribution to numerical precision;
- retaining or improving the downstream GFDR structure sufficiently for **all six primary quality gates to pass**.

This rules against the hypothesis that Stage-B's remaining activity failure necessarily requires replacing the geometric/directional substrate. It supports a factorized model in which scalar motion state/order is separately predicted and then composed with constrained geometry.

## 5. Frozen current-evidence diagnostic

The frozen V11 artifacts persist raw/current motion evidence only on `V8_BASE` outputs. They do **not** persist the raw `delta_point_map_srcA` amplitude on `SEED_BASIN_WSEED`; that route stores `U_obs_seed_weight`, which is a solver/projection weight and is not an equivalent substitute.

On the four `V8_BASE` Stage-B episodes:

- median finite true-active Spearman of persisted current evidence vs truth amplitude: **0.916010**
- baseline subset F activity: **0.685396**
- current-evidence rank composition F activity: **0.877234**
- baseline G line: **0.082006**
- unconstrained current-rank G line: **0.101577**

Thus current differential evidence contains strong scalar ordering signal on this route, but naive radial scaling can slightly cross the G-line quality threshold. That is evidence **for**, not against, the proposed constrained composition: amplitude should score/select candidates inside `H_i`, not freely scale XYZ.

For the seed route, `U_obs_seed_weight` has only **0.205201** median finite true-active Spearman and degrades F when treated as an amplitude ranking signal. It must not be mislabeled as current amplitude.

The previously recorded full-panel post-Stage-B diagnostic remains the correct evidence that the actual frozen N1D `delta_point_map_srcA` signal carries useful Stage-B amplitude ordering (`~0.52245` median finite per-episode Spearman). Reconstructing that raw signal for all routes is required for the next composition test.

## 6. Decision

### Demonstrated

**YES — scalar-motion factorization ceiling is demonstrated in direction-fixed continuous endpoint space.**

A scalar ordering intervention is sufficient to close the failed F-activity gate while preserving the existing displacement magnitude distribution and retaining all primary downstream quality gates.

### Not demonstrated

**NO — candidate-basin-safe composition is not yet demonstrated.**

The frozen V11 NPZ files do not persist each carrier's feasible candidate set `H_i`, so this experiment cannot assert that radially modified endpoints are valid members of the original geometric basin.

**NO — full-panel current-amplitude composition is not yet demonstrated.**

The raw current-amplitude signal must be recomputed for `SEED_BASIN_WSEED` episodes from the frozen checkpoint/raster/source path.

**NO — Hybrid V11 Stage-B is not requalified.**

Its original blind decision remains immutable. These are truth-open development diagnostics only.

## 7. Next required experiment

Before training the proposed `p_active + log_amp + dir` heads:

1. reconstruct/export the exact V11 feasible candidate set `H_i` (or equivalent basin candidates) for every Stage-B carrier;
2. recompute frozen N1D `delta_point_map_srcA` amplitude/activity for **all** V11 routes from the recovered checkpoint and raster inputs;
3. keep V11 geometric candidates fixed;
4. use current amplitude/activity only as a candidate energy/ranking term;
5. select the final `P_B` **inside `H_i`**;
6. evaluate the unchanged GFDR stack;
7. compare against V11 baseline, unconstrained scalar composition, and the oracle rank ceiling above.

If current amplitude closes most of the oracle gap **inside the feasible candidate set**, the architecture target is effectively identified: learn `p_active/log_amp` with ranking supervision and use it to score constrained geometry rather than adding another free XYZ head.

Only after that development result should a learned held-out amplitude head be trained, and only after stable development evidence should a new untouched qualification panel be preregistered.

## 8. Reproducible artifacts

GitHub persists the diagnostic in a text-safe, hash-guarded form:

- `post_stageb_factorization_ceiling_v1.py` — thin wrapper that concatenates/decodes the exact source payload and refuses execution unless the decoded SHA-256 matches;
- `source_bundle_b64/post_stageb_factorization_ceiling_v1.py.b64.part00` … `part03` — exact source payload;
- `POST_STAGEB_FACTORIZATION_CEILING_V1_RESULT_SUMMARY.json` — compact canonical aggregate/provenance result;
- this report.

The full per-episode result is deterministically regenerable from the frozen inputs and the decoded source; it is not duplicated in GitHub.

Canonical local artifact hashes at publication preparation:

- decoded diagnostic source SHA-256: `caffa717db433a56109f16eefa5ce68c4e13f24722ac40c46b85b2f6d00473a7`;
- regenerated full per-episode result JSON SHA-256: `a9520ae969e1c0483fc8b5521ffebd11f658ae8f976663bfdf4d3f56b940fdf4`.

The wrapper independently verifies the decoded source hash before compiling/executing it.
