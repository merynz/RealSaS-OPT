# RealSaS — Controlled DINOv2 Ladder Prereg Amendment V1.1

**Date:** 2026-08-29  
**Status:** `BINDING_BEFORE_ANY_DINO_CANDIDATE_OUTPUT__SUPERSEDES_ONLY_THE_ENUMERATED_V1_AMBIGUITIES`

Parent: `DINO_CONTROLLED_REPRESENTATION_LADDER_PREREG_V1.md`.

No DINO candidate output, optimizer step, candidate feature cache or candidate metric was opened before this amendment.

## Why this amendment exists

A self-audit found two phrases in V1 that were scientifically too loose:

1. `at least 95%` was not converted to an integer for the fixed 64 family-style held-out cells;
2. `not globally underfit` did not have a frozen numeric operator.

This amendment closes those loopholes before candidate evidence exists.

## Frozen integer coverage rule

`FIT_PROXY32` contains 32 families x 2 frozen styles = 64 family-style cells.

Therefore:

`HELDOUT_REPLAY_PASS_MIN = 61 / 64`.

A candidate with `60/64` or fewer is a primary FAIL even though 60/64 = 93.75% is visually close to 95%.

Hard typed-route/replay exceptions remain separately forbidden: `HARD_ROUTE_FAILURE_MAX = 0`.

## Frozen training-fit adequacy rule

`TRAIN_DIAG32` also contains the same fixed 32 nested training families x 2 frozen styles = 64 family-style cells.

At `PRIMARY_32768`, a rung is considered **training-fit adequate** iff:

- `TRAIN_DIAG32_DIRECT_REPLAY_PASS_COUNT >= 61 / 64`; and
- `TRAIN_DIAG32_HARD_ROUTE_FAILURE_COUNT = 0`.

No scalar loss, mean depth error or median P95 may override this direct-replay adequacy rule.

If a rung fails this training-fit adequacy rule, its held-out result is reported but it cannot receive `REPRESENTATION_ACCESSIBLE_V1` because the fixed learner/budget did not adequately fit its own frozen training diagnostic population.

If all four rungs fail training-fit adequacy, the allowed conclusion is:

`SHARED_LEARNER_OR_OPTIMIZATION_ADEQUACY_NOT_ESTABLISHED_AT_32768`

—not `DINO_REPRESENTATION_CAPACITY_FALSIFIED`.

## Updated primary PASS rule

At `PRIMARY_32768`, `REPRESENTATION_ACCESSIBLE_V1` requires all of:

1. `FIT_PROXY32_DIRECT_REPLAY_PASS_COUNT >= 61/64`;
2. `FIT_PROXY32_HARD_ROUTE_FAILURE_COUNT = 0`;
3. `TRAIN_DIAG32_DIRECT_REPLAY_PASS_COUNT >= 61/64`;
4. `TRAIN_DIAG32_HARD_ROUTE_FAILURE_COUNT = 0`.

All other V1 rules remain unchanged.
