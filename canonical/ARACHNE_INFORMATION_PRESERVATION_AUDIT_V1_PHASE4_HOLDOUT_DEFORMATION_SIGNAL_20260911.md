# RealSaS — Arachne Information-Preservation Audit V1 — Phase 4 Holdout Deformation Signal

**Date:** 2026-09-11  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_RUNNING_EXPERIMENT_CHANGE_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Current A0 4/8/16/32 run remains unchanged.**

## Finding

The current FIX2 token-capacity experiment does **not** backpropagate any direct deformation-consequence loss, and the disjoint holdout is never used as a training target. `hold_deform` is an evaluation-only co-metric.

Training per joint uses exactly 384 scalar queries: 192 GSA950 supervised rows + 192 Dense8K product-derived rows. The per-joint objective is `BCEWithLogits + 0.1*MSE + Dice` on that joint's scalar field. Joint-field normalization is not part of the training objective.

At evaluation, all 22 joint logits are first sigmoid-transformed and then normalized jointly to the simplex. Only after this cross-joint normalization is the LBS consequence metric computed. Therefore the gradient path is:

`per-joint scalar field loss -> field encoder/queries/decoder`

and **not**:

`holdout normalized joint competition -> LBS deformation -> loss`.

So the user's suspicion that the signal capable of directly optimizing `hold_deform` is not reaching the optimizer is literally correct. This is not an implementation disconnect in evaluation; it is the preregistered objective/data contract.

## Evaluation metric connectivity

The deformation co-metric itself is not frozen or disconnected. It computes teacher and predicted LBS from `wtruth` and `wpred`, then uses RMS predicted-vs-teacher deformation error divided by teacher motion RMS. Thus model-output changes can change the metric. The missing connection is backward supervision, not forward evaluation.

## Additional caution: synthetic probe semantics

The current `PROBE_TRANSFORMS` are synthetic per-joint **translations** indexed by serialized joint index. They are not articulated parent-relative bone rotations around actual pivots. This is useful as a deterministic consequence probe but has two implications:

1. redistribution of weight between joints with similar synthetic translation vectors can produce small or cancelling deformation changes even while row-wise weight error changes materially;
2. because probe transforms are assigned from `joint index`, the consequence metric is not intrinsically joint-permutation invariant unless transform assignment is permuted consistently with joint identity.

This does not invalidate within-run arm/checkpoint comparison because ordering and probes are fixed, but it limits interpretation as a physical deformation proof.

## Scientific interpretation

A falling disjoint-holdout row-L1 p95 with flat holdout deformation is therefore compatible with at least three mechanisms:

- the per-joint scalar objective improves marginal field values but not the coordinated post-normalization joint balance that matters to LBS;
- improvements occur on rows/joints that contribute little to the fixed probe's deformation vector, while structured errors remain on deformation-sensitive rows;
- synthetic translation-probe cancellation/insensitivity hides some weight improvements.

It is **not** evidence by itself that the decoder cannot influence holdout deformation, nor that token capacity is the blocker.

## Action

Do not alter the running 4/8/16/32 experiment. After the arms finish, perform a zero-optimizer diagnostic on the frozen selected artifacts:

- compare pre-normalization scalar error vs post-normalization row error;
- per-row/per-joint contribution to LBS error;
- deformation-error stratification by pure vs blend/boundary rows;
- dominant-joint and inactive-mass errors among the rows contributing most to LBS error;
- probe-sensitivity test with multiple deterministic transform banks;
- joint-permutation equivariance of the deformation co-metric;
- separately evaluate an articulated parent-relative rotation probe before deciding whether deformation consequence should enter a later A0/A1 training objective.

Any direct deformation loss or holdout-target training would be a new treatment and requires a new preregistration. No current A0 gate or winner rule is changed by this audit finding.
