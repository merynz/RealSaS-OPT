# RealSaS — Consumer Coupling Probe Prereg V1

**Date:** 2026-08-29  
**Status:** `SEALED_BEFORE_COUPLING_OUTCOME__SACRIFICIAL_ONLY`

## Question

Does the clean `G0 -> Compiler -> A0 -> Compiler -> deformation` route remain measurably coupled to skeleton geometry, or can A0 silently reweight a systematically bad G0 skeleton until the final posed surface is nearly indistinguishable from the clean route?

This is the missing step 6 of `CONSUMER_VALIDITY_INTERLOCK_V1.md`. It is not a product-quality claim and does not tune IRIS, G0, A0, Compiler solvers, or the depth bridge.

## Frozen witness

Use the same real clean fixture and exact historical Compiler closure already used by the clean consumer interlock:

- asset: `asset_551ea351b43a1787d0f55536`
- fixture transport SHA-256: `81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5`
- exact vendor raw SHA-256: `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`

## Frozen corruption

Construct `G0_BAD_COLLAPSE75_V1` from the clean G0 proposal.

- identify the proposal root as the unique highest `root_score` joint;
- keep the root position unchanged;
- for every non-root joint with clean position `p` and root `r`, set

`p_bad = r + 0.25 * (p - r)`;

- preserve proposal IDs, edges, scores, confidence, support-surface IDs, and hard-required topology exactly;
- only joint positions are changed;
- run the exact same Compiler skeleton qualification on clean and corrupted proposals.

This is deliberately a large, systematic skeleton-geometry corruption. The corruption magnitude itself must satisfy normalized joint-position RMS >= `0.05 * surface_bbox_diag`; otherwise the probe is invalid rather than tuned.

## A0 routes

Evaluate two corrupted-skeleton skin routes:

1. `BAD_RECOMPUTED_A0`: rerun the exact sacrificial A0 distance-softmax rule on the corrupted qualified skeleton.
2. `BAD_FROZEN_CLEAN_WEIGHTS`: copy the clean qualified skin weights by `source_proposal_id` onto the corrupted skeleton's newly minted canonical joint IDs, then requalify the copied proposal.

The frozen-weight route is diagnostic only. It estimates how much A0 reweighting compensates the skeleton corruption.

## Matched deformation probe

Use the same deterministic linear-blend probe family as the clean interlock, but with a frozen `15 degree` non-root rotation to increase coupling observability. For each qualified skeleton, derive the local rotation axis from the joint-parent vector exactly as in the clean probe and apply the matched rule to all non-root joints.

Compare final posed surface coordinates against the clean route.

Primary coupling observable:

`POST_COMP_RMSE_NORM = RMSE(P_bad_recomputed - P_clean) / surface_bbox_diag`

Secondary observables:

- `POST_COMP_P95_NORM`: P95 pointwise posed-position delta / bbox diagonal;
- `FROZEN_WEIGHT_RMSE_NORM`: corrupted skeleton with frozen clean weights vs clean route;
- `COMPENSATION_RATIO = POST_COMP_RMSE_NORM / FROZEN_WEIGHT_RMSE_NORM` when denominator is nonzero;
- mean per-surface-row A0 weight L1 change after mapping canonical IDs back to proposal IDs.

## Hard gate

The coupling probe PASS requires all of the following:

- clean skeleton qualification PASS;
- corrupted skeleton qualification PASS;
- clean A0 qualification = 512/512 rows;
- corrupted recomputed A0 qualification = 512/512 rows;
- corrupted frozen-clean-weight qualification = 512/512 rows;
- normalized skeleton corruption RMS >= `0.05`;
- all three posed surfaces finite;
- `POST_COMP_RMSE_NORM >= 0.005`;
- `POST_COMP_P95_NORM >= 0.010`;
- coupling proof is bound to the exact corrupted product state.

Interpretation: a deliberately severe bad G0 must remain visibly detectable in final surface response even after A0 is allowed to recompute its weights. A0 is allowed to compensate partially; `COMPENSATION_RATIO` is diagnostic and has no pass threshold. We do not penalize useful compensation, only silent erasure below the frozen detectability floor.

## Stop rule

No corruption factor, rotation angle, metric, threshold, A0 sigma/top-k, or Compiler setting may be changed after observing this probe outcome. If the hard gate fails, record the failure as evidence and redesign only in a new explicitly versioned preregistration.