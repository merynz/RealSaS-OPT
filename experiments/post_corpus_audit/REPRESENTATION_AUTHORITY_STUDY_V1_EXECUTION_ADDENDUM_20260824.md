# Representation Authority Study V1 — Execution Addendum

**Date:** 2026-08-24  
**Status:** `FROZEN_BEFORE_R0_R3_NEW_RESULTS__R4_FORMULATION_NOT_YET_FROZEN`  
**Parent prereg:** `REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md`

This addendum fills execution details that the parent prereg did not specify. It is frozen **before** running the new R0–R3 study and must not be changed in response to results. It does not authorize an IRIS optimizer step.

## 1. Controlled population

Use the later chronological authority `IRIS_USABLE_CONTROLLED_CORPUS_FREEZE_V1.md`.

Immediate legal population: **3930 assets**. Excluded before this study:

- 56 Stage-B6 repair-pending assets whose current A renders correspond to pre-repair geometry;
- 6 active non-Basis shape-key assets;
- 1 all-eight-view blank asset.

Representation evaluation uses only OPEN `FIT/TUNE` membership from that freeze. `CAL/DEV/EXTERNAL_HOLDOUT` remain unopened for this study.

## 2. Truth/cache authority

Input truth is the V2 geometry-only cache built from:

`pixel -> triangle+barycentric -> continuous canonical P`

with raster authority used only to prove legal visibility/same-surface support.

The cache must satisfy the V2 stage/cache audit before this study. Accepted `track_xy` remains exact continuous projection; no pixel-center re-quantization.

No joint, parent, skin, authored owner ID or teacher rig field may enter this study.

## 3. Candidate universe

For a source observation `(asset, source_view, physical_locus)` and target view:

- candidate universe = **all cached persistent physical loci legally visible in the target view**;
- GT track/locus IDs are used only to construct/evaluate the exact physical target and visibility relation;
- no candidate shortlist derived from the answer is allowed;
- no authored rig identity is used.

This is a representation-ceiling test over the geometry-only persistent-surface sample, not a dense-pixel product matcher result.

## 4. Set-valued truth

Inherited physical same-locus tolerance:

`SAME_LOCUS_TOL = 0.003` canonical units.

A predicted candidate is correct when its **exact canonical physical P** is within this tolerance of the query's exact physical P.

Therefore top-1/top-4/top-8 are ambiguity-aware set-containment metrics. The evaluator never penalizes selecting another candidate inside the legal same-locus set merely because it has a different cached index.

## 5. Query/view-pair panel

Every evaluated asset contributes a deterministic balanced query panel across:

- adjacent yaw distance 1;
- skip-one yaw distance 2;
- opposite yaw distance 4.

Default panel cap: `96 queries / asset` (32 target queries per category where available). If a category has fewer legal queries, unused quota is not silently moved to a different category in aggregate reporting; actual counts are reported.

Queries are chosen by stable SHA-256 ordering over `(asset_id, track_index, source_view, target_view)`, not current representation score.

R0/R1/R2 may run on all open controlled assets if runtime permits. The CLI may set a smaller deterministic pre-result sample for development, but any result with `max_assets > 0` is explicitly `DEVELOPMENT_ONLY` and cannot issue the final study decision. Final confirmatory R0–R3 output requires `max_assets=0` over all records in the supplied OPEN cache manifest.

## 6. Observation-noise semantics

The parent prereg says perturbations are independent. Execution meaning is now frozen:

- every `(asset, view, locus, arm)` observation receives an independent deterministic perturbation;
- **both source queries and target candidates** are perturbed;
- exact physical P remains evaluator-only truth;
- RNG seed is SHA-derived from immutable observation identity + arm, never iteration order.

### P noise

Gaussian isotropic canonical-coordinate noise with sigma exactly:

`0, 0.0005, 0.0010, 0.0025, 0.0050, 0.0100`.

### N noise

For a requested angular level, choose a deterministic random tangent direction orthogonal to N and rotate N by the **exact stated angle**:

`0°, 5°, 10°, 20°, 40°`.

Do not approximate angular noise by adding Gaussian xyz noise and renormalizing.

## 7. R0/R1/R2/R3 scoring

### R0 / R2 — P only

Score is canonical Euclidean observation distance:

`score_P(i,j) = ||P_obs_source_i - P_obs_target_j||_2`.

Lower is better.

### R1 / R3 — P + N

The parent prereg does not specify a learned/calibrated coefficient. To avoid post-result coefficient tuning, freeze **legacy diagnostic parity** from `iris_controlled_v1/representation_ceiling_v1.py`:

`score_PN = dP + 0.05 * (1 - cos(N_source, N_target))`.

`0.05` is therefore not promoted as a product-optimal physical constant. It is a frozen comparison coefficient inherited from the pre-audit representation ceiling. The result must report P-only and P+N arms separately; no coefficient search or best-of-weight reporting is allowed.

### R3 grid

Run the full preregistered cross product:

- all six P sigma values;
- all five N angular values.

No post-result pairing/subset selection.

## 8. Reciprocal / cycle semantics

For each representation arm:

- forward top-1: source -> target;
- reciprocal: use the selected target observation as query back into all legal source-view candidates; reciprocal success if reverse top-1 belongs to the original source query's legal same-locus set;
- cycle: for queries whose true physical locus has a legal third-view observation, choose the third view by stable SHA ordering among legal third views and evaluate source -> target -> third -> source; cycle success if final top-1 belongs to original legal same-locus set.

The true locus is used only to choose a **legal evaluation third view**, never as a matching score/anchor.

## 9. Required stratification

Every final result reports:

- pooled query metrics;
- per-asset metrics and family-tail distribution;
- provider/source registry;
- FIT vs TUNE;
- capability class from `CANONICAL_VARIANT_SELECTION.json` (`IRIS`, `GEPPETTO`, `ARACHNE` strongest available class);
- mesh connected-component bucket;
- observation coverage/support bucket;
- pair category.

Mean-only promotion remains forbidden.

## 10. Hard-tail manifest

Record every exact/noise query satisfying at least one:

- top-8 miss;
- top-1 physical error > 0.01 canonical units;
- reciprocal failure under exact R0/R1;
- cycle failure under exact R0/R1.

Each row contains asset, source/target/third view when relevant, provider, capability, component/coverage bucket, arm and legal ambiguity-set size. No RGB or rig-secret data are required in this manifest.

## 11. R4 integrity lock

**R4 is NOT defined by this addendum.**

The parent prereg names an `uncertainty-aware global relational address` but does not freeze its mathematical construction. Historical GFDR evidence proves that relational/global addressing can be causally useful in scoped settings, but it does not uniquely specify a clean current single-pose R4 implementation.

Therefore:

- R0–R3 may execute now under this addendum;
- if R2/R3 expose a reproducible hard tail that motivates R4, a separate `R4_FORMULATION_PREREG` must freeze the exact relation, confidence weighting, candidate use and no-oracle constraints **before R4 results are opened**;
- no newly invented relational formula may be retrospectively called the preregistered R4 arm.

This is deliberately stricter than silently filling the gap.

## 12. Decision discipline

R0–R3 runner may issue only measurement output plus one of:

- `R0_R3_COMPLETE__R4_NOT_NEEDED_IF_EXACT_GEOMETRY_CEILING_CLEAR`
- `R0_R3_HARD_TAIL__FREEZE_R4_FORMULATION_BEFORE_NEXT_RESULT`
- `APPARATUS_TARGET_REOPEN_REQUIRED`

The final parent-prereg decision (`P_GEOMETRY_SUFFICIENT`, `P_PLUS_R_REQUIRED`, `...SOI2`) is written only by the canonical interpretation report after verifying the full required evidence. The runner itself must not invent missing promotion thresholds after seeing results.
