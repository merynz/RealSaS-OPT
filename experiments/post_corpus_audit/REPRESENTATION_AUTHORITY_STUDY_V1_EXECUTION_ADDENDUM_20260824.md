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

## 5. Confirmatory asset and query panel

The parent prereg requires deterministic family-disjoint samples rather than an exhaustive all-open brute force. The final R0–R3 confirmatory panel is therefore frozen at:

- **256 OPEN controlled assets**;
- **48 queries per asset maximum**;
- 16 query slots for adjacent yaw distance 1;
- 16 for skip-one yaw distance 2;
- 16 for opposite yaw distance 4.

### Asset selection

Build strata from `(source_registry_id, split, strongest capability class)` where strongest capability is `ARACHNE > GEPPETTO > IRIS`.

1. Within every non-empty stratum, sort assets by `SHA256("repr-v1:" + asset_id)` and take one witness.
2. Fill remaining slots to 256 from the remaining OPEN controlled assets sorted by the same stable hash.
3. If fewer than 256 legal cached assets exist, the run is `APPARATUS_TARGET_REOPEN_REQUIRED`; do not silently reduce the confirmatory panel.

This guarantees representation of every available provider/split/capability stratum without changing the corpus' real distribution inside reported pooled metrics. Source-stratified and macro-stratified summaries are reported separately.

### Query selection

For each asset/category, enumerate legal `(track, source_view, target_view)` tuples and select the first 16 under stable `SHA256(asset_id|track|source|target|category)` ordering.

If a category has fewer than 16 legal queries, use all available queries and report the shortfall; unused quota is **not** moved to another category.

The final run is confirmatory only when:

- asset target = 256;
- query cap = 48;
- no debug `max_assets` override is used;
- the exact selected asset IDs are written to the result before any arm metrics are interpreted.

Smaller runs are `DEVELOPMENT_ONLY` and cannot issue the study decision.

## 6. Observation-noise semantics

The parent prereg says perturbations are independent. Execution meaning is now frozen:

- every `(asset, view, locus, arm)` observation receives an independent deterministic perturbation;
- **both source queries and target candidates** are perturbed;
- exact physical P remains evaluator-only truth;
- RNG must be independent of query/evaluation loop order.

### Deterministic RNG namespace

For executable vectorization without changing the statistical contract:

1. derive one 128-bit seed from the first 16 bytes of `SHA256("repr-v1-noise|" + asset_id + "|" + arm_id + "|V" + view_index)`;
2. initialize NumPy `Philox` with that seed;
3. generate the full observation-noise array in canonical ascending cached `track_index` order.

The cached track index is part of the immutable observation identity for this study. Therefore a track's perturbation is fixed by `(asset_id, arm_id, view_index, track_index)` and cannot change when query order, candidate order or evaluation batching changes. No Python/global RNG state is allowed.

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

The bucket boundaries are frozen before results as:

- connected components: `1`, `2-4`, `5-16`, `17+`;
- query physical-locus view support from `track_support`: `2-3`, `4-5`, `6-8`.

No bucket boundary may be changed after R0–R3 results are opened. Mean-only promotion remains forbidden.

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
