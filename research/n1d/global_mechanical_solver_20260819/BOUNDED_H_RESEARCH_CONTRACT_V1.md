# RealSaS N1D — Bounded H Research Contract V1

**Date:** 2026-08-19  
**Decision:** `FREEZE_F16_FOR_RESEARCH__PRODUCT_COMPRESSION_DEFERRED`  
**Status:** `OPEN_DEVELOPMENT_ARCHITECTURE_FREEZE__NOT_QUALIFICATION`

## Why this decision is being made now

Representation Sufficiency Battery V1 demonstrated that the fixed score-top4 contract is hard-tail insufficient while the same frozen refined top16 candidate pool is hard-tail sufficient on the broad 8-family e00 development panel.

Subsequent compression work was preregistered and tested rather than tuned informally:

1. Adaptive H V1:
   - fixed K8 fails hard-tail coverage;
   - descriptor-only adaptive retention fails;
   - descriptor + raw geometry escalation improves coverage but still fails the worst-family/gap gates and over-expands.
2. Expansion-need audit:
   - only 13/492 reliable carriers are F8 failures rescued by F16;
   - candidate spatial multimodality is informative, but the rare-event sample is too small for a stable family-disjoint expansion classifier.
3. Diversity Retention V1:
   - D8-C4 materially improves coverage at K8 but misses the worst-family gate;
   - diagnostic D12-C8 reaches the F16 hard-tail floor but exceeds the compact K<=10 target.
4. K10 causal follow-up:
   - D10-C4 comes within one `11032` carrier of the worst-family threshold but still fails the preregistered gate;
   - D10-C8 fails more strongly.
5. Score-depth follow-up:
   - fixed score K10 and K12 both fail the hard-tail gate;
   - therefore D12-C8's success is not reducible to rank depth alone.

Continuing to tune K10 selection rules on the same truth-open 8-family panel until one more carrier flips would create an avoidable overfitting risk.

## Frozen research candidate-retention contract

For the next factor-head experiments, retain the frozen refined **score top16 candidates per usable view** and construct the same pairwise rank-sufficient 3D hypothesis set `H_i`.

```text
per-view refined descriptor candidates: top16
coarse search: existing frozen top8 seed route
H_i construction: frozen pairwise rank-sufficient triangulation
H_i authority: sole Pose-B XYZ authority
final endpoint: not selected by this contract; later constrained/global solver owns collapse
```

No `p_active`, `log_amp` or `dir` head may bypass this H set to create free XYZ.

## Why F16 is acceptable as a research contract

The objective here is scientific closure, not final product micro-optimization.

On the broad open-development panel F16 has:

- pooled primary-2x containment: `0.98171`
- worst family: `0.94915`
- 8/8 families >= `0.90`
- best-worst gap: `5.08pp`
- representation-conditioned oracle: all 6 principal GFDR gates PASS.

A vectorized execution diagnostic measured pairwise H construction from already-computed top16 per-view candidates:

- 512 carriers: `0.1697s`
- approximately `0.331 ms/carrier`
- approximately `21.2 ms` per 64-carrier episode
- mean H size `2146`
- median H size `1536`
- p95 H size `4608`

This timing excludes neural forward, descriptor search and later scoring, so it is **not a product inference benchmark**. It is sufficient to show that the H materialization itself is not a blocker for the next research stage.

## Product compression is explicitly deferred

`K=16` is not declared a permanent product constant.

Compression becomes a separate optimization problem after the mechanical/factorized representation is scientifically stable. Future compression work must use either:

- additional open-development families;
- a dedicated untouched compression validation split; or
- a representation-preserving distillation/pruning criterion with its own preregistration.

The current 8-family truth-open panel must not be repeatedly mined to invent a K10 rule that flips one remaining carrier.

## Authorization consequence

Because a **bounded research H contract is now frozen**, the prerequisite that paused factor-head execution is satisfied.

The next authorized work is a small, strictly family/episode-disjoint factor-head experiment:

```text
p_active
log_amp
```

with `dir` remaining separate and H remaining frozen.

Large/end-to-end training is still forbidden.

## Qualification boundary

This freeze does not change any qualification result:

- Stage A remains PASS.
- Frozen Stage B remains FAIL/no-retune.
- Stage-B and e00 development truth remain open-development evidence only.
- Any new qualification requires a new untouched preregistered panel.

## Canonical supporting artifacts

- `REPRESENTATION_CONTRACT_V2.md`
- `ADAPTIVE_HYPOTHESIS_RETENTION_V1_REPORT.md`
- `EXPANSION_NEED_SEPARABILITY_AUDIT_V1_REPORT.md`
- `DIVERSITY_PRESERVING_RETENTION_V1_RESULT.json`
- `DIVERSITY_PRESERVING_RETENTION_V1_K10_FOLLOWUP_RESULT.json`
- `SCORE_RANK_DEPTH_RETENTION_V1_RESULT.json`
- `F16_H_CONSTRUCTION_BENCHMARK_V1.json`
