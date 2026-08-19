# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__F16_BOUNDED_H_RESEARCH_CONTRACT_FROZEN__SMALL_PACTIVE_LOGAMP_HEAD_AUTHORIZED__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; primary F-activity was `0.433160 < 0.50` while the other principal geometry/mechanics metrics passed.
- Stage-B/e00 truth is open-development evidence only and cannot be reused as blind qualification.
- sealed21 / external10 remain **CLOSED**.
- Any later blind qualification requires a **new untouched preregistered panel**.

## Representation decision

Canonical representation sufficiency remains:

`REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT`

Supported interface:

```text
P_A
bounded set-valued H_i / P_B_geom
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred
+ typed U_obs
```

Authority split:

```text
H_i                                -> physical Pose-B XYZ authority
p_active                           -> absolute motion/non-motion evidence
log_amp                            -> conditional magnitude/ranking evidence
dir                                -> direction evidence
U / margin / multimodality/support -> retention / abstention / confidence
compiler/global solver             -> final collapse
```

Free XYZ motion authority and early fixed top4 collapse remain forbidden.

## Bounded H research contract

Current research freeze:

`FREEZE_F16_FOR_RESEARCH__PRODUCT_COMPRESSION_DEFERRED`

File: `BOUNDED_H_RESEARCH_CONTRACT_V1.md`  
Commit: `b3cec959a9d593e0a2693c2bf894b9ea7673a4ac`

Why:

- fixed top4 hard-tail RED;
- score top8 / top10 / top12 fail the strict tail gates;
- Adaptive H V1 does not safely identify rare expansion cases;
- K8/K10 diversity improves hard-tail coverage but still misses the worst-family gate;
- D12-C8 matches the F16 worst-family floor but exceeds the compact target;
- repeatedly tuning K10 on the same truth-open 8-family panel would create overfit risk.

F16 broad e00 reference:

```text
pooled primary-2x = .98171
worst family       = .94915
families >= .90    = 8/8
best-worst gap     = 5.08pp
strict-1x pooled   = .94919
```

Representation-conditioned oracle constrained to this H passes all 6 principal GFDR gates.

Vectorized pairwise H materialization diagnostic, excluding neural forward/search/scoring:

```text
~0.331 ms / carrier
~21.2 ms / 64-carrier episode
mean H = 2146
median H = 1536
p95 H = 4608
```

Therefore H materialization is not a research-stage blocker. `K=16` is **not** declared a permanent product constant; compression is deferred to a dedicated later optimization split/problem.

## Closed compression diagnostics

### Adaptive H V1

Verdict:

`FAIL__NO_COMPACT_ADAPTIVE_CONTRACT_PASSES__KEEP_F16_REFERENCE`

- F8: pooled `.95528`, worst `.88136` — coverage FAIL.
- A1 descriptor adaptive: pooled `.96951`, worst `.89831` — coverage FAIL, efficiency PASS.
- A2 descriptor+geometry: pooled `.97358`, worst `.91525`, mean K `11.93` — coverage + efficiency FAIL.

### Expansion-need audit

Only `13/492` reliable carriers (`2.64%`) are F8 misses rescued by F16.

- score/reprojection features do not provide a stable family-disjoint rare-event expansion gate;
- candidate spatial multimodality is more informative (best single LOFO AUROC ~`.808`) but 13 positives are insufficient to freeze a robust expansion classifier.

### Diversity retention

- D8-C4: pooled `.97764`, 8/8 >=.90, worst `.91525`; efficient but tail FAIL.
- D10-C4: pooled `.97967`, worst `.93220`, gap `6.78pp`, H/F16 `.391`; one `11032` carrier short of the prereg worst-family gate.
- D12-C8: pooled `.97967`, worst `.94915`; coverage PASS, compact budget FAIL.

### Score rank depth

- score K10: pooled `.96545`, worst `.90164` — FAIL.
- score K12: pooled `.97154`, worst `.91525` — FAIL.

Conclusion: top16 safety is not explained by rank depth alone; preserving alternate modes matters, but current compact policies are not robust enough to replace F16 on this development panel.

## Current authorized next work

The bounded-H prerequisite is now satisfied for research.

Authorized:

1. preregister and run **small `p_active + log_amp` family/episode-disjoint probes**;
2. keep `dir` separate;
3. keep F16 H frozen and prohibit free XYZ;
4. test each factor causally before combined candidate scoring;
5. maintain worst-family/tail gates, not aggregate-only promotion.

Still forbidden:

- large/end-to-end training;
- retuning F16 H from truth;
- Stage-B requalification;
- sealed/external opening.

## Canonical current artifacts

- `REPRESENTATION_CONTRACT_V2.md`
- `BOUNDED_H_RESEARCH_CONTRACT_V1.md`
- `F16_H_CONSTRUCTION_BENCHMARK_V1.json`
- `ADAPTIVE_HYPOTHESIS_RETENTION_V1_REPORT.md`
- `EXPANSION_NEED_SEPARABILITY_AUDIT_V1_REPORT.md`
- `DIVERSITY_PRESERVING_RETENTION_V1_RESULT.json`
- `DIVERSITY_PRESERVING_RETENTION_V1_K10_FOLLOWUP_RESULT.json`
- `SCORE_RANK_DEPTH_RETENTION_V1_RESULT.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_CANONICAL_DECISION.md`

**Current frontier:** frozen F16 H + small factorized motion-state/magnitude heads; no big training.
