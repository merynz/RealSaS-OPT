# RealSaS N1D — New Global Mechanical Hypothesis-Set Solver Status

**Date:** 2026-08-19  
**Current canonical state:** POST-STAGE-B DEVELOPMENT / REPRESENTATION SUFFICIENCY BATTERY ACTIVE / FIXED-TOP4 H RED / SET-VALUED REPRESENTATION SURVIVES  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Current canonical frontier — 2026-08-19

Hybrid V11 remains the last frozen qualification treatment on this line.

- Stage A: **PASS** on 16/16 frozen episodes with exact standalone-runner parity.
- Stage B: **FAIL, immutable/no-retune**. Geometry/mechanics retained strong values, but primary F-activity Spearman was `0.433160 < 0.50`.
- Stage-B truth is now **OPEN DEVELOPMENT EVIDENCE** only. It must not be reused as a fresh blind qualification panel.
- sealed21 / external10 remain **CLOSED**.
- **Big/new architecture training is NOT AUTHORIZED** while the representation sufficiency battery is open.

## Representation sufficiency battery — current result

The battery was preregistered before the new population/tail measurements (`b87e222fdf7bf4c66a62314c88ee9e155a576c73`). Its current decision is **not** “representation PASS.” It has isolated one concrete representation-contract failure and one important rescue.

### Test D — broad family-disjoint learnability

On the canonical open-development N1D population (`232` episodes, `29` families; source `N1D_CANONICAL_DEV_PER_EPISODE.json`, SHA-256 `37d81db0733f6fbf433a944aa0797550f3beee15ff65377ecbcfada1064f4326`):

- strict 29-fold LOFO p_active proxy AUROC: **0.97411**;
- p_active balanced accuracy at one fixed 0.5 threshold: **0.88937**;
- multifeature active log-amp LOFO Spearman: **0.84524**;
- a naive multifeature Ridge collapses held-out family `12832` to **-0.10714** Spearman;
- however the single frozen raster-derived scalar `predicted_flow_mean` has `12832` Spearman **+0.78571**, aggregate **0.87594**, median family **0.85714**, minimum family **0.50**, with **29/29 evaluable families >=0.50** and no negative family.

Interpretation: broad family-disjoint amplitude-order information exists, but naive feature fusion/calibration can destroy a hard-tail family. This supports explicit ranking and calibrated `p_active`, not uncontrolled fusion.

### Test B — collision proxy

On coarse inference-safe episode summaries, closest 5% cross-family neighbors have active/silent mismatch **7.69%**; closest 10% mismatch **16.67%**. This remains **AMBER**, because this proxy omits the proposed explicit carrier-level `p_active`, wider candidate set and typed uncertainty fields. It is not yet an irreducible representation collision proof.

### Test C — current feasible-set hard tail

The Test-C evaluator mapping/local-scale addendum was preregistered before current H replay (`5dc6989b2026a2db7194130242243b0ab6902b09`). The unchanged current global-foreground route exports its real pairwise rank-3 hypothesis set `H_i` before final selection/refit.

Current fixed-top4 H results:

- `14702`: reliable mapping `63/64`, strict 1x containment **0.93651**, primary 2x containment **1.00000**;
- historical hard-tail `11032`: reliable mapping `59/64`, strict 1x containment **0.61017**, primary 2x containment **0.84746** (`50/59`);
- historical hard-tail `15290`: reliable mapping `61/64`, strict 1x containment **0.83607**, primary 2x containment **0.90164**.

The preregistered current-H Test-C decision is therefore **RED**: `11032 <0.90`, and the `14702 -> 11032` gap is **15.25 percentage points**.

Failure localization on `11032` shows the dominant miss occurs **before** downstream H selection. Contained carriers have median `3` visible views with a target-near (`<=4 px`) retained descriptor candidate and median target-to-top4 distance `2.62 px`; missed carriers have median `0` such views and median distance `11.00 px`. Thus the fixed final top-4 candidate truncation frequently prevents target-near geometry from entering `H_i` at all.

### Candidate-breadth causal counterfactual

A separate counterfactual was preregistered before breadth results (`85139f87da7d510d581535d73e066f16a325a1cd`). Frozen model/descriptor/evidence are unchanged.

**Retention-only arm:** same coarse top8 and the exact same refined pixel pool; retain final top16 instead of top4.

- `11032`: primary 2x containment **0.84746 -> 0.94915**, strict 1x **0.61017 -> 0.86441**;
- `15290`: primary 2x **0.90164 -> 1.00000**, strict 1x **0.83607 -> 0.98361**;
- `14702`: primary 2x remains **1.00000**, strict 1x **0.93651 -> 1.00000**.

A broader-search arm (`coarse top8 -> top32`, final top16) produces **no additional primary containment gain** on these witnesses. Therefore the required target-near evidence was already in the baseline refined descriptor pool and was being discarded by early top4 collapse.

**Current interpretation:**

- the **current fixed-top4 H representation contract is hard-tail insufficient**;
- the tested hard-tail does **not** demonstrate that raster-derived descriptor information is absent;
- the representation class survives if it remains more set-valued through the compiler boundary;
- the immediate architectural fix is to preserve a wider/uncertainty-aware bounded candidate set, not to add a free XYZ head or jump directly to a larger image model;
- a revised set-valued contract still requires a complete broad Test-C/A rerun before representation sufficiency can be supported.

## Post-Stage-B activity/amplitude factorization evidence

Prior to the sufficiency battery:

1. frozen V11 baseline was re-evaluated against the exact canonical evaluator with **12/12 episode parity, max diff 0.0**;
2. a direction-fixed exact-amplitude ceiling closes F activity and preserves all primary downstream quality gates;
3. a **truth-rank / frozen-magnitude-distribution / frozen-direction** arm preserves the complete V11 displacement-magnitude multiset to `2.78e-17` numerical error while moving F activity from `0.433160` to `1.000000`; F-kernel `0.838823`, D `0.068851`, R `0.043841`, G-direction `0.897785`, G-line `0.076640`, all principal primary quality gates PASS;
4. candidate-anchored current-amplitude composition on the three primary seed-route Stage-B episodes moves aggregate F activity `0.433160 -> 0.687180`, with F-kernel `0.70775`, D `0.10163`, R `0.04274`, G-direction `0.89947`, G-line `0.08024`: **6/6 principal quality metrics PASS**;
5. silence must be factored separately: raw-current p95 is approximately `1.384e-3` on primary `11214/e04` versus `3.068e-4` on near-zero `11214/e06` (~`4.51x` separation), while q95-normalized composition erases absolute scale and raises the near-zero predicted moved fraction `0.1719 -> 0.1875`.

This development evidence does **not** rewrite frozen Stage-B as PASS. The historical qualification also has a separate design defect: all three `12907` Stage-B episodes are truth-stratum `near_zero`, so its primary-only four-family G-coverage requirement was structurally unattainable.

## Current architecture target

Keep common-world dual-time geometry and factor motion state explicitly:

```text
P_A, P_B_geom / set-valued H_i,
N_A, N_B,
V_A, V_B,
Z,
p_active, log_amp, dir, U_pred
+ deterministic typed U_obs
```

with authority split:

```text
set-valued candidate/mechanical basin -> physical XYZ authority
p_active                             -> absolute motion / silence gate
log_amp                              -> conditional magnitude / ranking evidence
dir                                  -> direction evidence
U / match margin / multimodality     -> preserve ambiguity; control pruning/abstention
```

`p_active/log_amp/dir` are evidence, not independent XYZ authority. **Fixed early top4 collapse is forbidden by the current hard-tail evidence.** Final `P_B` remains constrained by multiview feasible geometry.

## Next required experiment

Before any large training:

1. define a bounded set-valued candidate contract using the proven top16-retention witness as a safe development reference, with uncertainty/margin-aware pruning rather than unconditional early top4 collapse;
2. rerun current-route Test C across the broader e00 family panel and difficulty strata under that frozen revised contract;
3. run representation-conditioned feasible-basin oracle Test A and carrier-level collision Test B with the revised set;
4. only if those are non-RED, train/evaluate the small strict family/episode-held-out `p_active + log_amp` head with ranking + silence supervision.

Any later qualification requires a **new untouched preregistered panel**. Stage-B cannot be recycled as blind qualification.

## Canonical post-Stage-B artifacts

- `POST_STAGEB_CURRENT_AMPLITUDE_DIAGNOSTIC.json`
- `REALSAS_IRIS_POST_STAGEB_TARGET_REPRESENTATION_LOSS_RESEARCH_V1_20260819.md.b64`
- `post_stageb_factorization_ceiling_v1.py`
- `POST_STAGEB_FACTORIZATION_CEILING_V1_RESULT_SUMMARY.json`
- `POST_STAGEB_FACTORIZATION_CEILING_V1.md`
- `POST_STAGEB_CANDIDATE_ANCHORED_MECHANICAL_BASIN_V1.md`
- `POST_STAGEB_CANDIDATE_ANCHORED_MECHANICAL_BASIN_V1_RESULT.json`
- `post_stageb_candidate_anchored_mechanical_basin_v1.py`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_PREREG.md`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_INTERIM_RESULT.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_INTERIM.md`
- `post_stageb_representation_sufficiency_battery_v1_interim.py`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_MAPPING_ADDENDUM.md`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_CURRENT_H_WITNESS.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_11032_FAILURE_LOCALIZATION.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_BREADTH_COUNTERFACTUAL_PREREG.md`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_BREADTH_COUNTERFACTUAL_RESULT.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_BREADTH_REPLICATION.json`

## Historical status preserved below

The following section is retained as provenance for the pre-V11 research frontier. Statements such as “Stage-B truth CLOSED” describe that historical point and are superseded by the current canonical frontier above.

---

**Historical date:** 2026-08-19  
**Historical status:** ACTIVE RESEARCH / NOT SOLVED YET  
**Historical Stage-B truth:** CLOSED

## Why this line exists

Historical source recovery is no longer the primary path. Code-level architecture audit and repeated local diagnostics converge on the same failure: the descriptor/multiview hypothesis set often contains a good physical endpoint, while per-carrier unary selection collapses the set incorrectly.

The new line therefore preserves set-valued 3D hypotheses and tests observation-only **global mechanical coherence** before committing `P_B`.

## Preserved historical evidence

- Historical descriptor-top4 hypothesis oracle: flow/zero ~0.4254, direction ~0.9283.
- Historical conclusion: per-carrier unary geometry selection is the wrong abstraction.
- Local 10763/e01 witness: candidate-set containment is strong and a global rank-1 mechanical selection can recover a strong G solution.
- Full 16-episode Stage-A V1 frozen qualification failed on F activity / G direction / G line; the later V11 route superseded this frontier.

## Historical persistence

Compact research bundle SHA-256: `833cfeb104c795323c4ded90b0608cb1c5ed3df2998cd82a44bd7aa28d01b5df`.  
V1 source SHA-256: `efa858c59489688f9937a207bf78f03cfe9614b24d12f22369d13142e4e897c1`.  
Qualification result SHA-256: `a9fcc2ad4ececae998e17a79f732d133f8a8a08c6220e440d93abb4262d09b33`.  
Freeze ledger SHA-256: `069ed3132d806ac10e809991ac1002bf03ef7e86d8ca0dc271f39f69e9922bd7`.

Library: `/RealSaS_OPT/N1D_20260819_GLOBAL_MECHANICAL_SOLVER_RESEARCH/`  
Drive: `RealSaS_IRIS_N1D_GLOBAL_MECHANICAL_SOLVER_RESEARCH_20260819/`
