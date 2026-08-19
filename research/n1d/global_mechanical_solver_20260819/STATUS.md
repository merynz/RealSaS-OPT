# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `POST_STAGE_B_REPRESENTATION_SUFFICIENCY_BATTERY__CURRENT_TOP4_RED__REVISED_SET_VALUED_TESTC_GREEN_8OF8__TEST_A_B_OPEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Immutable qualification state

- Stage A: **PASS**, frozen standalone parity 16/16.
- Stage B: **FAIL**, immutable/no-retune. Primary F-activity `0.433160 < 0.50`; the other principal geometry/mechanics metrics remained passing.
- Stage-B truth is now **OPEN DEVELOPMENT EVIDENCE ONLY** and cannot be recycled as blind qualification.
- sealed21 / external10 remain **CLOSED**.
- Any future blind qualification requires a **new untouched preregistered panel**.
- **Large/new architecture training is NOT AUTHORIZED** while representation sufficiency Tests A/B remain open.

## Why the representation sufficiency battery exists

Historical experiments contained both near-threshold misses and genuinely low family/tail metrics. The battery asks whether the proposed raster-derived representation genuinely lacks information on some tails, or whether the information exists but is lost by premature candidate collapse, feature fusion or calibration.

Parent prereg: `b87e222fdf7bf4c66a62314c88ee9e155a576c73`.

Proposed factorized contract:

```text
P_A
P_B_geom / set-valued candidate-mechanical basin H_i
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred + typed U_obs
```

Authority split:

```text
set-valued H_i / geometry          -> physical XYZ authority
p_active                           -> absolute motion / silence
log_amp                            -> conditional magnitude / ranking
dir                                -> direction
U / margin / multimodality/support -> pruning, ambiguity, abstention
```

`p_active/log_amp/dir` are evidence, not free XYZ authority.

## Test D — broad family-disjoint learnability: PARTIAL PASS / fusion warning

Canonical open-development population: `232` episodes, `29` families, source `N1D_CANONICAL_DEV_PER_EPISODE.json`, SHA-256 `37d81db0733f6fbf433a944aa0797550f3beee15ff65377ecbcfada1064f4326`.

Strict 29-fold LOFO episode-summary proxies:

- p_active AUROC: **0.97411**
- p_active balanced accuracy at fixed 0.5: **0.88937**
- multifeature active log-amp Spearman: **0.84524**
- naive multifeature Ridge collapses held-out family `12832` to **-0.10714** Spearman
- but the single frozen raster-derived `predicted_flow_mean` score has `12832` Spearman **+0.78571**
- raw `predicted_flow_mean` amplitude ranking across all evaluable families: aggregate **0.87594**, median family **0.85714**, minimum family **0.50**, **29/29 >= 0.50**, no negative family.

Interpretation: broad family-disjoint amplitude ordering information exists. Naive cross-family feature fusion/calibration can destroy a hard tail. This supports explicit ranking supervision and calibrated factorization rather than unrestricted feature mixing.

## Test B — collision audit: AMBER / carrier-level test still open

Coarse inference-safe episode-summary nearest-neighbor proxy:

- closest 5% cross-family pairs: active/silent mismatch **7.69%**
- closest 10%: mismatch **16.67%**

This is not an irreducible representation collision verdict because the proxy omits the proposed explicit carrier-level `p_active`, wider H, feasible-set width, match margin, multimodality and other typed-U fields. Carrier-level revised-contract collision audit remains required.

## Test C — hard-tail feasible-set containment

Mapping/local-scale addendum was preregistered before current H results: `5dc6989b2026a2db7194130242243b0ab6902b09`.

Current global-foreground route uses:

```text
coarse foreground stride 4
coarse descriptor top8
±4 px / step2 refinement
final retained candidates top4 per visible view
pairwise rank3 H_i
mean-reprojection selection
all-visible nearest-candidate LS refit
```

### Current fixed-top4 contract — RED

Broad e00 panel: `9908, 11032, 12772, 13203, 14404, 14702, 14758, 15290`.

Across `492/512` reliably mapped carriers:

- pooled primary 2x H-containment: **462/492 = 0.93902**
- worst family `11032`: **0.84746**
- `13203`: **0.86885**
- families >=0.90: **6/8**
- best–worst gap: **15.25 percentage points**
- pooled strict 1x containment: **0.82724**

This triggers the preregistered Test-C RED rule. **Fixed early top4 collapse is forbidden by the current hard-tail evidence.**

### Failure localization

On `11032`, the dominant misses occur before downstream H selection:

- contained carriers: median `3` visible views have a target-near retained candidate within 4 px; median target-to-top4 nearest distance `2.62 px`;
- missed carriers: median `0` such views; median nearest distance `11.00 px`.

Thus target-near descriptor evidence is often discarded before it can enter H.

## Candidate-breadth counterfactual — revised set-valued Test C GREEN on 8/8

Breadth arms were preregistered before results: `85139f87da7d510d581535d73e066f16a325a1cd`.

Retention-only arm changes **only** final retention `top4 -> top16`. Coarse top8, refined pixel pool, model, descriptor and rasters stay frozen.

Per-family primary 2x H-containment:

| family | current top4 | same-pool top16 |
|---|---:|---:|
| 9908 | 0.98413 | **1.00000** |
| 11032 | 0.84746 | **0.94915** |
| 12772 | 1.00000 | **1.00000** |
| 13203 | 0.86885 | **0.95082** |
| 14404 | 0.92063 | **0.96825** |
| 14702 | 1.00000 | **1.00000** |
| 14758 | 0.98387 | **0.98387** |
| 15290 | 0.90164 | **1.00000** |

Broad same-pool top16:

- pooled primary 2x: **483/492 = 0.98171**
- worst family: **0.94915**
- families >=0.90: **8/8**
- best–worst gap: **5.08 pp**
- pooled strict 1x: **0.94919**

Under the preregistered hard-tail rule, the **revised set-valued top16 Test C is GREEN on this 8-family e00 development panel**.

A broader coarse-search arm (`coarse top8 -> top32`, final top16) raises pooled primary only to **0.98374** and changes primary coverage only on `14758` (`0.98387 -> 1.00000`). Therefore the dominant causal failure is premature final candidate truncation; occasional coarse-beam narrowness is secondary and can be handled with uncertainty-triggered bounded expansion.

## Representation conclusion after Test C

The user's hard-tail concern was correct in a precise form:

- **current compact top4 representation is genuinely insufficient on some families**;
- the tested failures do **not** show that the paired rasters/frozen descriptor lack the required information;
- the same frozen evidence recovers both RED hard families simply by preserving more already-present hypotheses;
- therefore the **representation class survives only as a more set-valued, uncertainty-aware contract**;
- compiler/global mechanical reasoning should collapse H after consistency/deformation reasoning, not at an early fixed descriptor top-k boundary.

This is not permission to adopt fixed top16 as product architecture blindly. Top16 is a causal development witness/safe reference. The product contract should use bounded uncertainty/margin-aware retention and adaptive expansion.

## Prior Stage-B factorization evidence still stands

- direction-fixed truth-rank/frozen-magnitude ceiling moves F activity `0.433160 -> 1.000000` while preserving all six principal quality metrics;
- candidate-anchored current-amplitude composition moves primary Stage-B F activity `0.433160 -> 0.687180` with 6/6 principal metrics PASS;
- absolute silence must remain separate from normalized amplitude (`p_active` distinct from `log_amp`).

These are truth-open development findings and do not rewrite frozen Stage-B qualification.

## Overall battery decision now

```text
Test A  representation-conditioned revised-basin oracle : OPEN / AMBER
Test B  carrier-level collision with revised H + factors : OPEN / AMBER
Test C  current top4 H                              : RED
Test C' revised same-pool top16 H, broad 8-family  : GREEN
Test D  broad family-disjoint learnability         : PARTIAL PASS / fusion-calibration warning
```

**Overall:** `REPRESENTATION_SUFFICIENCY_NOT_YET_SUPPORTED__SET_VALUED_REPRESENTATION_SURVIVES__BIG_TRAINING_NOT_AUTHORIZED`.

## Next authorized work

1. Freeze a bounded set-valued development contract using top16 coverage as the reference and uncertainty/margin-aware pruning rather than unconditional top16 everywhere.
2. Run Test A: representation-conditioned feasible-basin oracle under the revised set, without free XYZ truth injection.
3. Run Test B: carrier-level cross-family collision audit including explicit activity/amplitude/direction and typed uncertainty fields.
4. Only if A/B are non-RED, train the small strict family/episode-held-out `p_active + log_amp` head with ranking + silence supervision.
5. Any later blind qualification uses a new untouched preregistered panel.

## Canonical battery artifacts

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
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_BROAD_8FAMILY_RESULT.json`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_TESTC_BROAD_8FAMILY_REPORT.md`
- `post_stageb_representation_sufficiency_testc_current_h_probe.py`
- `post_stageb_representation_sufficiency_testc_breadth_fast.py`

Historical details remain available in Git history and the preceding canonical reports; this status file intentionally points to the current scientific frontier rather than duplicating the complete historical ledger.
