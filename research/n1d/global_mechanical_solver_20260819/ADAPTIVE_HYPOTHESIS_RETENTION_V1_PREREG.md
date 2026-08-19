# RealSaS N1D — Adaptive Hypothesis Retention V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_DEVELOPMENT_TEST`  
**Representation contract:** `REPRESENTATION_CONTRACT_V2.md`  
**Qualification:** unchanged; this is not a blind qualification.

## Question

Can observation-only uncertainty preserve the hard-tail coverage of the demonstrated fixed-top16 reference while carrying materially fewer hypotheses on easy/clear carriers?

The test must distinguish three possibilities:

1. fixed `K=8` is already sufficient;
2. descriptor uncertainty alone can choose `K` safely;
3. descriptor uncertainty needs a second-stage multiview geometric expansion trigger.

## Frozen panel and lineage

Open-development e00 panel:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Use the same frozen checkpoint, Problem-A carrier mapping, descriptor representation, masks, refined search geometry and evaluator mapping used by Representation Sufficiency Battery Test C. No model weight update is allowed.

Truth is used only to compute final containment metrics after each arm has committed its candidate sets.

## Reference facts known before this prereg

- fixed top4: pooled primary-2x `0.93902`, worst family `0.84746`, 6/8 families >=0.90 — RED;
- retention-only top16: pooled `0.98171`, worst `0.94915`, 8/8 >=0.90 — GREEN;
- coarse top32 + final16 adds only a small secondary gain, localized mainly to family 14758.

These known facts motivate but do not tune the adaptive policy below.

## Observable uncertainty quantities

For every usable view/carrier, compute the same frozen refined descriptor scores that produce the top16 reference set. Let descending scores be `s1...s16`.

Descriptor quantities:

```text
margin4 = s4 - s5
margin8 = s8 - s9
entropy16 = normalized entropy(softmax(s1..s16 / T)), T=0.05
```

All percentile thresholds below are estimated **without truth** from the other seven families in a strict leave-one-family-out fold.

Geometry quantities after building the current tier's `H_i`:

```text
solver_mean_reprojection_px
usable_view_count
pair_hypothesis_count
H_empty
```

Their percentile thresholds are likewise estimated without truth from the other seven families.

## Arms

### F4 — historical compact baseline

`coarse_k=8`, per-view `final_k=4`.

### F8 — fixed compact candidate

`coarse_k=8`, per-view `final_k=8`.

### F16 — demonstrated reference ceiling

`coarse_k=8`, per-view `final_k=16`.

### A1 — descriptor-only adaptive retention

For each usable view, form an uncertainty percentile

```text
u = max(
  percentile_low_confidence(margin4),
  percentile_high_uncertainty(entropy16)
)
```

where percentiles are defined against the seven-family observation-only calibration pool.

Tier:

```text
u < 0.50       -> K=4
0.50 <= u < .80 -> K=8
u >= 0.80      -> K=16
```

No geometric post-expansion.

### A2 — descriptor adaptive + bounded geometric escalation

Start with A1 per-view tiers. Build `H_i` and the frozen reprojection selector diagnostics.

Escalate every usable view by one tier (`4->8`, `8->16`) and rebuild if **any** holds:

```text
solver_mean_reprojection_px > train-fold q75
usable_view_count < 3
pair_hypothesis_count < train-fold q25
H_empty
```

Repeat at most once more to a maximum `K=16`.

If already at `K=16` and either `H_empty` or `solver_mean_reprojection_px > train-fold q95`, widen only the coarse descriptor beam `8->32`, keep final `K=16`, and rebuild once.

No truth-conditioned escalation is permitted.

## Primary coverage gates

A candidate contract is hard-tail acceptable only if all hold on reliable mapped carriers:

```text
pooled primary-2x containment >= 0.975
worst-family primary-2x containment >= 0.94
families with primary-2x >=0.90 = 8/8
best-worst primary-2x gap <= 7 percentage points
```

Strict-1x containment is reported but is secondary.

## Efficiency gates

A bounded adaptive contract must also satisfy:

```text
mean retained K per usable view <= 10.0
median retained K per usable view <= 8
fraction usable views at K=16 <= 0.35
fraction carriers invoking coarse32 <= 0.10
```

Report mean/median H hypothesis count and ratio to F16, but do not use H-count ratio as a hard gate because pair-count depends strongly on visibility pattern.

## Decision rule

1. If F8 passes all primary coverage gates and efficiency gates, prefer F8 over adaptive complexity unless an adaptive arm has materially lower mean K (>=1.0 candidate/view lower) with no coverage loss >0.5 percentage points pooled or >1 point worst-family.
2. Otherwise, if A1 passes all primary + efficiency gates, freeze A1.
3. Otherwise, if A2 passes all primary + efficiency gates, freeze A2.
4. If only F16 passes coverage, adaptive retention is **not solved**; keep top16 as development reference and diagnose the failed uncertainty trigger before factor-head training.
5. No result here authorizes large/end-to-end training.

## Non-negotiable interpretation

A failure of adaptive compression is not evidence that the paired raster representation is insufficient. It means the current observation-only uncertainty policy cannot yet compress the already-demonstrated sufficient set safely.

A pass authorizes freezing the bounded `H_i` retention contract and proceeding to the separately scoped small `p_active + log_amp` head experiment.
