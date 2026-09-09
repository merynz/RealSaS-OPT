# Arachne Mage A0 V7 — Replay Replicate Objective Autopsy

Date: 2026-09-09
Classification: `V7_REPLAY_REPLICATE__NOT_ORIGINAL_FINAL`
Replicate ID: `V7_REPLAY_REPLICATE_613cdba7f7c3`
Model SHA-256: `613cdba7f7c35a1839e1b1c819f8a54b0fead16eb6eec7024cf6e7465824d7fb`
Original V7 run fingerprint: `18ed583141112c65352d903a97452691cfac5f64383a34ad2a7f0c57ff4abcc6`

This replicate was resumed from the persisted V7 step-16 checkpoint. It is an independent numerical branch and does not replace the lost original step-192 model artifact.

## Replicate final product metrics

- normalized/raw row-L1 mean: `1.60390687`
- normalized/raw row-L1 p95: `1.83731955`
- deformation-error ratio: `0.90828615`
- continuous pairwise L2 mean: `3.70258284`
- pre-normalization joint pairwise L1 mean: `0.07245497`
- all 934 supervised rows have row-L1 > 1.0; 576/934 (`61.67%`) exceed 1.5.
- raw pre-normalization row mass mean: `4.71145126`; p95: `9.93279599`, while teacher simplex mass is 1.

The original V7 final branch had mean `1.46417796`, p95 `1.84107062`, deformation ratio `0.85944122`, and continuous pairwise L2 mean `11.70577`. Thus average behavior and latent separation are branch-sensitive, while the p95 wall is nearly branch-stable (`1.8411` original vs `1.8373` replay).

## Dominant-joint collapse

The row CSV shows that the replay predicts **joint index 8 as the dominant joint on all 950 rows**.

Among the 934 supervised rows, teacher joint 8 is dominant on exactly 434 rows. Therefore the apparent dominant-joint accuracy `0.4646680942` is exactly `434/934`, not evidence of broad correct ranking.

This corrects an initially misleading subgroup readout: pure-row dominant accuracy was `0.977477`, but 434 of the 444 pure rows are precisely the rows whose teacher dominant joint is joint 8. The model therefore did not learn a generally correct pure-row ranker; it learned a global default-dominant joint.

Blend rows have dominant-joint accuracy `0.0`.

## High-entropy near-uniform normalized predictions

- prediction entropy vs normalized row-L1: Pearson `0.92675`, Spearman `0.80565`.
- pure-row prediction entropy mean: `2.72277`.
- blend-row prediction entropy mean: `2.99116`.
- maximum entropy for 22 joints is `ln(22) ~= 3.09104`.

Thus blend predictions are close to maximally entropic/uniform, with joint 8 only mildly winning the rank everywhere. This is distinct from the old V3 exact `1/22` collapse but remains a severe cross-joint calibration/ranking failure.

## Sampling/calibration mismatch

V7 trains each joint independently using approximately 50% active-support samples and 50% global/supervised samples, without an importance correction that restores the natural row distribution before the 22 fields are compared.

For joint `j`, an active row has expected per-row sampling probability greater than an inactive row by approximately:

`1 + N_supervised / N_active_j`.

Observed active/inactive per-row sampling ratios span:

- minimum: `3.1521x`
- median: `28.9366x`
- maximum: `312.3333x`

Therefore different joints are optimized under different effective class priors. Their sigmoid outputs are not naturally calibrated onto a common cross-joint scale, yet inference treats them as comparable positive fields and normalizes them across all 22 joints.

Per-joint evidence is consistent with this failure. Joint 8 has 434 active supervised rows and an inactive raw false-positive mean of `0.90196`; its overall predicted raw mean is `0.83658` versus teacher mean `0.46467`. Most other joints maintain large inactive floors around `0.16–0.20` even when their teacher means are only around `0.003–0.03`. The resulting row mass is consequently far above 1 and the normalized row remains high-entropy.

## Gradient-objective diagnostic

At the replicate final logits:

- cosine(current expected sampled scalar objective, coupled mean row-L1) = `0.18794`
- cosine(uniform scalar objective, coupled mean row-L1) = `0.44668`
- cosine(current expected sampled scalar objective, coupled CVaR10 row-L1) = `0.34069`
- cosine(uniform scalar objective, current expected sampled scalar objective) = `0.68582`

Interpretation: the current sampled scalar gradient is **not generally anti-aligned**, but it is very weakly aligned with the authoritative mean-row objective. The mismatch is therefore better described as severe gradient inefficiency / wrong-axis expenditure rather than a simple reversed gradient.

A local logit-space descent probe with normalized RMS step size `epsilon=0.02` gives:

- current sampled scalar descent: mean row-L1 `-0.00204`, p95 `-0.00551`, CVaR10 `-0.00719`.
- uniform scalar descent: mean row-L1 `-0.00487`, p95 `-0.00863`, CVaR10 `-0.00809`.
- coupled mean-row descent: mean row-L1 `-0.01068`, p95 `-0.00539`, CVaR10 `-0.00487`.
- coupled CVaR10 descent: mean row-L1 `-0.00244`, p95 `-0.00948`, CVaR10 `-0.00931`.

For mean row-L1, the coupled mean objective is about `5.2x` more effective locally than the current sampled scalar objective at the same logit RMS step. Simply removing the active-heavy sampling bias while retaining an independent scalar objective also improves local mean progress by about `2.4x`.

## Causal interpretation

Strongly supported current diagnosis:

1. FSQ/discrete transport collapse is no longer the live V7 blocker.
2. The forced-field decoder remains field-sensitive.
3. V7 optimization is branch-sensitive/numerically unstable in average behavior.
4. Separately, the approximately `1.84` p95 wall survives distinct numerical branches.
5. The replay exposes a cross-joint rank/calibration collapse: one joint dominates every row while normalized predictions remain near-uniform/high-entropy.
6. The 50/50 active-heavy independent scalar training distribution induces joint-dependent prior shifts and is mathematically misaligned with the final normalized 22-joint product objective.
7. The objective/sampling mismatch is now strongly supported as the next causal target, but should be sealed by an intervention rather than declared fully proven from one replay autopsy.

## Next experimental gate

Do **not** add more characters and do **not** reintroduce FSQ yet.

The next experiment should preserve the V7 continuous forced-field architecture and isolate the training objective/sampling contract. Minimum causal ablations should compare from a common bound starting state:

- current active-heavy independent scalar objective;
- unbiased/uniform scalar objective;
- full-row coupled normalized-row objective (with a separate mean-row and tail/CVaR diagnostic if needed).

A strong improvement under uniform/coupled training with the architecture otherwise frozen would seal the objective/sampling mismatch and avoid inventing a bespoke FIT1 optimizer or abandoning single-character FIT1.
