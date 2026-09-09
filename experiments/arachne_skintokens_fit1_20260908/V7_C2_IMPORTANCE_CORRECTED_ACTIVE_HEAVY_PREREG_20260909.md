# Arachne Mage A0 FIT1 — V7-C2 Importance-Corrected Active-Heavy Warm-Start Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent C1 result commit: `0110935ea092d761ea58cff10790b25347c9d904`
Parent C1B result commit: `399731339c2aeaca13bd51e3293edfbfbf073a9b`

## Question
After the 2000-step V7 C0 run has learned near-correct geometry/ownership but leaves rare high inactive-joint leakage, is that leakage caused materially by the known uncorrected 50% active-support + 50% global sampling prior?

C1 showed that exact teacher-support projection reduces p95 `0.3240 -> 0.07916` and deformation `0.10529 -> 0.02997`. C1B showed that fixed teacher-free post-hoc sparse projections improve but cannot recover the correct support-cardinality tradeoff. The next causal variable is therefore the sampling prior during training, not architecture or output remapping.

## Exact warm-start binding
Both arms start independently from the same completed V7 C0 2000-step FP32 model:
- model SHA-256: `ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9`
- architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
- config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`
- parameter count: `278,010,880`
- cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`
- Mage A0 only; 934 supervised rows, 22 joints.

Each arm uses a fresh AdamW optimizer state. The old 2000-step optimizer/scheduler is not resumed because its cosine LR has reached zero.

## Common continuation recipe
- optimizer steps: `384`
- AdamW: initial LR `2.5e-5`, weight decay `1e-4`
- cosine scheduler: `T_max=384`, eta_min=0
- rationale for LR: approximately the original long-horizon LR near step 1536 (`~2.54e-5`), giving a gentle continuation rather than resetting to the original `2e-4` peak
- every optimizer step remains 22-joint balanced
- same V7 BF16-autocast / FP32-master math-SDPA-only, TF32-off policy
- no FSQ / no quantizer
- same nested field-token prefix mechanism
- same frozen sample-index and prefix schedule in both arms
- 384 queries per joint: exactly 192 global-supervised + 192 active-support proposal draws, with the same sampling algorithm in both arms
- inference/product metric remains the existing `sigmoid(logit) -> normalize across 22 joints`; C1/C1B sparse remaps are diagnostic only and are not used for closure.

Required observations include step 0, 1, 4, 16, 32, 64, 96, 128, 192, 256, 320 and 384.

## Arms
### C2-CONTROL — biased active-heavy continuation
Exact existing sampled scalar objective:
- unweighted sampled BCEWithLogits
- `0.1 *` unweighted sampled probability-space MSE
- unweighted sampled Dice

This is the no-correction control from the same warm-start state/schedule.

### C2-TREATMENT — importance-corrected separable terms
The proposal distribution for joint `j` is

`q_j(i) = 0.5 / N_sup + 0.5 / N_active_j` for active supervised row `i`,

`q_j(i) = 0.5 / N_sup` for inactive supervised row `i`.

The target scalar-row distribution is uniform supervised:

`u(i) = 1 / N_sup`.

For every sampled scalar term use the exact Horvitz-Thompson-style importance multiplier

`w_j(i) = u(i) / q_j(i)`.

Treatment objective:
- `mean(w * BCEWithLogits_element)`
- `0.1 * mean(w * MSE_element)`
- **the exact same unweighted sampled Dice term as control**.

No weight clipping, self-normalization, ad-hoc rescaling or learned weighting is permitted. Expected `E_q[w] = 1` must be preflighted for every joint.

Dice is deliberately held common between arms because the sampled Dice ratio is nonlinear; silently changing it would bundle a second objective treatment. C2 therefore tests whether correcting the known sampling prior in the separable BCE/MSE pressure is already causally useful.

## Required telemetry
At every observation, for each arm:
- authoritative row-L1 mean, p95, CVaR10
- deformation-error ratio
- dominant accuracy, top-3 inclusion, mean rank
- pairwise spatial variation ratio and joint-std ratio
- raw mass and prediction entropy
- inactive predicted mass: mean, p95, and mean within the current p95 tail
- pure / 2-joint / 3+-joint row mean and p95
- active vs inactive sigmoid calibration summaries
- control/treatment scalar-loss components
- gradient-route finite/nonzero checks at first optimizer step

## Decision logic
1. **Importance-prior causality supported** if treatment materially lowers inactive leakage and improves p95/deformation relative to the matched control while ownership/top3/variation remain intact.
2. If treatment approaches the C1 teacher-support-oracle regime (deformation near/below `0.05`, p95 trending toward `~0.08`) but p95 remains above `0.05`, the next separate treatment may target within-support blend ratios.
3. If treatment does not outperform the control on leakage/tail metrics, do not add a blend objective yet without diagnosing why the exact importance correction failed to change the expected failure mode.
4. Closure still requires the unchanged product gates: row-L1 p95 `<=0.05`, deformation-error ratio `<=0.05`, compiler/simplex gates and three stable checks.

## Forbidden inside C2
- architecture changes
- token-count/width/depth changes
- FSQ reintroduction
- extra characters or A1 data
- softmax/sparsemax/simplex-projection inference substitution
- top-k or teacher-support masking
- coupled row/simplex auxiliary
- blend-ratio loss
- importance-weight clipping or self-normalization
- different sample/prefix schedules between arms

Scientific FAIL remains a valid result. No A1/generalization authorization is implied.