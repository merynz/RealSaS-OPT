# Arachne Mage A0 FIT1 — Causal Experiment Ledger

Date: 2026-09-09
Scope: Mage A0 FIT1 skin-field codec only. This ledger records why each architecture/training change was made so later work does not rediscover already-falsified explanations.

## Fixed scientific target

- Dataset/cache: `ARACHNE_MAGE_FS1_CONDITIONING_CACHE_V2.npz`
- Cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- Teacher weights SHA-256: `7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d`
- 22 joints; 934 supervised rows; 16 low-confidence rows.
- Acceptance gates are unchanged unless explicitly preregistered in a later experiment: qualified/raw row-L1 p95 <= 0.05, deformation-error ratio <= 0.05, simplex/compiler repair gates, three stable checks.
- Scientific FAIL is a valid result and must not be converted into an infrastructure exception.

## V3 — SkinTokens-strength baseline

Source commit: `786c5d5a226c9d2bd9cf1d28a7697c711fd77d01`
Architecture: `RealSaS.SkinFieldCodec.SkinTokensStrength.v3`
Training: one joint field per optimizer step; BCE + L1; FSQ `(8,8,8,5,5,5)`; four field tokens; shared 384-token geometry condition; decoder normalized 22 independent scalar fields across joints.

Observed full 4096-step result:
- `raw_row_l1_p95_auth = 1.9090909361839294`
- `qualified_row_l1_p95_auth = 1.9090909090909087`
- `raw_deformation_error_ratio_auth ~= 0.97188836`
- no terminal closure.
- FSQ collapsed almost entirely to one code after early training.

Interpretation:
`42/22 = 1.909090909...` is the exact row-L1 signature of one-hot teacher weights versus a uniform `1/22` prediction. Therefore the normalized output had collapsed to uniform weights. Gradient-route preflight was finite/nonzero, so this was not a disconnected optimizer/backprop bug.

Decision: investigate sparse reconstruction pressure and FSQ/representation collapse; do not extend V3 training.

## V4 — sparse-aware reconstruction

Source commit: `56bf6676901fbc3cab35701cde4e98f11bb953a1`
Architecture: `RealSaS.Arachne.SkinFieldCodec.v4`
Changes versus V3:
- decoder importance sampling: 384 points, 50% active/dense support + 50% supervised/global;
- objective: BCE + `0.1*MSE` + Dice, Dice epsilon `1e-4`;
- nested field-token prefix dropout 1..4;
- explicit continuous/pre-FSQ and quantized/post-FSQ identity telemetry.

Observed full 4096-step result:
- final output remained the exact uniform signature (`1.909090936...`).
- continuous pairwise latent separation contracted strongly during training.
- FSQ reached one shared sequence and stayed collapsed.

Interpretation:
sparse-aware loss alone did not prevent collapse. The failure was not explained solely by zero-dominated sampling.

Decision: test whether per-joint optimizer updates were causing shared-model catastrophic interference.

## V4JB — joint-balanced optimizer contract

Contract commit: `474f8da26e4f486dbb210371753298da95add01f`
Architecture remains V4.
Training change:
- every optimizer update sees all 22 joints;
- each joint contributes `loss/22`, gradients accumulate sequentially, then one AdamW step;
- 192 optimizer updates x 22 joints = 4224 field reconstructions, approximately preserving the previous field-exposure budget.

Observed full 192-update result:
- final FSQ identity recovered substantially: 10 unique joint sequences / 9 unique token IDs;
- continuous pairwise L2 mean ~= 4.23;
- quantized pairwise L2 mean ~= 7.15;
- nevertheless output stayed exactly uniform: `raw_row_l1_p95_auth = 1.9090909361839294`;
- training BCE rose to ~10.4 with max losses ~13.4.

Interpretation:
joint-balanced training repaired a major part of representation collapse, but different quantized latents still produced effectively identical normalized skin predictions. This falsified the claim that FSQ collapse alone explained the frozen metric.

Decision: inspect decoder/loss output semantics directly.

## V5 — logits-first loss

Source commit: `63ef23c2607aa12d455df9886a6908e704f7957c`
Architecture: `RealSaS.Arachne.SkinFieldCodec.v5`
Changes:
- decoder returns logits;
- BCE is `binary_cross_entropy_with_logits` in FP32;
- MSE/Dice use `sigmoid(logits)`;
- probability clamp removed;
- pre-normalization decoder telemetry added.

Reason:
V4 loss used `pred.clamp(1e-6, 1-1e-6)` before BCE. At saturation, clamp can zero the gradient. A dedicated regression confirmed V5 preserves gradients for deliberately wrong `+20/-20` logits (`+0.5/-0.5`).

Observed early run (stopped after decisive signature; no need to complete 192):
- step 1: quantized pairwise L2 ~= 4.11 with 3 unique FSQ sequences, but pre-normalization joint pairwise probability L1 only ~= `1.63e-6`;
- step 4: 4 unique sequences / quantized L2 ~= 3.22, but pre-normalization joint pairwise L1 ~= `4.05e-8`;
- saturation fraction `abs(logit)>12` was 0.0 at these checks;
- by step 12 FSQ had collapsed to one sequence and remained collapsed through the observed run.

Interpretation:
the clamp bug was real but not the principal cause of uniform output. The decoder was effectively ignoring joint-specific field tokens even while those tokens were measurably different.

Decision: make joint-specific field memory structurally mandatory in the decoder.

## V6 — forced-field decoder

Source branch tip after implementation: `e8f229226caadcc3b63d42ebb3687889515c99f4`
Architecture: `RealSaS.Arachne.SkinFieldCodec.v6`
Changes:
- shared geometry/condition path first builds query features;
- scalar logit then requires a separate field-only cross-attention and bilinear readout;
- no field-independent scalar-logit bypass;
- field input projection is bias-free and field LayerNorms are non-affine;
- zero field tokens imply exactly zero logits by construction/preflight;
- logits-first loss and joint-balanced training retained.

Observed early run (stopped after step 32 because signature was decisive):
- step 1: pre-update 12 unique FSQ sequences; after update 3 unique sequences;
- step 1 pre-normalization joint pairwise L1 = `0.0957143` and joint std ~= `0.1076` — many orders of magnitude above V5. Therefore the forced-field decoder fix worked: different field tokens now produce different fields.
- step 1 normalized result moved away from the old exact signature (`raw_row_l1_p95_auth ~= 1.93729`, deformation ratio ~= 0.93595), also proving the decoder output was no longer hard-wired to uniform.
- by step 4 FSQ had collapsed to one sequence; quantized pairwise L2 = 0 while continuous pairwise L2 was still ~= 3.97.
- after FSQ collapse, pre-normalization joint diversity returned to exactly zero and the old `1.909090936...` uniform signature returned.
- continuous pairwise L2 subsequently contracted from ~9.77 at step 1 toward ~1.02 by step 32, interpreted as secondary collapse after quantization removed any benefit to preserving identity.

Interpretation / causal localization:
1. Encoder can produce distinct per-joint continuous fields.
2. Forced-field decoder can and does use distinct field tokens.
3. FSQ maps these distinct continuous fields onto the same discrete code extremely early.
4. Once all joints receive the same transported token, decoder outputs become identical; joint normalization yields `1/22`; gradients no longer reward continuous identity separation, causing secondary encoder contraction.

Decision: **remove FSQ from the FIT1 learning path**. Do not treat this as a cosmetic bypass. For V7, FSQ is absent from the model graph and the decoder receives the 4x512 continuous field tokens directly. Compression/quantization is deferred to a later post-fit quantization/distillation stage after the continuous codec demonstrates closure.

Practical note: 22 joints x 4 tokens x 512 scalars = 45,056 scalars. At FP16 this is ~88 KiB per character, negligible for the current demo. Therefore FIT1 scientific closure has higher priority than early token compression.

## V7 — continuous field transport

Source commit: `b86e3c7c0f54053fff5fecf81b6d81e50562d977`
Architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
Training contract:
- no FSQ module or quantizer parameters in the model graph;
- `encode_field()` outputs continuous 4x512 tokens directly to the forced-field decoder;
- 22-joint balanced optimizer update;
- 384 decoder queries per field, approximately 50% active/dense + 50% global/supervised;
- BCEWithLogits + `0.1*MSE` + Dice;
- nested prefix dropout 1..4 retained;
- 192 optimizer updates / 4224 field reconstructions; cosine schedule reaches LR=0 at the final update.

Observed full result:
- status: `NO_A0_V7CFFJB_TERMINAL_CLOSURE`; final step 192; streak 0; A1 unauthorized.
- final raw/qualified row-L1 p95 ~= `1.84107062`.
- final row-L1 mean ~= `1.46417796`.
- final deformation-error ratio ~= `0.85944122`.
- best observed p95 was ~= `1.82779122` at step 88, when LR ~= `1.1305e-4`; p95 then plateaued around 1.84 rather than converging to the acceptance region.
- row-L1 mean improved from ~= `1.83945` at step 1 to ~= `1.46418` at step 192 (~20.4% relative improvement).
- deformation ratio improved from ~= `0.97168` to ~= `0.85944` (~11.6% relative improvement), with best observed ~= `0.85647` around step 144.
- continuous identity survived: final continuous pairwise L2 mean ~= `11.7058`, min ~= `0.2315`, near-equal pair fraction at 1e-6 = 0.
- decoder field sensitivity survived: final pre-normalization joint pairwise L1 ~= `0.1039`; zero-field logits remain exactly zero; saturation fraction `|logit|>12` is 0 at final.
- final pre-normalization probability mean ~= `0.20599`. Across 22 joints this corresponds to average raw per-row mass ~= `4.53` before simplex normalization, while the teacher row mass is 1 by definition.
- final sampled scalar-field training loss appears numerically moderate (`mean_loss ~= 0.789`, sampled `mean_l1 ~= 0.229`) even though authoritative normalized row-L1 remains extremely poor (`mean ~= 1.464`, p95 ~= 1.841`).

Critical correction to the first interpretation:
This is **not** accurately described as "only a hard 5% tail remains." Row-L1 is bounded by 2 for simplex distributions. With mean row-L1 ~= 1.464, the error must still be widespread: even under the most favorable possible distribution, at least ~46% of rows must exceed L1=1.0 and at least ~64% must exceed L1=0.5. Therefore V7 broke the exact-uniform collapse but did not approach FIT1 closure globally.

What V7 proves:
1. Removing FSQ fixes the discrete transport collapse: continuous joint identities remain distinct through the full run.
2. The forced-field decoder remains causally sensitive to field tokens.
3. The old exact `1/22` attractor is no longer the only reachable output.
4. Representation collapse is no longer the primary blocker.

What V7 does **not** prove:
- It does not show that the remaining problem is a small hard-row tail.
- It does not justify simply extending the same run: the scheduler already drives LR to zero at step 192.
- It does not show that larger support joints are the main failure. Late per-joint sampled loss has essentially no correlation with active support size in the bound Mage cache.
- It does not establish whether the remaining global underfit is mainly architecture capacity, optimization recipe, or objective/inference mismatch.

Strongest remaining hypothesis after V7:
**scalar-field training objective / sampling distribution is misaligned with the authoritative normalized 22-joint matrix objective.** The decoder trains each joint independently on an active-heavy sampled query distribution, then inference normalizes all 22 positive scalar fields across joints. Final raw probability mass (~4.53 per row on average) is far from the teacher simplex mass 1. This gives the optimizer a route to reduce independent BCE/MSE/Dice while leaving relative cross-joint ratios—and therefore authoritative row-L1/deformation—bad.

This is a hypothesis, not yet a sealed root cause. A targeted diagnostic should compare:
- raw per-row field-sum distribution before normalization;
- normalized row-L1 versus raw scalar reconstruction by row;
- pure vs blend rows;
- dominant-joint accuracy;
- hard-row entropy / active-joint count;
- counterfactual full-row coupled training objective versus the existing independent sampled scalar objective.

Historical sanity check:
The earlier non-SkinTokens Arachne A0 baseline reached row-L1 p95 around `0.1147` with deformation ratio around `0.04871`. Therefore the Mage target is demonstrably much more fit-able than V7's `1.84 / 0.859` result. V7 should not be treated as "nearly solved but tail-limited"; the current SkinTokens-inspired FIT1 recipe is still dramatically worse than the earlier baseline on the same product gates.

Decision: **do not start V8 yet.** First run a row-level diagnostic on the V7 final checkpoint/result to distinguish broad objective/sampling misalignment from decoder expressivity/optimization limits. Do not reintroduce FSQ during this diagnostic.

## Current causal chain (short form)

`V3 uniform collapse`
-> sparse-loss hypothesis tested by V4: insufficient
-> per-joint optimizer interference tested by V4JB: representation diversity improved but output still uniform
-> clamp/saturation tested by V5: real numerical bug fixed, but decoder still ignored field tokens
-> forced field-conditioning tested by V6: decoder causality restored
-> V6 immediately exposed FSQ as the remaining transport bottleneck
-> V7 removed FSQ and eliminated representation collapse / exact-uniform lock
-> V7 nevertheless remained globally far from FIT1 closure; current leading target is objective/sampling vs normalized-row acceptance mismatch, not a narrow hard-tail-only failure.
