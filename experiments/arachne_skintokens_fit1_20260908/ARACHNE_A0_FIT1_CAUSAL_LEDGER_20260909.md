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

## V7 preregistered next test — continuous field transport

Planned architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
Required invariants:
- no FSQ module or quantizer parameters in the model graph;
- `encode_field()` outputs continuous 4x512 tokens directly to the forced-field decoder;
- joint-balanced 22-field optimizer update retained;
- sparse 50/50 decoder query sampling retained;
- BCEWithLogits + 0.1*MSE + Dice retained;
- nested prefix dropout retained;
- acceptance gates unchanged;
- telemetry must include continuous pairwise L2, pre-normalization joint pairwise L1/std, logit saturation, and zero-field ablation.

Expected causal readout:
- If continuous pairwise separation and pre-normalization joint diversity remain alive while row-L1/deformation improve, FSQ was the blocking transport bottleneck.
- If continuous representations still collapse before useful fit, the next target is encoder/training dynamics rather than decoder or quantization.
- Do not reintroduce FSQ into FIT1 merely because the continuous codec passes. Quantization becomes a separate post-fit gate.

## Current causal chain (short form)

`V3 uniform collapse`
-> sparse-loss hypothesis tested by V4: insufficient
-> per-joint optimizer interference tested by V4JB: representation diversity improved but output still uniform
-> clamp/saturation tested by V5: real numerical bug fixed, but decoder still ignored field tokens
-> forced field-conditioning tested by V6: decoder causality restored
-> V6 immediately exposed FSQ as the remaining transport bottleneck
-> V7 removes FSQ to test continuous FIT1 closure.
