# Arachne Mage A0 FIT1 — V7 Objective Causal Triplet Result

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Source/contract commit: `f2a573022f718cd390904cc4c6bfe427436bce11`
Architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
FIT character count: 1 (Mage A0)
FSQ: absent
Shared initial FP32 model SHA-256: `bfdc24213edd0e6e881794b8d9ac8bf91266fa212276f573f7f9e6c4d9f85197`
Shared compute policy: math SDPA only, flash/mem-efficient/cuDNN SDPA off, TF32 off.

## Purpose

Separate two hypotheses after the V7 replay autopsy:

1. uncorrected 50% active-heavy sampling is the main cause of product-level failure;
2. independent scalar-field training is misaligned with the authoritative normalized 22-joint row objective.

Three arms used identical serialized initial weights, identical V7 architecture, AdamW/cosine schedule, 192 optimizer steps, and identical frozen nested-prefix schedule.

## Arm C0 — biased scalar control

Contract: V7 active-heavy sampled scalar objective: 384 queries/joint, ~50% active + ~50% global, BCEWithLogits + 0.1*MSE + Dice.

Final:
- status: `NO_A0_CAUSAL_ARM_TERMINAL_CLOSURE`
- row-L1 mean: `1.5863117757138767`
- row-L1 p95: `1.8367122726980596`
- deformation ratio: `0.897921085357666`
- raw mass mean: `4.682127952575684`
- dominant-joint accuracy: `0.4646680951118469`
- unique predicted dominant joints: `1`
- predicted dominant histogram: joint 8 on all 934 supervised rows
- prediction entropy mean: `2.8432605266571045`
- pure mean/p95: `1.3707486391067505 / 1.6182490587234497`
- blend mean/p95: `1.781638264656067 / 1.8403315544128418`
- best p95: `1.8262910289484808` at step 76
- best mean: `1.58597954979608` at step 188
- best deformation: `0.89754319190979` at step 188

Interpretation: under the shared deterministic compute policy, the V7 control reproduces the same qualitative rank-collapse seen in the replay autopsy: one dominant joint everywhere, high entropy, broad product-level failure. Active-heavy sampling does create materially more field differentiation than the other two arms.

## Arm A — uniform/full-supervised scalar objective

Contract: all 934 supervised rows for every joint; same independent BCEWithLogits + 0.1*MSE + Dice; no active-heavy sampler.

Final:
- status: `NO_A0_CAUSAL_ARM_TERMINAL_CLOSURE`
- row-L1 mean: `1.8356897881393244`
- row-L1 p95: `1.9033999670296908`
- deformation ratio: `0.9712908267974854`
- raw mass mean: `1.5777982473373413`
- dominant-joint accuracy: `0.4646680951118469`
- unique predicted dominant joints: `1`
- predicted dominant histogram: joint 8 on all 934 supervised rows
- prediction entropy mean: `3.090862512588501`
- pure mean/p95: `1.9033629894256592 / 1.9034501314163208`
- blend mean/p95: `1.774369478225708 / 1.8203688859939575`
- continuous pairwise L2 mean: `0.8049048185348511`
- pre-normalization joint pairwise L1 mean: `0.0013767335331067443`

C0 -> A causal delta:
- mean: `+0.24937801242544766` (worse)
- p95: `+0.06668769433163124` (worse)
- deformation: `+0.07336974143981934` (worse)
- entropy: `+0.24760198593139648` (more uniform)
- raw mass: `-3.1043297052383423`
- dominant accuracy: unchanged

Interpretation: simply removing active-heavy sampling is decisively harmful. Sparse positive support becomes too weak relative to the many inactive scalar targets; continuous identity contracts and predictions approach the uniform-simplex signature. Therefore the active-heavy sampler is not merely a bug; it provides essential positive-support learning pressure in this FIT1 regime. The remaining issue is that the sampler is uncorrected and thus distorts calibration across joints.

## Arm B — coupled normalized row-L1 through sigmoid fields

Contract: all 934 supervised rows; obtain all 22 scalar logits, apply sigmoid independently, normalize the 22 positive probabilities jointly, optimize mean row-L1 with exact two-pass VJP.

A100 preflight proved:
- all model gradient routes finite/nonzero;
- two-pass logit replay max abs = `0.0`;
- direct-vs-two-pass toy VJP gradient delta = `0.0`;
- therefore the observed failure is not a two-pass implementation mismatch.

Final:
- status: `NO_A0_CAUSAL_ARM_TERMINAL_CLOSURE`
- row-L1 mean: `1.8395368244785233`
- row-L1 p95: `1.909090985916555`
- deformation ratio: `0.9718883633613586`
- raw mass mean: `21.9999942779541`
- raw mass p95: `21.999996185302734`
- prediction entropy mean: `3.0910425186157227`
- pre-normalization probability mean: `0.999999821...`
- pre-normalization joint pairwise L1 mean: `7.739667751138768e-08`
- decoder logit abs mean: `15.248443603515625`
- decoder saturation fraction `|logit| > 12`: `1.0`
- unique predicted dominant joints: `15`, but differences are numerical crumbs on an otherwise uniform distribution
- pure row p95: `1.9090909957885742`

Critical step-1/step-4 trajectory:
- pre-update step-1 coupled mean row-L1: `1.8728759288787842`
- step-1 coupled logit gradient RMS: `1.6323583622579463e-05`
- after the first AdamW update, decoder logit abs mean ~= `11.65`, probability mean ~= `0.9999875`, saturation fraction ~= `0.3356`, row-L1 p95 ~= `1.90909084`
- by step 4, probability mean ~= `0.9999993`, saturation fraction ~= `0.9962`, coupled logit gradient RMS ~= `4.27e-11`
- from there the arm is trapped at the exact-uniform product signature.

Interpretation: the coupled objective idea was tested through the existing `sigmoid(logit) -> normalize across joints` parameterization and exposed a new failure mode. Mean row-L1 is invariant to common positive scaling after normalization, while sigmoid supplies a saturating coordinate map. A large common-mode logit increase sends every field probability toward 1, leaves normalized rows near 1/22, and kills the logit gradient through sigmoid saturation. The first AdamW step is enough to enter this basin. This falsifies `sigmoid -> normalize + pure coupled row-L1` as a viable direct FIT1 objective at the existing LR/parameterization.

## Causal conclusions

1. **Removing active-heavy sampling alone is not the fix.** It makes FIT1 substantially worse and pushes the model toward near-uniform fields.
2. **The active-heavy sampler has a useful role:** it amplifies rare positive support. Its flaw is missing importance correction / calibration consistency across joints.
3. **Pure coupled row-L1 is conceptually product-aligned but the tested sigmoid-field parameterization is pathological.** It has a common-mode saturation escape and collapses in one optimizer step.
4. Therefore the previous broad hypothesis `objective/sampling mismatch` survives only in a refined form. The next intervention must preserve sparse-positive learning pressure while removing prior bias, and any coupled simplex objective must avoid sigmoid common-mode saturation.
5. FIT1 remains one Mage character. Nothing in this experiment supports increasing the character count as the next move.

## Next preregistered direction

Do not start broad architecture changes yet. The next minimal causal tests should target the two newly localized mechanisms:

- **importance-corrected active-heavy scalar sampling**: retain the 50/50 active/global query curriculum but weight each sampled element by the inverse of its sampling probability so the expected scalar objective matches the full supervised distribution;
- **simplex-native coupled parameterization**: compare joint-relative logits with a cross-joint softmax (or an equivalent gauge-fixed log-ratio formulation) instead of `sigmoid -> normalize`, so common-mode shifts cancel algebraically and cannot create a sigmoid saturation trap.

These should be tested as isolated arms under the same one-character FIT1 contract, same V7 architecture, same serialized initial weights, same prefix schedule, and unchanged product gates.

A1 remains unauthorized.
