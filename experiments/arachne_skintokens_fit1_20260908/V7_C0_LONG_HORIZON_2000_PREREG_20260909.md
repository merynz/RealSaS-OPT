# Arachne Mage A0 FIT1 — V7 C0 Long-Horizon 2000-Step Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`

## Question
Does the sealed V7 C0 biased-scalar training recipe fail because 192 optimizer steps are simply too short, or does the same objective continue to optimize mean/deformation while leaving cross-joint ownership/ranking and p95 effectively stuck?

## Frozen controls
- Architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
- Config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`
- Parameter count: 278,010,880
- No FSQ / no quantizer
- Mage A0 only; same cache/binding/teacher truth as V7 C0
- Same serialized FP32 initialization as the completed C0 causal-triplet arm
- Same active-heavy sampler: 384 queries/joint, 50% active-support + 50% global supervised
- Same independent scalar objective: BCEWithLogits + 0.1*MSE + Dice
- Same 22-joint balanced optimizer update
- Same AdamW: lr=2e-4, wd=1e-4
- Same nested field-token prefix mechanism and deterministic seed policy
- Same math-SDPA-only / TF32-off compute policy used by the completed causal triplet
- Same authoritative product metrics/gates; scientific FAIL remains valid output

## Only scientific treatment
- `MAX_STEPS: 192 -> 2000`
- `CosineAnnealingLR T_max: 192 -> 2000`

This is a restart from the exact shared initial FP32 state, not a resume from step 192 (the 192-step schedule already reaches LR=0 there).

## Required telemetry
At checkpoints, record at minimum:
- authoritative normalized row-L1 mean, p95, CVaR10
- deformation-error ratio
- raw dominant-joint histogram / unique dominant joints
- dominant accuracy
- teacher-dominant top-3 inclusion and mean rank
- predicted/teacher mean pairwise row-L1 variation ratio
- predicted/teacher mean joint-std ratio
- joint-8 global mean-logit offset relative to the mean of the other 21 joints
- raw probability mass mean/p95
- continuous field identity separation
- LR and objective trace

Checkpoint observation schedule should include at least: 1, 4, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2000. Additional periodic observations are permitted.

## Decision logic
1. **Horizon causality supported** if, after step 192, p95 and ownership telemetry show sustained qualitative improvement: dominant histogram diversifies in a teacher-aligned way, dominant accuracy/top-3 improve, variation ratios rise materially toward teacher, and p95 resumes a meaningful downward trend.
2. **Objective/ranking mismatch strengthened** if mean/deformation continue to improve while p95 remains near the old ~1.8 wall and raw ranking stays joint-8 dominated / variation remains compressed.
3. A lower mean alone is not evidence of ownership recovery. p95/CVaR and ranking telemetry are authoritative for this distinction.
4. Do not introduce softmax-native coupled training, importance correction, architecture changes, token-count changes, FSQ, different optimizer, or extra characters inside this experiment.
5. If the long-horizon control falsifies the 192-step horizon hypothesis, the next intervention may target the cross-joint training objective/coordinate system under a separate preregistration.

## Relevant prior evidence
- Completed C0 192-step final: mean ~1.58631, p95 ~1.83671, deformation ~0.89792, raw dominant joint 8 on 934/934 rows.
- Geometry causality autopsy: query position causally controls output, but predicted row-to-row spatial variation is only ~0.1606 of teacher.
- Normalization-coordinate autopsy: raw logits themselves are joint-8 argmax on 934/934 rows; plain positive-alpha softmax cannot repair this ranking. Teacher-free per-joint mean-logit removal diversifies argmax to 19 joints but does not improve dominant accuracy, showing the residual ranking is also insufficient.

No A1/generalization authorization is implied by this preregistration.
