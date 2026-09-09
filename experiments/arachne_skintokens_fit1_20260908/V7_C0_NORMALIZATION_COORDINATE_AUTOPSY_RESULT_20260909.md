# V7 C0 Normalization-Coordinate Autopsy — Result

Date: 2026-09-09
Scope: read-only / zero-training diagnostic on the sealed C0 final model.
Preregister commit: `a70583cd7765d12e89b7767765cc3424bc7258d4`
C0 model SHA-256: `0e5f11cd5925b35235a83527b2ec6ef7903f7214734262576e9b8c45a77c9e53`

## Question
Does C0 already contain correct cross-joint ownership information in raw logits that is merely damaged by the current `sigmoid -> normalize` simplex mapping, or is the raw cross-joint ranking itself wrong?

## Guardrails
- Positive-alpha `softmax(alpha * Z)` is monotone within each row and therefore cannot change the raw-logit argmax.
- Row-centering before softmax is algebraically invariant; measured max delta was exactly 0 for alpha = 0.5, 1, 2, 4, 8.
- Per-joint global bias removal is diagnostic only: `B_j = mean_i Z_ij` over the 934 supervised C0 logits. Teacher values are not used to construct `B_j`.
- Alpha-driven sharpening may mechanically change entropy/variation and must not be treated as ownership recovery without dominant/top-k evidence.

## Raw-logit ranking
- Raw dominant accuracy: `0.4646680942`.
- Raw dominant histogram: `{8: 934}`.
- Therefore the joint-8 collapse exists before sigmoid/normalization.

## Current sigmoid-normalize mapping
- mean row-L1: `1.5863117757`
- p95: `1.8367122727`
- dominant accuracy: `0.4646680942`
- unique dominant joints: `1`
- teacher-dominant top-3 inclusion: `0.6745182013`
- teacher-dominant mean rank: `4.1584582441`
- pred/teacher pairwise variation ratio: `0.1606102404`
- pred/teacher joint-std ratio: `0.0760121277`

## Plain softmax(alpha * Z)
Plain softmax never changes the 934/934 joint-8 argmax, as required by monotonicity.

Representative points:
- alpha 0.5: mean `1.6781346`, p95 `1.8532455`, dominant accuracy `0.4646681`, unique dominants `1`, variation ratio `0.0156764`.
- alpha 1: mean `1.3297535`, p95 `1.9297178`, dominant accuracy `0.4646681`, unique dominants `1`, variation ratio `0.0413426`.
- alpha 2: mean `1.0792402`, p95 `1.9965543`, dominant accuracy `0.4646681`, unique dominants `1`, variation ratio `0.0054877`.
- alpha 4: mean `1.0706679`, p95 `1.9999971`, dominant accuracy `0.4646681`, unique dominants `1`.
- alpha 8: mean `1.0706638`, p95 `2.0000000`, dominant accuracy `0.4646681`, unique dominants `1`.

Interpretation: increasing alpha mostly sharpens the already-wrong global joint-8 rank. The lower mean at high alpha is misleading: it rewards rows whose teacher dominant is joint 8 while driving the tail/blend/other-owner rows toward the maximum L1 error of 2. Plain softmax is therefore not a post-hoc rescue of hidden correct ownership.

## Teacher-free per-joint global-bias removal
The C0 per-joint mean-logit offsets are extreme:
- joint 8 mean logit: `+1.8980915`
- most other joints: approximately `-1.47` to `-1.90`
- total mean-logit bias range: `3.7958786`

Subtracting `B_j = mean_i Z_ij` changes the dominant histogram from one joint to 19 unique joints, proving a very large global joint offset is present.

However ownership correctness does not improve:
- bias-removed raw dominant accuracy: `0.4625267666` (slightly below original `0.4646680942`)
- unique dominant joints: `19`
- bias-removed alpha 1 top-3 inclusion: `0.5449678801` versus original/plain `0.6745182013`
- bias-removed alpha 1 mean: `1.8347870`, p95 `1.9075713`
- bias-removed alpha 8 mean: `1.7844036`, p95 `1.8967172`
- bias-removed alpha 8 variation ratio: `0.0741567`, still far below teacher variation.

Therefore the 934/934 joint-8 collapse is partly driven by a strong global joint offset, but removing that offset does not uncover a correct spatial ownership ranking. The residual cross-joint score structure remains broadly wrong.

## Causal conclusion
The hypothesis "C0 already has the correct ownership logits and only the final sigmoid-normalize coordinate mapping destroys them" is falsified.

What survives:
1. Geometry is causally present in the logits (proved by the preceding geometry autopsy).
2. Field identity is causally present (proved by V6/V7 forced-field tests).
3. A huge joint-8 global offset is present.
4. But cross-joint relative ranking is not yet learned correctly before normalization.

Consequently, a production change from sigmoid-normalize to softmax may still be a better training coordinate system, but it cannot be justified as a mere output remapping of the current C0 solution. It would need to change the training dynamics/objective so the network learns a different raw rank structure.

## Next experiment decision
Do not treat post-hoc softmax as a solved intervention. The remaining clean confound is training horizon: C0 used only 192 AdamW/cosine steps, with LR reaching zero at 192. Run a fresh long-horizon C0 from the same initial weights with the same architecture, same active-heavy scalar objective/sampling, and only `MAX_STEPS = T_max = 2000` changed.

Track at preregistered checkpoints:
- authoritative mean/p95/deformation
- raw dominant accuracy and histogram / unique dominant joints
- teacher-dominant top-3 and mean rank
- pred/teacher pairwise variation ratio and joint-std ratio
- joint-8 mean-logit offset versus the other joints
- pure/blend mean and p95

Decision rule:
- If raw ranking diversifies and p95/variation improve materially at longer horizon, 192 steps was a substantive optimization limitation.
- If mean/deformation continue improving while p95, raw joint-8 dominance, and ownership variation stay stuck, objective/training-coordinate mismatch becomes strongly causal and the next intervention should be simplex-native coupled/softmax training (with forced-field score-space causality preserved), not still more C0 time.
