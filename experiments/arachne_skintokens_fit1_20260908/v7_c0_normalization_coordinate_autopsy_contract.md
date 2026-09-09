# Arachne Mage A0 V7 C0 — Normalization Coordinate Autopsy Contract

Date: 2026-09-09
Branch: exp/arachne-skintokens-cleanroom-fit1-20260908
Scope: zero-training read-only diagnostic on the sealed C0 final model.

## Fixed artifact binding
- C0 arm: `C0_BIASED_SCALAR`
- C0 final model SHA-256: `0e5f11cd5925b35235a83527b2ec6ef7903f7214734262576e9b8c45a77c9e53`
- C0 source commit: `f2a573022f718cd390904cc4c6bfe427436bce11`
- Architecture/config: `RealSaS.Arachne.SkinFieldCodec.v7`, config hash `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`
- Cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- Target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`
- 934 supervised rows, 22 joints.

## Question
Does C0 already contain useful cross-joint ownership structure in its final raw logits that is being obscured mainly by the current simplex mapping, or is the raw cross-joint ranking itself wrong/global-bias dominated?

## Important invariant
For any row, `sigmoid(z)/sum(sigmoid(z))`, `softmax(alpha*z)` for `alpha>0`, and `softmax(alpha*(z-rowmean(z)))` are all strictly monotone in each row's logits and therefore preserve the raw-logit argmax ordering. They cannot change the dominant-joint histogram by themselves. Any recovery of dominant ownership requires an intervention that changes cross-joint offsets/ranking, such as the diagnostic per-joint bias removal below.

## Read-only mappings to evaluate
On the exact final raw logit matrix `Z[934,22]`:

1. Current mapping: `sigmoid(Z) / sum_j sigmoid(Z_j)`.
2. `softmax(alpha*Z)` for preregistered `alpha = {0.5, 1, 2, 4, 8}`.
3. Row-centering sanity: `softmax(alpha*(Z-rowmean(Z)))`; must match `softmax(alpha*Z)` within numerical tolerance.
4. Diagnostic per-joint global-bias removal: `B_j = mean_i Z_ij` computed from the supervised C0 logits only, then `softmax(Z-B)`.
5. Bias-removed alpha sweep: `softmax(alpha*(Z-B))` for the same preregistered alpha set. This is diagnostic only and is not authorization to add a learned/post-hoc bias correction to production.

No teacher values are used to compute `B`; teacher is evaluation-only.

## Metrics
For every mapping:
- authoritative supervised row-L1 mean, p95, and CVaR10;
- dominant-joint accuracy;
- unique predicted dominant-joint count and histogram;
- teacher-dominant top-3 inclusion rate;
- prediction entropy;
- predicted row-to-row mean pairwise L1 and ratio to teacher;
- mean per-joint std across rows and ratio to teacher;
- pure-row and blend-row mean/p95;
- row-wise teacher-dominant logit margin/rank diagnostics when applicable.

## Interpretation guardrails
- Increasing alpha mechanically sharpens distributions and can mechanically raise the variation ratio. This is not evidence of better ownership by itself.
- Because plain softmax preserves rank, dominant accuracy/unique dominant count cannot improve over raw C0 rankings unless rankings are changed by a cross-joint offset intervention.
- If `softmax(alpha*Z)` improves L1/variation but leaves 934/934 joint-8 dominance, the coordinate map changes calibration/contrast but does not solve ownership ranking.
- If subtracting `B_j` materially diversifies the dominant histogram and improves dominant accuracy/top-3 inclusion without using teacher to construct `B`, then a global per-joint offset is causally implicated in the joint-8 rank collapse.
- If bias removal does not help, the cross-joint rank structure itself is wrong, and post-hoc normalization is not the primary blocker.
- No training or architecture change is authorized solely by this diagnostic; its role is to decide whether softmax-native training should precede the long-horizon C0 test.

## Decision ordering after this autopsy
- Strong post-hoc recovery from softmax/bias-corrected coordinates -> prioritize simplex-native/score-space training experiment.
- Little or no post-hoc recovery -> retain the 2000-step C0 horizon test as a meaningful next causal experiment before redesigning architecture.
