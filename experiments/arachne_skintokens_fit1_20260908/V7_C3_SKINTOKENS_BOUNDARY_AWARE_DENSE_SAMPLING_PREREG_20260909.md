# Arachne Mage A0 FIT1 — V7-C3 SkinTokens Boundary-Aware Dense Sampling

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent localization result commit: `26d994b25f83096623ac78d804c715660ede319a`
C2 treatment base-model SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
Upstream SkinTokens audited commit: `273b691d35989d71cd17ff2895fdc735097b92d1`

## Causal question

Does replacing the existing exact-positive dense pool with a SkinTokens-inspired **support + near-support boundary pool**, while holding architecture, objective, importance correction, optimizer recipe, query budget, global samples, prefix schedule, and base checkpoint fixed, reduce the remaining weak-true-vs-nearby-false support-ordering error?

This treatment is authorized because the preregistered frozen localization returned `STRONG` alignment:
- 94.5% of missed true influences are in the weakest quartile of their own positive field;
- 60.0% are in the nearest boundary quartile and 87.27% in the nearest boundary half;
- 58.49% of false-kept influences are in the nearest-support quartile of their own true field;
- support displacement rises from 0% on one-joint rows to 63.64% on four-joint rows.

## Important limitation: proxy port, not exact triangle replay

The current bound Mage cache does not contain the original triangle-face adjacency used by upstream `SamplerMix.sample_on_skin()`. Therefore C3 does **not** claim exact upstream sampling parity.

Instead, C3 ports the same geometric idea to the available normalized point cloud:
1. for joint `j`, define exact positive support on supervised Mage rows by `teacher_weight_j > 1e-8`;
2. compute the axis-aligned bounding-box diagonal of those positive support positions;
3. set neighborhood radius `r_j = min(0.1, 0.1 * support_bbox_diagonal_j)`;
4. include every supervised row whose Euclidean distance to the nearest positive-support row is `< r_j`;
5. sample the dense half from this support-plus-neighborhood pool.

The constants `0.1 / 0.1` are the **public upstream code defaults** for `max_distance` and `rate_distance`. The repository does not expose evidence that a different override was used for final paper training, so C3 must not call them proven training hyperparameters.

## Matched arms

Both arms start from the exact C2 treatment final checkpoint and use fresh optimizer state.

### Control — `C3_CONTROL_C2_ACTIVE_ONLY`
- dense pool: exact positive supervised rows (`w_j > 1e-8`);
- 192 dense + 192 global rows per joint/update;
- C2 importance-corrected BCE/MSE;
- sampled Dice unchanged.

### Treatment — `C3_TREATMENT_BOUNDARY_PROXY`
Identical except:
- dense pool: point-cloud support + near-support neighborhood defined above.

No other treatment is allowed.

## Fixed training contract

- Architecture: exact Arachne V7 continuous forced-field codec, 278,010,880 parameters.
- FSQ: absent.
- Base checkpoint: exact C2 treatment final model SHA above.
- Mage cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`.
- Target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`.
- Supervised rows: 934; joints: 22.
- Decoder queries per joint/update: 384 = 192 dense + 192 global.
- Objective: BCEWithLogits + `0.1*MSE` + Dice.
- BCE/MSE importance correction: exact `w_i = u_i / q_i` for each arm's own proposal distribution; target `u` remains uniform over supervised rows.
- Sampled Dice: unchanged/unweighted exactly as C2 treatment.
- Nested field-token prefix: 1..4, matched across arms.
- Optimizer: fresh AdamW, LR `2.5e-5`, weight decay `1e-4`.
- Scheduler: cosine to zero over 384 optimizer updates.
- Steps: 384.
- BF16 autocast with FP32 model/optimizer master state.
- Same global-row draws and same prefix schedule across arms. Dense draws use the same deterministic per-step/per-joint seed policy but necessarily map into different dense pools.
- No architecture, loss coefficient, output map, top-k, threshold, character set, or token-count change.

## Importance algebra

For a joint with `N` supervised rows and dense-pool size `M`, with dense fraction `d=0.5`:

- for every supervised row: global contribution `(1-d)/N`;
- for rows in the dense pool: additional `d/M`;
- therefore `q_i = (1-d)/N + 1[i in dense_pool] d/M`;
- uniform target `u_i = 1/N`;
- BCE/MSE importance weight `w_i = u_i/q_i`.

Preflight must verify for every joint and both arms:
- `sum(q)=1`;
- `E_q[w]=1`;
- pointwise `q*w=u` to numerical tolerance;
- all exact-positive rows are contained by the treatment pool;
- treatment pool is a strict expansion for at least one joint;
- no unsupervised row enters either pool.

## Evaluation contract

The authoritative FIT1 gates remain the existing **raw normalized 22-joint** product metrics. Top-4 remains diagnostic only and does not weaken the gate.

Required at step 0 and milestones:
- raw row-L1 mean / p95 / CVaR10;
- deformation-error ratio;
- dominant accuracy / top3 / mean rank;
- variation ratio / joint-std ratio;
- inactive predicted mass mean and p95-tail inactive mass;
- production-diagnostic top-4 p95/deformation;
- top-4 true-support recall, full-support containment, and displacement-row count;
- support-size slices 1/2/3/4.

Step-0 C2 baseline must replay approximately:
- raw p95 `0.22197734`;
- raw deformation `0.08492328`;
- production top-4 p95 `0.17656562`;
- production top-4 deformation `0.07743423`;
- top-4 displacement rows `55`.

## Acceptance / interpretation

Scientific FAIL is valid and must not throw.

Primary causal success requires the treatment to beat the matched control on the residual signature, especially:
- lower raw p95 and deformation;
- fewer top-4 support-displacement rows / better true-support containment;
- reduced inactive-tail mass;
without degrading dominant/top3 ownership.

FIT1 closure still requires the unchanged primary gates (`p95 <= 0.05`, `deformation <= 0.05`) plus existing compiler/simplex/stability requirements.

Interpretation:
- If treatment materially beats control and specifically reduces displacement rows, the upstream boundary-sampling mechanism is causally supported for this blocker.
- If treatment does not beat control on displacement/tail metrics, reject the proxy sampler as the next answer; do not stack a new loss on top.
- If support ordering is largely repaired but raw p95 remains near the teacher-support oracle floor (~0.08), the remaining blocker becomes genuine within-support blend calibration and must be diagnosed separately before any custom objective.

## Explicit prohibitions

- No custom blend-ratio auxiliary.
- No architecture/FSQ/token-count change.
- No 4k blind continuation.
- No K or threshold sweep.
- No silent top-4 gate substitution.
- No extra characters or pretraining.
- No claim that this point-cloud proxy is exact SkinTokens face-mask sampling.