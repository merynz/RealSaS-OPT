# RealSaS — Arachne Mage A0 FIT1 — V7-C4 SkinTokens Face-Barycentric Biased Dense Supervision Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Status at preregistration: **NO C4 OPTIMIZER STEP HAS RUN**

## Why C4 exists

V7-C3 tested a 950-node point-cloud proxy for SkinTokens-style support/near-support sampling while applying exact `w=u/q` importance correction to BCE/MSE in both arms. C3 completed with no FIT1 closure. The treatment did not beat the matched control on the primary metrics: final raw row-L1 p95 changed from `0.18175699718563249` to `0.19077774486726193`, and raw deformation-error ratio from `0.06118907406926155` to `0.06876000761985779`; top-4 displacement changed only `48 -> 47`. The C3 combined result SHA-256 supplied for this preregistration is `5fe9c350e2f09a1fd858d86d39c965c9c429321c76851994379a0466d925eaa0`.

C3 therefore falsifies the **point-cloud proxy under importance-corrected reconstruction**. It does not test two upstream SkinTokens mechanisms that C3 intentionally omitted:

1. dense supervision is sampled on real mesh faces, including triangle-interior points with barycentrically interpolated skin targets;
2. the dense sample distribution is allowed to bias the training objective instead of being algebraically cancelled back to a uniform supervised-row objective by `u/q`.

C4 isolates these two missing mechanisms.

## Frozen scientific target and model

All three arms start independently from the exact C2 treatment model:

- model SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
- architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
- parameter count: `278,010,880`
- config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`
- no FSQ / no architecture change / no token-count change
- cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- cache binding SHA-256: `c7e3bf10fc8edf16f862b4ce58b3aabfeb2aabf14cc8e744e53e3afc0764ac9f`
- target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`
- teacher-W content SHA-256: `7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d`
- 934 supervised GSA rows, 16 low-confidence rows, 22 canonical joints.

The exact training-only full source authority is the sealed Mage `normalized.npz`:

- SHA-256: `528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f`
- full source: 5321 vertices / 5763 faces / 5321x41 skin
- exact 41->22 identity is recovered only through frozen Geppetto generation/proposal lineage into `QualifiedSkeletonIR.source_proposal_id`; geometry/Hungarian identity is forbidden.
- exact generation-row -> teacher-source-control order: `[1,19,20,21,22,2,3,9,10,11,12,13,14,4,5,6,7,8,15,16,17,18]`.
- the 19 non-selected controls must have exact zero total skin mass.
- exactly 42 source vertices have zero selected-22 skin mass; faces touching any such vertex are excluded, leaving exactly 5683 deformation-supported faces.

Teacher/source mesh and dense source skin are **training/evaluator-only**. They are not product inference inputs.

## Upstream SkinTokens mechanism being ported

Pinned upstream commit: `273b691d35989d71cd17ff2895fdc735097b92d1`.

Source anchors:

- `src/data/sampler.py` blob `ef7bc0d8fccd2bd4e0b2bd13887c1c96b2caa6dc`
- `src/rig_package/utils.py` blob `80a89f9fa5c3a3e2a6ef0646071ff638e640f3d1`

The port preserves these source semantics for the dense lane:

1. `face_has_skin = any(skin[faces] > 0)` for the selected joint;
2. if active faces exist, collect their unique vertices and build nearest distance to that active vertex set;
3. include any face with at least one vertex at distance `< min(0.1, active_vertex_bbox_diagonal * 0.1)`;
4. sample faces proportional to `norm(cross(v1-v0, v2-v0))`;
5. sample triangle-interior coordinates with two uniform random lengths and reflect pairs whose sum exceeds 1 exactly as upstream;
6. barycentrically interpolate the selected joint skin target using those same random lengths;
7. use the sampled face normal for the dense query normal, matching upstream `face_normals[face_index]` behavior.

RealSaS-specific adaptation: the mask operates only on the 5683 deformation-supported teacher faces. `num_vertex_samples=0` is preregistered to isolate the missing triangle-interior signal within the fixed 192-dense-query budget; this is **not** a claim that upstream training used zero vertex samples in every configuration.

## Three matched arms

All arms use 384 optimizer steps, fresh AdamW, LR `2.5e-5`, weight decay `1e-4`, `CosineAnnealingLR(T_max=384)`, BF16 autocast with FP32 masters, math SDPA, TF32 disabled, 22 field reconstructions per optimizer step, 384 decoder queries per field (`192 global + 192 dense`), the same nested field-token prefix schedule, and the unchanged V7 field encoder/condition encoder/decoder.

### A — `C4_REFERENCE_IMPORTANCE_ACTIVE_ONLY`

Exact C3-control reconstruction lane:

- global 192: uniform without replacement from the 934 supervised GSA rows;
- dense 192: exact positive supervised GSA rows for that joint, replacement only if the active pool is smaller than 192;
- BCE/MSE use exact `w=u/q` importance weights for the fixed-row mixture;
- Dice is sampled and unweighted, unchanged from C2/C3.

The A row schedule MUST reproduce C3-control schedule SHA-256 `e7a257f20a447e5b34ff850658851e937691fc60aed4c6286d6a8043b071cef2`; common global/prefix SHA-256 MUST reproduce `0875c257bb53f06439bb3de9eb28d4bc642cfb8807cff1c703a3c3932d74268e`.

### B — `C4_BIASED_ACTIVE_ONLY`

Uses the **exact same GSA query rows, order, global draws, dense draws, shuffle, and prefix as Arm A**. The only change is:

- no `u/q` correction on BCE/MSE; every sampled query receives unit sample weight.

A vs B therefore isolates whether deliberate active-heavy objective bias was cancelled by C2/C3 importance correction.

### C — `C4_SKINTOKENS_FACE_BARY_BIASED`

Uses the same 192 global GSA draws, same prefix, and same 384-slot shuffle seed as A/B. The 192 dense slots are replaced by source-faithful face samples from the per-joint SkinTokens face mask described above.

For each dense sample:

- query xyz is the triangle-interior point in the exact normalized source coordinate system;
- V7 decoder query geometry is `[2*xyz_normalized, face_normal_xyz, valid=1]`, matching the existing 7-D V7 geometry-query contract;
- target is the exact barycentrically interpolated canonical-22 joint skin scalar;
- BCE/MSE sample weight is 1 (no importance correction);
- Dice remains sampled and unweighted.

The field encoder observations remain the same frozen GSA field contexts in all three arms. C4 changes decoder supervision only.

## Determinism

Seed: `20260909`.

Sub-seed function is the first four bytes, little-endian, of SHA-256 over `"{seed}:{step}:{joint}:{stream}"`.

Frozen streams:

- 1 = common global GSA rows
- 2 = common field-token prefix
- 3 = A/B active-GSA dense rows
- 4 = common 384-slot shuffle
- 5 = C face-area draw + barycentric random lengths

The notebook must compute and bind a SHA-256 over the complete C dense schedule (global row IDs, selected source face IDs, barycentric random lengths, slot permutation, prefix). The hash is part of the notebook/runtime fingerprint and must be locally reproduced before GPU execution.

## Acceptance gates — unchanged FIT1 gates

No C4-specific metric may replace the FIT1 gate. For any arm to close Arachne FIT1 it must satisfy the existing raw and qualified 22-joint acceptance contract at the terminal observation with a three-observation passing streak:

- raw row-L1 p95 <= 0.05
- qualified row-L1 p95 <= 0.05
- raw deformation-error ratio <= 0.05
- qualified deformation-error ratio <= 0.05
- finite outputs; no negative weights
- raw simplex max residual <= 1e-6
- qualified simplex max residual <= 1e-6
- qualified row count = 950
- Compiler total correction L1 <= 1e-4
- Compiler mean row correction L1 <= 1e-7
- Compiler max row correction L1 <= 1e-6
- Compiler sparsification discarded mass = 0.

Top-4 is diagnostic only and cannot close FIT1.

Scientific FAIL is a valid terminal result and must not raise an infrastructure exception.

## Preregistered causal interpretation

Primary causal comparisons are terminal A vs B and B vs C, with the whole trajectories retained.

- If B improves raw p95 and raw deformation versus A without degrading dominant-joint accuracy/top-3 inclusion, the evidence supports **objective-bias cancellation by importance correction** as a blocker.
- If C improves raw p95 and raw deformation versus B and also reduces false-joint displacement or p95-tail inactive mass without ownership degradation, the evidence supports **missing face/topology/barycentric dense supervision** beyond mere active-heavy bias.
- If B improves but C does not, retain the simpler biased fixed-GSA mechanism and reject the source-face port for this FIT1 target.
- If C improves but B does not, the off-grid face/barycentric signal is the supported mechanism.
- If neither B nor C improves over A, reject this sampling family; do not automatically stack a custom blend-ratio loss. Reopen a separately preregistered within-support calibration diagnosis.
- If support-ordering diagnostics become nearly repaired while raw p95 stalls near the prior teacher-support oracle floor (~0.08), treat that as evidence for a remaining within-support blend-ratio/calibration problem, not justification to weaken the 0.05 FIT1 gate.

## Canonical FIT1 candidate rule

If multiple arms terminal-pass, choose the **simplest passing intervention** in this preregistered order:

`A reference importance-corrected active-only` -> `B biased active-only` -> `C SkinTokens face-bary biased`.

All passing models are still retained with SHA-256. This selection rule is only for Arachne FIT1 candidate promotion.

## Product boundary

**C4 is an Arachne FIT1 experiment, not a product experiment.**

Even if an arm terminal-passes, the strongest allowed statement is:

`Arachne FIT1 scientifically closed for this preregistered arm.`

It MUST NOT emit or imply `PRODUCT_PASS`, product acceptance, or substitution for a separate production/end-to-end product experiment with its own independent contract.

A1 remains unauthorized by this notebook.
