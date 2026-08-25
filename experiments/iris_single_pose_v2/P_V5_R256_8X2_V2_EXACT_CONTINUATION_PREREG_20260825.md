# P-V5 R256 8x2 V2 Exact-Continuation V1 — Preregistration (2026-08-25)

## Scientific question
Does the **exact same V2 TAIL optimizer trajectory**, resumed from the immutable failed `TAIL_4096` checkpoint with optimizer/scaler/RNG state intact, cross the strict 16/16 `P_p95 <= 0.005` threshold under a bounded additional budget?

This is a localization gate. It cannot rewrite the fresh 8x2 V2 scientific FAIL.

## Frozen parent authority
Parent run: `IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_CERTIFICATION_V2`.

Required parent facts:
- status: `P_V5_R256_8X2_V2_CERTIFICATION_INSUFFICIENT`;
- selected label: `TAIL_4096`;
- selected total optimizer steps: `6144`;
- selected pass count: `14/16`;
- selected worst-cell P95: `0.005129679851233959`;
- only failing asset: `asset_36fb02305846592b1ecdf3d4`, both styles;
- parent run-complete SHA-256: `339b6a2f929e732e9ac0bb5c540e85b3b473e86acc6508391de9807b20dbb77b`;
- parent decision SHA-256: `77626d41cf87facac5e5d5247a20268bf3654cd94ab24ab16ff8031d8d37678c`;
- exact parent checkpoint SHA-256: `8c3872412124644885b66aea1b00413a8da5ab91a7471e83fcae001d43544049`.

The checkpoint must contain model, AdamW optimizer, GradScaler, and Python/NumPy/Torch CPU/Torch CUDA RNG state.

## Frozen data authority
Same exact V1/V2 8x2 membership, 16 cells, 4096 visible truth rows/view, same style pair, same ordering.

Before any optimizer step, regenerated truth NPZ SHA-256 values must equal:
- `asset_551ea351b43a1787d0f55536`: `f32a90c5b15422f84a2ef051ff8908553eb2a31801dee83554d7a493f8356751`
- `asset_36fb02305846592b1ecdf3d4`: `18c69d343f7624061af2c45dca0c49ae64768f903593d27ef566ccca391d807e`
- `asset_0679fdef64f19a4832a6d521`: `d95dd8cb13b7e85d6193fa585f9046458b5f36303940e0ff9d8e75381f3eef6c`
- `asset_76313e4bd82b82fcd1659c70`: `034a11a5efc076f2d5615aad33ae7743fd59f0c61e69b1ff59a2f237a81b5945`
- `asset_6f086a5b1a66378ffe04d7e4`: `ea9723e082d76aab7bdf2030c623447b31534f261a52f6e197ccf403d3b83a69`
- `asset_425122d500ecf5767404f9c0`: `49bcbd25c832e45198d9763b5c1f8ea9cf1a7bebedb9c00b943e01ba9ac2adf3`
- `asset_5a19f8c5254be7bf30c504f5`: `e50075ccccb35b6ddbc60ffd4e42d587b285147d214fab0abab46b6efbe23b85`
- `asset_f8a40d6c5d815fe79c8b5e42`: `15f5b7d7174c399366734854558963afb57cd06d89aa0bff20f50453b4ec74c2`

A zero-step parent re-evaluation must reproduce 14/16 PASS and worst-cell P95 within `5e-5` of the recorded parent value before optimizer authorization.

## Frozen optimization
- architecture: unchanged R256 P-V5;
- objective: unchanged P-only camera-forward depth objective;
- effective optimizer update: all 16 cells, 8-cell x2 gradient accumulation;
- optimizer: **restore exact AdamW state from parent**; no fresh moments;
- scaler: restore exact GradScaler state;
- RNG: restore exact saved states after all setup/re-evaluation and immediately before first continued optimizer step;
- LR: unchanged `3e-5`;
- additional optimizer steps: **1024**;
- eval continuation steps: `128, 256, 512, 768, 1024`;
- run the full +1024 schedule even if an earlier checkpoint passes.

## PASS / FAIL
PASS iff at least one preregistered continuation checkpoint has all 16 cells individually `P_p95 <= 0.005`.

FAIL otherwise.

Candidate selection for persisted BEST: minimum `(worst_cell_P_p95, aggregate_P_p95, total_optimizer_steps)` over the preregistered continuation checkpoints.

## Interpretation
- PASS: the original V2 remains immutable FAIL, but exact continuation establishes that the final shared trajectory was still optimization-budget limited. Freeze/preregister unseen-family generalization before opening unseen data.
- FAIL: blind budget extension is no longer the default intervention. Remain at 8x2 and localize true shared capacity/interference; architecture/capacity intervention becomes the primary next branch. PatchMatch still requires geometry-localized evidence.

## Firewalls
- no fresh optimizer moments;
- no checkpoint weight-only restart;
- no architecture change;
- no PatchMatch;
- no augmentation;
- no camera JSON or hidden teacher camera;
- N/U/Z frozen;
- no TUNE/CAL/DEV/sealed/unseen-family access;
- scientific optimizer steps before all authority checks: 0.
