# P-V5 R256 — 1 Asset × 2 Styles Joint Overfit V1 — Preregistration
Date: 2026-08-25
Status: FROZEN BEFORE OPTIMIZER STEP 1

## Parent evidence
The prior frozen one-cell gate missed `P_p95 <= 0.005` at `0.005682396539486942`, then the preregistered optimizer-localization gate recovered the same exact checkpoint without changing P ontology, V5 reconstruction or R256 representation.

Optimizer-localization authority:
- zero-step parent reproduction absolute difference = `0.0`;
- control fresh AdamW lr `3e-4`, 512 steps: minimum P p95 `0.005420877947472036` — FAIL;
- fresh AdamW lr `1e-4`: first PASS at restart step 256, minimum `0.00420133795123547`;
- fresh AdamW lr `3e-5`: first PASS at restart step 64, minimum `0.003946938854642211`;
- canonical label: `LATE_STAGE_LR_FLOOR_SUPPORTED`;
- status: `P_V5_R256_ONE_CELL_RECOVERY_PASS`.

This authorizes the next promotion rung only: one frozen asset × two frozen styles.

## Scientific question
Can one shared R256 image-conditioned P/depth learner jointly overfit the same asset in both rendering styles to the canonical per-cell P precision target?

This tests same-asset style sharing/interference. It does not test family generalization.

## Frozen membership
Asset:
- `asset_76313e4bd82b82fcd1659c70`
- split `FIT`

Styles:
- `cel_clean`
- `ink_cel`

Exactly 2 asset-style cells. Both styles are present in every optimizer step. No style-specific model parameters or style-ID conditioning are allowed.

## Representation / model
Unchanged true full-R P path:

```text
R256 RGBA -> full-resolution image stem --------┐
R256 RGBA -> encoder -> y2@128 -> upsample@256 ├-> R256 fusion -> scalar depth@256
                                                ↓
                                    V5 analytic canonical P@256
```

Screen-plane coordinates, canonical yaw and native-image-derived `h_native` remain analytic/deterministic. The learner predicts camera-forward scalar depth only.

N/U/Z heads remain frozen and outside the objective.

## Data and supervision
- learner input: derivative 512 RGBA resized once to 256 with PIL bilinear;
- native 1024 RGBA is used only to estimate `h_native` once per style;
- canonical yaws V0..V7 = `0,45,...,315` degrees;
- 4096 deterministic visible raster-authority samples/view;
- the two styles use the same geometric sample loci because truth geometry is shared and the sampling seed remains `sha256(f"{asset_id}|pv5-depth|{view}")`;
- no augmentation.

Forbidden:
- TUNE/CAL/DEV/EXTERNAL;
- sealed splits;
- `camera.json`;
- teacher camera half extent as model input;
- style-specific weights or style labels as learner inputs.

## Fresh initialization
The two-style learner starts from a fresh deterministic model initialization, not from the one-style recovered checkpoint.

Seed: `20260825`.

Reason: the gate must test joint two-style fit rather than adaptation from a network already specialized to `cel_clean`.

## Frozen repaired optimizer schedule
The one-cell localization supports a high-LR main fit followed by a lower-LR fresh-optimizer tail.

Phase MAIN:
- AdamW
- lr `3e-4`
- betas `(0.9,0.95)`
- weight decay `0`
- 2048 optimizer steps

Phase TAIL:
- restart AdamW with fresh moments on the current model weights
- lr `3e-5`
- betas `(0.9,0.95)`
- weight decay `0`
- 512 optimizer steps

Every optimizer step uses one batch containing both asset-style cells (`B=2`, each with all 8 views).

Objective:
- FP16 AMP neural forward;
- FP32 camera-forward depth SmoothL1;
- beta `0.01`;
- loss averaged across both cells and all valid sampled loci.

No LR search or additional arm is authorized inside this gate.

## Frozen evaluation schedule
Authority evaluations:
- INIT, optimizer step 0;
- MAIN steps `512, 1024, 2048`;
- TAIL restart steps `64, 128, 256, 512`.

Each evaluation reports:
- per-cell P p50/p90/p95/mean/max;
- aggregate P metrics;
- worst-cell P p95.

Checkpoint selection minimizes:
1. worst-cell P p95;
2. aggregate P p95;
3. total optimizer steps.

## PASS / FAIL
Threshold: `0.005`.

PASS iff at least one preregistered candidate checkpoint satisfies **both**:
- `cel_clean P_p95 <= 0.005`;
- `ink_cel P_p95 <= 0.005`.

Aggregate P p95 alone cannot PASS the gate.

PASS label:
`P_V5_R256_TWO_STYLE_OVERFIT_PASS`

FAIL label:
`P_V5_R256_TWO_STYLE_OPTIMIZATION_INSUFFICIENT`

## Promotion policy
PASS:
- one-asset/two-style joint learner sufficiency is closed;
- preregister `8 assets × 2 styles R256` only.

FAIL:
- remain at one asset × two styles;
- localize style interference / joint-capacity / optimizer evidence;
- do not broaden to 8 assets;
- do not reopen closed P ontology, V5 analytic geometry or certified R256 representation without contradictory evidence.

## Research order
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
