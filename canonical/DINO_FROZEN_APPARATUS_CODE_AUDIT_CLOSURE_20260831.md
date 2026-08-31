# RealSaS — DINO Frozen Apparatus Code Audit V1

**Date:** 2026-08-31  
**Decision:** `BLOCK_AND_FIX`  
**New DINO training authorized:** **NO**  
**Machine-readable audit SHA-256:** `8b0d6f40383db65688afdcbbd384f7ee4a3dd0f3398867cde264dbb69b047848`

This audit follows the S1 early termination. Its purpose is not to reinterpret S1, but to inspect the components that were treated as fixed controls before opening another representation or decoder experiment.

## Source authority

| Artifact | SHA-256 |
|---|---|
| `dino_shared_depth_learner_v1.py` | `af820b2e739253e716b03399fb1add355412027cf5e5da7409a6eec8b8a0a41f` |
| `dino_controlled_ladder_s_b_l_g_v1_2.py` | `9c0a5e941fc71b1813e539530b0b5fba093df7978fca2ada241a812df94ef790` |
| `DINO_SHARED_LEARNER_APPARATUS_SPEC_V1.json` | `c66560939e41f38e8f191299a610447f2216049dbfb7c64e61737be28f440b47` |
| `DINO_CONTROLLED_LADDER_TRAINING_AUTHORIZATION_V2.json` | `0790c9667120cf138d1286511d427faaf8ad0f519d00b3ce0135138dab69ecc0` |

Evaluator correction content SHA:
`32bbed01c52bd6ad65ad1896ea3c1ad7ec8b35138788c673e3d87a581610df1e`.

## A. Representation extraction/cache — PASS WITH NAMED LIMITATIONS

- Official DINO source revision and weight hashes are checked at runtime.
- Foundation input begins from native `1024x1024 RGBA`, but the DINO branch consumes RGB only, resizes bicubic+antialias to `518x518`, then applies ImageNet normalization.
- The cached representation is last-layer `x_norm_patchtokens`, exactly `37x37 x native_dim`, FP32 lossless.
- Candidate tokens are zero-padded to 1536 and L2-normalized per token in FP32.
- Cache roundtrip and byte hashes are checked.

Limitation: only the last-layer 37x37 token field is exposed. There is no earlier-layer or multiscale DINO access. After 1024->518 and patch-14 extraction, one token axis corresponds to roughly 27.7 native pixels, so subpatch detail must be supplied by the native path.

This is an interface hypothesis, not a causal finding.

## B. Sampling/supervision — SEMANTICS PASS, COVERAGE LIMITATION

The exact historical sampler is:

- 4096 authoritative raster rows per view;
- if visible support >=4096: deterministic uniform sample without replacement;
- if visible support <4096: deterministic coverage-repeat;
- no augmentation;
- no edge/thin/grazing stratification;
- no hard mining or mechanics-derived weighting.

Every view therefore contributes exactly 4096 loss slots, regardless of visible area.

The missing audit item is empirical structural coverage: S1 did not quantify how many selected rows lie near silhouettes, thin structures or grazing regions. Uniform visible-row sampling can underrepresent low-area but mechanically sensitive structures.

## C. Shared learner / spatial access — BLOCK FOR ANOTHER CAPACITY LADDER

Important correction: the local DINO field **is retained at 37x37**. It is incorrect to say that all DINO information is collapsed to 16x16.

Actual route:

```text
last-layer DINO 37x37
   -> shared 1536->256 stem
   -> within-view axial reasoning at 37x37
   -> local 37x37 -------------------------+
                                             -> context fuse at 37x37
   -> cross-view context 37->16 -> fusion
                         -> bilinear 16->37 -+
   -> bilinear query sample
   + native 25-point RGBA stencil
   + xy/yaw/scale
   -> pointwise MLP
   -> scalar depth
```

Parameter audit:

| Block | Params | Fraction |
|---|---:|---:|
| total | 5,046,739 | 100% |
| token stem | 926,720 | 18.36% |
| within-view | 1,054,720 | 20.90% |
| cross-view | 1,188,864 | 23.56% |
| context fuse | 1,713,152 | 33.95% |
| native encoder | 29,640 | **0.587%** |
| final depth decoder | 133,643 | 2.648% |

The native `1024` path is therefore not a dense spatial decoder. It observes only 25 RGBA samples/query (center + 8 directions at radii 1,4,16) through a 29,640-parameter MLP. Neighboring output queries have no explicit coupling in the final head.

This is sufficient to declare a **material shared-access confound** for a backbone-capacity ladder. It does **not** prove that this path caused the observed S/B plateau.

## D. Loss / optimizer / runtime — IMPLEMENTATION PASS, OBJECTIVE + REPEATABILITY LIMITATIONS

Frozen objective:

`mean SmoothL1(d_pred,d_truth), beta=0.01`.

Inside beta:

`|grad| = |e| / beta`.

Therefore:

- `e=.002 -> |grad|=.2`
- `e=.0002 -> |grad|=.02`

Above `.01`, gradient magnitude saturates at `1`.

So the earlier statement “critical errors receive almost no gradient” is rejected. The real concern is that a **uniform pointwise mean** is misaligned with a consumer whose qualification is tail- and derivative-sensitive.

Optimizer/runtime are internally coherent:

- AdamW `(0.9,0.95)`, wd=0;
- LR `3e-4` for steps 1-2048;
- fresh AdamW moments and LR `3e-5` for steps 2049-32768;
- AMP FP16 shared stack, FP32 target/loss, GradScaler;
- four 4-cell microbatches -> one 16-cell logical update;
- resume restores optimizer/scaler and Python/NumPy/Torch CPU/CUDA RNG state.

However global deterministic algorithms are intentionally not forced because CUDA `grid_sample` backward may be nondeterministic. S and B are each one execution. Therefore a tiny delta such as TRAIN `17 -> 18` is not evidence of a reliable ordering. The safe S1 claim remains only **no material S->B improvement**.

Before a future small-effect comparison, run a repeatability calibration or use a deterministic substitute.

## E. Evaluator — PASS

The evaluator is the strongest frozen component in this audit:

- every authoritative visible raster row is queried densely;
- predicted `P` is reconstructed with exact `camera.json` frame and exact `half_extent`;
- the frozen DTB-ND1 robust local-plane route is replayed;
- Geppetto/Arachne proxy comparison is per-family against exact-depth zero reference;
- FIT eligibility is 54 cells, threshold 52;
- TRAIN eligibility is 56 cells, threshold 54;
- ineligible cells are neither PASS nor FAIL;
- hard typed-route failures are forbidden;
- scalar depth diagnostics cannot replace direct consumer replay.

No evaluator issue was found that explains the S/B plateau.

## F. Camera-scale input — PASS WITH FIX REQUIRED IN V2

The learner does **not** receive `camera.json["half_extent"]` directly. The runner derives a scalar from alpha bounding boxes:

`0.5 / max(V0 width, V2 width, maximum view height)`.

This initially looked like a missing camera-authority bug. A real master witness was checked:

- exact V0 camera half extent: `0.54`;
- exact V2 camera half extent: `0.54`;
- V2 alpha width: `948/1024`;
- derived observable half extent: `0.5400843882`.

So this is not evidence that S1 had a scale failure. But the camera is known by contract; re-deriving its scale from style pixels is unnecessary and can become style-dependent. V2 must use exact camera authority directly.

## G. Corpus substrate — SEPARATE BLOCK / NO RETROACTIVE REINTERPRETATION

The parallel clean-character audit has already shown that structural eligibility is not the same as a clean single-character visual substrate.

This **cannot** be used to reinterpret S1 after the fact.

For causality:

1. first decoder-only experiment keeps the exact S1 membership;
2. clean-C0 becomes a separate H0 intervention later.

Never change decoder and corpus membership in the same causal step.

## Final causal-confound table

| Component | Verdict | Can explain plateau? | Action |
|---|---|---|---|
| DINO source/weights/cache | PASS + interface limitation | possible via last-layer-only access | keep fixed first |
| 4096 sampler | known coverage limitation | plausible | measure coverage; keep fixed in decoder-only test |
| shared decoder/native access | **BLOCK** | **strong plausible confound** | replace/test causally |
| mean SmoothL1 beta=.01 | known objective limitation | plausible | keep fixed first, ablate later |
| optimizer/resume | PASS | no major issue found | keep |
| CUDA repeatability | limitation | can obscure tiny deltas | calibrate |
| evaluator / DTB replay | **PASS** | no issue found | keep |
| image-derived camera scale | minor avoidable proxy | not supported as S1 cause | replace with exact camera input |
| corpus purity | separate H0 confound | plausible | separate intervention |

## Required sequence

No new four-rung ladder is authorized.

The next sequence is:

1. freeze a high-resolution spatial decoder/native-access V2;
2. first causal run: **B, same S1 membership, same current loss**;
3. separately train/evaluate:
   - DINO ON / native ON,
   - DINO OFF / native ON,
   - DINO ON / native OFF;
4. treat native-only success as a legitimate result;
5. after decoder causality, run **beta-only** loss ablation;
6. then run **tail/CVaR-only** ablation with beta held fixed;
7. only reopen S/B/L/g if one revised single rung reaches adequate accessibility.

## Claim firewall

Permitted:

> S1 S->B flatness is conditional on SharedLearnerV1 and its sealed apparatus.

> SharedLearnerV1 contains material high-resolution access limitations that must be causally tested before another capacity ladder.

Forbidden:

> The decoder is already proven to be the dominant cause.

> Loss is proven secondary.

> DINO is necessary or unnecessary.

> Corpus contamination explains S1.

> L/g would fail.
