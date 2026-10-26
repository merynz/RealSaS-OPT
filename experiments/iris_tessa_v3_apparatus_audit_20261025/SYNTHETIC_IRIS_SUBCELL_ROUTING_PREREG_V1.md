# RealSaS — IRIS V3 Post-Foundation Subcell Routing Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** tiny synthetic CPU learned mechanism court; no product/ModelRelease authority.

## Question

Assume the frozen foundation token already contains distinct subpatch information. Can the current **source-token -> scene-query attention -> non-overlapping 2x upsample** mechanism route four independent subcell states to distinct carrier positions?

This isolates the post-foundation scene route. It does **not** test whether DINOv2 actually preserves those subpatch states.

## Why this court exists

The current scene uses a 32x32 learned query grid followed by `ConvTranspose2d(kernel=2,stride=2)` to a 64x64 carrier. A rank audit shows that one 512-D scene-query vector can in principle control all `4*96=384` output scalars in its 2x2 block, so non-overlap alone does not prove a resolution failure. The remaining question is information routing.

## Synthetic sample

Use a **4x4** source-token grid. Each source token carries:

`[x, y, b00, b01, b10, b11]`

where the four `b` values are independent fair bits in `{-1,+1}` and correspond to the four 2x2 subcells owned by that source position.

- source-token order is randomly permuted every batch;
- source XY remains attached to each token;
- target is the resulting **8x8** binary field;
- independent random bits deliberately remove smoothness/interpolation shortcuts.

The 4x4/8x8 scaling is a bounded hosted-CPU mechanism analogue of the reference 32x32/64x64 route. It is not a claim of reference-scale equivalence.

## Arm A — CURRENT_LIKE_QUERY_UPSAMPLE

Tiny current-like mechanism:

- source projection to `d_model=64` using content+XY;
- learned 4x4 scene queries plus coordinate MLP;
- 2-layer TransformerDecoder, 4 heads, GELU, norm-first;
- reshape to 4x4;
- `ConvTranspose2d(64,1,kernel=2,stride=2)`;
- 8x8 logits trained directly with BCE.

There is no source-source self-attention before scene queries, matching the relevant placement property of current V3.

## Arm B — RAW_BLOCK_POSITIVE_CONTROL

No scene-query compression. A shared MLP reads each raw source token and emits four logits, which are placed into its 2x2 target block using the attached XY/index correspondence. This establishes that the task/data are learnable.

## Arm C — RANDOM_LABEL_FIREWALL

Use Arm A inputs unchanged but replace the 8x8 target with independent fair random labels absent from the inputs.

## Seeds

Exactly `731, 947, 1217, 1597, 2017`.

## Optimizer / budget

- CPU torch profile;
- AdamW, lr `1e-3`, betas `(0.9,0.999)`, weight decay 0;
- batch `64`;
- exactly `1200` steps;
- BCE-with-logits;
- no scheduler, clipping, early stopping or outcome-dependent tuning;
- evaluation exactly `4096` fresh samples per seed/arm.

## Metrics

- bit accuracy over all 8x8 outputs;
- BCE;
- per-seed and aggregate mean/min/std;
- parameter count.

## Preregistered gates

Firewall C: every seed bit accuracy in `[0.48,0.52]`; otherwise `IRIS_SUBCELL_FIREWALL_INVALID`.

Positive control B: every seed >= `0.98`, mean >= `0.995`; otherwise `IRIS_SUBCELL_TASK_OR_CONTROL_INVALID`.

Current-like route A is `POSTFOUNDATION_SUBCELL_ROUTING_ESTABLISHED_TINY_V1` when:

- every seed >= `0.95`;
- mean >= `0.98`.

If firewall/control pass but A misses either threshold: `POSTFOUNDATION_SUBCELL_ROUTING_NOT_ESTABLISHED_TINY_V1`.

## Interpretation boundary

A PASS means only that the tiny current-like post-foundation mechanism can route four independent states per source/query cell. It does not prove current 32/64 resolution is sufficient for real subjects and does not prove DINOv2 contains the required information.

A FAIL authorizes a separately preregistered query-density / fusion / carrier treatment study before any backbone replacement.

No outcome authorizes Truth139/FIT8 training, product activation, ModelRelease promotion, or a reference IRIS architecture change.
