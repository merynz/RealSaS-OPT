# RealSaS — TESSA V3 Independent-Compression Addressability Probe Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_PROBE_OUTPUTS`  
**Execution class:** tiny synthetic CPU learned probe; no product/ModelRelease authority.

## Question

Can the **current TESSA V3 memory mechanism** — independently compressing scene, surface and presentation streams before concatenation — preserve enough endpoint-addressable information for a downstream reader to retrieve the payload belonging to a presentation-referenced surface locus?

This is intentionally narrower than the product hypothesis. It tests one mechanism: endpoint join through independent typed compression.

## What this does not test

- reference 1024-d / 24-layer TESSA capacity;
- topology token generation;
- IRIS H quality;
- causal truth quality;
- product inference;
- whether early/mid fusion is better;
- FIT8 genericity.

A PASS proves only tiny-scale mechanism capability. A FAIL triggers apparatus investigation; it is not by itself a falsification of reference TESSA.

## Exact source mechanism

Use the repository's current `TESSATriStreamMemoryV3` and `_TypedMemoryStreamV2` unchanged for the treatment and control.

No custom replacement compressor is allowed.

## Synthetic sample

Each sample contains 16 surface loci on a fixed 4x4 normalized coordinate lattice in `[-0.5, 0.5]^2`.

For each sample:

1. assign an independent balanced payload bit `p_i in {-1,+1}` to every locus;
2. randomly permute surface-token order so array index cannot identify a locus;
3. uniformly select one target locus `j`;
4. create a presentation token containing the exact coordinate of locus `j` and a constant type marker, but **not** its payload;
5. classification target is `p_j`.

Surface token fields:

`[x, y, payload]`

Presentation token fields:

`[x_target, y_target, 1]`

Scene stream contains four deterministic zero/nuisance tokens with the same input width. It is present only because the reference memory is tri-stream; it carries no target information.

Fresh payloads, permutations and target indices are generated for every training/evaluation batch.

## Treatment and negative control

### Treatment — TRUE_ENDPOINT

Presentation coordinate is the true selected locus coordinate.

### Control — SHUFFLED_ENDPOINT

Presentation coordinate is an independently sampled locus coordinate from the same sample, constrained to differ from the true selected locus. The target remains `p_j`.

The marginal distribution and token count of the presentation stream therefore match treatment, while endpoint information is destroyed.

No architecture, optimizer or batch difference is permitted between arms.

## Tiny TESSA memory configuration

Use `TESSAStructuralizerConfigV3` with:

- `d_model = 64`
- `n_heads = 4`
- `decoder_layers = 2` (not used by this memory-only probe, retained as valid config)
- `mlp_ratio = 2`
- `dropout = 0`
- `coordinate_bins = 128`
- `max_faces = 128`
- `max_vertices = 128`
- `scene_memory_tokens = 4`
- `surface_memory_tokens = 6`
- `presentation_memory_tokens = 2`
- `conditioning_token_budget = 12`
- `memory_source_width = 32`
- `memory_refinement_layers = 1`
- `local_attention_window = 64`
- `query_chunk_size = 16`
- `gradient_checkpointing = false`

The surface compression ratio is therefore `16 -> 6`, close in spirit to the reference `1024 -> 384` ratio without claiming scale equivalence.

## Reader

After `TESSATriStreamMemoryV3` returns concatenated memory, use one identical small reader in both arms:

1. one learned 64-D query token;
2. one 4-head `MultiheadAttention` from query to the complete compressed memory;
3. residual + LayerNorm;
4. MLP `64 -> 64 -> 1` with GELU;
5. binary logit for payload sign.

The reader is intentionally separate from the 24-layer topology decoder to isolate memory addressability.

## Initialization and seeds

Run exactly three seeds:

`731, 947, 1217`

For each seed, treatment and control start from identical parameter initialization. Dataset RNG streams are deterministic and arm-independent except for the presentation-coordinate intervention.

## Optimizer and budget

- PyTorch CPU profile pinned by `requirements/torch-cpu.txt`;
- AdamW;
- learning rate `1e-3`;
- betas `(0.9, 0.999)`;
- weight decay `0`;
- batch size `128`;
- exactly `1000` optimizer steps;
- BCE-with-logits;
- no early stopping;
- no LR schedule;
- no gradient clipping;
- no architecture/weight tuning after outputs are opened.

## Evaluation

For each seed and arm, evaluate on exactly 4096 newly generated samples with no gradients.

Report:

- accuracy;
- BCE;
- predicted-positive fraction;
- target-positive fraction.

## Preregistered interpretation

`MECHANISM_ACCESSIBLE_TINY_V1` only if:

- every TRUE_ENDPOINT seed has accuracy >= 0.90;
- mean TRUE_ENDPOINT accuracy >= 0.95;
- every SHUFFLED_ENDPOINT seed lies in `[0.45, 0.55]` accuracy;
- treatment improvement over paired control is >= 0.35 for every seed.

If treatment fails while control remains chance, status is `TINY_LATE_FUSION_ADDRESSABILITY_NOT_ESTABLISHED` and the next step is to inspect compression/fusion placement under a separately preregistered treatment — not to tune this run.

If control exceeds the allowed chance band, status is `SYNTHETIC_FIREWALL_INVALID` and no architecture conclusion is permitted.

## Authority

PASS does not authorize:

- Truth139 training;
- FIT8 training;
- a claim that reference TESSA preserves endpoint addressability at 1024->384 scale;
- a claim that late fusion is optimal;
- product activation or model promotion.

It only removes one basic mechanism-level concern and permits the audit to proceed to more realistic scale/structural probes.
