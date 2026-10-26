# RealSaS — TESSA V3 Fusion Placement Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** tiny synthetic CPU learned mechanism court; no product/ModelRelease authority.

## Authorization

The clean V2 endpoint-addressability probe established its task/firewall and positive control, but current independent compression did not meet the preregistered accessibility threshold (`mean=0.75537109375`, seed results `0.7393 / 0.9104 / 0.6165`). This authorizes a treatment study of compression/fusion placement. It does **not** authorize a reference-TESSA architecture change.

## Scientific question

Holding source information, output token budget, downstream reader, data, optimizer, training budget and seeds fixed, does allowing **surface and presentation sources to interact during compression** improve endpoint-addressable information preservation relative to current independent typed-stream compression?

This court tests **fusion placement**, not topology generation and not reference-scale capacity.

## Fixed synthetic task

Use the exact clean V2 task:

- 16 surface loci on a fixed 4x4 normalized coordinate lattice;
- independent fair payload bits `p_i in {-1,+1}`;
- random surface-token permutation every batch;
- uniformly random target locus `j`;
- presentation source contains `(x_j,y_j,1)`;
- target is payload `p_j`;
- scene source is four zero/nuisance tokens.

Surface source token: `[x, y, payload]`.  
Presentation source token: `[x_target, y_target, 1]`.

## Fixed tiny config

- `d_model=64`
- `n_heads=4`
- `dropout=0`
- `memory_source_width=32`
- `memory_refinement_layers=1`
- scene memory tokens `4`
- surface/presentation combined memory tokens `8`
- total conditioning budget `12`
- current-late output partition remains surface `6`, presentation `2`
- no topology decoder used.

## Shared downstream reader

All treatment arms use the **same two-hop reader class** from V2:

1. learned query reads the memory region designated presentation;
2. resulting query reads the memory region designated surface;
3. residual + LayerNorm + MLP predicts payload sign.

The reader does not see raw source tokens.

## Arms

### A — CURRENT_LATE_INDEPENDENT

Repository `TESSATriStreamMemoryV3` unchanged.

- scene compressed independently `raw -> 4`;
- surface compressed independently `raw 16 -> 6`;
- presentation compressed independently `raw 1 -> 2`;
- streams concatenated only after compression.

This is the current V3 mechanism.

### B — JOINT_TYPED_COMPRESSION

Scene remains unchanged and separately compressed to 4 tokens.

Surface and presentation each retain their own source projection into the same `memory_source_width=32`, with learned source-type embeddings. Their projected raw source tokens are concatenated **before** resampling.

A single shared multi-head cross-attention compressor uses eight learned output queries:

- six queries carry a learned surface-query type embedding;
- two queries carry a learned presentation-query type embedding;
- all eight queries may attend to the full concatenated typed surface+presentation source memory;
- one shared refinement stack follows;
- outputs retain the exact 6/2 partition expected by the same downstream reader.

Thus B changes only **where cross-stream interaction becomes possible**. Source information, output token count and reader semantics remain fixed.

### C — RAW_TWO_HOP_POSITIVE_CONTROL

Retain V2 raw positive control unchanged. It validates task/readout learnability and is not a product-architecture candidate.

### D — RANDOM_LABEL_FIREWALL

Run on the joint-compression architecture with byte-identical true-endpoint inputs but an independently generated fair random label absent from the inputs. This validates that joint compression does not create a synthetic leakage channel.

## Parameter accounting

Report trainable parameter count for A and B. No accuracy normalization by parameter count is performed. Interpretation is conservative:

- if B has **more** parameters and wins, fusion placement is not isolated strongly enough; result is `JOINT_FUSION_PROMISING_CAPACITY_CONFOUNDED` unless a later parameter-matched court confirms it;
- if B has <= A parameters and wins under the thresholds below, a capacity explanation cannot account for the gain in the simple direction.

No post-outcome width adjustment is allowed in V1.

## Seeds

Exactly `731, 947, 1217, 1597, 2017`.

Five seeds are used because V2 exposed large seed variance.

Within each seed:

- A and B use the same generated training/evaluation sample streams;
- A, B and their readers are initialized deterministically from the same seed but are not forced to share incompatible parameter tensors;
- C uses the same true-target sample stream;
- D uses the same input stream as B plus an independent sealed label RNG.

## Optimizer / budget

- pinned CPU torch profile;
- AdamW;
- LR `1e-3`;
- betas `(0.9, 0.999)`;
- weight decay `0`;
- batch `128`;
- exactly `1500` optimizer steps;
- BCE-with-logits;
- no schedule, clipping, early stopping, hyperparameter search or outcome-dependent tuning.

Evaluation: exactly `4096` fresh samples per seed/arm.

The increase from V2's 1000 to 1500 steps is preregistered before observing any V1 fusion-placement output and applies identically to A/B/C/D.

## Metrics

Per seed/arm:

- accuracy;
- BCE;
- predicted-positive fraction;
- target-positive fraction.

Aggregate:

- mean accuracy;
- minimum seed accuracy;
- standard deviation across seeds;
- paired per-seed B-A gain;
- mean paired B-A gain;
- parameter counts A/B.

## Preregistered interpretation

### Firewall

D must lie in `[0.45, 0.55]` accuracy for every seed. Otherwise: `FUSION_COURT_FIREWALL_INVALID` and no mechanism verdict.

### Positive control

C must achieve every seed >= `0.90` and mean >= `0.95`. Otherwise: `FUSION_COURT_READER_OR_TASK_INVALID`.

### Joint typed compression establishes endpoint accessibility

B must achieve:

- every seed >= `0.90`;
- mean >= `0.95`.

### Fusion-placement superiority

Only if firewall and positive control pass, call `JOINT_TYPED_COMPRESSION_SUPERIOR_TINY_V1` when all are true:

1. B satisfies the accessibility thresholds above;
2. B-A paired gain is positive in at least 4/5 seeds;
3. mean paired B-A accuracy gain >= `0.10`;
4. B trainable parameter count <= A trainable parameter count.

If 1–3 hold but B has more parameters: `JOINT_FUSION_PROMISING_CAPACITY_CONFOUNDED`.

If B improves but does not meet all thresholds: `FUSION_PLACEMENT_SIGNAL_INCONCLUSIVE_TINY_V1`.

If B does not improve meaningfully: `JOINT_COMPRESSION_NOT_SUPPORTED_TINY_V1`.

## What a positive result authorizes

A positive tiny result authorizes a **separately preregistered real-data / reference-scale fusion treatment**, not a production rewrite.

## What no outcome authorizes

No outcome authorizes:

- Truth139/FIT8 training;
- ModelRelease promotion;
- product activation;
- canonical topology authority;
- a claim that reference TESSA fails;
- replacing current V3 on main without a subsequent court.
