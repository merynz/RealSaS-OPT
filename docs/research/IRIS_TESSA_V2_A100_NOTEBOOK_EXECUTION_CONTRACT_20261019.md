# IRIS / TESSA V2 — A100 Run-All Notebook Execution Contract

**Date:** 2026-10-19  
**Branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Authority:** research execution contract only  
**Scientific state at authoring:** no direct-surface artifact executed; no TESSA V2 optimizer step executed; no TESSA V2 scientific PASS.

## 1. Purpose

The next A100 notebook is an execution vehicle for already reviewed source and preregistered courts. It must not invent architecture, thresholds, truth, resolution, or optimizer policy after runtime evidence is observed.

The notebook answers two ordered questions:

1. Is the exact completed V8 carrier capable of producing an admissible high-resolution direct IRIS surface on all Fit8 subjects under the selected T512→sparse-T1024 decoder?
2. If and only if that A-arm is admissible, can full-reference TESSA V2 learn a free-running structuralization that reduces face count while preserving the preregistered source-geometry/mask/topology court?

A partial TESSA A/B PASS does **not** promote TESSA to mandatory product authority and does **not** mint canonical M.

## 2. Source work that must be complete before notebook delivery

The notebook may be delivered only from a source head where the dedicated V2 gate and the broad architecture/model contracts are green.

Required source/preregistration stack:

- exact V8→raw-field sparse-surface adapter;
- fixed full-resolution `T512→sparse T1024` scientific policy with no CLI low-resolution escape;
- exact read-only V8 / prepared-V4 / Stage01 input binding;
- direct-surface A100 engineering preflight;
- direct-surface 8/8 materializer;
- independent surface-geometry court;
- exact Stage01 8-view silhouette court using the existing conservative SAT raster semantics;
- TESSA V2 global-sequence reconstruction from sealed V4 topology windows;
- 4608-context / 512-scored-token training windows that actually exercise causal global history beyond the 4096 exact local window;
- strict cached incremental decoder with parity against prefix-recompute autoregressive reference behavior;
- model-evidence-only TESSA corpus builder: sealed V8 scene planes + direct-surface witness + sealed V8 H/mechanical evidence; teacher topology target only;
- locked machine-readable TESSA V2 preregistration;
- full-reference A100 forward/backward/Adam-state engineering preflight with zero optimizer steps;
- fail-closed pretraining admission court;
- locked 2400-step resume-safe trainer;
- strict all-eight-subject free-running evaluator with no teacher prefix;
- preregistered partial A/B court.

Green source contracts prove only that these paths are internally consistent. They do not prove runtime scientific quality.

## 3. Runtime gates that necessarily remain for the A100 notebook

These cannot be honestly closed on the source/CI runner and therefore occur in notebook execution, in this order:

```text
exact Drive authority resolution
    ↓
one-time local byte copy + SHA verification
    ↓
exact source checkout at notebook-pinned green SHA
    ↓
A100 / host RAM / disk engineering admission
    ↓
exact V8 direct-surface field-query preflight
    ↓
8/8 full T512→sparse-T1024 direct materialization
    ↓
A-arm geometry + 64-view source-mask courts
    ↓
TESSA V2 corpus materialization from A-arm witness
    ↓
full-reference TESSA V2 BF16 forward/backward preflight, optimizer steps = 0
    ↓
pretraining admission court
    ├─ BLOCK → no optimizer step
    └─ PASS
         ↓
locked 2400-step TESSA V2 training
         ↓
final-step strict incremental free-run on all 8 subjects
         ↓
preregistered partial A/B court
```

There is no path from a failed admission gate to training by lowering surface resolution, shrinking TESSA, relaxing thresholds, or substituting teacher-forced loss.

## 4. Exact immutable input authority

### Completed V8

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

### Prepared Fit8 V4

- preparation source `7029ee3f48787aa9f7e27eecef89155eb9368a96`;
- manifest SHA256 `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`;
- preparation receipt SHA256 `b3d8e72f980bdb194319e6c128257a20b2d46a71c7b86d1e2ed9ea6d1e16c93f`;
- eight shard IDs/hashes are fixed by `canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`.

### Stage01 source-observation authority

- `FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.zip`;
- SHA256 `9d9c9cbf63c979326cd2f39f5435006b81dddb201028a34ebb4cf10e1b434936`;
- exact admitted 8-view masks/cameras and raw source assets.

Stage01 does not contain a pre-materialized depth or render-normal truth court. The notebook must not fabricate one.

## 5. Direct A-arm policy

The direct baseline is exact completed V8 only:

```text
sealed V8 scene core
  → raw learned field
  → SparseRegularTetraPolicyV5
  → T512 coarse / sparse T1024 refinement
  → model surface candidate
```

No V3 migration, TESSA, extraction delta, teacher topology, or uncalibrated uncertainty enters this baseline.

A-arm training admission thresholds are preregistered in `canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json`. They include finite geometry, zero non-manifold/degenerate debt, bounded bidirectional source-surface distance, and source-mask IoU/gap requirements.

## 6. TESSA V2 fit policy

Reference architecture remains fixed:

- width 1024;
- 16 heads;
- 24 decoder layers;
- FFN 4096;
- 512 scene + 512 surface/mechanics memory tokens;
- 1024 coordinate bins;
- 4096 exact local context;
- causal 256-token global-history blocks;
- full-capacity training only.

Training corpus conditioning is model-produced evidence only:

- sealed V8 scene planes;
- 8192 deterministic area-weighted direct-surface witness samples per subject;
- direct-surface implicit normals;
- sealed V8 raw H at witness points;
- V8-trained stretch/tangent/bend/rigidity evidence.

Teacher topology is target-only. Teacher rig/skin is not conditioning. Contact persistence remains validity-false/abstain. Uncalibrated uncertainty is excluded.

Locked optimization:

- AdamW;
- peak LR `1e-4`, warmup 120, cosine to `1e-5`;
- 2400 optimizer steps;
- round-robin eight subjects = exactly 300 optimizer steps per subject;
- deterministic cyclic topology windows;
- BF16 autocast, FP32 parameters/optimizer state;
- gradient clip 1.0;
- checkpoint every 200 steps;
- telemetry every 10 steps.

Teacher-forced loss is telemetry, never the scientific PASS court.

## 7. Free-running and first A/B court

Only the final locked step is evaluated for the preregistered first court:

- greedy generation;
- strict grammar mask;
- no teacher prefix;
- cached incremental decoding with causal global-history summaries;
- EOS required for all eight subjects;
- maximum 131072 generated tokens.

First A/B court scope is deliberately limited to what is currently source-defensible:

- source-surface geometry;
- exact 8-view Stage01 source masks;
- decoded topology validity;
- face economy.

Preregistered policy requires at least 25% face reduction and tightly bounded geometry/mask regression with no new non-manifold edges, degenerate faces, open boundary chains, or geometric component-count change.

Possible PASS status:

`SCIENTIFIC_STRUCTURALIZATION_CANDIDATE__PRODUCT_PROMOTION_INCOMPLETE`

Even under PASS:

```text
TESSA mandatory = false
Compiler qualified = false
canonical M authorized = false
product promotion complete = false
unseen generalization claimed = false
```

## 8. Courts explicitly outside this first notebook result

The first notebook cannot honestly close:

- source-defensible depth court;
- render-normal court;
- full Compiler static qualification;
- Compiler support-binding/mechanics qualification;
- G3 authored-motion qualification;
- G5 product/multiview qualification.

Those are required before TESSA can be promoted from research structuralization candidate to product pipeline requirement.

## 9. Notebook I/O discipline

The notebook follows the V9 FastIO lesson:

1. mount Drive once;
2. resolve exact authority paths compactly;
3. copy large immutable inputs from Drive to local `/content` exactly once while hashing;
4. perform all repeated reads/hashes from local storage;
5. write checkpoints/evidence locally first and mirror sealed checkpoints/receipts to Drive at explicit boundaries;
6. resume only from hash/lineage-bound checkpoints.

Repeated hashing directly through Drive FUSE in the training hot path is forbidden.

## 10. Failure semantics

Every runtime gate is fail-closed.

A failure must leave an explicit receipt or visible exception naming the failed authority/gate. The notebook must not silently:

- reduce T512/T1024 resolution;
- shrink TESSA architecture;
- relax preregistered thresholds;
- change teacher truth;
- expose teacher topology/rig/skin to inference conditioning;
- resume from a checkpoint with different source/prereg/corpus/readiness lineage;
- convert a partial A/B PASS into product/Compiler/canonical-M authority.
