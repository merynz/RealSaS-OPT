# RealSaS — TESSA V3 Typed Endpoint-Binding Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** tiny synthetic CPU information-contract court; no product/ModelRelease authority.

## Motivation

Two prior preregistered courts established:

1. current independent surface/presentation compression does not reliably preserve endpoint-addressable information on the tiny lookup task (`mean=0.7554` in clean V2);
2. allowing generic joint surface+presentation interaction before compression improves the mean by about `+0.1231` with fewer parameters, but remains seed-unstable and does not satisfy the preregistered accessibility gate.

The intended causal graphics evidence contract already requires typed pre-M endpoint references (`SceneLocusRef`, `LocusPairRef`, `RayHitRef`). Therefore the next question is not simply whether a larger/freer Transformer can rediscover a join. It is whether **deterministically binding typed relation endpoints to their referenced surface witnesses before learned compression** removes the avoidable addressability burden.

## Scientific question

For a pairwise relation whose two endpoint references are valid and resolvable, does deterministic pre-compression endpoint binding yield more stable recoverability than:

- current independent late fusion; and
- generic joint typed compression?

A positive result favors a Bridge/data-contract repair over a broad TESSA rewrite.

## Synthetic pair-relation task

Use 16 surface loci on the same fixed 4x4 normalized coordinate lattice as the earlier addressability courts.

Each locus independently receives one fair payload bit `p_i in {0,1}`. Surface-token order is independently randomly permuted every batch while coordinates remain attached.

For every sample choose two **distinct ordered** endpoint indices `(i,j)` uniformly.

Unbound presentation source contains only:

`[x_i, y_i, x_j, y_j, 1]`

Target:

`y = XOR(p_i, p_j)`.

This target requires recovering both referenced endpoint payloads and cannot be solved from endpoint coordinates alone because payloads are independently re-sampled every sample.

Scene source remains four zero/nuisance tokens.

## Deterministic endpoint binding

The bound arm performs a non-learned exact join before learned presentation compression:

1. resolve each presentation endpoint coordinate against the raw shuffled surface source;
2. require exactly one matching surface locus per endpoint;
3. append the two referenced payload values to the presentation source.

Bound presentation token:

`[x_i, y_i, x_j, y_j, p_i_signed, p_j_signed, 1]`.

The binder has **zero trainable parameters** and uses no target label. It exposes only source features already legally reachable through the typed endpoint references.

This intentionally changes the information contract. It is testing whether typed addressability should be resolved by the Bridge rather than rediscovered by TESSA.

## Fixed tiny config

- `d_model=64`
- `n_heads=4`
- `dropout=0`
- `memory_source_width=32`
- `memory_refinement_layers=1`
- scene memory tokens `4`
- surface memory tokens `6`
- presentation memory tokens `2`
- total conditioning budget `12`
- no topology decoder.

## Arms

### A — CURRENT_LATE_UNBOUND

Current `TESSATriStreamMemoryV3` semantics:

- surface source dim 3: `[x,y,payload_signed]`;
- presentation source dim 5: unbound pair endpoint coordinates;
- streams independently compressed before concatenation.

### B — GENERIC_JOINT_UNBOUND

The preregistered joint typed-compression treatment generalized to the pair presentation input:

- surface and unbound presentation source tokens may interact in one shared compressor before the 6/2 output partition;
- no deterministic endpoint join.

### C — ENDPOINT_BOUND_LATE

- surface stream is compressed independently exactly as in A;
- deterministic zero-parameter binder constructs the 7-D bound presentation token;
- bound presentation is independently compressed to two presentation memory tokens;
- concat/reader remains late as in current V3.

This isolates whether correct endpoint binding can repair the information contract without requiring generic early/joint fusion.

### D — RAW_BOUND_POSITIVE_CONTROL

A small MLP reads the deterministic bound presentation token directly and predicts XOR. No compressed surface/presentation memory. This validates task/target learnability.

### E — BOUND_RANDOM_LABEL_FIREWALL

Byte-identical C inputs, but target is an independently generated fair random label absent from all inputs.

## Shared downstream reader

A/B/C use the same `TwoHopReader` class from the prior clean courts:

1. learned query attends presentation memory;
2. resulting query attends surface memory;
3. residual+LayerNorm+MLP -> binary logit.

No arm-specific reader tuning.

## Seeds

Exactly `731, 947, 1217, 1597, 2017`.

## Optimizer / budget

- pinned CPU torch profile;
- AdamW, lr `1e-3`, betas `(0.9,0.999)`, weight decay 0;
- batch `128`;
- exactly `1500` steps;
- BCE-with-logits;
- no schedule, clipping, early stopping, hyperparameter search or outcome-dependent tuning;
- exactly `4096` fresh eval samples per seed/arm.

## Metrics

Per seed/arm:

- accuracy;
- BCE;
- predicted-positive fraction;
- target-positive fraction.

Aggregate:

- mean/min/std accuracy;
- paired C-A and C-B gains;
- trainable parameter counts;
- binder parameter count (must be 0).

## Gates

### Firewall

E must be in `[0.45,0.55]` for every seed. Otherwise `ENDPOINT_BINDING_FIREWALL_INVALID`.

### Positive control

D must achieve every seed >= `0.98` and mean >= `0.995`. Otherwise `ENDPOINT_BINDING_TASK_INVALID`.

### Bound endpoint accessibility

C must achieve:

- every seed >= `0.95`;
- mean >= `0.98`.

### Stable endpoint-binding superiority

Call `ENDPOINT_BOUND_LATE_SUPERIOR_TINY_V1` only when firewall/control pass and:

1. C satisfies the accessibility gate;
2. C beats A in at least 4/5 seeds with mean paired gain >= `0.10`;
3. C beats B in at least 4/5 seeds with mean paired gain >= `0.05`;
4. binder trainable parameter count is exactly 0.

This comparison intentionally does **not** require C total learned parameters <= A/B because the bound presentation input is wider by contract. Parameter counts are reported and the scientific claim is about information binding, not capacity-matched architecture superiority.

If C satisfies its own accessibility gate but paired superiority is weaker: `ENDPOINT_BINDING_ESTABLISHED_BUT_SUPERIORITY_INCONCLUSIVE_TINY_V1`.

If C misses its accessibility gate with valid controls: `ENDPOINT_BINDING_NOT_SUFFICIENT_TINY_V1`.

## What a positive result means

A positive result authorizes designing the real Bridge around explicit typed endpoint binding before TESSA presentation compression, followed by a separately preregistered real-data/reference-scale court.

It does **not** by itself authorize generic early fusion, a TESSA production rewrite, Truth139/FIT8 training, ModelRelease promotion or product activation.
