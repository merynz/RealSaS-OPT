# TESSA Typed Endpoint-Binding Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `SYNTHETIC_TESSA_ENDPOINT_BINDING_PREREG_V1.md`  
**Workflow:** `iris-tessa-v3-endpoint-binding-v1`  
**Run:** `40222240886`  
**Status:** `ENDPOINT_BOUND_LATE_SUPERIOR_TINY_V1`

## Question

When a presentation relation refers to two surface loci, should TESSA be expected to rediscover the endpoint-to-surface join after independent source compression, or should typed endpoint-local evidence be bound before compression?

The task used 16 shuffled surface loci with independent random payload bits. A pair token supplied endpoint coordinates and the target was the XOR of the two endpoint payloads. This removes semantic/category shortcuts and makes exact endpoint access necessary.

## Result

Five preregistered seeds:

| Arm | Mean accuracy | Minimum accuracy |
|---|---:|---:|
| CURRENT_LATE_UNBOUND | 0.503223 | 0.485596 |
| GENERIC_JOINT_UNBOUND | 0.507373 | 0.485596 |
| ENDPOINT_BOUND_LATE | **1.000000** | **1.000000** |
| RAW_BOUND_POSITIVE_CONTROL | **1.000000** | **1.000000** |
| BOUND_RANDOM_LABEL_FIREWALL | 0.506396 | 0.495117 |

Paired endpoint-bound minus current-late gains:

`+0.499268, +0.492920, +0.491455, +0.514404, +0.485840`

Paired endpoint-bound minus generic-joint gains:

`+0.499268, +0.498291, +0.465332, +0.514404, +0.485840`

The deterministic binder has **0 trainable parameters**. All preregistered gates passed: firewall, positive control, endpoint-bound accessibility, superiority versus current late compression and superiority versus generic joint compression.

## Interpretation

For an exact endpoint-dependent relation, adding generic cross-stream attention is not an adequate substitute for preserving the relation's addressability. The controlled failure is an information-contract problem rather than evidence that TESSA needs more capacity.

The preferred real treatment is therefore:

1. preserve typed endpoint refs in presentation evidence;
2. convert endpoint coordinates from the TESSA domain into the IRIS query domain explicitly;
3. evaluate `Q(Z, x_a)` and `Q(Z, x_b)` (and only justified endpoint-local mechanics/evidence);
4. attach those endpoint-local payloads to the presentation relation **before** TESSA presentation compression;
5. keep TESSA proposal-only and Compiler-authoritative.

This result does not prove that generic joint fusion is never useful for other non-addressable/global presentation evidence. It specifically establishes the value of explicit binding for endpoint relations.

## Authority

This court does not authorize a TESSA product rewrite, Truth139 training, FIT8 training or product promotion.
