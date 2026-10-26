# TESSA Dependency-Span / Free-Running Mechanics Audit Result V1

**Run:** `40318317853`  
**Head SHA:** `1408da7b58d7aca30be87fbccfe5efa4771d9f37`  
**Artifact digest:** `sha256:0c7a48dfe3ffefabbd7bc8970ef7cc90596f4366b5e88cce2ddac8b4a4f14f89`  
**Status:** `TESSA_4096_WINDOW_RETAIN_PRETRUTH__NO_INCREASE_EVIDENCE_V1`

## Grammar / sequence stress

All procedural teacher sequences replayed through the strict streaming grammar and exceeded the current local window:

| case | faces | tokens | restarts | grammar |
|---|---:|---:|---:|---|
| long connected strip | 1500 | 4513 | 1 | PASS |
| restart-heavy | 520 | 5206 | 520 | PASS |
| mixed | 1060 | 4313 | 161 | PASS |

The adjacent serialization's local structural coordinate dependency is six tokens (the previous trailing-edge vertex triples). Grammar phase/count state is explicit deterministic state rather than learned-history responsibility.

With `local_attention_window=4096` and `query_chunk_size=256`, the minimum direct post-window historical span is `3841` tokens: about `640.17x` the serialization-local structural span.

The shipped incremental implementation also passed reference parity across the window boundary (`9` post-window prefixes; max absolute error `4.77e-07`).

## Decision

Retain local window `4096` as the conservative V3.1 pre-Truth baseline. There is no mechanism evidence to increase it.

This court does not authorize decreasing the window: learned long-range shape/style/topology correlations must be measured with authored/real truth before reducing capacity.
