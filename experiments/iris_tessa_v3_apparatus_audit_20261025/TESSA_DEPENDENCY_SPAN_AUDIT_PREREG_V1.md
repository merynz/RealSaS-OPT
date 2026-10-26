# TESSA Dependency-Span / Free-Running Mechanics Audit V1 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** architecture-search only.

## Questions

1. Does the strict TESSA V1-compatible serialization/grammar itself require learned autoregressive history anywhere near the current 4096-token local window?
2. Does the exact cached/windowed decoder remain numerically equivalent to the reference decoder across the window boundary?
3. Is there current mechanism evidence that requires increasing TESSA local history before authored causal truth exists?

## Courts

### A — serialization-local dependency census

Generate deterministic procedural teacher meshes including:

- a long connected triangle strip whose encoded sequence exceeds 4096 tokens;
- a restart-heavy disconnected-triangle mesh whose encoded sequence exceeds 4096 tokens;
- a mixed connected/restart topology.

Replay every token through `TESSAGrammarStateV2` and require exact completion and face/restart accounting.

The V1 adjacent-face serialization emits one new xyz triple per continuation face. Its trailing-edge coordinate evidence is constructionally contained in the previous two vertex triples, so the serialization-local structural coordinate span is six tokens. Grammar phase/count state is explicit deterministic state and is not delegated to learned history.

### B — shipped incremental/reference parity

Execute `assert_windowed_incremental_reference_parity_v2(device='cpu')` and require PASS with post-window prefixes included.

### C — window margin

For current `local_attention_window=4096`, `query_chunk_size=256`, compute the minimum direct historical token range available to a post-window query under chunk-end anchoring: `window - chunk + 1`.

Compare that lower bound against the six-token serialization-local structural span.

## Selection logic

- If grammar replay and incremental parity pass, and minimum post-window direct history is far above the serialization-local structural span, there is **no mechanism evidence to increase the local window** pre-Truth.
- This court does **not** authorize reducing 4096: learned long-range shape/style/topology correlations require authored/real data to measure.
- Therefore the conservative V3.1 pre-Truth choice is to retain 4096 unless this audit falsifies it.

## Non-claims

This court does not prove 4096 is globally optimal, does not replace learned dependency measurement on authored truth, and does not claim scientific topology accuracy of an untrained checkpoint.
