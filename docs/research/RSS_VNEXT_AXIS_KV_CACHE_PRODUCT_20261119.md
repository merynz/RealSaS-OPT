# AXIS V6: exact-weight streaming autoregressive decoder

The 2048 event product cap now holds a 190-joint reference rig plus
FSQ SkinTokens, but the original `generate_events` recomputed the full
Transformer prefix after every newly decoded token. This was not an
appropriate deployment algorithm for long auxiliary-chain rigs.

`models/rss_vnext/axis_causal_kv_cache_v1.py` uses the **identical trained
TransformerEncoderLayer projections and norms**, caching causal self-attention
K/V once per layer. Every new event computes one query against earlier keys
and itself. The shared AXIS inference still performs the identical
cross-attention over the original IRIS/TESSA scene memory and uses the same
heads, pointer grammar, descendant CHAIN endpoint mask and FSQ codebook.
No new weights or checkpoint format are introduced.

In an unpadded, batch-1, eval-mode sequence, every cached key is in the
newest token's causal past. Thus no triangular attention mask is needed
for that single query row. Training keeps the unchanged full-prefix
forward path; inference uses a streaming cache that does not materialize a
second independently trained causal rig model.

The new regression compares every prediction head and hidden state after
every prefix to the original full-prefix forward on the same model
and causal source events. It separately guards max-sequence lengths,
batch size and eval/no-grad authority.

Important boundaries:
- Floating-point tolerance/parity is verified by CI, not assumed.
- RAM/VRAM and speed remain to be measured on qualified 190+ joint real
  Morphology using pinned inference hardware. The cache replaces
  full-prefix recomputation; it is not a claim of measured FPS.
- Source-native rig/skin and control teacher admission are unchanged.
- Physics, Motion and Graphics follow the same compiled puppet contract.
