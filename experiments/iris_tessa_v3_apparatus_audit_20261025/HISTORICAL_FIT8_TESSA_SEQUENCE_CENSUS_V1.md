# Historical FIT8 TESSA Sequence Census Reconciliation V1

**Audit date:** 2026-10-25  
**Evidence date:** 2026-10-22  
**Authority:** historical corpus evidence only; no V3/reference qualification.

## Source

Read-only Drive artifact:

- folder: `fit8_teacher_census_v1_20261022`
- result file ID: `1ACF2isoqSq6brGfeMW3nestrc5aY0FZY`
- log file ID: `177VaAoLwDVtCtV-NLa3W9s6bSlIrgDBt`
- experiment: `STRUCTURAL-EVIDENCE-A-B-FIT8-TEACHER-CENSUS-V1`
- corpus manifest SHA256: `0800e42fbc479320c4f8880db4e4b74bf4756eb72380b4f597b35102e44d5663`
- evaluator source SHA: `ff7be648b9dce421936e2ce6662dc09a1e9ab6e3`
- training execution source SHA: `eb11004e4d8c7fba6f6ba95fa8eacaa0c24fcb3d`
- presentation manifest SHA256: `21434efbfd409504d11bd883f35041089dbe7329e829c37446eb21c59b19b9f6`
- artifact explicitly reports `qualification_pass_claimed=false`.

TESSA V3 intentionally retains the V2 autoregressive topology vocabulary/decoder lineage, so these serialized teacher lengths are relevant historical evidence for current V3 scale planning. They are not treated as a fresh regeneration on the current audit SHA.

## Teacher serialized lengths

The artifact reports evaluated valid tokens over all teacher windows:

| Subject | Valid tokens | Coordinate tokens | Structural tokens | Census windows |
|---|---:|---:|---:|---:|
| Snake | 8,192 | 7,704 | 488 | 16 |
| Bat | 7,262 | 6,522 | 740 | 15 |
| T-Rex | 8,678 | 8,214 | 464 | 17 |
| Horse | 26,222 | 19,608 | 6,614 | 52 |
| Spider | 11,893 | 11,304 | 589 | 24 |
| GreenBlob | 7,426 | 7,002 | 424 | 15 |
| Shark | 3,057 | 2,892 | 165 | 6 |
| George | 53,516 | 47,190 | 6,326 | 105 |

## Consequence for current product decode cap

Current product Shape allows at most `131072` generated topology tokens.

All eight historical FIT8 serialized teacher sequences are below this cap. The largest observed subject is George at `53,516`, approximately 40.8% of the cap.

Therefore:

> the arithmetic worst-case mismatch between `max_faces=32768` and `maximum_new_tokens=131072` is a real declared-scale ceiling, but it is **not currently supported as a material FIT8 blocker** by the available eight-subject teacher corpus.

Do not increase product decode budget or redesign topology tokenization merely because the theoretical max-face serialization can exceed the cap.

## What remains open

This census does **not** close local-attention adequacy.

Current TESSA reference self-attention has a nominal 4096 local window with chunk-end direct-history semantics of approximately 3841..4096 tokens per layer. Sequence length alone does not reveal whether useful dependencies exceed that span.

Still required before declaring the sequence apparatus ready:

1. derive or measure dependency-span classes from teacher serialization (component/chart/restart/closure/repeated-structure relations);
2. report their distribution by morphology;
3. stratify current structural errors by dependency span if an executable reference checkpoint is used;
4. only then preregister a window/position/tokenization treatment if long-range dependencies are actually material.

## Authority

This historical evidence partially closes G2 only for **FIT8 sequence-length/decode-budget materiality**. It does not authorize:

- changing TESSA attention/window/tokenization;
- Truth139/FIT8 training;
- V3 qualification;
- ModelRelease or product promotion.
