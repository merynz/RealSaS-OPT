# V2 source recovery V2 reconstruction note

Parent scientific authority: `08620737ce526b0c0cc6c551f53910e34a1107fe`.

Recovery V1 failed fail-closed and was not tuned. Recovery V2 is a **new source authority justified by independent pre-existing source evidence**, not by selecting a parameter to match V1's failed historical metrics.

## Independent source evidence

ChatGPT Library contains `realsas_gfdr_v2.py`, file id `file_00000000067c824684daaed9c0e7bd8d`, created `2026-08-17T04:43:10.402819+00:00`, before the V2 global-solver experiments and before Recovery V1.

Exact evidence source SHA-256:

`954302fa7962841179521a49e8c989e44535138f0d733f7cd4d76746062593b0`

Its frozen GFDR-V2 surface-graph rule defines:

- `GFDRV2Config.k_neighbors = 12`;
- nearest-neighbor distances excluding self;
- per-point local scale = median of those k-neighbor distances;
- zero/degenerate local scales replaced by the median positive local scale, with `1.0` fallback.

The corrupted historical V2 source suffix independently preserves the tail `[:,:k];return np.median(nn,axis=1)`, so the pre-existing GFDR rule supplies a concrete source-backed recovery hypothesis for the missing `local_scale` helper.

## V1 -> V2 source delta

Recovery V2 changes **only** `local_scale` relative to frozen Recovery V1:

- V1: guessed `k=8`, median eight nearest non-self distances, no degenerate fallback.
- V2: exact pre-existing GFDR semantics above with `k=12` and positive-scale fallback.

Frozen unified-diff SHA-256:

`8fc54ca574dcaf0b103ccac9c0a2073aca9af8781ebbb60be2f83970855e9a0e`

No other scientific logic, threshold, ordering, factor, graph rule, candidate rule, evaluator rule, or checkpoint changed.

## Exact source transport

Authoritative Recovery V2 decoded source SHA-256:

`0747bf3754443053fa5a992f03df4cacddc95f75969572fe4de11253f1e441b4`

Because a pre-freeze manual one-file GitHub transfer introduced a transcription typo, that non-authoritative copy was deleted before persistence/prereg. The exact source is stored in GitHub as eight base64 parts and verified in GitHub Actions by concat/decode SHA guards, compile, and V1->V2 diff guard.

GitHub CI verification run: `32307397649`  
Verified decoded-source artifact: `9385179953`  
Base64 stream SHA-256: `e1e3712f273f860102631892fc25445548821ebae9ea8195f78ba5f04dd24ec4`

The CI artifact was downloaded and compared byte-for-byte with the local source; `cmp` passed.

## Fail-closed rule

Recovery V2 gets one frozen historical parity execution after the three-store persistence gate and a new preregistration. Any Gate A/B/C/D mismatch invalidates V2. No post-outcome tuning is allowed. Corrected observable-geometry 0/48 remains forbidden unless Recovery V2 passes all source-recovery gates.

`sealed21 = CLOSED`  
`external10 = CLOSED`
