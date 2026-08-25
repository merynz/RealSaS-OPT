# RealSaS IRIS P-V5 R256 — 36fb × Two-Style Sufficiency Result — 2026-08-25

**Decision:** `PASS`  
**Status:** `P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_PASS`

Frozen asset: `asset_36fb02305846592b1ecdf3d4`  
Frozen styles: `cel_clean`, `ink_cel`  
Fresh model: yes  
Architecture change: no  
PatchMatch: no

## Selected authority

Selected preregistered checkpoint: `TAIL_2048`  
Total optimizer steps: `4096`

- aggregate P95: `0.002595278900116682`
- worst-cell P95: `0.0025984761072322723`
- pass count: `2/2`
- threshold: `0.005` per cell

Per-cell:
- `36fb / cel_clean`: `0.002594304538797585` — PASS
- `36fb / ink_cel`: `0.0025984761072322723` — PASS

Drive run-complete SHA-256: `bbcb57a64b3326007f2b17ae3e4249890d29d3650773ae17f792521f22609c97`  
Drive decision SHA-256: `77abd5649c14fcbbd27eaa76a0e1a4485390c8c85be05414099e4d84063dbf94`

## Causal interpretation

This falsifies the hypothesis that `36fb` is intrinsically non-extractable under the current P-V5 R256 representation. The same asset that remained the dominant blocker in the shared 8×2 run fits both frozen styles with large margin when trained alone from a fresh model.

Combined with the preregistered continuation result — where all 16 shared cells improved simultaneously and the worst-cell trajectory remained descending — the strongest current explanation is shared optimization/schedule burden, not a localized geometric-information failure and not a demonstrated hard shared-capacity wall.

## Consequence

Authorize a fresh 8×2 V2 certification with:
- unchanged model and data membership;
- unchanged P-only objective;
- MAIN `2048 @ 3e-4`;
- continuous TAIL `4096 @ 3e-5` with fresh moments at TAIL start;
- resumable checkpoints containing optimizer/scaler/RNG state.

Unseen-family remains CLOSED until V2 certification passes.
