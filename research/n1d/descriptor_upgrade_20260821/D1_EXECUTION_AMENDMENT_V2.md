# D1 Execution Amendment V2 — 2026-08-21

This amendment does not change the frozen descriptor-repair ladder (`D0 -> D1 -> D2 -> D3 -> D4`). It corrects execution/preflight assumptions discovered before any D1 optimizer step.

## Corrections

1. **Family folder resolution is EPISODE_INDEX-authoritative.** Four-digit family IDs are zero-padded in Drive paths (e.g. family `9842` is stored at `family_09842/`). No runtime path may be inferred from the integer ID alone.
2. **Per-family train/holdout episode counts are split-authoritative.** Family `12781` has seven episodes and the canonical split records `5 train + 2 holdout`; therefore a global hard-coded `e00..e05` train set is forbidden. For each fit family, use the canonical `EPISODE_INDEX` order and `fit_train_episode_counts[fid]`, reserving the remaining canonical episodes as holdout.
3. **Matched legacy continuation control is required for causal attribution.** Comparing frozen N1D BEST directly against 24 additional D1 epochs conflates objective treatment with additional training. V2 therefore trains two branches from the exact same canonical N1C BEST warm-start under the exact same V2 data schedule, RNG schedule, architecture, optimizer, and checkpoint-selection rule:
   - `C1_LEGACY_CONTROL`: canonical N1D observation loss.
   - `D1_OBJECTIVE`: identical loss except the descriptor `Z_match` term is replaced by the frozen D1 objective.
   `D1 - C1` is the descriptor-objective treatment effect. Frozen canonical N1D BEST remains a current-reference D0 diagnostic only.

## Truth / scope

- No optimizer step occurred under the invalid V1 execution assumptions.
- `cal` remains untouched.
- `sealed21` remains CLOSED.
- `external10` remains CLOSED.
- D2/D3/D4 are not authorized or run by this amendment.

## Frozen authorities

- Canonical corpus index content SHA256: `907bca7323c8e7b3832ad276d44af9a5733e606b12c763a492775387f4afb765`
- Canonical N1D BEST SHA256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- Shared N1C BEST warm-start SHA256: `6b0528610ba14a60dc57f7218bb88ed9bd9401076d887da276dd3d8cc5e81e68`
- D1 objective source SHA256: `b84a3c3bf3cf2d29b43f2b0804b9e62fe92c495f78ed1451bc4cc68366cb2b19`
