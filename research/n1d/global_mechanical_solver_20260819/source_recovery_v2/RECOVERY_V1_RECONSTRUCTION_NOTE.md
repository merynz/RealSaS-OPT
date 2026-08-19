# V2 source recovery V1 reconstruction note

Parent scientific authority: `08620737ce526b0c0cc6c551f53910e34a1107fe`.

Recovered source SHA-256: `578fdddcd17d922603fb463ee1f2cae5b88e0e2c96a850900716abef5e7588be`.

## Persisted exact material reused

- Intact prefix/suffix of the persisted but corrupted V2 source bundle.
- Frozen canonical N1D checkpoint SHA-256 `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`.
- Exact Problem-A visual-hull/DIS front-door behavior recovered from the preserved Stage-A recovery lineage.
- Exact F16 descriptor refinement and pairwise triangulation mathematics from `adaptive_hypothesis_retention_v1.py`.

## Two reconstructed missing helpers frozen before parity execution

1. `bilinear_flow_samples`: raw-pixel bilinear sampling of the DIS flow field. This is the unit-consistent form required by V2 `pair_R`, whose candidate displacement is also in native pixels. Synthetic algebraic equivalence to the preserved Stage-A bilinear sampler is exact.
2. `local_scale(P,k=8)`: median distance to the 8 nearest non-self surface points. The surviving corrupted suffix contains `[:,:k];return np.median(nn,axis=1)`; V1 freezes `k=8` before any historical parity result is computed.

## Fail-closed rule

This is a recovery candidate only. No post-outcome tuning is allowed. Any frozen historical parity miss invalidates V1. The corrected observable-geometry 0/48 audit is forbidden unless V1 passes.
