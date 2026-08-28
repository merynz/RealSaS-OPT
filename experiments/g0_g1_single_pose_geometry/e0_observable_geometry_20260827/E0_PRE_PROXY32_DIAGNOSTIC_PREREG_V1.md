# RealSaS E0 — Pre-Proxy32 Diagnostic Prereg V1

**Date:** 2026-08-28  
**Status:** `FROZEN_BEFORE_DIAGNOSTIC_OUTCOMES__FIT_ONLY__PROXY32_DOWNSTREAM_CLOSED`

This diagnostic suite is interpretation-only. It cannot change the already-frozen Proxy32 qualification margins, metric roles, D0/D1/D2 definitions, MUTUAL_P003, or qualification population. It uses only frozen FIT train/selection assets already opened by the downstream-proxy training apparatus.

## Population

- Diagnostic train: first 64 assets in the frozen train order after the already-frozen common Arachne target-support eligibility (`min(D0_valid,D1_valid,D2_valid)>0`).
- Diagnostic evaluation: all frozen 59 FIT selection assets.
- Exact IDs: `E0_PRE_PROXY32_DIAGNOSTIC_SPLIT_V1.json`.
- Proxy32 downstream metrics are not accessed.

## D-A — implicit-normal decodability probe

Question: how much exact geometric normal information is already decodable from the current 36D consumer slot?

For each D1 anchor, exact target N is reconstructed evaluator-only from master mesh geometric vertex normals and the frozen D1 carrier triangle/barycentric authority. Consumer inputs never receive this authority.

Two matched pointwise probes:

```text
P3   : canonical P xyz only -> MLP 64 -> 64 -> N
X36  : current full 36D slot -> same MLP -> N
```

- seed 1862; identical architecture/optimizer budget;
- train on diagnostic train64; evaluate on selection59;
- oriented cosine loss;
- report oriented angular median/p90/p95 and sign-invariant angular median/p90/p95.

Interpretation is descriptive. If X36 materially exceeds P3, the current substrate contains decodable implicit orientation information (support/raster/depth provenance). B3 must then be interpreted as marginal value of explicit N beyond that implicit channel, not value relative to zero normal information.

## D-B — fixed-budget D0 allocation diagnostic

Question: is the D0->D1 Arachne improvement mainly a 512-slot allocation effect?

Construct only on the diagnostic 64+59 assets:

```text
D0@512  = existing frozen area-uniform full-mesh pack
D0@2048 = diagnostic-only area-uniform full-mesh sample with identical feature/target semantics
D1@512  = existing frozen visible-union FPS pack
```

No official E0 pack or qualification artifact is modified.

### Geometric coverage diagnostic

For each selection D1@512 anchor, report nearest-P distance to D0@512 and D0@2048; aggregate median, p95 and coverage at canonical distances 0.01/0.02/0.05.

### Arachne diagnostic

Train matched Arachne proxies on D0@512, D0@2048 and D1@512 using the same architecture and original 6-epoch optimizer schedule, but only diagnostic train64. Select all three checkpoints by CE on the same common D1@512 selection carrier and evaluate all three on that same carrier. This makes the comparison about training-sample allocation rather than different evaluation surfaces.

### Geppetto diagnostic

Train matched Geppetto isolation proxies on the three input budgets with the same model and 8-epoch optimizer schedule, using the complete point set for each diagnostic arm (512 / 2048 / 512) and common GT skeleton targets. This is explicitly a diagnostic input-budget experiment, not the official 256-point-subsampled Geppetto gate.

Interpretation:
- D0@2048 approaches D1@512: visible-union FPS is mainly more sample-efficient.
- D0@2048 materially exceeds D1@512: hidden/full-mesh surface carries useful information that the 512-slot D0 reference could not express.

Neither outcome changes frozen qualification margins.

## Firewalls

- Proxy32 downstream evaluation remains closed.
- DEV32 closed.
- no product-domain filtering.
- no IRIS learner.
- no official checkpoint replacement.
- diagnostics cannot rescue or veto the future frozen Proxy32 qualification result.
