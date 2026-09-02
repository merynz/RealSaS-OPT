# Arachne A1 — first-family CPU prereg V1

Status: `FROZEN_BEFORE_A1_OPTIMIZER`
Canonical repository: `merynz/RealSaS-OPT`
Fit branch: `fit/single-family-mage-v1-20260902`

## Scope

First executable FIT witness only. No held-out/generalization claim and no Mage-specific semantic/topology rule is permitted.

## Staged conditioning authority

This rung is explicitly `ORACLE_S__CURRENT_QUALIFIED_G` and exists to test the Arachne learned mapping after the SkinFieldCodec representation ceiling has passed.

Surface sample is generated generically from exact Master authority:
1. keep vertices with positive current mechanical-core skin mass;
2. keep vertices observed by at least one of the exact eight `raster_authority.npz` views, where observation is derived only from visible triangle -> vertex incidence;
3. deterministic farthest-point sample 384 vertices, seeded by the vertex farthest from the observed-set bbox center, with deterministic index tie behavior;
4. use canonical position and exact vertex normal;
5. support-view bits come from the eight raster authorities;
6. local relation diagnostic uses deterministic symmetric kNN-8 on sampled normalized positions, with score `exp(-distance / median_knn_distance)`; this is marked oracle-S relation evidence and is not a product/source-mesh authority.

Current G uses the generic mechanical-core truth rebind: skin-supported controls plus required graph bridges, assembly-only root excluded. Geppetto/Compiler ownership of product canonical IDs remains unchanged.

Joint support feature follows the existing Geppetto supervision convention: nearest-8 admitted S nodes, not dense skin truth.

## Frozen loss and model contract

- current `ArachneCandidateV2` default config;
- exact same just-qualified `SkinFieldCodecV1` decoder object, frozen;
- teacher latent from codec encoder;
- latent heteroscedastic NLL + dense field reconstruction + verified-LBS deformation consequence;
- no source-bone name or semantic slot input.

## Acceptance

At or before the authorized CPU ladder boundary:
- row-L1 mean <= 0.05;
- row-L1 p95 <= 0.15;
- verified-LBS deformation RMS <= 0.005;
- max simplex residual <= 1e-5;
- negative weight count = 0;
- latent NLL finite and lower than its step-0 value;
- uncertainty finite.

Ladder checkpoints: 0, 128, 256, 512, 1024. Extension beyond 1024 requires recording the 1024 result first; criteria must not be relaxed after seeing results.

A1 PASS does not authorize COMPLETE_E2E_FIT until IRIS and the learned-output Compiler/proof/runtime route also pass.
