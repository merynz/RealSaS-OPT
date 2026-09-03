# RealSaS — R6 Arachne One-Family U0/U1 Result — 2026-09-03

**Status:** `PASS / ONE_FAMILY_U0_U1_CLOSED`

## Authority

Preregistration:
`canonical/R6_ARACHNE_ORACLE_SUBSTRATE_PREREG_20260903.md`

Workflow run: `33778934035`  
Job: `100727539409`  
Runner region: `eastus2`

Witness: `branch_blend_4`  
Seed: `20260922`

Shipping Codec hash:
`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

Shipping Arachne hash:
`ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`

## U0_REFERENCE_FULL_SURFACE

Surface rows: `160`.

Compiler-qualified oracle G:
- joint count `4`;
- deform root count `1`;
- lineage `14732f1ccff3336e07f96a1b821cd02840378ed148e853ef84b5cd034106a144`.

### Shipping Codec A0

`PASS`

- pass step `704`;
- stable PASS checks `3`;
- raw row-L1 p95 `0.0390706882`;
- qualified row-L1 p95 `0.0390706770`;
- raw deformation ratio `0.0158581547`;
- qualified deformation ratio `0.0158581492`;
- Compiler total correction L1 `5.6889257e-06`.

### Shipping Arachne A1

`PASS`

- pass step `320`;
- stable PASS checks `3`;
- raw row-L1 p95 `0.0799687132`;
- qualified row-L1 p95 `0.0799687132`;
- raw deformation ratio `0.0205726642`;
- qualified deformation ratio `0.0205726624`;
- latent p95 error `0.1286491454`;
- mean latent uncertainty `0.0769450143`;
- Compiler total correction L1 `6.0896855e-06`.

Raw and qualified W independently satisfy the frozen acceptance bands; Compiler repair does not explain PASS.

## U1_OBSERVATION_ORACLE_SUBSTRATE

Surface rows: `159`.

Substrate telemetry:
- U0 full nodes `160`;
- U1 admitted nodes `159`;
- visible fraction `0.99375`;
- support views min/mean/max `2 / 3.61006 / 4`;
- DTB-ND1 normal coverage `69/159 = 0.43396`;
- local relation count `0`.

Compiler-qualified oracle G:
- joint count `4`;
- deform root count `1`;
- lineage `0c47b8c3b104f3e6d1a27a091c32fb106cb31a96edb05637fb96cc10a4b027ae`.

### Shipping Codec A0

`PASS`

- pass step `608`;
- stable PASS checks `3`;
- raw row-L1 p95 `0.0189781543`;
- qualified row-L1 p95 `0.0189781580`;
- raw deformation ratio `0.0073177908`;
- qualified deformation ratio `0.0073177796`;
- Compiler total correction L1 `6.1340397e-06`.

### Shipping Arachne A1

`PASS`

- pass step `512`;
- stable PASS checks `3`;
- raw row-L1 p95 `0.0577549525`;
- qualified row-L1 p95 `0.0577549562`;
- raw deformation ratio `0.0314872973`;
- qualified deformation ratio `0.0314872898`;
- latent p95 error `0.1323413253`;
- mean latent uncertainty `0.0456352197`;
- Compiler total correction L1 `6.3367770e-06`.

Again raw and qualified W independently satisfy the acceptance bands.

## Causal verdict

`U0_CODEC_A0 = PASS`  
`U0_ARACHNE_A1 = PASS`  
`U1_CODEC_A0 = PASS`  
`U1_ARACHNE_A1 = PASS`

Therefore the one-family result supports:

`ARACHNE_APPARATUS_CAPACITY_ON_REFERENCE_RICH_SURFACE = PASS`

and

`ARACHNE_APPARATUS_CAPACITY_ON_CURRENT_OBSERVATION_ORACLE_SUBSTRATE = PASS`.

The result does not establish predicted-IRIS sufficiency; U2 was not run.

The stronger material-self-occlusion claim remains unproven because the current capsule shell U1 retains `159/160` full-surface samples. This limitation is preserved rather than hidden.

## Authorization

The separately preregistered heterogeneous U1 Arachne rung is authorized and is the next authority:

`canonical/R6_ARACHNE_ORACLE_SUBSTRATE_HETEROGENEOUS_PREREG_20260903.md`.
