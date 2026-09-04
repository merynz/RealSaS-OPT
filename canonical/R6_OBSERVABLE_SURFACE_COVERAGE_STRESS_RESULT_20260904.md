# RealSaS — R6 Observable-Surface Coverage Stress — Result

**Date:** 2026-09-04  
**Status:** `PASS_CORE75__PASS_CORE60__SYNTHETIC_GSA_COVERAGE_TOLERANCE_ESTABLISHED__REAL_FAMILY_UNPROVEN`

## Authority

Preregistration: `canonical/R6_OBSERVABLE_SURFACE_COVERAGE_STRESS_PREREG_20260904.md`.

Workflow: `r6-observable-surface-coverage-stress-v1`  
Successful run: `33899456941`  
Job: `101109834745`  
Head commit: `15df5e71f06a05a0e8bb0d094663423a0cbd6fc8`  
Runner: `realsas-wsl-1660ti`  
Artifact: `9947091574`  
Artifact digest: `sha256:792bc8fc0f37e399834ec6ff02f77da3d37811895817fca05995866c98f2c59b`

The prior GitHub-hosted attempt `33899383621` failed before any job step started and produced no scientific output. It is infrastructure evidence only.

## Frozen scientific route

The stress was applied to `ObservationEvidenceIR` before compilation. The observed arm was:

`ObservationEvidenceIR -> RealSaS.GeometricSubstrateAssembler.current -> deterministic DTB-ND1 where qualified -> RiggingSurfaceIR S -> consumer`.

No post-GSA node deletion was used. No hidden-surface completion, source-mesh consumer input, source topology injection, learned visibility/occlusion, model-width change, threshold change or optimizer change was introduced after outputs.

Witness: `branch_blend_4`.

The natural eight-view oracle observation retained `159/160` shell samples before the preregistered contiguous deprivation masks. The stress therefore creates a materially stronger coverage contrast than the historical 99.375% U1 gate.

## GSA telemetry

### CORE_75

- U0 full samples: `160`
- naturally observable before stress: `159`
- stress-hidden groups: `39`
- GSA output surface nodes: `120`
- retained fraction vs U0: `0.750000`
- support-view min / mean / max: `2 / 3.625 / 4`
- DTB-ND1 normals: `33/120 = 0.275`
- observed local relations: `0`
- hidden-surface completion: `false`
- direct source-mesh consumer input: `false`
- GSA lineage: `3a139714b1376860d3cb6cdbf667677b792ef1db6e07e33e7b1c99e968f600a1`

### CORE_60

- U0 full samples: `160`
- naturally observable before stress: `159`
- stress-hidden groups: `63`
- GSA output surface nodes: `96`
- retained fraction vs U0: `0.600000`
- support-view min / mean / max: `2 / 3.5729166667 / 4`
- DTB-ND1 normals: `21/96 = 0.21875`
- observed local relations: `0`
- hidden-surface completion: `false`
- direct source-mesh consumer input: `false`
- GSA lineage: `f86d275709f00c6251aed7465664ff842a1eb6fe0a605f2113c1d437b59e6d96`

## Geppetto result

Frozen shipping config hash:
`6506e3764f7219b758d27c0e5d9dc7e7babc16e2616cd568b00da3dd789bbbf4`.

### U0_REFERENCE_FULL_SURFACE

- surface nodes: `160`
- pass step: `96`
- stable PASS checks: `3`
- generated controls: `4/4`
- final matched MAE: `0.0272199623`
- final matched p95: `0.0375987776`
- Compiler-qualified mechanical state: `roots=1, illegal_parents=0, unsupported=0`

### U1 -> GSA -> CORE_75

- surface nodes: `120`
- pass step: `96`
- stable PASS checks: `3`
- generated controls: `4/4`
- final matched MAE: `0.0156753529`
- final matched p95: `0.0220660195`
- Compiler-qualified mechanical state: `roots=1, illegal_parents=0, unsupported=0`

### U1 -> GSA -> CORE_60

- surface nodes: `96`
- pass step: `96`
- stable PASS checks: `3`
- generated controls: `4/4`
- final matched MAE: `0.0137343490`
- final matched p95: `0.0184208937`
- Compiler-qualified mechanical state: `roots=1, illegal_parents=0, unsupported=0`

Preregistered Geppetto verdict:
`STRONG_SYNTHETIC_GSA_GEPPETTO_COVERAGE_TOLERANCE`.

The lower matched errors in the stressed arms are not interpreted as evidence that deleting surface is beneficial. The causal claim is limited to retained downstream capacity / PASS under the frozen stress.

## SkinFieldCodec / Arachne result

Shipping Codec hash:
`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`.

Shipping Arachne hash:
`ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`.

Arachne uses the same frozen oracle mechanical G convention as the earlier R6 U1 Arachne gate. It therefore measures skin/deformation capacity on the stressed GSA surface and does not rescue or substitute for the separately measured Geppetto result.

### CORE_75 — 120 admitted rows

Codec A0: `PASS`
- pass step `544`, stable `3`
- raw row-L1 p95 `0.0250483025`
- qualified row-L1 p95 `0.0250482708`
- raw deformation ratio `0.0063259178`
- qualified deformation ratio `0.0063259383`
- Compiler total correction L1 `5.0240196e-06`

Arachne A1: `PASS`
- pass step `512`, stable `3`
- raw / qualified row-L1 p95 `0.0716092363`
- raw deformation ratio `0.0245011654`
- qualified deformation ratio `0.0245011393`
- latent p95 `0.1526485682`
- mean latent uncertainty `0.0484967940`
- Compiler total correction L1 `5.2411924e-06`

### CORE_60 — 96 admitted rows

Codec A0: `PASS`
- pass step `416`, stable `3`
- raw row-L1 p95 `0.0354084149`
- qualified row-L1 p95 `0.0354084186`
- raw / qualified deformation ratio `~0.0106130922`
- Compiler total correction L1 `3.6241254e-06`

Arachne A1: `PASS`
- pass step `352`, stable `3`
- raw row-L1 p95 `0.0469641462`
- qualified row-L1 p95 `0.0469641834`
- raw deformation ratio `0.0150277596`
- qualified deformation ratio `0.0150277680`
- latent p95 `0.1135558486`
- mean latent uncertainty `0.0544756912`
- Compiler total correction L1 `3.5390840e-06`

Preregistered Arachne verdict:
`STRONG_SYNTHETIC_GSA_ARACHNE_COVERAGE_TOLERANCE`.

Raw and qualified outputs independently satisfy the frozen acceptance bands; Compiler correction is small and does not explain PASS.

## Causal interpretation

For this frozen synthetic witness, the hypothesis that the current GSA/consumer route requires near-complete full-surface coverage is falsified through the tested range. A contiguous region deprivation leaving only `60%` of the U0 surface still permits:

- GSA compilation without hidden completion;
- sparse deterministic normal recovery;
- frozen Geppetto `4/4` mechanically qualified control generation;
- shipping Codec/Arachne PASS on all admitted rows;
- low deformation error without Compiler rescue.

This is materially stronger than the historical U1 result with `159/160` retained samples.

## Limits / remaining obligations

This result does **not** establish RigAnything equivalence or arbitrary real-character sufficiency.

1. The witness is a synthetic capsule-shell branch, not a real textured character.
2. The two deprivation masks are controlled contiguous missing-region stresses, not a photorealistic garment/self-occlusion renderer.
3. This high-contrast strengthening gate used one witness; heterogeneous high-contrast coverage remains unmeasured.
4. `local_relation_count = 0` in both stressed GSA outputs. Therefore the GSA observed-local-relation / MWB2 topology seam remains open and is not rescued by this result.
5. Arachne uses oracle mechanical G in this gate; Geppetto independently passed the same stressed surface, but a fully chained learned G->W stress was not executed here.
6. `U2_PREDICTED_IRIS_SUBSTRATE` remains unproven. This experiment injects oracle-correct observation depth/support into GSA and does not include IRIS prediction error.
7. A real Family-1 U0-vs-U1 GSA comparison remains required before using this result as evidence for the actual first-fit character.

## Architecture consequence

This result does not justify adding learned visibility/occlusion authority or hidden-surface teacher completion to IRIS. The immediate evidence instead supports preserving the current observation-grounded IRIS -> GSA separation while measuring the remaining real-family and GSA-topology obligations.

The all-ray-intersection teacher remains diagnostic, not promoted training truth, under this result.
