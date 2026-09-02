# RealSaS — Single-family Mage CPU fit V1

Status: `IN_PROGRESS__GEPPETTO_U0_PASS__CODEC_A0_PASS__ARACHNE_AND_IRIS_OPEN`

Canonical repository: `merynz/RealSaS-OPT`
Canonical base main commit: `de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3`
Fit branch: `fit/single-family-mage-v1-20260902`

## Scientific scope

This is the first executable FIT witness only. It does not weaken or replace the generic architecture claim, does not authorize held-out/generalization claims, and must not introduce Mage-specific semantic names, fixed topology, or a product joint cap.

Witness:
- canonical asset: `asset_96b983142e9fcd29ecf52f48`
- source candidate: `kaykit_cc0:cf898585da33fab50c724d31`
- source SHA-256: `cf898585da33fab50c724d31605fb931eb2912e6d2280092141e98ca81ad507d`
- split: `FIT`

## Geppetto U0

A real-family CPU fit exposed an accumulator-aliasing defect in `GeppettoLossV2`: all channel accumulators were initialized from the same Tensor and then updated with `+=`. The fix is storage-independent scalar accumulators and a regression test.

The mechanically qualified target axis is not all source bones. The generic training axis is the current deform-mechanical core: skin-supported controls plus required supported-control bridges; assembly-only roots and helper/IK-only controls are not promoted as deform controls. This is a structural/truth-rebind rule, not a Mage name list.

Accepted CPU witness metrics at step 1024:
- generated controls: 22
- count absolute error: 0
- matched normalized MAE: 0.0301506463
- matched normalized P95: 0.0429609455
- parent accuracy: 1.0
- root accuracy: 1.0
- STOP at target count: 0.9998625517

Metric artifact SHA-256: `42f3faf3dd5b61d69275123a0681f7aac405501d842e2b3cdbf2cd9cb838a548`
Ephemeral local checkpoint SHA-256: `8eb022af8b5a8a4e7ff2bb0129c84bb536176c40cf1ac940a0279efa6ed7d28a`

The checkpoint hash is recorded for provenance; qualification authority is source + prereg + reproducible metrics, not possession of an opaque checkpoint.

## SkinFieldCodec R6-A0

Current-qualified-G axis: 22 controls. At step 1024:
- row L1 mean: 0.0280882102
- row L1 P95: 0.1317590028
- verified-LBS deformation RMS: 0.0001163242
- simplex max abs residual: 1.1920929e-7

Frozen acceptance targets used for the remediation rung:
- row L1 mean <= 0.05
- row L1 P95 <= 0.15
- deformation RMS <= 0.005
- simplex residual <= 1e-5

Result: `PASS`.
Metric artifact SHA-256: `df7aacae9385ad14eb53c40bca150442e86cba4943adaf4b7042415bef5c3a41`

## Open gates

- Arachne A1: optimizer + dense-W/deformation qualification on the same current-S/current-qualified-G axis.
- IRIS: exact textured observation/render-frame truth and frozen foundation feature authority; proxy/fake features are forbidden.
- COMPLETE_E2E_FIT: closed until IRIS + Geppetto + Arachne learned outputs pass Compiler/proof/runtime.
- Generalization: closed.
