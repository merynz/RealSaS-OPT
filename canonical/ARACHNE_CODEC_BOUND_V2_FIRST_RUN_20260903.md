# RealSaS — Arachne / SkinFieldCodec Bound V2 first clean run

**Date:** 2026-09-03  
**Status:** `FAIL_A0__SHARP_STABILITY_ONLY`  
**Harness:** `test_arachne_v2_behavioral_panel_bound_v2.py`  
**Workflow:** `33756424156`  
**Job:** `100651837229`  
**Runner:** `westus`

## 1. Authority correction

This is the first full behavioral execution after correcting the confirmed V1 synthetic surface-row binding bug.

The scientific panel itself was not changed:
- same three witnesses;
- same seeds;
- same Codec/Arachne sizes;
- same AdamW settings;
- same A0 1536-step horizon;
- same A1 2048-step horizon;
- same 32-step evaluations;
- same A0/A1 thresholds;
- same requirement for three consecutive PASS evaluations.

Only teacher W and rest rows are rebound by exact canonical `surface_id` into `conditioning.surface_ids` order.

## 2. `chain_blend_3`

**Status:** `PASS`

Binding order is identity because `N=9`.

Codec A0:
- sustained PASS at step `480`;
- final checked row-L1 p95 `0.0369992293`;
- deformation ratio `0.0102748750`;
- max simplex residual `1.19e-07`;
- negative weights `0`.

Arachne A1 shipping / Compiler / LBS:
- sustained PASS at step `320`;
- qualified row-L1 p95 `0.0510563254`;
- qualified deformation ratio `0.0130262375`;
- Compiler total correction L1 `5.2867e-07`;
- qualified rows `9`;
- negative weights `0`.

## 3. `branch_blend_4`

**Status:** `PASS`

Canonical surface permutation:
`[0,1,10,11,2,3,4,5,6,7,8,9]`

Codec A0:
- sustained PASS at step `768`;
- final checked row-L1 p95 `0.0214583222`;
- deformation ratio `0.0076737860`;
- max simplex residual `5.96e-08`;
- negative weights `0`.

Arachne A1 shipping / Compiler / LBS:
- sustained PASS at step `352`;
- qualified row-L1 p95 `0.0422459207`;
- qualified deformation ratio `0.0129286591`;
- Compiler total correction L1 `5.5786e-07`;
- qualified rows `12`;
- negative weights `0`.

This clean full-chain result causally falsifies the historical V1 claim that `branch_blend_4` exposed a Codec representation-capacity failure.

## 4. `sharp_fork_5`

**Status:** `FAIL_A0__STABILITY`

Canonical surface permutation:
`[0,1,10,11,12,13,14,2,3,4,5,6,7,8,9]`

The clean trace no longer shows the historical catastrophic plateau. Deformation becomes small and row-L1 repeatedly crosses the frozen threshold.

Representative late evaluations:
- step `1280`: PASS, p95 `0.0453201`, deformation `0.00907888`;
- step `1312`: FAIL, p95 `0.0831804`;
- step `1344`: PASS, p95 `0.0400775`, deformation `0.00595768`;
- step `1376`: FAIL, p95 `0.0501573` — threshold miss only `0.0001573`;
- step `1408`: PASS, p95 `0.0384682`, deformation `0.00585171`;
- step `1440`: FAIL, p95 `0.0562494`;
- step `1472`: FAIL, p95 `0.0862843`;
- step `1504`: FAIL, p95 `0.0583804`;
- step `1536`: PASS, p95 `0.0448788`, deformation `0.00790533`.

At the final evaluation all scalar A0 acceptance thresholds pass, but only one consecutive PASS is present. No A1 run is authorized because the preregistered three-consecutive-PASS rule remains binding.

Therefore the clean remaining seam is:

`CODEC_A0_WORST_ROW_STABILITY != A0_ACCEPTANCE_STABILITY`

It is no longer supported to describe the failure as inability to represent the sharp field.

## 5. Objective/acceptance mismatch now under causal test

A0 optimization currently minimizes:
- Codec reconstruction objective;
- deformation MSE.

A0 acceptance additionally requires a hard row-L1 p95 ceiling.

Unlike A1, A0 has no explicit top-row/tail term. A1 already carries generic `top_fraction_row_l1_tail_v1` pressure.

A causal A/B diagnostic has been opened with correct surface-ID binding:
- baseline = current A0 objective;
- intervention = current A0 objective + generic top-10% row-L1 tail, weight `1.0`;
- same witnesses, seeds, model, optimizer, LR, WD, horizon and frozen acceptance rule.

The frozen Bound V2 panel is not changed by this diagnostic.

## 6. Current authorization boundary

- `chain_blend_3`: PASS;
- `branch_blend_4`: PASS;
- `sharp_fork_5`: clean FAIL_A0 stability;
- Arachne/Codec behavioral closure: `NOT YET PASS`;
- architecture refreeze: `BLOCKED`;
- formal Family-1 selection: `BLOCKED`.
