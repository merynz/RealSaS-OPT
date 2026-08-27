# RealSaS-OPT — Current State

**Date:** 2026-08-27  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_DOWNSTREAM_PROXY_V1_SEALED_READY__FIT_ONLY__TRAINING_NEXT__PROXY32_DEV32_CLOSED`

## Read this first

This file is the single continuation authority.

The active E0 question is whether the **observable single-pose A×8 common-frame substrate preserves enough rigging-relevant information** relative to a full-mesh ceiling. No E0 product PASS/FAIL has been declared. Product Geppetto and Arachne models do not yet exist; the next experiment uses fixed research information-isolation proxies only.

## Frozen geometry / persistence chain

```text
D0 = E0-0 full/closed canonical surface + oracle observation provenance
D1 = E0-a exact observable surface + oracle physical persistence
D2 = E0-b exact observable surface + frozen MUTUAL_P003 admission
```

Causal interpretation:

```text
D0 -> D1 = observable coverage/support tax
D1 -> D2 = deterministic persistence/admission tax
```

Corrected E0 geometry authority is Calibration-8 V1.3. The old fixed-half-extent V1.2 run is invalid. Forward E0 does not consume `camera.json`; orthographic scale is recovered from observable P + raster authority.

## Frozen E0-b admission

Selected rule: `MUTUAL_P003`.

On the frozen FIT calibration-8, BASE -> MUTUAL_P003:
- micro precision `0.906806 -> 0.935181`;
- micro recall `0.963237 -> 0.960053`;
- false pairs `964 -> 648`;
- true-positive pairs `9380 -> 9349`;
- definite cross-component false pairs `35 -> 26`;
- definite cross-component accepted rate `0.3384% -> 0.2601%`.

Teacher identity is forbidden from admission. `support>=2/3` is not part of the rule.

Canonical safety report:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/E0_B_SAFETY_CALIBRATION_RESULT_V1.md`

## Downstream proxy package — SEALED READY

Canonical package directory:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/`

Primary files:
- `E0_DOWNSTREAM_PROXY_PREREG_V1.md`
- `E0_DOWNSTREAM_PROXY_CONTRACT_V1.json`
- `E0_DOWNSTREAM_PROXY_PACKAGE_V1.json`
- `E0_DOWNSTREAM_PROXY_NOTEBOOK_PREFLIGHT_V1_1.md`
- `E0_DOWNSTREAM_PROXY_NOTEBOOK_PREFLIGHT_V1_1.json`

Exact executable notebook distributed with the experiment:
`RealSaS_E0_DOWNSTREAM_PROXY_V1.ipynb`

Notebook SHA-256:
`26f7bc974fffe7be4467724846a16dd8e4abbab32dc1db56d99e202ae0e08710`

Contract SHA-256:
`cf204ad1ef7d8460e402fb6c3db7361d122b3284116aa6e840af032d0fb39a2a`

Implementation status:
`REIMPLEMENTED_FROM_FROZEN_S0_B_CONTRACT__NOT_BYTE_IDENTICAL_S0`

The original executable S0 probe implementation/checkpoints were not recovered. The fixed S0-B research discipline and capacities were reimplemented explicitly; this package must not be described as byte-identical S0 reuse.

## Frozen production-domain FIT population

Source membership is the pre-existing FIT train512 ladder. Ordered train512 SHA:
`de6d536173ab9cd35385f56f501f9b68fee55303dc76211cb15f061eccbed096`

Master ledger SHA:
`475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`

Prospective downstream-truth eligibility is metadata-only:

```text
capabilities.iris == true
AND capabilities.geppetto == true
AND capabilities.arachne == true
```

This yields 433/512 truth-capable FIT assets. No E0 metric, morphology, image content or downstream outcome was used for eligibility.

Frozen SHA-order split:
- train: 374; SHA `d63f99344a345071b895e1e547368feefd3f1b3df31c60a70e80a420d7d1a28f`
- selection: 59; SHA `d3cf11be8d0a25c557c7ece7f8c913e1cc674dcf6fd225014ab1607bce08777c`

Historical calibration-8 remains unchanged for E0 geometry. Only four members possess both Geppetto and Arachne truth and therefore may enter the rigging proxy calibration:
- `asset_551ea351b43a1787d0f55536`
- `asset_0679fdef64f19a4832a6d521`
- `asset_76313e4bd82b82fcd1659c70`
- `asset_f8a40d6c5d815fe79c8b5e42`

This 4/8 intersection is frozen from pre-existing capability metadata, not downstream outcomes.

## Fixed proxy contracts

### 36D point adapter

```text
P xyz                                      3
support mask V0..V7                       8
matched normalized raster XY V0..V7      16
support fraction                          1
camera-forward depth dot(P,F_v) V0..V7    8
                                           --
total                                     36
```

### Arachne information-isolation proxy

GT product skeleton/parents fixed. `36D + 21D point/bone relation = 57D`; MLP `57 -> 96 -> 96 -> 1`, GELU, softmax over legal deform controls.

Frozen schedule: seed `1862`, identical initialization across D0/D1/D2, AdamW lr `2e-3`, wd `1e-4`, 6 epochs, batch 24 asset rows, 96 target-valid points/row. BEST only by frozen 59-FIT selection CE.

This is not product Arachne/SkinTokens.

### Geppetto information-isolation proxy

36D points -> hidden48; 48 learned queries; one 4-head cross-attention block + FFN; xyz per query. GT count is an experimental oracle control and admits exactly the first K query slots before Hungarian assignment.

Frozen schedule: seed `1862`, identical initialization across D0/D1/D2, AdamW lr `1.5e-3`, wd `1e-4`, 8 epochs, batch 16 asset rows, 256 points/row. BEST only by frozen 59-FIT selection joint-mean error.

This is not product skeleton/count/topology generation.

## Apparatus corrections / preflight

Several issues were caught before scientific execution and corrected:
1. prep staging must preserve canonical `asset_id` as `asset_dir.name`, because E0 deterministic samplers namespace seeds by asset ID;
2. Arachne row sampling and Geppetto eval subsampling are arm-independent, so matched arms receive matched stochastic draws;
3. Geppetto GT-count oracle may not choose an outcome-favorable subset from all 48 queries;
4. the production batch SurfaceBuilder must preserve scalar matcher semantics.

Frozen real-551 scalar↔batch parity record SHA:
`04651c06428f61070d3f2a5b3852b30bcab3ba5158099711a0fdf587d34d710b`

Exact notebook V1.1 preflight:
- all 9 code cells compile;
- embedded 12-file authority bundle verifies;
- downstream tests `4 passed`;
- frozen real-551 32-anchor scalar↔batch bit-exact parity record verifies;
- real 551 512-anchor D0/D1/D2 reference parity versus corrected E0 V1.3 passes;
- real-data proxy forward/backward smoke passes for D0/D1/D2;
- `teacher_identity_consumed_by_D2_admission = false`;
- scientific training not executed in preflight;
- numerical non-inferiority margins not frozen;
- Proxy32 and DEV32 closed.

## Next executable action

Run the **exact sealed notebook** on Colab GPU.

Expected sequence:
1. stage only frozen 374 train + 59 selection + 4 truth-capable calibration assets;
2. build 512-point D0/D1/D2 packs;
3. train matched from-scratch Arachne proxy arms;
4. train matched from-scratch Geppetto proxy arms;
5. inspect calibration4 D0→D1 and D1→D2 paired deltas;
6. do **not** auto-freeze a numerical margin;
7. only after inspection write a separate non-inferiority-margin decision record;
8. Proxy32 may be opened only after that separate freeze.

## Parked questions

### N head / S0-B3

Normal information is downstream-useful, but whether IRIS needs a direct learned N head remains open. The strongest structured `N(P)` falsification is parked until the D0/D1/D2 substrate sufficiency gate is inspected. Do not freeze IRIS as P-only yet.

### Product-domain eligibility

Do not retroactively alter the frozen E0/calibration populations. After E0 closure, open a prospective `PRODUCT_DOMAIN_V1` audit for later train/dev/generalization panels. Rigging-irrelevant assets such as giant view-occluding rectangular props may be excluded only by criteria frozen before new-panel outcomes are inspected.

## Authorization state

`D2_D5_CORRESPONDENCE_LOCALIZATION = COMPLETE`

`E0_CALIBRATION8_V1_2 = INVALIDATED`

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_SAFETY_CALIBRATION = COMPLETE_FIT_ONLY`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROXY_V1 = SEALED_READY_FOR_FIT_EXECUTION`

`E0_DOWNSTREAM_PROXY_SCIENTIFIC_TRAINING = NOT_YET_EXECUTED`

`DOWNSTREAM_NONINFERIORITY_MARGINS = NOT_FROZEN`

`E0_PROXY32_QUALIFICATION = CLOSED`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`N_B3 = PARKED_UNTIL_AFTER_D0_D1_D2_INSPECTION`

`PRODUCT_DOMAIN_V1 = DEFERRED_PROSPECTIVE_AFTER_E0`

`C1_C2_C3 = NOT_YET_EXECUTED`

`MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`

`FULL_PATCHMATCH = NOT_AUTHORIZED`
