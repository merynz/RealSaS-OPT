# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_DOWNSTREAM_PROXY_V1_2_CAMERA_RECOVERY_SEALED_READY__FIT_ONLY__TRAINING_NEXT__PROXY32_DEV32_CLOSED`

## Read this first

This file is the single continuation authority.

The active E0 question is whether the **observable single-pose A×8 common-frame substrate preserves enough rigging-relevant information** relative to a full-mesh ceiling. No E0 product PASS/FAIL has been declared. Product Geppetto and Arachne models do not yet exist; the next experiment uses fixed research information-isolation proxies only.

The original downstream-proxy V1 executable is superseded by the V1.1 Drive/FUSE fix. V1.1 then successfully built/published 189/437 frozen assets and false-failed at asset 190 because the camera-scale estimator required 128 ratios after an arbitrary `abs(grid)>=0.05` cutoff; only 84 survived on a small-screen-footprint asset. No downstream scientific training or outcome inspection had begun. V1.2 is now executable authority: it keeps the frozen population, D0/D1/D2, MUTUAL_P003, proxy models and optimizer schedules unchanged, but replaces the ratio heuristic with a robust through-origin slope estimator `q=h*g`.

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

## Downstream proxy scientific contract — unchanged

Scientific contract SHA-256:
`cf204ad1ef7d8460e402fb6c3db7361d122b3284116aa6e840af032d0fb39a2a`

Scientific builder SHA-256:
`0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`

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

## V1 execution failure — localized

The original V1 Colab run exited in production prep before any first-asset pack/report was committed to Drive. The outer notebook showed only `CalledProcessError`.

Post-failure localization used the first frozen train asset `asset_d7d4192f9dac8b146a17bc41`:
- current master `primary_geometry.npz` SHA exactly matches the frozen source-cache authority;
- all V0..V7 `raster_authority.npz` SHAs exactly match frozen source-cache authority;
- the exact sealed `build_compact_pack(..., anchor_count=512)` succeeds on those exact bytes;
- `prep_v1/packs` and `prep_v1/reports` contained no committed first-asset artifact.

Therefore the observed V1 failure is localized to the **Drive/FUSE publication path before first commit**, not the scientific builder, source population, D0/D1/D2 definitions, or MUTUAL_P003.

`E0_DOWNSTREAM_PROXY_V1 = SUPERSEDED_EXECUTION_APPARATUS_DRIVE_FUSE_FAILURE`

## Operational amendment V1.1 — SEALED READY

Canonical amendment files:
- `E0_DOWNSTREAM_PROXY_OPERATIONAL_AMENDMENT_V1_1.md`
- `E0_DOWNSTREAM_PROXY_OPERATIONAL_AMENDMENT_V1_1.json`
- `run_e0_downstream_proxy_prep_v1_1.py`
- `E0_DOWNSTREAM_PROXY_NOTEBOOK_PREFLIGHT_V1_2.md`
- `E0_DOWNSTREAM_PROXY_NOTEBOOK_PREFLIGHT_V1_2.json`
- `E0_DOWNSTREAM_PROXY_PACKAGE_V1_1.json`

Operational amendment SHA-256:
`e1a7e5c6ed61e08bac48927f38e240ed6e4974fb5402aeeacac363ddbe387662`

Prep runner V1.1 SHA-256:
`404ad50b8f7a92804530f0b26eb9e8ebd620a196cfa4040bad0ab24f5917d2f6`

Executable notebook:
`RealSaS_E0_DOWNSTREAM_PROXY_V1_1.ipynb`

Notebook SHA-256:
`83c49b04b29debbca5c7942b853df0f51d884d8bab1bba0be5d3b668be422151`

Preflight V1.2 SHA-256:
`e3e18c554c0d5b421c801282acd3dd34877c57c6568d88b102d2f950e4ea52e8`

V1.1 changes only execution apparatus:
1. build scientific NPZ/JSON bytes locally first;
2. publish Drive artifacts with direct write + readback SHA verification + bounded retry;
3. forbid Drive-side `tmp.replace` / rename as publication primitive;
4. stream child stdout/stderr in the notebook and persist logs;
5. build scientific checkpoints locally first, then verified-copy them to Drive.

Scientific invariants unchanged:
- 374 FIT train / 59 FIT selection / 4 truth-capable historical calibration;
- D0/D1/D2 unchanged;
- teacher-free `MUTUAL_P003`, cycle threshold `0.003` unchanged;
- 36D adapter unchanged;
- Arachne/Geppetto proxy capacities unchanged;
- seed, optimizer, epochs, checkpoint selection unchanged;
- no calibration gradient;
- numerical non-inferiority margins unfrozen;
- Proxy32/DEV32/TUNE/CAL/EXTERNAL closed.

V1.1 exact-runner smoke on the first real frozen train asset produced a pack bit-identical to direct scientific builder output:
`d9f1a40ee1db408fcdff7a541af91ebb2c33d1b8949d2236447a3a8dced6d018`

## Camera recovery V1.2 — SEALED

Trigger asset: `asset_de72098ec0bd43ce9fc30c60`. Old estimator failure: `insufficient stable half-extent ratios: 84`. V0 actually contains 2393 visible rows; the failure was caused by the old fixed `|grid|>=0.05` conditioning cutoff, not missing camera information.

New observable-only estimator solves the camera half-extent as a robust regression through the origin:

```text
q_x = dot(P,right) = h * g_x
q_y = -dot(P,up)   = h * g_y
```

Near-center observations naturally contribute low leverage through `g^2`; there is no fixed screen-coordinate cutoff and no ratio-count gate. Deterministic Huber IRLS provides outlier resistance. Fail-close authority remains native reprojection P95 plus x/y slope consistency and numerical information energy. `camera.json` remains forbidden in the forward path.

Real trigger-asset V0 diagnostic:
- visible rows: `2393`;
- recovered h: `0.5400000774096994`;
- post-hoc camera authority: `0.5400000643730164`;
- absolute error: `1.3036683e-08`;
- native reprojection P95: `0 px`.

Compatibility check: on the previously passing real frozen train asset `asset_d7d4192f9dac8b146a17bc41`, old and new estimators produce a byte-identical 512-anchor compact pack SHA `d9f1a40ee1db408fcdff7a541af91ebb2c33d1b8949d2236447a3a8dced6d018`.

V1.2 resume records are bound to both the downstream builder SHA and the exact `e0_geometry.py` SHA. V1.1 packs must not be resumed under V1.2.

Canonical V1.2 authorities:
- `E0_DOWNSTREAM_PROXY_CAMERA_RECOVERY_AMENDMENT_V1_2.md/json`;
- `E0_DOWNSTREAM_PROXY_NOTEBOOK_PREFLIGHT_V1_3.md/json`;
- `E0_DOWNSTREAM_PROXY_PACKAGE_V1_2.json`;
- `run_e0_downstream_proxy_prep_v1_2.py`;
- executable `RealSaS_E0_DOWNSTREAM_PROXY_V1_2.ipynb` SHA `a46fe75a7ce5a0318d79c6d8b055bc593bd8c57942edc85c1d93353bf742047f`.

## Next executable action

Run **only** `RealSaS_E0_DOWNSTREAM_PROXY_V1_2.ipynb` on Colab GPU. Do not rerun V1 or V1.1.

Expected sequence:
1. use fresh `prep_v1_2` / `checkpoints_v1_2` execution paths;
2. stage only frozen 374 train + 59 selection + 4 truth-capable calibration assets;
3. build 512-point D0/D1/D2 packs;
4. train matched from-scratch Arachne proxy arms;
5. train matched from-scratch Geppetto proxy arms;
6. inspect calibration4 D0→D1 and D1→D2 paired deltas;
7. do **not** auto-freeze a numerical margin;
8. only after inspection write a separate non-inferiority-margin decision record;
9. Proxy32 may be opened only after that separate freeze.

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

`E0_DOWNSTREAM_PROXY_V1 = SUPERSEDED_EXECUTION_APPARATUS_DRIVE_FUSE_FAILURE`

`E0_DOWNSTREAM_PROXY_V1_1 = SUPERSEDED_CAMERA_RECOVERY_RATIO_HEURISTIC_FALSE_FAIL_AT_190_OF_437`

`E0_DOWNSTREAM_PROXY_V1_2 = SEALED_READY_FOR_FIT_EXECUTION`

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
