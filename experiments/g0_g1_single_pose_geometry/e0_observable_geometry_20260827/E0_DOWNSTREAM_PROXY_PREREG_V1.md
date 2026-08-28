# RealSaS E0 — Downstream Information-Sufficiency Proxy Prereg V1

**Date:** 2026-08-27  
**Status:** `FROZEN_BEFORE_DOWNSTREAM_PROXY_OUTCOMES__BATCH_PARITY_PREFLIGHTED__FIT_ONLY__PROXY32_DEV32_CLOSED`  
**Scope:** matched from-scratch proxy calibration for E0-0 / E0-a / frozen-safe E0-b.  
**Important:** these are research information-isolation proxies, **not** product Geppetto/Arachne implementations.

## 1. Authority and why this is a reimplementation

The original S0-B report survives, but an executable byte-authority for the S0 probe implementation/checkpoints was not recovered from the canonical starter/Library search. Therefore this experiment is explicitly:

`REIMPLEMENTED_FROM_FROZEN_S0_B_CONTRACT`

It preserves the S0 discipline (matched capacity, matched initialization, from-scratch training, GT-count Geppetto isolation, GT-skeleton Arachne isolation), but it does **not** claim byte-identical reproduction of S0.

Frozen source facts:
- S0 Arachne: 57D point/joint pair slots -> MLP `96 -> 96 -> 1`, seed 1862, AdamW lr 2e-3, wd 1e-4, 6 epochs, 24 rows/batch, 96 points/row, BEST by selection soft-label CE.
- S0 Geppetto isolation: 36D point slots -> 48-query attention decoder, hidden 48, seed 1862, AdamW lr 1.5e-3, 8 epochs, 16 rows/batch, 256 points/row.
- The full/product Geppetto count/topology model and product Arachne/SkinTokens model do not yet exist.

## 2. Production-master downstream truth authority

The current `RealSaS_MASTER_CORPUS_1024_V3/master/assets/<asset>/primary_geometry.npz` already contains the downstream fields needed by these proxies when the frozen ledger capabilities permit them:

- Geppetto truth: `bone_heads`, `bone_tails`, `parents`, `deform_mask`.
- Arachne truth: the above plus `skin`.
- IRIS/E0 surface construction is still forbidden from consuming those fields.

The master corpus field firewall itself defines:
- IRIS allowlist: geometry only;
- GEPPETTO allowlist: geometry + bone/parent/deform fields;
- ARACHNE allowlist: GEPPETTO fields + skin.

No old M4/M5 truth cache is required for this E0 proxy gate.

## 3. Frozen FIT population

Source membership:
- `P_V5_R256_FIT_SCALE_LADDER_MEMBERSHIP_V1.json`
- raw SHA-256: `3cfc3dc75be431993a9a863d08f381cd800abe6bb6a56c80c06d124f6cf3c100`
- canonical-json SHA-256: `8531360f1c61dc4cdb699c095790c35ab3af0da5a7a290b420e5777bb4e9169b`
- frozen train512 ordered asset-id SHA-256: `de6d536173ab9cd35385f56f501f9b68fee55303dc76211cb15f061eccbed096`

Master ledger:
- `MASTER_VARIANTS.jsonl`
- SHA-256: `475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`
- 4008 rows.

Truth-availability eligibility is frozen **before any downstream proxy outcome**:
`capabilities.iris == true AND capabilities.geppetto == true AND capabilities.arachne == true`.

This yields 433/512 FIT assets. Exclusion is target-availability only; no image geometry, difficulty, E0 metric, morphology, or downstream outcome is consulted.

The 433 eligible IDs are SHA-256-sorted by `asset_id`, mirroring S0's deterministic family-disjoint split discipline:
- train: 374, SHA-256 `d63f99344a345071b895e1e547368feefd3f1b3df31c60a70e80a420d7d1a28f`
- selection: 59, SHA-256 `d3cf11be8d0a25c557c7ece7f8c913e1cc674dcf6fd225014ab1607bce08777c`

The exact lists are frozen in `E0_DOWNSTREAM_PROXY_SPLIT_V1.json`.


### Frozen source-hash reduction

The 433 FIT train+selection assets are byte-bound by `E0_DOWNSTREAM_PROXY_SOURCE_HASHES_V1.json`, reduced deterministically from the existing P-V5/R256 FIT scale cache manifest. It contains only source geometry/raster hashes and provenance metadata required by this gate; it does not add any new population or outcome information.

- reduced record count: 433
- reduced manifest SHA-256: `19a5391c635f0cca9e04663915208de70a2c671b72253e4c4c3c761dcefa882c`
- parent cache manifest membership canonical SHA-256: `8531360f1c61dc4cdb699c095790c35ab3af0da5a7a290b420e5777bb4e9169b`
- `camera_json_consumed = false`

## 4. Calibration eligibility amendment

The historical E0 calibration8 was selected for IRIS/geometry calibration, not downstream truth availability. Frozen master-ledger capabilities show only 4/8 are both Geppetto- and Arachne-capable:

1. `asset_551ea351b43a1787d0f55536`
2. `asset_0679fdef64f19a4832a6d521`
3. `asset_76313e4bd82b82fcd1659c70`
4. `asset_f8a40d6c5d815fe79c8b5e42`

Ordered SHA-256: `65af2d7f34ccee54f2c836deb1d58c1d12a319e11c11576a8f13cdce171445eb`

The other four remain in the E0 geometry/persistence calibration and are **not removed from E0**. They simply cannot enter a rigging-truth proxy that requires fields they do not possess.

This capability intersection is metadata-only and frozen before downstream proxy outcomes.

## 5. Geometry arms

All arms expose exactly 512 point slots in the RealSaS canonical/object frame.

### D0 — full-mesh ceiling
- identical deterministic area-uniform E0-0 P sampler;
- full-mesh face/bary provenance is retained only for target interpolation / upper-bound oracle visibility;
- exact oracle 8-view visibility + raster XY provenance is granted so the information ceiling is not artificially feature-poorer than observable arms.

### D1 — observable + oracle persistence
- exact E0-a P anchors;
- oracle physical support/matched rows.

### D2 — observable + frozen-safe deterministic persistence
- exact same P anchors as D1;
- base E0-b deterministic matches;
- source self witness retained;
- every nonself match must survive observation-only reverse matching with
  `||P_return - P_anchor|| <= 0.003`;
- no teacher triangle/bary/component/skin/joint identity enters D2 admission.

Thus:
- D0 -> D1 asks coverage/support sufficiency.
- D1 -> D2 asks persistence sufficiency under the already-frozen `MUTUAL_P003` rule.


### Operational implementation freeze

The scientific matcher semantics remain the scalar E0 oracle/derived contracts. For tractable 512-anchor production execution, the downstream pack builder uses a vectorized implementation that preserves candidate ordering, gates, scoring and first-argmin tie behavior. Before promotion, real `asset_551ea351b43a1787d0f55536` at 32 anchors produced **bit-exact equality across every compact-pack field** against the scalar reference. The scalar reference remains test-only authority and is not a second experimental arm.

The prep staging directory must retain the canonical `asset_id` as `asset_dir.name`, because E0 deterministic samplers namespace their seeds by asset ID. Generic staging names are forbidden.

Matched stochastic schedules are also frozen across geometry arms: Arachne point-sampling labels and Geppetto evaluation subsampling labels omit the arm name, so D1/D2 do not receive different row draws merely because of treatment naming. Geppetto's GT-count oracle admits exactly the first `K` query slots for a `K`-joint target before Hungarian assignment; it may not choose an outcome-favorable subset from all 48 queries.

## 6. Fixed 36D point adapter

Every point in every arm receives exactly:

```text
P canonical xyz                         3
support mask V0..V7                    8
matched normalized raster XY V0..V7   16
support fraction                        1
camera-forward depth dot(P,F_v) V0..7  8
-----------------------------------------
total                                  36
```

Unsupported XY slots are zero. Camera-forward depth is deterministic from P + frozen view geometry and is not an extra learned/teacher field.

D0's support/XY are oracle upper-bound metadata; D1's are oracle persistence; D2's are deterministic `MUTUAL_P003`.

## 7. Arachne information-isolation proxy

GT product skeleton is fixed. Legal controls are `deform_mask == true`.

57D pair slot:
- 36D point adapter;
- 21D point/bone relation:
  head xyz, tail xyz, point->head, point->tail, point->midpoint,
  distances to head/tail/midpoint, unit bone axis.

Model:
`57 -> 96 -> 96 -> 1`, GELU, softmax across legal controls.

Training:
- seed 1862;
- identical initialization bytes across D0/D1/D2;
- AdamW lr 2e-3, wd 1e-4;
- 6 epochs;
- batch 24 asset rows;
- 96 target-valid points per row;
- BEST selected only by the frozen 59-FIT selection soft-label CE.

Targets:
- D0 skin interpolated at the deterministic full-mesh face/bary sample;
- D1/D2 use identical skin truth interpolated at the shared observable source carrier;
- zero-total-skin target points are evaluator-invalid for Arachne only and are counted, never replaced using outcome information.

Metrics:
CE, weight MAE, top1, GT mass in predicted top4, and deterministic weight-induced displacement diagnostic.

## 8. Geppetto information-isolation proxy

This is not full skeleton generation.

Model:
- point encoder 36 -> 48 -> 48;
- 48 learned queries, hidden 48;
- one 4-head cross-attention block + FFN;
- xyz output per query.

Training:
- seed 1862;
- identical initialization bytes across D0/D1/D2;
- AdamW lr 1.5e-3, wd 1e-4;
- 8 epochs;
- batch 16 asset rows;
- 256 points/row;
- BEST selected only by frozen 59-FIT selection joint-mean error.

GT count/assignment is an experimental oracle control. Hungarian assignment is permitted because this is information isolation, not product inference.

If an asset has >48 deform controls, the probe target is deterministically farthest-point-subsampled in canonical joint-head space to 48. The original count is reported. No asset is excluded for exceeding probe query capacity.

Metrics:
joint mean, family-p95, joint-p95 mean, PCK@0.05, PCK@0.08.

## 9. Calibration and non-inferiority order

This run opens only the four truth-capable historical calibration members after all:
- source bytes,
- split,
- feature layout,
- architectures,
- optimizer budgets,
- checkpoint-selection rules,
- D2 admission rule

are frozen.

The run **does not auto-select or auto-freeze a numerical non-inferiority margin**.

After calibration output is inspected, numerical margins may be frozen in a separate decision record. Only then may the truth-capable intersection of `qualification_proxy32` be opened.

## 10. Firewalls

- FIT only.
- `qualification_proxy32 = CLOSED`.
- `DEV32 = CLOSED`.
- TUNE/CAL/DEV/EXTERNAL not used.
- no pretrained consumer.
- no MapAnything.
- no IRIS learner.
- no product Geppetto/Arachne claim.
- teacher rig truth is target/evaluator-only.
- D2 admission consumes no teacher identity.
- no family is added/removed using E0 or downstream outcome.
