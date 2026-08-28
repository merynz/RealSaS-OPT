# E0 Downstream Proxy — Arachne Target-Support Amendment V1.3

Status: **FROZEN BEFORE ANY DOWNSTREAM TRAINING OUTCOME**.

The V1.2 prep completed all 437 compact packs. Scientific calibration then stopped before the first optimizer epoch could complete because `asset_b2528524fbbb0184a3cd14ac` had no valid Arachne target rows in D0. No selection/calibration metric, downstream comparison, non-inferiority margin, Proxy32, or DEV32 outcome was opened.

A full audit of the already-frozen V1.2 prep index shows exactly three of the 374 FIT train assets lack a supervised Arachne target in at least one matched arm:

- `asset_2821f8599863966b86bef876`: D0=0, D1=0, D2=0 valid skin rows;
- `asset_b2528524fbbb0184a3cd14ac`: D0=0, D1=0, D2=0;
- `asset_fbffec7069568f0e2c5baed8`: D0=12, D1=0, D2=0.

All 59 FIT selection assets and all 4 truth-capable historical calibration assets have at least one valid Arachne target in every arm.

## Frozen rule

Arachne training eligibility is treatment-symmetric and target-availability-only:

```text
eligible_for_arachne_train(asset) :=
    count(D0_skin_valid) > 0
AND count(D1_skin_valid) > 0
AND count(D2_skin_valid) > 0
```

Therefore:

- frozen base FIT train population remains 374;
- Arachne matched supervised train subset is 371/374;
- the same 371 asset IDs are used for D0, D1 and D2;
- Geppetto remains 374/374;
- FIT selection remains 59/59;
- calibration remains 4/4;
- no asset is removed from E0 itself;
- no compact pack is rebuilt;
- the completed 437 V1.2 packs remain the substrate authority.

This amendment is **not** retrospective ProductDomain filtering. It does not inspect image content, morphology, giant-plane status, E0 difficulty, model loss, or downstream performance. It only prevents an optimizer step for an asset/consumer pair for which the frozen pack contains no supervised target in one or more treatment arms.

`PRODUCT_DOMAIN_V1` remains a separate prospective future audit.

## Firewalls

- D0/D1/D2 substrate definitions unchanged;
- `MUTUAL_P003` unchanged;
- pack population and pack bytes unchanged;
- model capacities unchanged;
- optimizer schedules unchanged;
- random seed unchanged;
- Geppetto train population unchanged;
- selection/calibration populations unchanged;
- numerical non-inferiority margins not frozen;
- Proxy32 and DEV32 remain closed;
- no downstream calibration outcome inspected before this rule was frozen.
