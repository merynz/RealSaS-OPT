# E0 downstream proxy — operational amendment V1.1

Date: 2026-08-28

## Trigger

The sealed V1 notebook reached production pack preparation and the child prep process exited before any pack/report artifact was committed to Drive. The outer notebook surfaced only `CalledProcessError`.

Post-failure localization used the first frozen train asset `asset_d7d4192f9dac8b146a17bc41`:

- current master `primary_geometry.npz` SHA exactly equals the frozen source-cache SHA;
- all V0..V7 `raster_authority.npz` SHAs exactly equal the frozen source-cache SHAs;
- the exact sealed scientific builder `build_compact_pack(..., anchor_count=512)` succeeds on those exact bytes;
- Drive `prep_v1/packs` and `prep_v1/reports` contained no committed first-asset artifact.

Therefore the observed failure is localized to the operational Drive/FUSE publication path, not the scientific substrate builder or frozen source authority.

## Authorized operational correction

V1.1 changes only storage / diagnostics:

1. preserve the exact scientific builder source SHA `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`;
2. build pack/report bytes on local `/content` first;
3. publish to Drive with direct `write_bytes` followed by readback SHA verification and bounded retry;
4. forbid Drive-side `tmp.replace` / rename as a publication primitive;
5. stream child stdout/stderr into the notebook and persist a local child log, so future child exceptions are not hidden behind `CalledProcessError`;
6. train/checkpoint locally first, then hash-verified direct-copy checkpoint artifacts to Drive.

## Scientific invariants — unchanged

- population: 374 FIT train / 59 FIT selection / 4 truth-capable historical calibration;
- D0, D1, D2 definitions unchanged;
- D2 admission remains teacher-free `MUTUAL_P003`, cycle threshold `0.003`;
- 36D adapter unchanged;
- Arachne and Geppetto proxy capacities unchanged;
- seed / optimizer / epoch budgets / checkpoint selection unchanged;
- no calibration family gets gradient;
- numerical non-inferiority margins remain unfrozen;
- Proxy32 / DEV32 / TUNE / CAL / EXTERNAL remain closed.

This amendment does **not** authorize scientific reinterpretation or a new population. It repairs only the execution apparatus.
