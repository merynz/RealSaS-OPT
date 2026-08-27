# E0 downstream proxy notebook preflight V1.2

Status: **PASS — operational FUSE fix only; science unchanged**.

- notebook: `RealSaS_E0_DOWNSTREAM_PROXY_V1_1.ipynb`
- notebook SHA-256: `83c49b04b29debbca5c7942b853df0f51d884d8bab1bba0be5d3b668be422151`
- scientific contract SHA-256: `cf204ad1ef7d8460e402fb6c3db7361d122b3284116aa6e840af032d0fb39a2a`
- operational amendment SHA-256: `e1a7e5c6ed61e08bac48927f38e240ed6e4974fb5402aeeacac363ddbe387662`
- prep runner V1.1 SHA-256: `404ad50b8f7a92804530f0b26eb9e8ebd620a196cfa4040bad0ab24f5917d2f6`
- scientific builder SHA-256 unchanged: `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`

## Failure localization

The V1 Colab run exited in production prep before any first-asset pack/report appeared in Drive. The first frozen train asset was reconstructed independently from Drive: its primary geometry and all eight raster authority hashes exactly match the frozen cache authority, and the exact sealed `build_compact_pack(..., 512)` succeeds. The failure is therefore localized to the Drive/FUSE publication apparatus, not to D0/D1/D2 science.

## V1.1 correction

The prep runner now builds local bytes first and publishes with direct write + readback SHA + bounded retry. Drive-side `tmp.replace` is forbidden. Child output is streamed to the notebook so a future child exception is visible.

Exact V1.1 runner smoke on the first real train asset produced a pack bit-identical to direct builder output: `d9f1a40ee1db408fcdff7a541af91ebb2c33d1b8949d2236447a3a8dced6d018`.

## Firewalls

Scientific contract, population, D2 `MUTUAL_P003`, model capacities, optimizer schedules and selection rules are unchanged. Scientific training was not run during this preflight. Proxy32/DEV32 remain closed; numerical non-inferiority margins remain unfrozen.
