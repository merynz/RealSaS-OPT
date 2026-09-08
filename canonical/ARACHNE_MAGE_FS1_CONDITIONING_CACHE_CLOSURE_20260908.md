# RealSaS — Arachne Mage FS1 Conditioning Cache Closure — 2026-09-08

**Status:** `PASS_DETERMINISTIC_DOUBLE_REPLAY__FS1_CACHE_SEALED__A0_STILL_BLOCKED`

The preregistered cache builder at commit `0a6c6d02da4cb251e28f842be0c47a61b74da974` was executed twice against the sealed FS1 target and the exact S/G inputs. Both cache NPZ and manifest outputs were byte-identical.

## Sealed cache

- cache NPZ SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- cache binding SHA-256: `c7e3bf10fc8edf16f862b4ce58b3aabfeb2aabf14cc8e744e53e3afc0764ac9f`
- manifest SHA-256: `5b3e63d60fce76226e16aac5d754a5f9cde114ccd8327a3242fad8a714af7d6b`
- surface features: `950 x 20`
- joint features: `22 x 8`
- pair geometry: `950 x 22 x 10`
- teacher weights: `950 x 22`
- supervised rows: `934`
- LOW rows: `16`

## Explicit conditioning semantics

- V1 semantics: `ArachneConditioningAdapter.conditioning_hashes[0]`
- V1 hash: `475d71523a38c5763756713e2047b337662ad6c01735aaa89671c6297326e42c`
- V2 semantics: `ArachneConditioningAdapterV2.conditioning_hashes[0]`
- V2 hash: `9e979b824e6e30fcaf1158e4bceeb080ce5fa1cff463290e1c4a7c2ac4f5f4e9`
- pair geometry raw SHA-256: `416d5e5848ec579eb67e5e402dbf6e60908f4d9a30dbe0650a1d19be7183e4e7`

The historical `7b82...`, `89973...` and `12484...` values remain historical evidence only because their byte-generating builder/hash-payload semantics were not retained. They do not authorize FS1 A0.

No optimizer step occurred. A0 remains blocked until prereg/runner fingerprints are rebound to the sealed FS1 target + this cache and CPU resume/terminal-idempotence regressions pass.
