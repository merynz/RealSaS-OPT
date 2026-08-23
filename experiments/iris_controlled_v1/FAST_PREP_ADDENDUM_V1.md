# IRIS Controlled V1 — Fast Preparation Addendum

Date: 2026-08-23
Status: ENGINEERING PATCH BEFORE OPTIMIZER; scientific target/splits/losses unchanged.

## Why this exists

The original serial cache builder was scientifically valid but operationally pathological on a Colab-mounted Google Drive: each asset opened many small files serially, then re-read those files for per-source SHA, then compressed sixteen 512×512 RGBA images into a per-asset NPZ. The observed throughput was about 20 assets / 16 minutes (~48 s/asset), implying a >40-hour open-split preparation.

This is an engineering/I/O failure, not evidence about IRIS or the corpus.

## Frozen changes

1. **Ceiling first.** Build only the deterministic 64 open assets required by the exact representation ceiling. Do not prepare 3248 assets before the information gate.
2. **Local SSD cache.** Derived cache lives under `/content/IRIS_CONTROLLED_V1_FAST_CACHE`; canonical source remains Drive.
3. **Bounded parallel source reads.** Default 8 threads; user may lower/raise `--workers` within 1–16.
4. **Vectorized correspondence lookup.** Mathematically equivalent to the previous strict-`<` candidate loop; an actual-source parity check runs before preparation and must report zero visibility/row mismatch and <=1e-6 error difference.
5. **No redundant per-source re-hash during cache build.** Parent corpus/split/package SHA authorities remain frozen; every derived cache artifact itself is SHA-256 sealed.
6. **256×256 derived RGBA cache.** The preregistered neural input resolution is 256. Source 512 RGBA is deterministically resized once with PIL RGBA bilinear and cached as uint8. This preprocessing choice is frozen before optimizer step 1.
7. **Uncompressed local NPZ.** Avoid CPU zlib cost; cache is derived/ephemeral and can be deterministically regenerated from frozen source authority.
8. **Full cache only after ceiling PASS.** The same 64 local cache files are reused when expanding to all FIT+TUNE assets.

## What does NOT change

- 3930 corpus eligibility authority;
- FIT 2935 / TUNE 313 / CAL 246 / DEV 270 / EXTERNAL 166;
- sealed policy;
- exact triangle+bary surface truth;
- P/N authority;
- 512 geometry samples/view;
- 48 anchors/view;
- maximum 384 tracks;
- network architecture;
- losses and warmup;
- optimizer settings;
- representation ceiling thresholds.

## Patch hashes

- `prepare_iris_controlled_v1_fast.py`: `8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7`
- `launch_iris_controlled_v1_fast.py`: `1f35904429912b626dab815ed6af871d3e67fd61729963d5e2973017e421c690`

The fast launcher separately verifies the original frozen base package manifest before doing any work.
