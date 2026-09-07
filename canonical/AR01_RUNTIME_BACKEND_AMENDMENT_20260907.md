# AR-01 Runtime Backend Amendment — 2026-09-07

**Status:** `PRERUN_HARDWARE_POLICY_CORRECTION__NO_AR01_OPTIMIZER_STEP_EXECUTED_BEFORE_AMENDMENT`

## Correction

The originally delivered AR-01 notebook incorrectly introduced an A100-SXM4-40GB GPU-name allowlist under a parity rationale. Recent authoritative Geppetto continuation experiments explicitly allow CUDA devices without an A100 requirement and have run on NVIDIA L4.

AR-01 therefore uses the same hardware policy:

- CUDA is required.
- NVIDIA L4 is valid.
- No GPU-name allowlist is part of scientific validity.
- Runtime GPU identity and VRAM remain recorded in artifacts.
- AR0 and AR1 must run under the same runtime/backend for the paired causal comparison.

## Scientific invariants unchanged

This amendment changes **hardware admission only**. It does not change:

- AR0 / AR1 model class or initialization
- skeleton-causal feedback intervention
- Mage FIT1 witness / forced 41 target
- structural serialization
- loss or loss weights
- seed
- AdamW hyperparameters / epsilon switching rule
- check cadence / checkpoint cadence / 16,384-step horizon
- free-running scientific evaluation
- terminal structural PASS criteria
- 48-check terminal stability requirement
- preregistered categorical outcome operator

Because the A100 check failed before training began, this correction occurred before any AR-01 optimizer step or scientific observation and did not condition the protocol on an AR-01 result.

## Delivered notebook

`RealSaS_MAGE_GEPPETTO_AR01_SKELETON_CAUSAL_CUDA_NO_GPU_ALLOWLIST_FINAL.ipynb`

SHA-256: `1660c344421b9dfeb110c60ece30014687ed913828d692298bdf2a8a28f4755d`

This amendment artifact SHA-256: `d01fe9e3a780811097ec43ea7cca73fb7b127d42bb7f6a31e4ea86f909614508`