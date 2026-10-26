# Oracle-Z Query Calibration Court V4 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** representation-search only.

## Motivation

Oracle-Z V3 showed that the direct-XYZ positive control failed the surface-response target. That does not identify a carrier failure; it can also mean the shared query operator `Q` is under-capacity for the target family.

Before changing the target or comparing triplane/dense carriers again, factor query capacity explicitly.

## Frozen target / data

Reuse V3 calibration scenes, sampling, target function, scales, seeds and train/eval pools unchanged. No carrier is present: all arms receive exact XYZ.

## Arms

Per-scene controls, same train/eval data:

- `Q96_D2_PLAIN`: V3 query baseline;
- `Q256_D4_PLAIN`: larger smooth MLP;
- `Q128_D3_FOURIER4`: xyz plus sin/cos Fourier bands 1,2,4,8;
- `Q192_D3_FOURIER4`: larger Fourier query.

Learning-rate grid: `3e-3, 1e-3`.  
Training: 5000 steps per arm/LR, fixed seeds, no weight decay.

## Validity / selection

For each arm choose its best preregistered LR by mean eval nRMSE.

A query arm is surface-calibrated only if:

- mean nRMSE <= 0.10;
- worst scene nRMSE <= 0.14;
- mean local surface nRMSE <= 0.16;
- random-label firewall evaluated with the chosen arm/LR has mean nRMSE >= 0.90.

If several arms pass, select the lowest trainable parameter count; if within 5% parameters, lower mean nRMSE wins.

## Interpretation

- If a larger/Fourier direct-XYZ Q passes, freeze that query class for the next matched carrier court. V3 is then diagnosed primarily as query-capacity confounding.
- If no arm passes, do not compare carriers. Reopen target/sampling formulation with an explicit easy-to-hard target ladder.

No carrier, Product, ModelRelease, Truth Builder or FIT8 authority changes here.
