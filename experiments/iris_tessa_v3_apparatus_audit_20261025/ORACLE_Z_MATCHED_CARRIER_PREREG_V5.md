# Oracle-Z Matched Carrier Court V5 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** representation-search only.

## Question

With the surface target now calibrated by `Q192_D3_FOURIER4`, does the current triplane scene carrier impose a reproducible surface/thin-support limitation relative to a parameter-matched dense 3-D carrier when both are decoded by the same shared query operator?

## Frozen data / target

Reuse the V3 scene families, five seeds, four thickness/radius factors, train/eval pools, target function and normalization unchanged.

`h_ref = 2/24`; factors = `4,2,1,0.5`; families = `SURFACE_STRUCTURED`, `THIN_SURFACE`.

## Query operator

Freeze the V4 selected query **class**:

- width 192;
- three hidden SiLU layers;
- Fourier4 xyz basis `[xyz, sin/cos(2^k*pi*xyz), k=0..3]`;
- shared weights across all scenes for carrier arms.

The Q weights themselves are trained within each arm; only the architecture/basis is frozen.

## Arms

### U — PER_SCENE_DIRECT_XYZ_UPPER_BOUND

One Q192/Fourier4 query per scene, exact XYZ input, no carrier. This is a target-learnability upper bound only and is never a deployable carrier candidate.

### N — SHARED_Q_COORDINATE_ONLY_NEGATIVE_CONTROL

One shared Q192/Fourier4 query across all scenes with a zero 48-D carrier feature. It has coordinates but no scene-specific state.

### T — TRIPLANE_SHARED_Q

Per scene: `3 x 16 x 24 x 24 = 27,648` trainable carrier values. Bilinear XY/XZ/YZ samples concatenate to 48 carrier channels. Shared Q receives `[carrier48, Fourier4(xyz)]`.

### D — DENSE3D_SHARED_Q

Per scene: `16 x 12 x 12 x 12 = 27,648` trainable carrier values. Trilinear sample gives 16 channels and is expanded by a fixed non-trainable 3x repeat to 48 channels. Shared Q receives `[carrier48, Fourier4(xyz)]`.

Thus T and D have exactly equal scene-carrier parameter counts and exactly the same trainable shared-Q architecture/input width.

## Optimization

- LR: `3e-3`;
- no weight decay;
- direct upper bound: 5000 steps;
- coordinate-only control: 5000 steps;
- carrier arms: 8000 steps;
- fixed batch/pool/eval construction from V3;
- deterministic fixed seeds;
- runtime is not a scientific gate; workflow uses the 360-minute runaway guard.

## Measurements

For every family/factor:

- mean/worst normalized RMSE;
- mean/worst local-surface normalized RMSE;
- per-seed values.

Also record:

- exact trainable carrier parameter counts;
- shared-Q parameter count;
- scene-carrier shuffle ablation for T/D (roll carrier state across scenes while preserving coordinates/query weights);
- coordinate-only negative-control error.

## Target-valid points

A family/factor is eligible for carrier interpretation only when U satisfies:

- mean nRMSE <= 0.12;
- worst nRMSE <= 0.18;
- mean local nRMSE <= 0.18.

A carrier supports an eligible point under the same gates.

## Apparatus gates

The court is invalid for a carrier verdict unless:

1. T and D carrier parameter counts are exactly equal;
2. at least the `4*h_ref` point for both families is target-valid under U;
3. at least one of T/D supports both `4*h_ref` family points;
4. coordinate-only N does not satisfy the support gate on all eligible points;
5. for every carrier considered successful, rolling scene carrier state increases its overall mean nRMSE by >=25% relative, proving the carrier rather than coordinates alone carries scene-specific state.

## Selection

Select `DENSE3D` only when, within at least one family, there are **two adjacent target-valid factors** such that:

- D passes both;
- T fails both;
- D improves mean nRMSE over T by >=25% on both;
- D passes scene-shuffle dependence.

Otherwise, if apparatus gates are valid and there is no such robust adjacent-scale dense advantage, retain the current `TRIPLANE` provisionally by simplicity/continuity.

If neither carrier supports the easy calibrated points, or the controls fail, return apparatus inconclusive and do not force a carrier selection.

## Non-claims

This court selects only the pre-Truth scene carrier form. It does not prove the real shared-scene hypothesis, final extraction resolution, product quality, Truth Builder labels, FIT8 genericity, or ModelRelease authority.
