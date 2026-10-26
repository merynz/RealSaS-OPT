# Oracle-Z Carrier Robustness Court V6 — preregistration

**Status:** SEALED AFTER V5, BEFORE V6 OUTCOME  
**Authority:** final pre-Truth carrier selection only.

## Why this court exists

V5 was valid and its preregistered large-effect rule retained triplane provisionally, but dense crossed more absolute support gates while the mean differences were small. That pattern can be caused by optimization variance around hard thresholds.

V6 therefore does **not** alter the representation, target, query class or data family. It asks whether the small dense advantage is reproducible across independent optimizer seeds on only the V5 target-valid family/scale points.

## Frozen target-valid set from V5

Interpret only:

- `SURFACE_STRUCTURED`: factors `4,2,1,0.5 * h_ref`;
- `THIN_SURFACE`: factors `4,1 * h_ref`.

The two V5 target-invalid thin points remain excluded; V6 cannot make them valid retroactively.

## Arms

Exactly the V5 matched carrier arms:

- `TRIPLANE_SHARED_Q`: `3x16x24x24 = 27,648` carrier values per scene;
- `DENSE3D_SHARED_Q`: `16x12x12x12 = 27,648` carrier values per scene;
- identical shared `Q192_D3_FOURIER4` architecture;
- same fixed non-trainable dense 16->48 feature repeat;
- same LR `3e-3`, 8000 steps and sampling construction.

## Replication

Run three independent optimizer seeds. For each seed, T and D receive the same sampled training batches. Report:

- overall mean nRMSE across the frozen valid scene set;
- per family/scale mean nRMSE;
- count of the six V5-valid points passing the same support gate;
- paired dense relative gain `(T-D)/T`.

## Final carrier rule

Select `DENSE3D` only if all are true:

1. dense overall nRMSE is lower on **all three** optimizer seeds;
2. mean paired dense relative gain across the three seeds is >=5%;
3. dense passes at least as many of the six support groups on every optimizer seed;
4. averaged across optimizer seeds, dense is not >10% worse than triplane on any one frozen family/scale group.

Otherwise retain `TRIPLANE` as the final **pre-Truth** carrier by continuity/simplicity.

This 5% threshold is a robustness/materiality threshold, not a replacement for V5's large-effect causal selection rule. V5 already established that no dramatic adjacent-scale dense advantage exists.

## Output

V6 must emit exactly one of:

- `DENSE3D_SELECTED_PRETRUTH_V6`;
- `TRIPLANE_SELECTED_PRETRUTH_V6`.

The output becomes the carrier dependency for the continuous joint-system court.

## Non-claims

This is still synthetic pre-Truth representation evidence. Authored causal truth can reopen the carrier later, but FIT8 must not silently change it without a new court.
