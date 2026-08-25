# IRIS Single-Pose V2 — P Formulation V5 CI283 G1 Result Interpretation

Date: 2026-08-24

## Canonical result

`P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`

The frozen optimizer-zero real-corpus G1 completed on the exact 16 FIT-only sentinel panel. The scientific result JSON was persisted before a notebook-only finalization typo (`allowed = {{...}}`) raised `TypeError: unhashable type: 'set'`. The typo occurred after `result = json.load(...)`; it did not affect the scientific runner or its output.

## Integrity

- asset count: 16
- styles: `cel_clean`, `ink_cel`
- learner resolution conditions: 1024, 512, 256
- total asset × style × resolution cells: 96
- fatal assets: 0
- optimizer steps: 0
- training authorized: false
- TUNE consumed: false
- sealed splits opened: false
- camera half-extent model input: false
- native scale source resolution: 1024
- scale re-estimated after resize: false

Persisted result SHA-256:

`e233e5f9dfc15f7888d637e1c047094836f40152b4ab5a066bf49b8569c671c7`

## Frozen P gate result

Frozen gate: P p95 <= `0.005` in every asset/style/resolution cell.

Observed distribution across all 96 cells:

- max P p95: `0.0013928374974057078`
- median P p95: `7.315552629734155e-05`
- p90: `0.0008157795993611216`
- p95: `0.0013928374974057078`
- min: `6.636565376538783e-05`

Therefore all 96/96 cells satisfy the unchanged P precision gate.

Worst asset remains `asset_00aa1b666ba193851a498194` with native-derived `h=0.5589519650655022` and P p95 `0.0013928374974057078`; the exact same scalar is transported at 1024, 512 and 256.

Second hard scale case `asset_76313e4bd82b82fcd1659c70` uses native-derived `h=0.6183574879227053` and P p95 `0.0008157795993611216` across all three learner-resolution conditions.

Native observable-scale diagnostic against teacher camera half-extent across 32 asset/style measurements:

- max absolute error: `0.0018578393178214636`
- median: `8.438818565392747e-05`
- p95: `0.0014639099979944054`

## Canonical interpretation

`P-V5 native-scale-once geometry formulation is CLOSED/PASS.`

This closes the formulation bug chain found after CI202:

1. free absolute XYZ regression was badly conditioned;
2. analytic screen-plane + learned depth survived;
3. fixed `h=0.54` was falsified;
4. native image-derived scale survived;
5. post-resize scale re-estimation was falsified;
6. native-scale-once transport passes the original P p95 <=0.005 target at 1024/512/256.

No threshold was loosened and no asset-specific rescue branch was introduced.

## Notebook finalization bug

The first CI283 notebook contained a Python literal typo in the post-result finalization cell:

```python
allowed = {{
    "P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED",
    "P_V5_NATIVE_SCALE_ONCE_GEOMETRY_FAIL",
}}
```

This constructs an outer set containing an inner set and raises `TypeError: unhashable type: 'set'`.

A recovery notebook must not rerun the scientific G1 if the persisted result JSON already exists. It should validate the existing result, write the completion marker, and copy the authority files only.

## Next gate

Training is still not automatically authorized by this geometry closure. The next optimizer-bearing experiment must receive a separately frozen FIT-only depth/P overfit preregistration. TUNE remains closed during that diagnostic.
