# P Formulation V3 — CI237 G1 Apparatus Supersession

Date: 2026-08-24
Status: `CI237_V1_RESULT_PRESERVED__SCIENTIFIC_GEOMETRY_NOT_OPENED__CI244_V2_CURRENT`

## What CI237 produced

The CI237 notebook completed and persisted the preregistered label:

`P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`

This file/result is preserved and must not be deleted or rewritten.

## Why it is not a scientific geometry failure

All 16/16 panel assets failed before any gauge, projection or analytic-reconstruction measurement was opened.

The only fatal class was the V1 auditor condition:

```text
set(primary_geometry.npz fields) == {vertices, faces}
```

The master corpus `primary_geometry.npz` is intentionally a superset and may contain canonical transforms, vertex normals and rig/mechanics metadata. The frozen preregistration did **not** require the source NPZ to contain only two fields. It required the G1 process to **read only `vertices` and `faces`**.

Consequently CI237 V1 returned empty (`n=0`) geometry summaries and cannot support a claim that canonical gauge, raster-P projection, camera geometry or the P-V3 analytic reconstruction failed.

Classification:

`APPARATUS_FALSE_REJECT__MASTER_NPZ_SUPERSET_MISTAKEN_FOR_CONSUMPTION_LEAK`

## Corrected semantics

The master NPZ may be a superset. G1 now:

- opens the NPZ with `allow_pickle=False`;
- requires `vertices` and `faces` to exist;
- loads values only for `vertices` and `faces`;
- records extra field **names** for provenance;
- never loads/interprets extra field values.

A new regression inserts an object-dtype hidden field that cannot be loaded under `allow_pickle=False`. The regression passes only if the auditor consumes `vertices/faces` and ignores hidden values. Missing `vertices` or `faces` remains fatal.

## CI244 release authority

The corrected optimizer-zero closure artifact is P Formulation V3 Closure Bundle V2, built and isolated-tested by GitHub Actions run #244 from source head:

`985997ba87faa92cc004ad9ab70efeacc400886f`

Artifact:

`iris-v2-p-formulation-v3-closure-bundle-v2`

Artifact ID:

`9527242584`

ZIP SHA-256:

`e04bc7c9638038688304b558cb816f8bccaf5e253bc2195977c0c2328f693667`

The panel, thresholds, P formulation, camera contract and allowed scientific labels are unchanged from the frozen preregistration. TUNE/sealed splits remain unopened and optimizer steps remain zero.

## Authority rule

CI237 V1 remains historical evidence of an apparatus false reject. CI244 V2 is the only executable authority for reopening the same frozen G1 measurement after that pre-measurement bug.

Training remains forbidden until corrected G1 yields `P_V3_FORMULATION_GEOMETRY_CLOSED`.
