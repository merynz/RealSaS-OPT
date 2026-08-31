# RealSaS — IRIS Reprojection V2 Gate-0 Synthetic Closure

**Date:** 2026-08-31  
**Status:** `PASS_SYNTHETIC_GATE0__REAL_CORPUS_GATE0_REQUIRED__NO_OPTIMIZER_AUTHORIZED`

## Authority

Commit under test: `409d17f50a9ad5a80bc43367a1e2ba6a70b16bb0`  
GitHub Actions run: `33406189796`  
Job: `gate0-cpu` / `99534250994`

Machine-readable result:
`experiments/iris_reprojection_v2_20260831/GATE0_SYNTHETIC_PREFLIGHT_RESULT_V1.json`.

Result content SHA-256:
`ef5765a0a107c4b57aebdadf8d857a613e61bb16493e5f3a396a4f88232c3d76`.

## Regression result

`9 / 9 PASS`:

- projection/backprojection roundtrip;
- common-world-Z row invariance across the 8-view orbit;
- hull padding monotonicity;
- synthetic sphere truth containment + domain reduction;
- thin-structure cell strata;
- robust evidence permutation invariance;
- robust evidence with missing views;
- streamed raw-descriptor memory budget;
- explicit UNKNOWN/resolution-unsupported state separation.

The separate synthetic preflight executable also returned `PASS_SYNTHETIC_GATE0`.

## Synthetic hull frontier

| padding px | truth containment | active lattice fraction |
|---:|---:|---:|
| 0 | 0.7845 | 0.233 |
| 1 | 1.0000 | 0.255 |
| 2 | 1.0000 | 0.296 |
| 4 | 1.0000 | 0.341 |
| 8 | 1.0000 | 0.501 |

This synthetic example demonstrates why containment and computational value must be reported together; it does **not** select the real-corpus padding.

## Streaming sanity

For a synthetic accounting example with 2,000,000 active candidates, 8 views and 64-D FP32 raw descriptors:

- per-candidate raw descriptor bytes: `2048`;
- 64 MiB working budget -> `32768` candidates/chunk;
- `62` chunks;
- max raw descriptor working bytes: `67108864`.

This validates the intended rule that raw eight-view descriptors need not be stored in a canonical 3D volume.

## Thin-structure warning retained

The synthetic sweep deliberately contains sub-cell structures even at spacing `.004`. Therefore synthetic PASS does **not** authorize `.004`, `.008` or `.016` for product geometry. The real-corpus Gate 0 must quantify authoritative projected/geometry thickness strata before spacing selection.

## Interpretation firewall

Supported:

> The deterministic reprojection/hull/lattice/evidence primitives are internally coherent on the sealed synthetic regression suite.

Not supported:

- real-corpus hull containment;
- real-corpus search reduction;
- any coarse spacing selection;
- DINO-S sufficiency;
- learned V2-A feasibility;
- product qualification.

## Next gate

`REAL_CORPUS_GATE0` is required before learned V2-A source/loss/runtime may be sealed.

**Scientific optimizer steps remain: 0.**  
**V2 optimizer authorization remains: NO.**
