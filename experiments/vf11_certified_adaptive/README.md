# VF-11 Certified Adaptive Research Lane

Research-only branch for the VF-11 extraction/discretization re-scope.

## Claim boundary

- No product authority is minted.
- No Knight witness or mainline run is started.
- The historical R512/X1 authority is not mutated; the full 8-phase X1 run remains on HOLD.
- Shipping zero-surface extraction code is unchanged.

## V1 closure

`range_engine_v1.py` is retained as historical research apparatus only.

The frozen Knight certifiability run `20260921T114055Z` returned 100% `UNKNOWN`
through depth 7. The implementation is therefore sealed as
`FALSIFIED__PRACTICALLY_VACUOUS_BOUNDS`, not as evidence against certified
adaptive extraction itself.

The v1 failure had two distinct causes:

1. naive forward interval propagation destroyed useful correlation through
   LayerNorm / dense nonlinear layers;
2. a float boundary bookkeeping path produced spurious
   `TRIPLANE_BORDER_REGION` outcomes at exact align_corners=False pixel-center
   boundaries.

See `canonical/VF11_RANGE_ENGINE_V1_CLOSURE_20260921.json`.

## C0 range lane

Before reopening regularity, v2 asks only:

> Can we prove that a cell does not contain the zero set?

`range_engine_c0_v2.py` therefore emits only:

- `PROVEN_EMPTY_POSITIVE`
- `PROVEN_EMPTY_NEGATIVE`
- `PROVEN_ZERO_EXISTS`
- `UNKNOWN`

It uses exact triplane interpolation-knot splitting, shared affine
x/y/z + xy/xz/yz generators, the exact variance nonnegativity invariant,
correlation-preserving linear propagation, SiLU secant residual enclosures,
and unresolved-only micro-refinement.

`PROVEN_REGULAR` is deliberately forbidden in v2. C1 regularity work may
start only after C0 is shown to be useful and tightening.

The formulas are research-conservative, but v2 still uses ordinary float64
rather than a directed-rounding backend. It cannot mint shipping proof
authority.

## Long lane

The next long run is `KNIGHT_FIELD_C0_RANGE_RESEARCH_V2`.

It uses the existing frozen Knight field/checkpoint only:

- no training,
- no Knight witness execution,
- no product threshold selection,
- no extractor bake-off,
- exact repo/checkpoint/input hashes,
- deterministic parent-cell sampling,
- logged Run-All notebook,
- final result bundle on Drive.

The run measures EMPTY / ZERO-EXISTS / UNKNOWN, proof depth, range-box cost,
regime splitting, throughput, and empirical contradiction alarms.
