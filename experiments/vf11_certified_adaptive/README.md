# VF-11 Certified Adaptive Research Lane

Research-only branch for the VF-11 extraction/discretization re-scope.

## Claim boundary

- No product authority is minted.
- No Knight witness or mainline run is started.
- The historical R512/X1 authority is not mutated; the full 8-phase X1 run is on HOLD by a newer research preregistration.
- Shipping zero-surface extraction code is unchanged.

## Short lane

- `range_engine_v1.py`: conservative field-value and fixed-direction derivative bounds for the current bilinear-triplane + LayerNorm + SiLU SDF decoder.
- `profile_v1.py`: deterministic per-depth certifiability profiling.
- `tests/research/test_vf11_range_engine_v1.py`: synthetic/adversarial containment and determinism smoke bank.

The current interval formulas are research evidence only because v1 does not yet use a directed-rounding IEEE-754 kernel.

## Long lane

The next long run is `KNIGHT_FIELD_CERTIFIABILITY_RESEARCH_V1`.
It will use an existing frozen Knight field/checkpoint only, with no training and no product threshold selection.
Execution will be delivered as a logged Run-All notebook bound to an exact repo commit and exact input hashes.
