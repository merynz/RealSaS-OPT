# Authority Model

## Canonical owner

`compiler/realsas_compiler_core/` owns current product identity, graph qualification, lineage hashes, capability contracts and exact-state proof/export eligibility.

## Subordinate services

`compiler/realsas_compiler_services/` may measure, solve, diagnose, attribute, repair, serialize or export. A service may return evidence or a proposed transformation; it does not mint competing canonical identity.

## Runtime

Runtime is a projection/consumer. It can reject malformed or unsupported packages, but it does not redefine the product.

## Historical source

Historical code has four possible statuses:

- `PROMOTE_REBIND`: useful semantics, adapted to current types/ownership.
- `SOURCE_DIFF_SELECT`: multiple historical implementations compete; compare formulation/tests and promote one.
- `ARCHIVAL_ONLY`: valuable evidence, no current executable role.
- `DO_NOT_PROMOTE_MONOLITH`: structurally conflicts with current ownership and must not be resurrected wholesale.

A historical SHA is provenance, not execution permission.
