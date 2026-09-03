# Repository Structure Policy

## Goals

1. A new engineer can locate the current authority in under a minute.
2. Historical evidence remains reproducible without becoming an alternate runtime.
3. Production code, services, experiments, tests and evidence never share ambiguous ownership.
4. Moving/renaming code is allowed only with import/workflow migration and regression coverage.

## Naming

- `core` means semantic/canonical authority.
- `services` means subordinate deterministic execution behind core contracts.
- `reference` means conformance/reference implementation, not production owner.
- `historical` means provenance-only unless an explicit promotion manifest says otherwise.
- dated canonical files are immutable evidence once closed; supersession happens by index, not silent rewrite.

## Cleanup policy

Do not mass-delete or mass-rename while scientific gates are active. First establish indexes and authority boundaries; then move only paths whose consumers can be migrated atomically.

## Promotion policy

Every historical promotion requires: source SHA -> file SHA -> classification -> current typed consumer -> causal/regression gate -> explicit authority status.
