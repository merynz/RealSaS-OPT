# Canonical Authority Index

`canonical/` contains sealed decisions and evidence. It is intentionally not an executable package.

## Current top-level architecture

- `SYSTEM_ARCHITECTURE_V4_20260902.md` — product/type boundary.
- `CANONICAL_REALSaS_COMPLETION_PLAN_20260902.md` — generic source completion plan.
- `BEHAVIORAL_HARDENING_LEDGER_20260903.md` — preserved behavioral-hardening evidence; its earlier ordering that deferred repository/runtime restoration is superseded **for continuation order** by `RESTORATION_STATE.md`. The evidence inside remains historical truth.

## Active restoration

- `COMPILER_RUNTIME_PROMOTION_PLAN_V1_20260903.md`
- `COMPILER_RUNTIME_PROMOTION_MANIFEST_V1_20260903.json`
- `COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`

## Supersession rule

Closed dated records are not silently rewritten. A later decision supersedes continuation authority by explicit index entry while preserving the old record byte-for-byte.
