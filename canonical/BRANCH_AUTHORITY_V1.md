# RealSaS — Branch Authority V1

**Canonical continuation branch:** `main`

Only `CURRENT_STATE.md` on `main` is continuation authority for new scientific, compiler, runtime and product work.

## Historical branch rule

Every non-main branch is retained as historical/experimental evidence unless a future commit on `main` explicitly promotes content from it. Side branches must not be treated as competing current truth simply because they contain newer-looking filenames, experiments or partial restorations.

Examples of historical/experimental branches:
- `g0-g1/single-pose-geometry` — preserved scientific lineage through E0/N-B3 and compiler integration precursor.
- `architecture/compiler-ir-solver-canonical-20260825` — preserved compiler architecture/restoration design lineage.
- `integration/compiler-runtime-canonical-20260828` — preserved integration work evidence.

Historical branches are not deleted merely to simplify navigation. Provenance is preserved; continuation authority is centralized by `main`.

## Promotion rule

If a future side experiment produces useful results, promotion requires an explicit `main` commit that records:
1. the source branch/commit,
2. the evidence or result being promoted,
3. compatibility/reconciliation with current `main`,
4. the new `CURRENT_STATE.md` continuation decision.

No implicit branch switch is allowed.
