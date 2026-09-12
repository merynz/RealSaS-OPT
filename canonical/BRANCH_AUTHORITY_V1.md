# RealSaS — Branch Authority V1

**Canonical repository-wide continuation authority:** `CURRENT_STATE.md` on `main`

`main` owns the continuation decision and machine registration. An explicitly registered non-main branch may carry the **current executable experiment/repair lineage** while it is named consistently by `main/CURRENT_STATE.md` and `canonical/AUTHORITY_MAP_V1.json`.

## Current active exception

`repair/mage-full-subject-reclosure-20260912` is currently registered as `ACTIVE_EXPERIMENT` for `MAGE_FIT2_PIPELINE_REFIT`.

This means:

- the branch is current executable repair/refit work, not merely historical evidence;
- repository-wide stop/go and continuation semantics still come from `main/CURRENT_STATE.md`;
- exact experimental/source claims must be read from the active branch's prereg/result/authority artifacts;
- branch recency alone grants no authority — the explicit main registration does;
- when the experiment closes, main must reconcile result/provenance/registry/journal/ledgers and either promote/refreeze or return the branch to evidence-only status.

## Default non-main branch rule

Every non-main branch **not** explicitly registered by current main authority is retained as historical/experimental evidence by safe default. Side branches must not be treated as competing current truth simply because they contain newer-looking filenames, experiments or partial restorations.

Examples of historical/experimental branches:

- `g0-g1/single-pose-geometry` — preserved scientific lineage through E0/N-B3 and compiler integration precursor;
- `architecture/compiler-ir-solver-canonical-20260825` — preserved compiler architecture/restoration design lineage;
- `integration/compiler-runtime-canonical-20260828` — preserved integration work evidence.

Historical branches are not deleted merely to simplify navigation. Provenance is preserved; continuation authority remains centralized by `main`.

## Starting an active experiment branch

A side branch becomes current executable authority only when one atomic main-side reconciliation names:

1. experiment/gate id;
2. exact branch;
3. current status/question;
4. what the experiment does **not** prove;
5. matching `CURRENT_STATE.md` stop/go state.

`canonical/AUTHORITY_MAP_V1.json` is the machine registration source. `canonical/LIVE_AUTHORITY_MAP.md` is only its generated live view.

## Promotion / closure rule

If an active or historical side experiment produces useful results, closure/promotion requires an explicit `main` transaction that records, as applicable:

1. source branch/commit and frozen apparatus;
2. exact evidence/result/provenance;
3. compatibility/reconciliation with current product/scientific authority;
4. experiment registry + scientific journal;
5. architecture/experiment ledgers when interpretation changes;
6. `CURRENT_STATE.md` continuation decision;
7. explicit promotion/refreeze/supersession where applicable.

No implicit branch switch or implicit promotion is allowed.
