# RealSaS — Branch Authority V1

**Canonical repository-wide continuation authority:** `CURRENT_STATE.md` on `main`

`main` owns continuation decisions and machine registration. An explicitly registered non-main branch may carry the **current executable experiment/repair lineage** while it is named consistently by `main/CURRENT_STATE.md` and `canonical/AUTHORITY_MAP_V1.json`.

## Current active exception

`repair/mage-full-subject-reclosure-20260912` is registered as `ACTIVE_EXPERIMENT` for `MAGE_FIT2_PIPELINE_REFIT`.

This means:

- the branch is current executable repair/refit work, not merely historical evidence;
- repository-wide stop/go and continuation semantics still come from `main/CURRENT_STATE.md`;
- exact experimental/source claims must be read from the active branch's prereg/result/authority artifacts;
- branch recency alone grants no authority — explicit main registration does;
- executable implementation on the branch is not silently merged/promoted to main;
- when the experiment closes, main must reconcile result/provenance/registry/journal/ledgers and explicitly promote/refreeze or return the branch to evidence-only status.

Current active-branch substate:

- corrected H1 observable authority CLOSED;
- GSA8192 real V0..V7 Stage-0 evidence CLOSED;
- fresh Geppetto FIT2 RUNNING, PASS not claimed;
- `FIT2_MESH_PRODUCT_RECLOSURE_CONTRACT_AND_IMPLEMENTATION` CLOSED PASS on the active branch, proven by self-hosted run `34716890157` (`41 passed in 4.36s`);
- corrected real FIT2 mesh result still pending fresh G/W;
- fresh Arachne, professional motion, exact runtime/export reclosure and PRODUCT_PASS remain open.

Implementation-level mesh contract closure does **not** promote its code into `main` and does **not** establish a real corrected mesh result.

## Default non-main branch rule

Every non-main branch **not** explicitly registered by current main authority is retained as historical/experimental evidence by safe default. Side branches must not be treated as competing current truth simply because they contain newer-looking filenames, experiments or partial restorations.

Examples of historical/experimental branches:

- `g0-g1/single-pose-geometry` — preserved E0/N-B3 and compiler-integration precursor lineage;
- `architecture/compiler-ir-solver-canonical-20260825` — compiler architecture/restoration lineage;
- `integration/compiler-runtime-canonical-20260828` — integration work evidence;
- `seal/geppetto-reference-strength-fit1-20260908` — historical sealed Geppetto FIT1 evidence;
- `exp/arachne-skintokens-cleanroom-fit1-20260908` — historical Arachne/A0 research lineage;
- `demo/investor-single-specimen-e2e` — historical demo lineage only.

Historical branches are not deleted merely to simplify navigation. Provenance is preserved; continuation authority remains centralized by `main`.

## Starting an active experiment branch

A side branch becomes current executable authority only when one main-side reconciliation names:

1. experiment/gate id;
2. exact branch;
3. current status/question;
4. what the experiment does **not** prove;
5. matching `CURRENT_STATE.md` stop/go state.

`canonical/AUTHORITY_MAP_V1.json` is the machine registration source. `canonical/LIVE_AUTHORITY_MAP.md` is only its generated view.

## Promotion / closure rule

If an active or historical side experiment produces useful results, closure/promotion requires an explicit `main` transaction recording, as applicable:

1. source branch/commit and frozen apparatus;
2. exact evidence/result/provenance;
3. compatibility/reconciliation with current product/scientific authority;
4. experiment registry + scientific journal;
5. architecture/experiment ledgers when interpretation changes;
6. `CURRENT_STATE.md` continuation decision;
7. explicit implementation promotion/refreeze/supersession where applicable.

No implicit branch switch or implicit promotion is allowed.
