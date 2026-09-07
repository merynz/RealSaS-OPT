# RealSaS — Live Authority Map

> **GENERATED FILE — DO NOT HAND EDIT.**  
> Policy/source of truth: `canonical/AUTHORITY_MAP_V1.json`. Repository branch names/heads are read live from `origin` with `git ls-remote --heads`; branch trees are not fetched.  
> State fingerprint: `c208826dca2797abb905a4d7fc50cc0c903b875720a281975d463b422f169fec`

**Continuation authority:** `CURRENT_STATE.md` on `main`.  
A recent branch, green Action, notebook, report, or source file is **not** continuation authority unless the manifest + `CURRENT_STATE.md` explicitly say so.

## Execution authority

- Runner mode: `SELF_HOSTED_LOCAL`
- Required labels: `self-hosted, linux, x64, realsas`
- Known runner: `realsas-wsl-1660ti`
- Operator path hint: `~/actions-runner`
- Budget policy: Do not fan out authority-map or routine science workflows across every experimental branch push. Prefer main authority changes, branch create/delete events, explicit manual dispatch, or already-required self-hosted scientific jobs.

## Rehydration order

1. `canonical/REHYDRATION_PACKET.md`
2. `CURRENT_STATE.md`
3. `canonical/CONTEXT_COVERAGE_AUDIT.md`
4. `canonical/LIVE_AUTHORITY_MAP.md`
5. `canonical/CONTEXT_STATE_V1.json`
6. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
7. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
8. `canonical/EXPERIMENT_REGISTRY_V1.json`
9. `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`

## Live experiment register

_No active experiments registered._

## Branch inventory — observed live

**CANONICAL: 1** / **EVIDENCE_ONLY: 3** / **EVIDENCE_ONLY_UNREGISTERED: 67** / **DELETE_CANDIDATE: 9**

### CANONICAL

| Branch | Head | Classification reason |
|---|---|---|
| `main` | `f7876ab694bb` | canonical continuation branch |

### EVIDENCE_ONLY

| Branch | Head | Classification reason |
|---|---|---|
| `demo/investor-single-specimen-e2e` | `47892ffa16b7` | Demo lineage only; never scientific continuation authority. |
| `exp/geppetto-ar01-skeleton-causal-v1-20260907` | `58ea876d3e61` | AR-01 closed at contract 41055bd073538d6b with verdict AR01_NO_TERMINAL_CLOSURE; retain source/prereg/provenance only. |
| `exp/lossless-rigging-evidence-v1-20260906` | `d840d96ece57` | Lossless RiggingSurfaceIR / V3 evidence lineage; no longer the active continuation experiment. |

### EVIDENCE_ONLY_UNREGISTERED

| Branch | Head | Classification reason |
|---|---|---|
| `agent/n1d-global-mechanical-solver-research-20260819` | `08620737ce52` | observed live; not explicitly registered active |
| `agent/n1d-observable-functional-audit-v2-20260820` | `bf3ef16ae250` | observed live; not explicitly registered active |
| `agent/n1d-observable-functional-quotient-rebuild-20260819` | `161d4d5d9893` | observed live; not explicitly registered active |
| `agent/n1d-v2-source-parity-recovery-20260820` | `32b06472b7ff` | observed live; not explicitly registered active |
| `architecture-v3-svg-20260901` | `f7473bc93a71` | observed live; not explicitly registered active |
| `architecture-v4-single-family-e2e-20260902` | `c84658b8b95c` | observed live; not explicitly registered active |
| `architecture/compiler-ir-solver-canonical-20260825` | `7163fac1f333` | observed live; not explicitly registered active |
| `architecture/v4-generic-strength-source-completion-20260902` | `7f39a846ad05` | observed live; not explicitly registered active |
| `audit/final-completion-plan-20260902` | `00e930e788d3` | observed live; not explicitly registered active |
| `audit/geppetto-v2-frozen-base-7f39` | `7f39a846ad05` | observed live; not explicitly registered active |
| `audit/iris-architecture-discipline-20260824` | `d7bba01c85e9` | observed live; not explicitly registered active |
| `baseline/h0-first-family-v1-20260904` | `a3cec7021f0d` | observed live; not explicitly registered active |
| `behavioral/geppetto-v2-integrity-v1-20260903` | `2b5d46718683` | observed live; not explicitly registered active |
| `cleanroom/iris-geometry-field-v1-20260904` | `e72636d8366d` | observed live; not explicitly registered active |
| `compiler-runtime-migration-audit-20260901` | `3267b3f1ec16` | observed live; not explicitly registered active |
| `compiler-runtime-migration-closure-20260901` | `255910d22fea` | observed live; not explicitly registered active |
| `consumer-interlock-v0-20260829` | `81ec9fd98d90` | observed live; not explicitly registered active |
| `dino-controlled-ladder-prereg-20260829` | `753b384d864c` | observed live; not explicitly registered active |
| `dino-zero-step-preflight-20260829` | `396ac67b6d65` | observed live; not explicitly registered active |
| `e2e/mage-scene-first-v1-20260905` | `c6b5108f672b` | observed live; not explicitly registered active |
| `feature/living-compile-v4-editor` | `cd56be7300c3` | observed live; not explicitly registered active |
| `first-family-fit/charactergen-backbone-v1` | `15896da77dca` | observed live; not explicitly registered active |
| `first-family-fit/v1-20260904` | `d8ba99dd4fe0` | observed live; not explicitly registered active |
| `first-family-fit/v1-20260904-work` | `953f2ded4911` | observed live; not explicitly registered active |
| `first-fit-base/main-20260904` | `1f1d8b5bd0df` | observed live; not explicitly registered active |
| `first-fit-base/main-20260904-v2` | `461fd25264f4` | observed live; not explicitly registered active |
| `fit/single-family-mage-v1-20260902` | `f2cd746f8beb` | observed live; not explicitly registered active |
| `g0-g1/single-pose-geometry` | `e423cd462002` | observed live; not explicitly registered active |
| `geometric-substrate-rename-20260901` | `7a27dd9fc7b4` | observed live; not explicitly registered active |
| `geppetto-arachne-v0-1-20260830` | `777e1bb56d9c` | observed live; not explicitly registered active |
| `geppetto-r6-teacher-projection-port-20260901` | `bfa51e74be02` | observed live; not explicitly registered active |
| `hardening/pre-fit-closure-v1-20260904` | `d4da94279e1f` | observed live; not explicitly registered active |
| `integration/compiler-runtime-canonical-20260828` | `e423cd462002` | observed live; not explicitly registered active |
| `integration/compiler-runtime-heavy-promotion-20260901` | `47892ffa16b7` | observed live; not explicitly registered active |
| `iris/mapanything-ortho-apache` | `3b66a4d2beb5` | observed live; not explicitly registered active |
| `legacy-geppetto-arachne-reconcile-20260901` | `f687ff8e78a2` | observed live; not explicitly registered active |
| `m4-closure-20260829` | `95a487b92ab0` | observed live; not explicitly registered active |
| `m4-execution-seal-20260829` | `d203b8233509` | observed live; not explicitly registered active |
| `m4r-closure-20260829` | `1fcd18a4cc70` | observed live; not explicitly registered active |
| `mwb0-closure-backlog-20260831` | `916ecc2c562a` | observed live; not explicitly registered active |
| `mwb1-identity-baseline-20260901` | `82b888c67afd` | observed live; not explicitly registered active |
| `mwbo-typed-seam-20260831` | `095ef0d439c8` | observed live; not explicitly registered active |
| `next/depth-bridge-route-correction-20260829` | `dcce855b0c93` | observed live; not explicitly registered active |
| `next/m4-grid-prereg-20260829` | `f26fd2837619` | observed live; not explicitly registered active |
| `next/proxy-evaluator-restoration-20260829` | `8f3d288c10de` | observed live; not explicitly registered active |
| `next/structured-depth-bridge-20260829` | `e362c473c2fd` | observed live; not explicitly registered active |
| `ops/temp-trigger-n1d-v2-recovery-20260820` | `536b40b6a6bf` | observed live; not explicitly registered active |
| `ops/trigger-n1d-v2-recovery-v2-verify-20260820` | `12a813a8f501` | observed live; not explicitly registered active |
| `promote/iris-scene-first-signed-main-v3-20260905` | `dfb087c6a23e` | observed live; not explicitly registered active |
| `promote/iris-scene-first-signed-v3-20260905` | `7cfb7efedbe9` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903` | `83aa411cc06d` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-directional-binding-firewall` | `800d3ccb042e` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-export-v2-safety` | `2eb09fe3e8c6` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-export-v2-safety2` | `2eb09fe3e8c6` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-export-v2-safety3` | `579a8697d7e3` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-motion-bake-backup` | `b46a39b7f1e7` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-p0-directional-binding` | `f0ce10051b4a` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-stage1-backup` | `2b5d46718683` | observed live; not explicitly registered active |
| `restoration/compiler-runtime-promotion-v1-20260903-stage1-treebase` | `2b5d46718683` | observed live; not explicitly registered active |
| `restoration/iris-mainline-promotion-v1-20260903-safety` | `a965a22e7a9a` | observed live; not explicitly registered active |
| `single-family-e2e-fit-v1-20260902` | `5fa4bf788328` | observed live; not explicitly registered active |
| `single-family-e2e-models-v1-20260902` | `362556db9b31` | observed live; not explicitly registered active |
| `source/generic-completion-v1-20260902` | `85a25b419a47` | observed live; not explicitly registered active |
| `tmp-do-not-use` | `953f2ded4911` | observed live; not explicitly registered active |
| `tmp/appearance-witness-fetch-20260826` | `ec623d005bdb` | observed live; not explicitly registered active |
| `tmp_should_not_create` | `a5edb2b23fc4` | observed live; not explicitly registered active |
| `tooling/export-frozen-source-7f39` | `43a1e5b2376f` | observed live; not explicitly registered active |

### DELETE_CANDIDATE

| Branch | Head | Classification reason |
|---|---|---|
| `do-not-use` | `91b4593d5b2a` | matches /^(do-not-use\|dummy-unused\|ignore-this)$/ |
| `dummy-unused` | `3609f2cc0cfc` | matches /^(do-not-use\|dummy-unused\|ignore-this)$/ |
| `e2e/mage-scene-first-v1-20260905-audit-temp` | `91b4593d5b2a` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |
| `first-family-fit/v1-20260904-debugtmp` | `953f2ded4911` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |
| `first-family-fit/v1-20260904-diag-scratch` | `953f2ded4911` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |
| `ignore-this` | `91b4593d5b2a` | matches /^(do-not-use\|dummy-unused\|ignore-this)$/ |
| `noop` | `c6b5108f672b` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |
| `noop2` | `c6b5108f672b` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |
| `scratch-mistake` | `91b4593d5b2a` | matches /.*(scratch\|debugtmp\|audit-temp\|noop\|mistake).*/ |

## Drift / validity

**VALID — manifest, active branches, required authority files, and `CURRENT_STATE.md` references agree.**

## Binding anti-conflation rules

- Only main/CURRENT_STATE.md is continuation authority.
- Scientific/mainline GitHub Actions run on the local self-hosted RealSaS runner; hosted-runner failures are apparatus/config only.
- Routine authority tracking must not create workflow-run fan-out across every experimental branch push.
- A branch is never active merely because it exists or has a recent commit.
- A closed experiment branch becomes evidence-only unless explicitly promoted through a separate transaction.
- Source existence is not experimental evidence.
- Component evidence is not a full-formulation verdict.
- A repository prereg/hash outranks detached chat/generated drafts when they disagree.
- Unindexed historical evidence means UNKNOWN/UNRECONCILED, never ABSENT.
- Promotion requires source + tests + evidence/result + CURRENT_STATE + provenance/revocation reconciliation.

## Update semantics

- Branch heads/classification are regenerated from live remote refs; do not manually copy branch SHAs into authority prose.
- Creating a new non-main branch is safe by default: it appears as `EVIDENCE_ONLY_UNREGISTERED` until explicitly registered.
- Starting an experiment requires adding it to `active_experiments` **and** naming its gate + branch in `CURRENT_STATE.md`; otherwise validation fails.
- Closing/promoting an experiment requires one atomic reconciliation of the machine manifest, human ledgers, result/provenance, source/tests, and `CURRENT_STATE.md`.
- `DELETE_CANDIDATE` is advisory only. No branch is deleted automatically.

