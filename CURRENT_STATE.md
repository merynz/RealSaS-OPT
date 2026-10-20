# RealSaS-OPT — Current State

**Date:** 2026-10-21  
**Production authority:** `main` until reviewed merge  
**Long-term research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Tie-safe V2 implementation source:** `7ae9b4886ec15725fcd95b7cfaeeb7aeab02b050`  
**Structural-evidence V2 implementation PR:** #135 — **MERGED**  
**Active research lane:** `SHARED_STRUCTURAL_EVIDENCE`  
**Active workstream:** `STRUCTURAL_EVIDENCE_INPUT_MATERIALIZATION_V2_HANDOFF`  
**Execution policy:** only the new **zero-optimizer V2 evidence materialization notebook** is the next execution boundary. No Relation/TESSA training, scientific diagnostic/court, optimizer/backward step, H-GEO-P4, SURF-A2, or scientific PASS/FAIL run is authorized.

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V3.json`.  
Active research plan: `docs/research/IRIS_SHARED_STRUCTURAL_EVIDENCE_PLAN_20261020.md`.  
Tie-safe materialization binding: `canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json`.

## Product invariant

```text
8-view observations
        ↓
IRIS shared structural scene
        + qualified metric geometry evidence
        ↓
geometry + graphics + mechanics evidence
        ↓
role-free structuralization / topology reconstruction
        ↓
Compiler qualification
        ↓
ONE canonical animation-ready M/G/W + graphics structure
        ↓
motion
        ↓
3D mechanics → source-preserved editable 2D presentation
```

There is no independent deformable visual mesh. Compiler alone mints canonical M. Named semantic classes may be auxiliary metadata; they are not structural authority.

## Immutable shared-scene authority

- IRIS V8 source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- shared scene core sealed after M1.

## Geometry boundary

`H-GEO-P3-SCALE-XFER` remains sealed `MIXED`; no P4, SURF-A2 or TESSA training was authorized. Geometry remains behind the typed `QUALIFIED_SURFACE_PROVIDER → XYZ / normals / local scene evidence / mechanics` interface. The current endpoint diagnostic is not silently promoted into a universal product gate.

## Structural-evidence bridge

PR #132 established the source bridge:

- pairwise front/order separate from global visibility;
- role-free typed presentation evidence with explicit validity;
- TESSA V3 scene + surface/mechanics + presentation memory;
- matched `1024` total conditioning-token budget for null-control and presentation-evidence arms;
- Compiler/runtime presentation-consumption regressions.

No scientific result follows from source existence.

## V1 materialization — byte-valid, rejected as scientific input

`STRUCTURAL-EVIDENCE-INPUT-MATERIALIZATION-V1` completed with **0 optimizer / 0 backward** and Drive byte-level round-trip.

Authority record: `canonical/STRUCTURAL_EVIDENCE_V1_POSTRUN_AUDIT_20261021.json`.

The V1 corpus contains:

- `484,139` sampled rays;
- `477,090` foreground-hit rays;
- `1,194,462` surface hits;
- `558,951` base pairs / `1,117,902` oriented rows;
- `167,116` rows where pairwise front and global visibility decouple;
- maximum stack depth `33`.

The payload audit found one blocking authority edge case: **25 Bat base pairs / 50 oriented rows** contain distinct-component equal-depth hits. V1 sorted such hits by face ID and promoted serialization rank into front / rank-zero visibility. Therefore:

`V1_BYTE_VALID__POSTRUN_PAYLOAD_AUDIT_REJECTED_FOR_SCIENTIFIC_INPUT`

V1 is immutable historical evidence only. It is not authorized as a Relation/TESSA training input or scientific court input.

## Tie-safe V2 source — merged and frozen

PR #135 merged tie-safe executable source at:

`7ae9b4886ec15725fcd95b7cfaeeb7aeab02b050`

V2 deliberately reuses the exact V1 ray sampling/intersection machinery; only authority semantics change:

- equal-depth pairwise front → **ABSTAIN**;
- unique minimum-depth hit → visible, valid;
- tied minimum-depth hits → visibility **ABSTAIN**;
- deeper hits → invisible, valid even when the minimum is tied;
- invalid target values are zero;
- V1-style arrays without explicit validity fail closed;
- the V2 admission validator opens the NPZ payload and checks CSR/depth/order/tie/visibility/pair-symmetry semantics in addition to hashes.

The pair point domain remains exact legacy IRIS-normalized source world. Before TESSA presentation memory, XYZ must be multiplied by `0.5`, matching the existing TESSA V2 corpus coordinate bridge.

Dedicated `iris-structural-evidence-v2-source-gate`, TESSA structuralization, IRIS V2 and model-mainline gates were green before the implementation merge.

## Immediate next boundary

`canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json` is now `READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS` and binds implementation source `7ae9b4886ec15725fcd95b7cfaeeb7aeab02b050`.

The next notebook may only:

1. hydrate and hash-verify the exact sealed Stage01 / legacy Fit8 / authored-teacher / prepared authority;
2. run the V2 all-hit producer at `8192` candidate rays/view and grid `32`;
3. emit eight `.all_hit_relation_evidence_v2.npz` artifacts plus `ALL_HIT_RELATION_EVIDENCE_V2_MANIFEST.json`;
4. run the deep fail-closed V2 admission validator;
5. persist exact hashes and a typed Drive round-trip receipt.

It must execute **0 optimizer steps, 0 backward steps**, no Relation/TESSA training, no scientific diagnostic/court and no research-hypothesis PASS/FAIL classification.

After and only after the V2 corpus is admitted and payload-audited, source-only work may bind the presentation input and lock the future matched-capacity A/B training/court preregistration. The later training notebook requires a separate user handoff.

## Current non-authorizations

- H-GEO-P4 training;
- SURF-A2 scientific run;
- Relation training;
- TESSA V2/V3 training;
- Structural-Evidence A/B court;
- fresh scientific diagnostic execution;
- threshold relaxation after results.
