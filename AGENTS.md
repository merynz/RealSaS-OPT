# RealSaS-OPT Agent Entry Contract

This repository is structured so a new AI agent/chat/session can reconstruct current scientific and product context without relying on conversational memory.

## Mandatory first-read order

Before making any architecture, experiment, branch or scientific-state claim:

1. `canonical/REHYDRATION_PACKET.md`
2. `CURRENT_STATE.md`
3. recent tail of `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
4. `canonical/MAGE_FIT2_REAL_E2E_REOPENING_CONTEXT_20260912.md`
5. `canonical/AUTHORITY_MAP_V1.json`
6. `canonical/CONTEXT_STATE_V1.json`
7. `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`
8. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
9. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
10. `canonical/EXPERIMENT_REGISTRY_V2.json`
11. `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md` when historical FIT1 interpretation/provenance matters
12. `canonical/FIT1_COMMIT_LINEAGE_V1.md` / `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` when locating exact older evidence

Current executable experiment branch is explicitly named by main authority. At present:

`repair/mage-full-subject-reclosure-20260912`

For current FIT2 executable details, also read on that branch:

- `canonical/MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1.json`
- `canonical/MAGE_FIT2_MESH_PRODUCT_RECLOSURE_AUDIT_20260912.md`
- `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`
- current `CURRENT_STATE.md`
- relevant `experiments/mage_full_subject_reclosure_v1/` artifacts
- `compiler/realsas_compiler_core/mesh/product_qualification.py`

`canonical/EXPERIMENT_REGISTRY_V1.json` and `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` are historical continuity/provenance inputs. Follow current machine pointers instead of guessing from filenames.

If generated views are stale, regenerate locally:

```bash
python3 tools/render_fit1_commit_lineage.py
python3 tools/build_knowledge_artifact_catalog.py
python3 tools/audit_context_coverage.py
python3 tools/render_authority_map.py --write
python3 tools/render_rehydration_packet.py --write
```

## Current state guard — do not start from stale FIT1 product readiness

The first real eight-view Mage FIT1 end-to-end product execution visibly failed despite historical local Geppetto/Arachne closures. The correct interpretation is:

**historical FIT1 local closures preserved -> real E2E product contradiction -> corrected observable H1/GSA -> fresh same-Mage FIT2 pipeline reclosure.**

Current high-level state:

- corrected H1 observable product surface — CLOSED PASS;
- corrected GSA8192 real V0..V7 Stage-0 evidence — CLOSED PASS;
- fresh Geppetto FIT2 — RUNNING from scratch; historical checkpoint loading forbidden; PASS not claimed;
- fresh Arachne FIT2 — BLOCKED ON GEPPETTO;
- FIT2 mesh product qualification contract/implementation — CLOSED PASS on active branch;
- corrected real FIT2 mesh result — PENDING fresh G/W;
- professional motion — OPEN;
- exact runtime/export reclosure — OPEN;
- PRODUCT_PASS — NOT CLAIMED;
- unseen/FIT8/LOFO — BLOCKED until same-Mage product reclosure.

Never answer “FIT1 closed, next unseen” as current state.

## Learned subsystem shorthand expansion — mandatory

A RealSaS learned model is **not** the entire subsystem around that model.

- **IRIS / H1** means: eight observations + exact cameras -> learned geometry/support evidence -> deterministic Compiler/GSA assembly, validation and provenance -> `RiggingSurfaceIR`. Current Mage product authority is the corrected observation-bound H1/GSA8192 lineage on the active FIT2 branch. Historical V2/V3 packages remain provenance/support evidence where referenced; the old incomplete-subject Mage geometry is not current product authority.
- **Geppetto** means: `RiggingSurfaceIR` -> learned skeleton/control/root/parent/mechanical-salience proposal -> Compiler exact graph qualification -> `QualifiedSkeletonIR` / canonical IDs.
- **Arachne** means: qualified S+G -> learned skin/deformation proposal -> Compiler skin/reference/simplex qualification -> `QualifiedSkinIR`; historical V5 FIT1 is scoped old-lineage evidence, not current corrected W.
- **Directional mesh** means: Compiler-owned view-local deformation/render discretization derived from admitted surface + exact observation authority -> `QualifiedEditableMeshIR`.
- **Compiler** is deterministic qualification/canonicalization/proof/routing authority; it is **not permission to invent missing learned semantics**.
- **Runtime/export** consumes qualified identities and may project/bake them; it is never a second topology or semantic authority.

Binding memory guard:

> **Geppetto proposes; Compiler qualifies the canonical rig.**

Apply the equivalent distinction to IRIS/GSA, Arachne/skin and future learned motion.

Before moving responsibility across layers, inspect `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` and ask whether the move creates a second semantic owner, hides learned failure with deterministic repair, or violates fail-close.

## Mesh product authority guard — mandatory

The real E2E contradiction proved that mesh is product-critical. It is the deformation/render domain carrying artist-visible pixels.

On the active branch, low-level:

`qualify_mwb2_observation_cdt_mesh(...)`

is only legal CDT/surface-support compatibility evidence. It is **not product closure**.

Product admission is:

`qualify_fit2_product_mwb2_observation_cdt_mesh(...)`

from `compiler/realsas_compiler_core/mesh/product_qualification.py`.

It must:

1. bind exact `ObservationRasterDomain` authority;
2. validate observation mask/source-alpha identity;
3. reconstruct promoted mesh raster coordinates from exact `SurfaceSupportBinding` + admitted S raster bindings;
4. independently rerasterize the exact promoted mesh;
5. recompute recall / precision / IoU / component recall / largest connected uncovered region;
6. remeasure topology/triangle quality;
7. apply frozen FIT2 product gates fail-closed;
8. reseal exact observation provenance into mesh lineage.

`candidate.residual_report` is diagnostic only and cannot establish product coverage.

Current frozen mesh product thresholds are preregistered on the active branch. Do not relax them after seeing corrected FIT2 results without a new explicit preregistration/fresh lineage.

Self-hosted contract/implementation evidence: run `34716890157`, runner `realsas-wsl-1660ti`, static compile PASS, `41 passed in 4.36s` at head `b1dc7fca6b6497d97c7be727d66fd9c0b64c3268`.

This is **contract/implementation PASS only**, not corrected real Mage mesh PASS.

## Supported inserted vertices / component identity

`LOCAL_CONVEX_INTERPOLATION` is the legal inserted-vertex route:

`P(v) = Σ a_i P(S_i)` and `W(v) = Σ a_i W(S_i)`, with `a_i >= 0`, `Σ a_i = 1`.

Rest geometry, raster placement and transferred skin must use the same admitted support simplex. A CDT/kernel-emitted point does not gain product authority merely because it exists.

Full safe-S topology defines component identity **before** directional visibility clipping. Unsafe/UNKNOWN relations may not join components. Aggregate coverage may not hide loss of small mechanically/semantically distinct visible pieces. Mage hat/cape/book/wand remain explicit witnesses once fresh G/W exists.

## Runtime/export identity firewall

After product qualification, runtime/export may not:

- secretly retriangulate;
- create an alpha-clipped replacement mesh;
- barycentrically transfer mechanics onto a different topology;
- mint renderer-local topology authority;
- silently drop qualified visible components.

Frame 0 and dynamic proof must bind the same admitted S/G/W/M/B/component/motion lineage.

## Motion guard

Current `ROTATION_ONLY_CURRENT_PRESET_V1` is a **mechanical deformation probe**, not professional animation evidence.

Future professional motion should preserve the same proposal/Compiler distinction: authored or learned motion semantics may propose intent/curves, while deterministic qualification owns retarget legality, contacts, joint limits, root trajectory, attachment constraints and proof.

Do not call product motion closed before its own prereg/result/evidence transaction.

## FIT / scientific continuity rule

The scientific FIT1 gate began at commit:

`f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee`

The first executable FIT base is:

`de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3`

`canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md` is the semantic epoch map. `canonical/FIT1_COMMIT_LINEAGE_V1.md/.json` is exhaustive discovery/provenance for FIT1-descendant changes.

Historical FIT1 successes/failures remain evidence. A later product contradiction may narrow their product interpretation without rewriting the sealed scientific outcome.

FIT1 and same-Mage FIT2 are not unseen-family generalization.

## Scientific claim discipline

Never conflate:

- source exists;
- mechanism implemented;
- mechanism tested;
- formulation tested;
- contract/implementation PASS;
- real witness result PASS;
- promotion/refreeze;
- same-witness product closure;
- unseen-family generalization;
- full PRODUCT_PASS.

For every experiment claim answer:

1. What local question was asked?
2. What global program goal did it serve?
3. Which arms/controls were compared?
4. Which mechanisms were present/held fixed?
5. What could it falsify?
6. What could it not prove?
7. What exact result occurred under which gate/evidence?
8. Did it change architecture authority, product authority, or only isolate a variable?
9. What next dependency follows and why?

## Chronology rule — append-only live memory

Current chronology is append-only in `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`.

- New events require exact RFC3339 UTC timestamps.
- Corrections/retractions are new events; never silently rewrite history.
- Before ending a substantive session, append decisions changing architecture, experiment selection, promotion interpretation, product authority, next work or stop/go state.
- Record requests as requests; never upgrade them into PASS.
- The journal is continuity memory, not a miscellaneous diary.

## Census vs semantic memory

Never confuse discoverability with understanding.

- `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` is the high-signal artifact census.
- `canonical/FIT1_COMMIT_LINEAGE_V1.json` is chronological provenance.
- `canonical/CONTEXT_COVERAGE_AUDIT.md` applies the AOA disposition policy.
- Artifact coverage means recoverability, not scientific promotion.

For numerical/mechanistic historical claims, inspect exact source evidence rather than reconstructing from filename/memory.

## Dependency / model naming boundary

RealSaS-owned model/package/class names describe RealSaS responsibilities; they do not adopt external project branding merely because research inspired a clean-room mechanism.

- External names may appear in comparison reports, audits, bibliographies, preregistrations and historical scientific lineage when provenance matters.
- Direct incorporated dependencies keep upstream identity where legally/technically required.
- **DINO/DINOv2 is the current explicit direct model dependency** and is bound by `models/iris/v2/dinov2_foundation_v2.py` plus `THIRD_PARTY_NOTICES.md`.
- Any new direct external dependency requires `THIRD_PARTY_NOTICES.md` update before promotion.

See `docs/repository/DEPENDENCY_AND_IP_POLICY.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSE`.

## Branch rule

Only `main/CURRENT_STATE.md` is repository-wide continuation authority.

- Active executable branches are explicitly registered in `canonical/AUTHORITY_MAP_V1.json` and `CURRENT_STATE.md`.
- Current active branch: `repair/mage-full-subject-reclosure-20260912`.
- Unregistered non-main branches are evidence-only by safe default.
- Branch recency does not imply authority.
- Do not delete evidence branches automatically; classify/dispose first.
- Active-branch implementation does not silently merge/promote to main; implementation promotion is a separate transaction.

## Execution environment

Current authority workflows run on the user's local self-hosted runner:

- labels `self-hosted, linux, x64, realsas`;
- known runner `realsas-wsl-1660ti`;
- operator path `~/actions-runner`.

Do not create routine GitHub-hosted branch-push fan-out. Learned refits requiring larger VRAM may run externally/Colab, but artifacts/results must return and be sealed before claims.

## Completion transaction

A gate is not complete merely because a notebook/report exists. Closing a scientific/product gate requires reconciling, as applicable:

- preregistration/frozen thresholds;
- frozen source/apparatus;
- exact result/evidence/provenance;
- mandatory real V0..V7 evidence for current product stages;
- implementation/source commit;
- current experiment registry;
- append-only scientific journal;
- experiment authority ledger;
- architecture ledger when interpretation changed;
- machine authority/context state;
- `CURRENT_STATE.md` when stop/go changed;
- explicit promotion/refreeze/supersession where applicable.

A promotion additionally requires one obvious promoted source home, machine-verifiable evidence identity and regression protection.

The goal is not more documentation. The goal is **deterministic context reconstruction with scientific responsibility boundaries intact**.
