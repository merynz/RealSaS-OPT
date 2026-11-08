# RealSaS/BUSTED — Truth Foundry: reproducible source census and six-head matrix V1

## Purpose

**This branch is the persistent handoff**. Do not rely on a chat session or local notebook.

The goal is the **final fully qualified, lineage-safe corpus** for IRIS Geometry/Semantic/Graphics, TESSA, AXIS and MIRA. This V1 implementation is the inventory and head-target routing **foundation**, NOT completion of independent source/semantic/graphics/rights courts. It does not enable training or materialization.

### Authoritative files

- `tools/corpus/truth_foundry/physical_census_v1.py`: evidence-only, original source SHA-bound importer census for all *2,466* currently observed Master subjects. Distinguishes **mesh objects vs disconnected mesh components**, rest bones, sparse skin rows, materials, images/missing dependencies, action curves, constraints and modifiers. Missing import observations stay blank/unknown, never zero.
- `tools/corpus/truth_foundry/target_matrix_v1.py`: **3,993 Master + 370 Core = 4,363 source records × 37 proposed targets = 161,431 target slots**, including source facet, court needed, consumer/head, evidence-observation status and never-open training mask.
- `tests/corpus/test_truth_foundry_v1.py`: fail-closed regression tests.
- `.github/workflows/truth-foundry-source-census-v1.yml`: hosted Actions tests, private Drive receipt readback, exact source join, source census and matrix; outputs to `gdrive:RealSaS_MASTER_CORPUS_1024_V3/audits/truth_foundry_global_source_census_v1/<run_id>-<attempt>/`.

### Frozen input provenance

- Master3993 V2: `audits/global_wave4_truth_reconciliation_v2/42413075814-1/`, 2,466 import observations.
- Master/Core historical family, Core370 exact 370 machine roster, and 1,024 dense source arithmetic: `audits/global_existing_truth_inventory_v1_20261107/`.
- Wave1/2/3/4 original import summary receipts: `audits/wave{1,2,3,4}_source_rig_scene_v1/<verified_run_id>-1/`. Their **original source geometry is not fetched** by this audit.
- Cross-corpus 53 byte-identical edges are established in PR #393 but **not yet embedded as per-record overlay** here. Thus 4,310 = provisional source identity groups, not unique characters.

### Head ownership and high-risk policy

IRIS Semantic learns source-addressed **concept/instance/region** labels from independently reviewed evidence. A mesh object, material name, or file name alone is neither semantic source truth nor a drawer. `drawable_necessity_three_way` and `layer_source_visibility_relation` in the 37-target proposal are **Compiler.Graphics qualification/evidence routes**, not default semantic class logits. IRIS Graphics carries source appearance, visibility and candidate layer evidence; Compiler.Graphics qualifies and compiles executable drawable decomposition. Treat physical observations, capability contracts and representation policy separately.

### Remaining P0 tasks (in order)

1. Reconcile PR #393's 53 per-record SHA identity edges, cross-format/lineage duplicates, 15 QA-only and 28 rejected/repair-only records without contaminating train/test.
2. Independently classify all actual **character vs prop**, morphology, initial pose, anatomical front/up, source mesh parts and source authoring transformations. Use texture dependency resolution, materials and UV/polygon-corner mapping; separate source truth vs importer observations.
3. Produce source-face-addressed semantic regions with reviewed concepts and **explicit abstain/unlabeled**; compare group proposals from `models/iris/truth/semantic_source_groups_v1.py` (PR #377), independent cross-view raster masks, source object/material teacher hints quarantined from student inputs.
4. Build independent **Graphics** tests: native RGBA, approved source-paired stylized 2D art, full/off/isolated/reveal visibility, edit consequences, layer candidate graph, occlusion & draw-order, canonical camera/pose, CAA hash/alpha invariants.
5. Verify TESSA bend/twist/deformation-ready topology, AXIS executable rig/IK/constraints, MIRA sparse skin and negative deform regions, with each teacher bound to source SHA and independent court evidence.
6. Only then: target-specific PASS/MASKED/UNKNOWN/FAIL; rights and split qualification; immutable materialization bundles; training.

### Rules

- Never equate `SOURCE_IMPORT_OBSERVED` with independent family PASS or commercial training rights.
- Never treat missing IK/skin/animation/texture count as a negative semantic label or generic morphological fact.
- Never take one canonical front render as a substitute for 3D source-bound multi-view/deformation supervision.
- Do not promote PR #327 historical Core370 family pass without current per-asset court readback.
- Never silently merge candidate/QA/technical-reject rows with the 4,363 source records.

## Resume in a new chat

Read this file, review the Actions run and its private Drive SHA summaries, then advance the P0 courts from the **actual** matrix. Every qualified change must be a GitHub commit and an immutable private Drive receipt. No local/chat-only state is authoritative.
### First persistent matrix checkpoint (2026-11-08)

- Private Drive folder: https://drive.google.com/drive/folders/1UZt6hIGDOz4Kh1gWJYLGjy_cvLvcHNJA
- Verified downloaded 1,950,745-byte ZIP: https://drive.google.com/file/d/1i1MOeb8uB9kIA2O2GNOlibs6zZzqQLwN/view
- Contains V1 **PRELIMINARY** 4,363 record census, 37-target × 161,431 proposed route matrix, summary SHA digests, plus historic 4,363-ledger inventory. `master_per_asset_physical_census_loaded=0` in this checkpoint because Wave1–4 summary receipts have not yet been rejoined in the remote physical source census. Do not call this fully materialized or training-ready.
- GitHub regression run: https://github.com/merynz/RealSaS-OPT/actions/runs/42509635548, `local-contract-regressions` **SUCCESS**, `private-source-truth-matrix` **SKIPPED** on PR by design. Do not claim full original source census has run based on that result.

### Per-head 8K strategic target (2026-11-08)

- Superseding contract: `canonical/REALSAS_TRUTH_FOUNDRY_8K_PER_HEAD_COVERAGE_CONTRACT_V2.json` (previous V1 5K preserved as history).
- The planned goal is **8,000 independent eligible FIT source identity clusters per head**, after real source rights/lineage/target-specific courts; not 8,000 files or copies of clips. Applies to IRIS Geometry, IRIS Semantic, IRIS Graphics, TESSA, AXIS, MIRA separately; advanced targets (RigProgram, edit-reveal, drawable consequence) counted separately.
- Assumes Anymate or equivalent large source pool becomes genuinely usable; the paper reports 230,716 asset entries but this is not 230,716 licensed RealSaS source scenes. HF processed data exposes pointcloud/voxel/joints/connectivity/skin primarily, not assured 2D art/source-texture/semantic/full native controls.
- No fabricated admission to meet 8K; all existing head-core admitted counts remain 0 in this proposal. Preserve independent 2D visual qualification and source-rights gate.

### Full-scene deep audit and mandatory visual review (2026-11-08)

- New producer: `tools/corpus/truth_foundry/deep_source_audit_v1.py`; synthetic fail-closed tests `tests/corpus/test_deep_source_audit_v1.py`; GitHub Actions `.github/workflows/truth-foundry-deep-source-audit-v1.yml`.
- Scope: reads complete imported-source scene receipts `asset_*.full.source.json.gz` SHA-checked against the 2,466 imported `summary.json` receipts and Master physical census, and emits full per-asset JSONL + 3,993-row structured coverage table + independent-review task roster **privately in Drive**. The original source model bytes/actual source-authoring controls and renders are not silently inferred.
- First run: https://github.com/merynz/RealSaS-OPT/actions/runs/42513706525 — synthetic-contracts SUCCESS, private exact-deep-readback SKIPPED on PR by deliberate push-only trust guard. The complete deep Drive readback is not yet verified; do **not** call 2,466 deep JSONs completed on that basis.
- Actual existing physical census measurements: 2,466 imported Master, 1,306 multi-mesh-object, 890 single-mesh multi-island, 270 single-mesh single-island; **none are proven semantic parts/drawable units**. All 3,993 currently mark `character_vs_prop=UNKNOWN`, `morphology=UNKNOWN`, `approved_stylized_2d_truth=UNKNOWN`, `native_control_functionality=UNKNOWN`.
- Target review taxonomy & proof rubric: `docs/research/TRUTH_FOUNDRY_DEEP_SOURCE_AUDIT_SCOPE_V1.md`, including human/humanoid/animal/monster/mech/tank/aircraft/prop classifications, limbs/wings/tail, initial source pose, canonical camera, attachments, texture, 2D style parity, editable layer/occlusion consequences, native controls, independent source-face semantic regions.
- NEXT: inspect actual hosted private run output; then generate source-bound visual/semantic/geometry and edit-consequence review artifacts. Do not turn unknown into a negative class and do not raise head PASS merely by creating a metadata row.
