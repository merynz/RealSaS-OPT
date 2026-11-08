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

### Exact 4,363-asset baseline and 56,719 review stages, persisted (2026-11-08)

- Reproducible builder: `tools/corpus/truth_foundry/build_baseline_review_queue_v1.py`, consuming frozen Master3993 V2, Master3993 physical census and exact Core370 historical roster.
- Current immutable **preliminary** zip, Drive readback 774,121 bytes: https://drive.google.com/file/d/1lcwkos8BjSdbDSbXjfa5__WgNl-KgxTm/view
- Inside: `ALL_4363_DEEP_AUDIT_BASELINE_V1.csv` (4,363 source records), `ALL_4363_INDEPENDENT_REVIEW_QUEUE_V1.csv` (4,363×13 = 56,719 explicitly UNKNOWN review cells), SHA summaries. A local independently reproduced run produced **identical SHA256** for both CSV files. This is a complete *review task roster*, NOT completed visual reviews.
- Current measured Master physical cohort: 2,466 imported; 1,306 multi-mesh-object, 890 one-mesh-multiple-islands, 270 one-mesh-one-island. 1,884 action-positive, 1,428 image-reference-positive, 883 missing-image-ref-positive. These are **import observations**, not source rights, functional IK or semantics.
- Deep full compressed source receipt runner and 8 synthetic PASS tests are in PR #395; full 2,466 deep scene readback & private Drive archived run **not yet independently confirmed**. Its synthetic PR check is https://github.com/merynz/RealSaS-OPT/actions/runs/42513706525; private job was skipped on that PR event by design.
- Review ontology/schema: `canonical/REALSAS_TRUTH_FOUNDRY_PER_ASSET_VISUAL_SEMANTIC_COURT_V1.schema.json` — no semantic/2D/layer PASS without source-linked actual rendering/camera/reviewer/causal proof.


## Comprehensive evidence expansion — 264 facets across 14 courts (2026-11-08)

**Verified private Drive output:** https://drive.google.com/drive/folders/1kvIvxDrwNuE1qOlfs_zbQiQocGcNpbJk
**Frozen input source:** `SOURCE_RECORD_CENSUS_4363.csv`, SHA256 `b03fc7b378ab5e8f70bd83f55822372eb5516746cc3d0b96286a1e903d0d2d1f`.
**Catalog:** `canonical/REALSAS_TRUTH_FOUNDRY_COMPREHENSIVE_FACETS_V1.json`.
**Producer:** `tools/corpus/truth_foundry/comprehensive_facet_matrix_v1.py`.
**Regression:** `tests/corpus/test_comprehensive_facet_matrix_v1.py`.
**Hosted CI:** `.github/workflows/truth-foundry-comprehensive-facets-v1.yml`.

### Confirmed results (private `SUMMARY_V1.json` and source/receipt hashes)

- 3,993 Master + 370 Core original **source records**, not distinct characters or legally qualified samples.
- **264 separately owned evidence facets in 14 groups**, not merely the prior 13 broad review stages.
- **1,151,832 source × facet cells** with explicit importer-observation, historical ID/split, or UNKNOWN state; independent court UNKNOWN everywhere in this *new* matrix.
- **61,082 source × 14-group review task rows**, OPEN not conducted.
- **41,922 importer-numeric observation slots** (17 mapped counters × 2,466 Master physical imports). These are arithmetic observations only and cannot be counted as independently accepted head targets.
- **0** independently qualified head target courts, source-face semantic labels, approved drawable/edit-consequence tests, or training permissions minted in this producer.
- `ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz` (18,713,037 bytes on Drive) preserves one detailed record per asset with 264 states, separate observation value and actual independent-court status.
- `FACET_COVERAGE_BY_COHORT_V1.csv`, `GROUP_COVERAGE_BY_COHORT_V1.csv`, `ASSET_GROUP_REVIEW_QUEUE_V1.csv`, `ASSET_FACET_REVIEW_SUMMARY_V1.csv` and SHA manifest are immutable receipt outputs.

### 14 evidence domains (independent, source-addressed)

1. Identity / original SHA / creator / license / cross-format lineage / split and 169 protected external holdouts.
2. Original scene hierarchy, transforms, authored pose, rest frame, projection/camera matrix.
3. Physical character/vehicle/prop category, morphology, limb/wing/tail/shell and hybrid anatomies.
4. Source mesh, original polygon-corner and triangulation lineage, components, open/modular/nonmanifold, UDF/SDF authority and deform-ready topology.
5. Native rig constraints/controllers/IK/FK/driver evidence, skin weights, bind, and positive/negative deformation.
6. Motion clips, bone curves, contacts, speed, native-vs-baked provenance, grip and secondary behavior.
7. Materials, UV, pixel/texture dependencies, alpha/bleed, native RGBA and recolor intervention.
8. Bound source rendering, camera/depth/normals/face-ID/barycentrics, global ray-stack visibility, hidden areas.
9. IRIS Semantic face-addressed concept/instance/region, morphological roles, cross-view mask and unknown/negative labels.
10. IRIS Graphics source-bound 2D art, visual identity/style, source-vs-generated art and atlas checks.
11. Compiler.Graphics drawable candidate separation, keep/hide/isolate and edit-reveal causal proof, painter/z ordering.
12. CAA source appearance authority, alpha/confidence, seams, atlas hashes and revision stability.
13. TESSA/AXIS/MIRA/Compiler Mechanics control-to-surface tests and save/reopen/render/export roundtrip.
14. Qualification, heldout split, teacher/student leakage, legal rights, consumer/loss binding and eligible materialization.

### Honest boundaries and immediate next production step

The catalog provides **auditable completeness questions, not completed truth acquisition**. The majority of 1,151,832 cells remain UNKNOWN and none of the new independent courts has been executed by this producer. For genuine “as if individually inspected” data:

1. Read the complete 2,466 original imported-scene receipts from `deep_source_audit_v1.py`; do not substitute 22 numeric source counters for original polygons, source face IDs, or rigs. Independently recover the 1,527 unobserved or out-of-scope sources in lineage-safe order.
2. Generate true source-bound rest/pose/camera render and evidence contact sheets: native RGB/alpha, semantic candidate ID, face/depth/normal/UV, joint/skin and attachment overlays, multiple viewpoints, no false front or anatomy.
3. Review category/morphology/pose/2D style and source semantic face regions with explicit signed reviewer receipts. StdGEN++ semantic full/keep/hide/isolate is a source-derived *teacher mechanism*, not semantic ontology or automatic drawable authority.
4. Run behavior intervention courts: color/hide/transform/deform/detach, full and newly revealed source surfaces, all-hit global occlusion, compositor equivalence, attachment and negative-control consequences.
5. Independently bind each accepted target to Geometry/Semantic/Graphics/TESSA/AXIS/MIRA consumer losses and verify source rights, lineage and split. Only then report a >0 qualified count or materialization eligibility.

**Do not equate the 264 facet definition count or 41,922 import observations with 264 learned labels or 41,922 head-ready targets.**


### Verified full imported-scene readback (independent from aggregate import counts)

Private run receipt: https://drive.google.com/drive/folders/17avud3OMcgAV_klJp2D9QEQtpToQGqtE

- `MASTER2466_FULL_SOURCE_DEEP_AUDIT_V1.jsonl`: **2,466 imported-scene full compressed JSON receipt SHAs independently checked** against associated summary and per-asset source IDs; 1,527 Master records did not have corresponding imported-scene receipts.
- `MASTER3993_DEEP_AUDIT_COVERAGE_V1.csv` and `MASTER3993_ALL_STAGE_REVIEW_TASKS_V1.csv` (54,985 open prior-stage tasks) retained; source category, anatomical pose, genuine native IK, 2D style, semantic and drawable courts remain zero PASS in this audit.
- Structural, **not semantic**, breakdown for 3,993 Master: 270 one object/one connected surface, 890 one object/multiple disconnected islands, 95 multi-object/component-balanced, 1,178 multi-object/multiple islands, 33 importer component count inconsistent (review), 1,527 without imported-scene evidence.
- This readback **does not rehash original 3D asset bytes**, perform human visual inspection or establish source-native authoring semantics. It validates archived imported-scene observations.

**Do not combine** the 54,985 old stage tasks with the new 61,082 broader 14-group review queue as independent asset counts; they are alternative overlapping audit projections.
