# Corpus370 V5 delta and Stage50 source binding checkpoint — 2026-11-04

**Status: real observation/source-packet repair in progress; no family/model training authority.** PR [#327](https://github.com/merynz/RealSaS-OPT/pull/327) remains draft and `main` unchanged. Do not merge until peer review / complete end-to-end research Attempt.

## Stable original parents, never rewrite

- 370 exact source subjects: 294 completed historical observation packets, 76 original observation preparation failures (**47 mask, 29 bind-vertex**). Downstream 294 compute inputs previously comprised 268 computed-but-unqualified and 26 execution failures; counts are separate.
- Exact historical V3 index SHA: `5007b0863eddce70833ad48c4bc26af92a7abd79b618c8dc5c69ade7f43840a7`. Its original V2 parent: `129b173b5d789cf4dc020f883ee8b16db8156380dd0cae939319ad3c279c139a`.
- Roster SHA `925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355`; source archive SHA `5c92e77ae07b215fa4234d221faf27659bc0c47a54d2711d6e49db3d820b8fa9`.
- **Read-only real independent 294/294 old PASS audit** [Actions 41801692680](https://github.com/merynz/RealSaS-OPT/actions/runs/41801692680): exact original `imported.json`, all eight native PNG/masks, camera, scene snapshot and admission hashes PASS. No rerender or new qualification, 0 unproven old successes.

## Real source 8-view V4 and V5 delta pilot

- Exact original FBX Dragon and Swordfish [Actions 41801173925](https://github.com/merynz/RealSaS-OPT/actions/runs/41801173925): eight actual source views each, with unchanged original `maximum_mask_gap_px=2`, source/import hash-bound geometry and original Blender runtime. Dragon relative vertex error 2.42e-7, max mask gap 1.414 px; Swordfish 4.35e-8 and 1 px. **Observation PASS only**, no Platform Attempt.
- Real V5 delta coordinator [Actions 41802511152](https://github.com/merynz/RealSaS-OPT/actions/runs/41802511152) emitted **294 original V3 PASS + 2 distinct V4 repairs + 74 preserved original FAIL = 370**. Every new row stores original old FAIL receipt and exact V4 producer binding, while original PASS is explicitly `HISTORICAL_V3_PASS_BYTES_RECHECKED`. All old failure receipts preserved.
- **Full 76-original-FAIL real repair running:** [Actions 41802949504](https://github.com/merynz/RealSaS-OPT/actions/runs/41802949504). The self-hosted runner rehashes 294 old PASS and attempts only 76 old failures with exact pinned Blender + V2 FBX reader + V3 uncertainty-bounded precision court. The two proven pilot V4 packets reuse only when their hashes/code binding still match. **No final new PASS count claimed until full output readback.**
- V5 output lives under `/home/monster/realsas_authority/corpus370_v5_delta_repair_pilot/CORPUS370_PREPARED_OBSERVATION_PACKETS_V5.json` on the research runner; the workflow will upload the typed final index + failure receipts as an Actions artifact. A partial or failed run cannot silently promote a subject.

## Transport and Platform interface

- `tools/corpus/prepare_corpus370_platform_truth_bundle_v3.py`: new **370-universe source/provenance/observation family bundle V3**. Accepts independent V4-only or composite V5 observation index; for historical V3 PASS pulls *original* `objects/<id>/imported/<sid>/imported.json` and separately rehashes V3 packet/source; for new V4 PASS validates every view, candidate, camera, snapshot, import SHA and old failure receipt. Preserves explicit failed subjects. Binds `observations/<sid>/imported.json` exact bytes into stage `source_surface` row. Does not mutate V2 transport.
- `tools/platform_source_surface_cohort_bootstrap_v3.py` with `canonical/CORPUS370_SOURCE_SURFACE_TRUTH_CAMPAIGN_V3.json`: projects only observation-PASS subjects into **new** `CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V3.json` (Stage50 independent source surface, Stage51 measured capabilities, Stage49 separate family evidence) and binds exact original imported geometry into each subject's `source_surface.admitted_import_geometry`. No product/training authority, no full Platform release yet.
- New synthetic 370-case transport and V5 mixed-producer tests, including corrupted imported JSON, image, per-view mask, changed old policy, forged producer, corrupted original historical parent, and source lineage must fail closed. CI `corpus370-repair-smoke-v1` and `corpus370-postrun-regressions-v1` cover the released pieces.
- Independently courted real two-subject source-rest geometry pilot [Actions 41803626856](https://github.com/merynz/RealSaS-OPT/actions/runs/41803626856) was created; treat as **pending until completed**. It is not the complete Stage50 geometry/normal family or a Platform Attempt.

## Still open before repair notebook/materialization/training

1. Complete **all 76** failed-preparation V4 repair readback and classify remaining typed failures; do not lower 2px or 1e-5 thresholds. If some truly fail, preserve them.
2. Build/reopen the full V3 family bundle from the final V5 index (not yet executed), then issue exact Platform EngineRelease and real child Attempts against V3 graph. Old calibration failures must not erase Stage50 source surface, missing control families stay MASKED.
3. Integrate independent TESSA V5 source-relative sampled-contact court into *new* teacher consumer and inspect genuine sampled source replay; not an automatic collision-freedom PASS.
4. Complete source→canonical target projection and independent per-head true loss masks for IRIS/TESSA; local oriented probe never equals global signed distance.
5. Only afterward release the **370 repair/revalidation Run All Colab notebook** with SHA-bound parent/child receipts and resumable subject-level outcomes. Inspect notebook execution outcomes before separate materialization/lineage split/readiness and training.

A green CI measures code regression, **not** the number of subjects eligible to train.
